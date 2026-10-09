//! Snapshot based document jobs. The worker never owns the live session.
use crate::{EngineError, Result, Session};
pub use photocraft_algo::progress::{Progress, with_progress};
use photocraft_doc::{Document, LayerId};
use serde_json::{Value, json};
use std::sync::Arc;

type Work = Box<dyn FnOnce(&mut Document, &mut Option<LayerId>, &Progress) -> Result<Value> + Send>;
pub struct Job {
    pub label: String,
    work: Work,
}
impl Job {
    pub fn new(label: impl Into<String>, work: impl FnOnce(&mut Document, &mut Option<LayerId>, &Progress) -> Result<Value> + Send + 'static) -> Self {
        Self { label: label.into(), work: Box::new(work) }
    }
    pub fn run(self, doc: &mut Document, active: &mut Option<LayerId>, progress: &Progress) -> Result<Value> {
        check(progress)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            photocraft_algo::progress::with_progress(progress, || (self.work)(doc, active, progress))
        }))
        .unwrap_or_else(|_| Err(EngineError::Other("background work failed with an internal error; the document is unchanged".into())));
        check(progress)?;
        if result.is_ok() {
            progress.set_fraction(1.0);
        }
        result
    }
}
pub fn check(progress: &Progress) -> Result<()> {
    if progress.is_cancelled() { Err(EngineError::Other("cancelled".into())) } else { Ok(()) }
}
pub fn enabled() -> bool {
    std::env::var("PHOTOCRAFT_JOBS").as_deref() != Ok("0")
}
pub fn supports(id: &str) -> bool {
    id.starts_with("filter.")
        || matches!(
            id,
            "image.imageSize"
                | "image.canvasSize"
                | "layer.mergeDown"
                | "layer.mergeVisible"
                | "layer.mergeLayers"
                | "layer.flattenImage"
                | "layer.rasterize.smartObject"
                | "layer.smartObjects.rasterize"
        )
}
/// Internal id is supplied by `Session::prepare_job`, never trusted from command params.
pub fn prepare_command(session: &Session, params: &Value) -> Result<Job> {
    let id = params.get("__job_command").and_then(Value::as_str).ok_or_else(|| EngineError::Other("missing background command id".into()))?.to_owned();
    let spec = crate::commands::find(&id).ok_or_else(|| EngineError::UnknownCommand(id.clone()))?;
    let mut params = params.clone();
    if let Some(map) = params.as_object_mut() {
        map.remove("__job_command");
    }
    // Ordinary algorithm filters resolve session dependencies before leaving the UI thread.
    if crate::filters::params_for(&id, &params).is_some() {
        return crate::filters::prepare_job(session, &id, &params);
    }
    // Background commands act on the active document only: copy just that one (its tiles
    // are shared copy-on-write), without history, journal or the other open documents.
    let mut isolated = Session::new();
    let mut st = session.active().ok_or(EngineError::NoDocument)?.clone();
    st.busy = false;
    st.history = Default::default();
    isolated.docs = vec![st];
    isolated.active = Some(0);
    isolated.tools = session.tools.clone();
    isolated.clipboard = session.clipboard.clone();
    let mut prefs = session.prefs.get().clone();
    prefs.script_events.enabled = false;
    isolated.edit_prefs(|p| *p = prefs);
    Ok(Job::new(spec.label, move |doc, active, progress| {
        check(progress)?;
        let st = isolated.active_mut().ok_or(EngineError::NoDocument)?;
        st.doc = Arc::new(doc.clone());
        st.active_layer = *active;
        let result = isolated.execute(&id, params)?;
        check(progress)?;
        let st = isolated.active().ok_or(EngineError::NoDocument)?;
        *doc = (*st.doc).clone();
        *active = st.active_layer;
        Ok(result)
    }))
}

pub struct PreparedJob {
    pub before: Arc<Document>,
    pub after: Document,
    pub active: Option<LayerId>,
    pub job: Option<Job>,
    pub label: String,
    pub command: String,
    pub params: Value,
    restrict: Option<usize>,
    coalesce: Option<String>,
}
impl Session {
    pub fn prepare_job(&mut self, id: &str, params: Value) -> Result<Option<PreparedJob>> {
        if !enabled() {
            return Ok(None);
        }
        let spec = crate::commands::find(id).ok_or_else(|| EngineError::UnknownCommand(id.into()))?;
        let Some(prepare) = spec.job else { return Ok(None) };
        if !self.is_enabled(id) {
            return Err(EngineError::Disabled(id.into(), "document is busy or command is unavailable".into()));
        }
        let (before, after, active) = self.begin_edit()?;
        let mut run_params = crate::channel_cmds::inject_target(self, id, crate::commands::inject_kind(id, params.clone()));
        let restrict = crate::channel_cmds::color_restriction(self, id, &run_params);
        if !run_params.is_object() {
            run_params = json!({});
        }
        if let Some(map) = run_params.as_object_mut() {
            map.insert("__job_command".into(), json!(id));
        }
        let job = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prepare(self, &run_params)))
            .unwrap_or_else(|_| Err(EngineError::Other("could not prepare background work".into())))?;
        self.active_mut().ok_or(EngineError::NoDocument)?.busy = true;
        Ok(Some(PreparedJob {
            before,
            after,
            active,
            label: job.label.clone(),
            job: Some(job),
            command: id.into(),
            coalesce: params.get("coalesce").and_then(Value::as_str).map(str::to_owned),
            params,
            restrict,
        }))
    }
    pub fn release_job(&mut self, before: &Document) {
        if let Some(st) = self.docs.iter_mut().find(|st| st.doc.id == before.id) {
            st.busy = false;
        }
    }
    pub fn finish_job(&mut self, prepared: PreparedJob, result: Result<Value>) -> Result<Value> {
        self.release_job(&prepared.before);
        let value = result?;
        self.color_restrict = prepared.restrict;
        self.coalesce_request = prepared.coalesce;
        let doc_id = prepared.before.id;
        let committed = self.commit_edit(prepared.before, prepared.after, prepared.active, &prepared.label);
        self.color_restrict = None;
        self.coalesce_request = None;
        committed?;
        if let Some(st) = self.docs.iter_mut().find(|st| st.doc.id == doc_id) {
            st.log_damage(st.last_damage);
        }
        let previous = self.active().map(|st| st.doc.id);
        self.active = self.docs.iter().position(|st| st.doc.id == doc_id);
        crate::edit_menu_cmds::after_command(self, &prepared.command);
        crate::automate_cmds::after_command(self, &prepared.command);
        self.active = previous.and_then(|id| self.docs.iter().position(|st| st.doc.id == id));
        self.sync_preset_store();
        if !crate::brush_cmds::coalesce_journal(self, &prepared.command, &prepared.params) {
            self.journal.push((prepared.command, prepared.params));
        }
        Ok(value)
    }
}

impl PreparedJob {
    pub fn run(&mut self, progress: &Progress) -> Result<Value> {
        let job = self.job.take().ok_or_else(|| EngineError::Other("background job already ran".into()))?;
        job.run(&mut self.after, &mut self.active, progress)
    }
}
