//! Worker-owned import/export services. Existing automation services remain synchronous.
use crate::{BackgroundImportFn, BackgroundSaveFn, PhotocraftApp, file_open, notices};
use photocraft_doc::{DocId, Document};
use photocraft_engine::jobs::Progress;
use std::sync::{
    Arc,
    mpsc::{self, Receiver, TryRecvError},
};

enum Message {
    Preview(Document),
    Opened(Result<(Document, Vec<String>), String>),
    Saved(Result<Vec<String>, String>),
}
enum Operation {
    Open { preview: Option<DocId> },
    Save { before: Arc<Document>, revision: u64 },
}
struct Running {
    path: String,
    progress: Progress,
    receiver: Receiver<Message>,
    operation: Operation,
    initial_active: Option<DocId>,
}
#[derive(Default)]
pub(crate) struct IoJobs {
    running: Vec<Running>,
}
impl Drop for IoJobs {
    fn drop(&mut self) {
        for job in &self.running {
            job.progress.cancel();
        }
    }
}

impl IoJobs {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(&mut self, path: String, import: BackgroundImportFn, initial_active: Option<DocId>) -> Result<(), String> {
        let (sender, receiver) = mpsc::channel();
        let progress = Progress::default();
        let task_progress = progress.clone();
        let task_path = path.clone();
        std::thread::Builder::new()
            .name("photocraft-import".into())
            .spawn(move || {
                let preview = |document| {
                    let _ = sender.send(Message::Preview(document));
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| import(&task_path, &task_progress, &preview)))
                    .unwrap_or_else(|_| Err("import failed with an internal error".into()));
                let _ = sender.send(Message::Opened(result));
            })
            .map_err(|e| format!("could not start import: {e}"))?;
        self.running.push(Running { path, progress, receiver, operation: Operation::Open { preview: None }, initial_active });
        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&mut self, before: Arc<Document>, revision: u64, path: String, compact: bool, save: BackgroundSaveFn) -> Result<(), String> {
        let (sender, receiver) = mpsc::channel();
        let progress = Progress::default();
        let task_progress = progress.clone();
        let task_path = path.clone();
        let task_doc = before.clone();
        std::thread::Builder::new()
            .name("photocraft-save".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| save(&task_doc, &task_path, compact, &task_progress)))
                    .unwrap_or_else(|_| Err("save failed with an internal error; destination preserved".into()));
                let _ = sender.send(Message::Saved(result));
            })
            .map_err(|e| format!("could not start save: {e}"))?;
        self.running.push(Running { path, progress, receiver, operation: Operation::Save { before, revision }, initial_active: None });
        Ok(())
    }
    pub fn poll(&mut self, app: &mut PhotocraftApp, ctx: &egui::Context) {
        let cancel = !self.running.is_empty() && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
        let mut pending = Vec::new();
        for mut running in std::mem::take(&mut self.running) {
            if cancel {
                running.progress.cancel();
            }
            let mut complete = false;
            loop {
                match running.receiver.try_recv() {
                    Ok(Message::Preview(mut document)) => {
                        if running.progress.is_cancelled() {
                            continue;
                        }
                        if let Operation::Open { preview } = &mut running.operation {
                            if preview.is_some() {
                                continue;
                            }
                            document.name = file_open::display_name(&running.path);
                            let previous = app.session.active().map(|st| st.doc.id);
                            app.session.add_document(document, None);
                            if let Some(st) = app.session.active_mut() {
                                st.busy = true;
                                *preview = Some(st.doc.id);
                            }
                            if previous != running.initial_active
                                && let Some(index) = app.session.documents().iter().position(|st| Some(st.doc.id) == previous)
                            {
                                app.session.set_active(index);
                            }
                            app.sync_views();
                        }
                    }
                    Ok(Message::Opened(result)) => {
                        let active = app.session.active().map(|st| st.doc.id);
                        let preview_id = match running.operation {
                            Operation::Open { preview } => preview,
                            _ => None,
                        };
                        if let Some(id) = preview_id {
                            if let Some(index) = app.session.documents().iter().position(|st| st.doc.id == id) {
                                if let Some(st) = app.session.documents().get(index) {
                                    let before = st.doc.clone();
                                    app.session.release_job(&before);
                                }
                                app.session.close(index);
                            }
                        }
                        match result {
                            Ok((document, warnings)) if !running.progress.is_cancelled() => {
                                if app.open_imported(&file_open::display_name(&running.path), document, warnings).is_ok() {
                                    if let Some(st) = app.session.active_mut() {
                                        st.path = Some(running.path.clone());
                                    }
                                    app.push_recent(&running.path);
                                }
                            }
                            Ok(_) => {
                                app.ui.status = "Opening cancelled".into();
                                app.ui.status_error = false;
                            }
                            Err(_) if running.progress.is_cancelled() => {
                                app.ui.status = "Opening cancelled".into();
                                app.ui.status_error = false;
                            }
                            Err(error) => app.open_failed(&file_open::display_name(&running.path), &error),
                        }
                        if active != preview_id
                            && active != running.initial_active
                            && let Some(index) = app.session.documents().iter().position(|st| Some(st.doc.id) == active)
                        {
                            app.session.set_active(index);
                        }
                        app.sync_views();
                        complete = true;
                        break;
                    }
                    Ok(Message::Saved(result)) => {
                        if let Operation::Save { before, revision } = &running.operation {
                            app.session.release_job(before);
                            match result {
                                Ok(warnings) => {
                                    let index = app.session.documents().iter().position(|st| st.doc.id == before.id);
                                    if let Some(index) = index {
                                        let old_active = app.session.active_index();
                                        app.session.set_active(index);
                                        if let Some(st) = app.session.active_mut()
                                            && Arc::ptr_eq(&st.doc, before)
                                            && st.revision == *revision
                                        {
                                            st.path = Some(running.path.clone());
                                            st.saved_revision = *revision;
                                        }
                                        photocraft_engine::automate_cmds::document_saved(&mut app.session, index);
                                        if let Some(index) = old_active {
                                            app.session.set_active(index);
                                        }
                                    }
                                    app.ui.status = format!("Saved {}", running.path);
                                    app.ui.status_error = false;
                                    notices::io_warnings(app, &format!("Saved {}", file_open::display_name(&running.path)), &warnings);
                                }
                                Err(error) => {
                                    app.ui.status = error;
                                    app.ui.status_error = true;
                                }
                            }
                        }
                        complete = true;
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        match &running.operation {
                            Operation::Save { before, .. } => app.session.release_job(before),
                            Operation::Open { preview: Some(id) } => {
                                if let Some(index) = app.session.documents().iter().position(|st| st.doc.id == *id) {
                                    if let Some(st) = app.session.documents().get(index) {
                                        let before = st.doc.clone();
                                        app.session.release_job(&before);
                                    }
                                    app.session.close(index);
                                    app.sync_views();
                                }
                            }
                            Operation::Open { preview: None } => {}
                        }
                        app.ui.status = "File worker stopped unexpectedly".into();
                        app.ui.status_error = true;
                        complete = true;
                        break;
                    }
                }
            }
            if !complete {
                pending.push(running);
            }
        }
        self.running = pending;
        if let Some(running) = self.running.first() {
            egui::Window::new("File operation").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_BOTTOM, [0.0, -36.0]).show(ctx, |ui| {
                ui.label(file_open::display_name(&running.path));
                ui.add(egui::ProgressBar::new(running.progress.fraction()).animate(true));
                if ui.button("Cancel (Esc)").clicked() {
                    running.progress.cancel();
                }
            });
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }
}
