//! Background command queue, collected on the UI thread.
use crate::PhotocraftApp;
use photocraft_doc::Document;
use photocraft_engine::{
    Result,
    jobs::{PreparedJob, Progress},
};
use serde_json::Value;
use std::sync::{
    Arc,
    mpsc::{self, Receiver, TryRecvError},
};

struct Running {
    before: Arc<Document>,
    label: String,
    progress: Progress,
    receiver: Receiver<(PreparedJob, Result<Value>)>,
}
pub(crate) struct Jobs {
    running: Vec<Running>,
    pub force_sync: bool,
    pub io: crate::io_jobs::IoJobs,
}
// Derivable only outside `cfg(test)`, where `force_sync` is true.
#[allow(clippy::derivable_impls)]
impl Default for Jobs {
    fn default() -> Self {
        // This crate's unit tests drive commands and read their results in the same call:
        // they run synchronously there. The app (and integration tests) use background jobs.
        Self { running: Vec::new(), force_sync: cfg!(test), io: Default::default() }
    }
}
impl Jobs {
    pub fn start(&mut self, mut prepared: PreparedJob) -> std::result::Result<Value, String> {
        let progress = Progress::default();
        let before = prepared.before.clone();
        let label = prepared.label.clone();
        let (sender, receiver) = mpsc::channel();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let handle = progress.clone();
            std::thread::Builder::new()
                .name("photocraft-command".into())
                .spawn(move || {
                    let result = prepared.run(&handle);
                    let _ = sender.send((prepared, result));
                })
                .map_err(|e| format!("could not start background work: {e}"))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            let result = prepared.run(&progress);
            let _ = sender.send((prepared, result));
        }
        self.running.push(Running { before, label, progress, receiver });
        Ok(serde_json::json!({"queued": true}))
    }
}
impl Drop for Jobs {
    fn drop(&mut self) {
        for job in &self.running {
            job.progress.cancel();
        }
    }
}

impl PhotocraftApp {
    pub(crate) fn poll_jobs(&mut self, ctx: &egui::Context) {
        // Esc is read once and cancels what the progress windows show: every file operation and
        // the command job on display (the active document's, else the first).
        let busy = !self.jobs.running.is_empty() || self.jobs.io.is_running();
        let cancel = busy && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
        let mut io = std::mem::take(&mut self.jobs.io);
        io.poll(self, ctx, cancel);
        self.jobs.io = io;
        let active = self.session.active().map(|st| st.doc.id);
        let shown = self.jobs.running.iter().position(|r| active == Some(r.before.id)).unwrap_or(0);
        let mut pending = Vec::new();
        for (index, running) in std::mem::take(&mut self.jobs.running).into_iter().enumerate() {
            if cancel && index == shown {
                running.progress.cancel();
            }
            match running.receiver.try_recv() {
                Ok((prepared, result)) => {
                    // Esc can arrive after the worker sent its result but before this frame
                    // collects it. Cancelled snapshots must never reach history or the canvas.
                    let result = if running.progress.is_cancelled() { Err(photocraft_engine::EngineError::Other("cancelled".into())) } else { result };
                    match self.session.finish_job(prepared, result) {
                        Ok(_) => {
                            self.ui.status = format!("{} complete", running.label);
                            self.ui.status_error = false;
                            self.sync_views();
                        }
                        Err(error) => {
                            self.ui.status = error.to_string();
                            self.ui.status_error = true;
                        }
                    }
                }
                Err(TryRecvError::Empty) => pending.push(running),
                Err(TryRecvError::Disconnected) => {
                    self.session.release_job(&running.before);
                    self.ui.status = "Background worker stopped; the document is unchanged".into();
                    self.ui.status_error = true;
                }
            }
        }
        self.jobs.running = pending;
        if let Some(running) = self.jobs.running.iter().find(|r| active == Some(r.before.id)).or_else(|| self.jobs.running.first()) {
            let progress = running.progress.clone();
            egui::Window::new("Working")
                .id(egui::Id::new("photocraft_jobs"))
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -36.0])
                .show(ctx, |ui| {
                    ui.label(&running.label);
                    ui.add(egui::ProgressBar::new(progress.fraction()).show_percentage());
                    if ui.button("Cancel (Esc)").clicked() {
                        progress.cancel();
                    }
                });
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }
}

#[cfg(test)]
mod bughunt_ps {
    use super::*;
    use serde_json::json;

    #[test]
    fn late_cancel_after_complete_never_commits_200_iterations() {
        for _ in 0..200 {
            let mut session = photocraft_engine::Session::new();
            session.execute("file.new", json!({"width":17,"height":23})).unwrap();
            session.execute("layer.new.layer", json!({})).unwrap();
            session.execute("paint.stroke", json!({"points":[[2,3],[14,19]],"size":7})).unwrap();
            let mut prepared = session.prepare_job("filter.blur.gaussianBlur", json!({"radius":1})).unwrap().unwrap();
            let before = prepared.before.clone();
            let revision = session.active().unwrap().revision;
            let progress = Progress::default();
            let result = prepared.run(&progress);
            assert!(result.is_ok());
            let (sender, receiver) = mpsc::channel();
            assert!(sender.send((prepared, result)).is_ok());
            // Worker completed and sent before cancellation; delivery is pending.
            progress.cancel();
            let mut app = PhotocraftApp::new(session, Default::default());
            app.jobs.running.push(Running { before: before.clone(), label: "blur".into(), progress, receiver });
            app.poll_jobs(&egui::Context::default());
            assert!(app.jobs.running.is_empty());
            assert_eq!(app.session.active().unwrap().doc, before);
            assert_eq!(app.session.active().unwrap().revision, revision);
            assert!(!app.session.active().unwrap().busy);
        }
    }
}
