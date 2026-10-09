//! Latest-wins viewport filter work. Only the target's pixels change; ordinary GPU/CPU
//! compositing retains the complete layer tree, masks, blend modes and layer effects.
use photocraft_doc::{Document, LayerContent, LayerId};
use photocraft_engine::jobs::Progress;
use photocraft_geom::Rect;
use serde_json::Value;
use std::sync::{Arc, mpsc};

pub fn supported(doc: &Document, active: Option<LayerId>, command: &str) -> bool {
    active.and_then(|id| doc.layer(id)).is_some_and(|l| matches!(l.content, LayerContent::Raster(_)))
        && matches!(
            command,
            "filter.blur.gaussianBlur" | "filter.blur.boxBlur" | "filter.blur.motionBlur" | "filter.sharpen.unsharpMask" | "filter.other.highPass"
        )
}

struct Request {
    doc: Arc<Document>,
    active: LayerId,
    command: String,
    params: Value,
    visible: Rect,
    hash: u64,
    level: u32,
    proxy: Option<(u32, Rect, Arc<Document>)>,
    gpu: Option<crate::gpu_canvas::GpuCanvas>,
}

struct Completed {
    proxy: Arc<Document>,
    proxy_area: Rect,
    level: u32,
    result: Option<Arc<Document>>,
}

impl Request {
    fn run(&self, progress: &Progress) -> Option<Completed> {
        if progress.is_cancelled() {
            return None;
        }
        let original_fp = photocraft_engine::filters::params_for(&self.command, &self.params)?;
        let halo = match original_fp.halo() {
            photocraft_algo::Halo::Radius(r) => r,
            _ => return None,
        };
        let proxy_area = self.visible.inflate(halo.saturating_add((1i32 << self.level).saturating_mul(32)));
        let proxy = match &self.proxy {
            Some((level, area, proxy)) if *level == self.level && area.contains_rect(&proxy_area) => proxy.clone(),
            _ => Arc::new(photocraft_compose::mip_view::document_region(&self.doc, self.level, proxy_area).ok()?),
        };
        let mut params = self.params.clone();
        if self.level > 0
            && let Some(values) = params.as_object_mut()
        {
            for (key, value) in values {
                if super::is_pixel_param(key)
                    && let Some(v) = value.as_f64()
                {
                    *value = serde_json::json!(v / f64::from(1u32 << self.level));
                }
            }
        }
        let fp = photocraft_engine::filters::params_for(&self.command, &params)?;
        let source = proxy.layer(self.active)?.surface()?;
        let canvas = proxy.bounds();
        let selection =
            if self.level == 0 { self.doc.selection.clone() } else { self.doc.selection.as_ref().and_then(|s| s.mip_surface(self.level, proxy_area).ok()) };
        let selection_bounds = self.doc.selection.as_ref().map(|s| photocraft_raster::Surface::mip_bounds(s.content_bounds(), self.level));
        let bounds = selection_bounds.filter(|r| !r.is_empty()).unwrap_or(canvas);
        let content = photocraft_compose::bounds::content_bounds(source);
        let area =
            photocraft_algo::output_area(&fp, content, bounds, selection_bounds).intersect(&photocraft_raster::Surface::mip_bounds(self.visible, self.level));
        // apply_in reads the exact kernel halo and original canvas extent; cropping the source
        // to the output rectangle alone would introduce artificial edges around the viewport.
        if progress.is_cancelled() {
            return Some(Completed { proxy, proxy_area, level: self.level, result: None });
        }
        let gpu_filter = match fp {
            photocraft_algo::FilterParams::GaussianBlur { radius } => photocraft_gpu::filters::Filter::Gaussian { radius },
            photocraft_algo::FilterParams::BoxBlur { radius } => photocraft_gpu::filters::Filter::Box { radius },
            photocraft_algo::FilterParams::MotionBlur { angle, distance } => photocraft_gpu::filters::Filter::Motion { angle, distance },
            photocraft_algo::FilterParams::UnsharpMask { amount, radius, threshold } => photocraft_gpu::filters::Filter::Unsharp { amount, radius, threshold },
            photocraft_algo::FilterParams::HighPass { radius } => photocraft_gpu::filters::Filter::HighPass { radius },
            _ => return None,
        };
        let filtered =
            self.gpu.as_ref().and_then(|gpu| gpu.filter_surface(source, gpu_filter, area, canvas.union(&content), selection.as_ref()).ok()).unwrap_or_else(
                || {
                    photocraft_algo::progress::with_progress(progress, || {
                        photocraft_algo::apply_in(source, &fp, area, bounds, selection.as_ref(), canvas.union(&content))
                    })
                },
            );
        let result = if progress.is_cancelled() {
            None
        } else {
            let mut result = (*proxy).clone();
            *result.layer_mut(self.active)?.surface_mut()? = filtered;
            Some(Arc::new(result))
        };
        Some(Completed { proxy, proxy_area, level: self.level, result })
    }
}

struct Running {
    hash: u64,
    visible: Rect,
    progress: Progress,
    receive: mpsc::Receiver<Option<Completed>>,
}

pub struct RegionPreview {
    running: Option<Running>,
    next: Option<Request>,
    pub result: Option<Arc<Document>>,
    pub shown_hash: u64,
    pub factor: u32,
    proxy: Option<(u32, Rect, Arc<Document>)>,
    pub damage: Rect,
    shown_area: Rect,
}

impl Drop for RegionPreview {
    fn drop(&mut self) {
        if let Some(r) = &self.running {
            r.progress.cancel();
        }
    }
}

impl RegionPreview {
    pub fn pending(&self) -> bool {
        self.running.is_some() || self.next.is_some()
    }

    fn poll(&mut self, desired: u64) {
        let finished = self.running.as_ref().and_then(|r| match r.receive.try_recv() {
            Ok(result) => Some((r.hash, r.visible, result)),
            Err(mpsc::TryRecvError::Disconnected) => Some((r.hash, r.visible, None)),
            Err(mpsc::TryRecvError::Empty) => None,
        });
        if let Some((hash, visible, result)) = finished {
            self.running = None;
            if let Some(completed) = result {
                self.proxy = Some((completed.level, completed.proxy_area, completed.proxy));
                if hash == desired {
                    self.damage = self.shown_area.union(&visible);
                    self.shown_area = visible;
                    self.factor = 1u32 << completed.level;
                    self.result = completed.result;
                    self.shown_hash = hash;
                }
            }
        }
        if self.running.is_some() {
            return;
        }
        let Some(mut request) = self.next.take() else { return };
        request.proxy = self.proxy.clone();
        let progress = Progress::default();
        #[cfg(target_arch = "wasm32")]
        {
            self.damage = self.shown_area.union(&request.visible);
            self.shown_area = request.visible;
            if let Some(completed) = request.run(&progress) {
                self.proxy = Some((completed.level, completed.proxy_area, completed.proxy));
                self.factor = 1u32 << completed.level;
                self.result = completed.result;
                self.shown_hash = request.hash;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (send, receive) = mpsc::sync_channel(1);
            let hash = request.hash;
            let visible = request.visible;
            let worker_progress = progress.clone();
            let spawn = std::thread::Builder::new().name("photocraft-filter-preview".into()).spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| request.run(&worker_progress))).ok().flatten();
                let _ = send.send(result);
            });
            if spawn.is_ok() {
                self.running = Some(Running { hash, visible, progress, receive });
            }
        }
    }
}

pub fn update(
    app: &mut crate::PhotocraftApp,
    doc: Arc<Document>,
    revision: u64,
    active: Option<LayerId>,
    command: String,
    params: Value,
    visible: Rect,
    zoom: f32,
) {
    let Some(active) = active else { return };
    let level = if std::env::var("PHOTOCRAFT_VIEW_COMPOSE").as_deref() != Ok("0") && zoom.is_finite() && zoom > 0.0 && zoom < 0.5 {
        (1.0 / zoom).log2().floor().clamp(1.0, 16.0) as u32
    } else {
        0
    };
    let hash = format!("{command}{params}{visible:?}{active:?}{level}")
        .bytes()
        .fold(revision.wrapping_mul(0x9e3779b97f4a7c15), |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
    let replace = app.filter_preview.as_ref().is_none_or(|p| p.doc != doc.id || p.revision != revision || p.active != Some(active) || p.region.is_none());
    if replace {
        app.filter_preview = Some(super::FilterPreview {
            doc: doc.id,
            revision,
            hash: !hash,
            proxy: doc.clone(),
            active: Some(active),
            k: 1,
            result: None,
            region: Some(RegionPreview {
                running: None,
                next: None,
                result: None,
                shown_hash: 0,
                factor: 1,
                proxy: None,
                damage: Rect::EMPTY,
                shown_area: Rect::EMPTY,
            }),
        });
    }
    let gpu = app.gpu.clone();
    let Some(preview) = app.filter_preview.as_mut() else { return };
    let Some(region) = preview.region.as_mut() else { return };
    if preview.hash != hash {
        if let Some(running) = &region.running {
            running.progress.cancel();
        }
        region.next = Some(Request { doc, active, command, params, visible, hash, level, proxy: None, gpu });
        preview.hash = hash;
    }
    region.poll(hash);
}
