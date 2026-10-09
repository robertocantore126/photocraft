//! Visible-row thumbnail requests. Two dedicated workers avoid nested rayon work.
use photocraft_doc::LayerId;
use photocraft_geom::{Rect, Size};
use photocraft_raster::Surface;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, PoisonError, mpsc};

pub type Key = (LayerId, u8);
pub struct Finished {
    pub key: Key,
    pub revision: u64,
    pub image: Result<egui::ColorImage, String>,
}
struct Request {
    key: Key,
    revision: u64,
    surface: Surface,
    size: Size,
    mask: bool,
}
#[derive(Default)]
struct Pending {
    requests: VecDeque<Request>,
    latest: HashMap<Key, u64>,
    stop: bool,
}
struct Shared {
    pending: Mutex<Pending>,
    wake: Condvar,
    context: Mutex<Option<egui::Context>>,
}
pub struct ThumbQueue {
    shared: Arc<Shared>,
    sender: mpsc::Sender<Finished>,
    receiver: mpsc::Receiver<Finished>,
    workers: usize,
}
impl Default for ThumbQueue {
    fn default() -> Self {
        let shared = Arc::new(Shared { pending: Mutex::new(Pending::default()), wake: Condvar::new(), context: Mutex::new(None) });
        let (sender, receiver) = mpsc::channel();
        #[allow(unused_mut)]
        let mut workers = 0;
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var("PHOTOCRAFT_THUMB_WORKERS").as_deref() != Ok("0") && std::env::var("PHOTOCRAFT_BACKGROUND_THUMBS").as_deref() != Ok("0") {
            for index in 0..2 {
                let shared = shared.clone();
                let sender = sender.clone();
                if std::thread::Builder::new().name(format!("photocraft-thumb-{index}")).spawn(move || worker(shared, sender)).is_ok() {
                    workers += 1;
                }
            }
        }
        Self { shared, sender, receiver, workers }
    }
}
impl ThumbQueue {
    pub fn needs(&self, key: Key, revision: u64) -> bool {
        self.shared.pending.lock().unwrap_or_else(PoisonError::into_inner).latest.get(&key) != Some(&revision)
    }
    pub fn request(&self, key: Key, revision: u64, surface: Surface, size: Size, mask: bool, context: &egui::Context) {
        let mut pending = self.shared.pending.lock().unwrap_or_else(PoisonError::into_inner);
        if pending.latest.get(&key) == Some(&revision) {
            return;
        }
        pending.latest.insert(key, revision);
        pending.requests.retain(|request| request.key != key);
        // Calls originate only from painted rows. New visible rows supersede older offscreen work.
        if pending.requests.len() >= 512
            && let Some(old) = pending.requests.pop_back()
        {
            pending.latest.remove(&old.key);
        }
        pending.requests.push_front(Request { key, revision, surface, size, mask });
        drop(pending);
        *self.shared.context.lock().unwrap_or_else(PoisonError::into_inner) = Some(context.clone());
        self.shared.wake.notify_one();
        context.request_repaint();
    }
    pub fn collect(&self) -> Vec<Finished> {
        // Browser fallback is deliberately bounded to two requests per frame.
        if self.workers == 0 {
            for _ in 0..2 {
                let request = self.shared.pending.lock().unwrap_or_else(PoisonError::into_inner).requests.pop_front();
                if let Some(request) = request {
                    let _ = self.sender.send(render(request));
                }
            }
        }
        let mut finished = Vec::new();
        while let Ok(result) = self.receiver.try_recv() {
            if self.shared.pending.lock().unwrap_or_else(PoisonError::into_inner).latest.get(&result.key) == Some(&result.revision) {
                finished.push(result);
            }
        }
        if self.workers == 0
            && !self.shared.pending.lock().unwrap_or_else(PoisonError::into_inner).requests.is_empty()
            && let Some(context) = self.shared.context.lock().unwrap_or_else(PoisonError::into_inner).as_ref()
        {
            context.request_repaint();
        }
        finished
    }
    pub fn retain(&self, live: &std::collections::HashSet<Key>) {
        let mut pending = self.shared.pending.lock().unwrap_or_else(PoisonError::into_inner);
        pending.latest.retain(|key, _| live.contains(key));
        pending.requests.retain(|request| live.contains(&request.key));
    }
}
impl Drop for ThumbQueue {
    fn drop(&mut self) {
        let mut pending = self.shared.pending.lock().unwrap_or_else(PoisonError::into_inner);
        pending.stop = true;
        pending.requests.clear();
        drop(pending);
        self.shared.wake.notify_all();
    }
}
// Workers retain an egui Context only for repaint notification; no GPU or UI calls happen there.
#[cfg(not(target_arch = "wasm32"))]
fn worker(shared: Arc<Shared>, sender: mpsc::Sender<Finished>) {
    loop {
        let request = {
            let mut pending = shared.pending.lock().unwrap_or_else(PoisonError::into_inner);
            while !pending.stop && pending.requests.is_empty() {
                pending = shared.wake.wait(pending).unwrap_or_else(PoisonError::into_inner);
            }
            if pending.stop {
                return;
            }
            pending.requests.pop_front()
        };
        if let Some(request) = request {
            if sender.send(render(request)).is_err() {
                return;
            }
            if let Some(context) = shared.context.lock().unwrap_or_else(PoisonError::into_inner).as_ref() {
                context.request_repaint();
            }
        }
    }
}
fn render(request: Request) -> Finished {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| image(&request.surface, request.size, request.mask)))
        .unwrap_or_else(|_| Err("thumbnail worker failed while reading layer pixels".into()));
    Finished { key: request.key, revision: request.revision, image: result }
}
pub fn image(surface: &Surface, size: Size, mask: bool) -> Result<egui::ColorImage, String> {
    let base = Rect::from_size(size);
    let use_mip = std::env::var("PHOTOCRAFT_THUMB_MIPS").as_deref() != Ok("0") && std::env::var("PHOTOCRAFT_CPU_MIPS").as_deref() != Ok("0");
    let level = if use_mip { Surface::mip_level_for_size(base, 64) } else { 0 };
    let reduced;
    let source = if level > 0 {
        reduced = surface.mip_surface(level, base)?;
        &reduced
    } else {
        surface
    };
    let divisor = (1u32 << level) as f64;
    let (w, h) = (size.width.max(1) as f64 / divisor, size.height.max(1) as f64 / divisor);
    let scale = w.max(h) / 64.0;
    let (ox, oy) = ((64.0 - w / scale) * 0.5, (64.0 - h / scale) * 0.5);
    let mut pixels = vec![egui::Color32::TRANSPARENT; 64 * 64];
    let mut sample = vec![0.0f32; source.channels()];
    for ty in 0..64 {
        for tx in 0..64 {
            let x0 = ((tx as f64 - ox) * scale).max(0.0);
            let y0 = ((ty as f64 - oy) * scale).max(0.0);
            let x1 = ((tx as f64 + 1.0 - ox) * scale).min(w);
            let y1 = ((ty as f64 + 1.0 - oy) * scale).min(h);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let mut sum = [0.0f64; 4];
            let mut weight = 0.0;
            for y in y0.floor() as i32..y1.ceil() as i32 {
                for x in x0.floor() as i32..x1.ceil() as i32 {
                    let area = (x1.min(x as f64 + 1.0) - x0.max(x as f64)).max(0.0) * (y1.min(y as f64 + 1.0) - y0.max(y as f64)).max(0.0);
                    source.read_pixel(x, y, &mut sample);
                    let rgba = if mask {
                        let v = sample.first().copied().unwrap_or(0.0);
                        [v, v, v, 1.0]
                    } else {
                        photocraft_raster::to_rgba(&source.format(), &sample)
                    };
                    let alpha = f64::from(rgba[3]);
                    for c in 0..3 {
                        if let Some(value) = sum.get_mut(c) {
                            *value += f64::from(rgba.get(c).copied().unwrap_or(0.0)) * alpha * area;
                        }
                    }
                    if let Some(value) = sum.get_mut(3) {
                        *value += alpha * area;
                    }
                    weight += area;
                }
            }
            let alpha = sum.get(3).copied().unwrap_or(0.0);
            let q = |v: f64| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let color = |c: usize| if alpha > 0.0 { q(sum.get(c).copied().unwrap_or(0.0) / alpha) } else { 0 };
            if let Some(pixel) = pixels.get_mut(ty * 64 + tx) {
                *pixel = egui::Color32::from_rgba_unmultiplied(color(0), color(1), color(2), q(if weight > 0.0 { alpha / weight } else { 0.0 }));
            }
        }
    }
    Ok(egui::ColorImage::new([64, 64], pixels))
}
