//! Memory probe: how process memory grows with layer count on the GPU canvas.
//!
//! Builds a 4096² 8-bit document one partial layer at a time (log-uniform 256–2048 px noisy
//! rectangles, like the 4K stress recipe) and refreshes the GPU canvas after each layer, the
//! way the app does. Every `--every` layers it prints one line with the compositor's own byte
//! counts; a wrapper samples the process's private bytes at the same moment.
//!
//! ```sh
//! cargo run --release -p photocraft-ui-egui --example layer_memory_probe -- --layers 300 --refresh full --budget-mb 512
//! ```

use std::io::Write as _;
use std::time::Instant;

use eframe::egui_wgpu::RenderState;
use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Effect, FxCommon, Layer, LayerMask, Shadow};
use photocraft_geom::{Rect, Size};
use photocraft_ui_egui::gpu_canvas::GpuCanvas;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo).max(1) as u64) as i32
    }
}

fn noisy_layer(name: &str, r: Rect, rng: &mut Rng) -> Layer {
    let mut l = Layer::raster(name, PixelFormat::RGBA8);
    let (c0, c1) = (rng.next(), rng.next());
    let w = r.width() as usize;
    let mut bytes = vec![0u8; w * r.height() as usize * 4];
    let mut x = rng.next() | 1;
    for (i, px) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let t = (i % w) as f32 / w as f32;
        let n = (x % 64) as f32 - 32.0;
        let mix = |a: u64, b: u64, s: u32| ((((a >> s) & 255) as f32) * (1.0 - t) + (((b >> s) & 255) as f32) * t + n).clamp(0.0, 255.0) as u8;
        *px = [mix(c0, c1, 0), mix(c0, c1, 8), mix(c0, c1, 16), 255];
    }
    if let Some(s) = l.surface_mut() {
        s.write_interleaved(r, &bytes);
    }
    l
}

fn canvas() -> Option<(GpuCanvas, RenderState)> {
    let setup = photocraft_ui_egui::gpu_canvas::wgpu_setup();
    let rs = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| egui_kittest::wgpu::create_render_state(setup, Default::default()))).ok()?;
    eprintln!("adapter: {} ({:?})", rs.adapter.get_info().name, rs.adapter.get_info().backend);
    Some((GpuCanvas::new(&rs), rs))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layers: usize = arg(&args, "--layers").and_then(|v| v.parse().ok()).unwrap_or(300);
    let every: usize = arg(&args, "--every").and_then(|v| v.parse().ok()).unwrap_or(25);
    let full = arg(&args, "--refresh").is_none_or(|v| v == "full");
    let budget_mb: Option<u64> = arg(&args, "--budget-mb").and_then(|v| v.parse().ok());
    // 0 = never; the stress recipe uses a mask every 7th layer and a drop shadow every 20th.
    let mask_every: usize = arg(&args, "--mask-every").and_then(|v| v.parse().ok()).unwrap_or(0);
    let shadow_every: usize = arg(&args, "--shadow-every").and_then(|v| v.parse().ok()).unwrap_or(0);
    // Edits per layer after it is added, each on a copy of the document kept as History (50
    // states), so every edit makes new tiles that must be uploaded again, as in the app.
    let edits: usize = arg(&args, "--edits").and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut history: std::collections::VecDeque<Document> = std::collections::VecDeque::new();
    // Like the app: never block on the GPU, only poll without waiting.
    let no_wait = args.iter().any(|a| a == "--no-wait");
    let wait = |rs: &RenderState| {
        let _ = rs.device.poll(if no_wait { eframe::wgpu::PollType::Poll } else { eframe::wgpu::PollType::Wait { submission_index: None, timeout: None } });
    };
    let Some((g, rs)) = canvas() else {
        eprintln!("no GPU adapter");
        return;
    };
    if let Some(mb) = budget_mb {
        g.set_memory_budget(mb << 20);
    }
    // `--open doc.pcraft [--strip groups,adjust,smart,fx,mask]`: refresh a saved document
    // (optionally without some kinds of layers) three times instead of building one.
    if let Some(path) = arg(&args, "--open") {
        let t_open = Instant::now();
        let loaded = if path.ends_with(".pcraft") {
            photocraft_format::load_path(std::path::Path::new(&path)).map_err(|e| e.to_string())
        } else {
            std::fs::read(&path).map_err(|e| e.to_string()).and_then(|b| photocraft_io::import_owned(&path, b).map(|r| r.document).map_err(|e| e.to_string()))
        };
        let mut doc = match loaded {
            Ok(d) => d,
            Err(e) => {
                eprintln!("open {path}: {e}");
                return;
            }
        };
        eprintln!("opened in {:.1} s", t_open.elapsed().as_secs_f64());
        let strip = arg(&args, "--strip").unwrap_or_default();
        doc.layers = strip_layers(std::mem::take(&mut doc.layers), &strip);
        let _ = writeln!(std::io::stdout(), "pass,resident_mb,fx_mb,refresh_ms,kind,layers={}", doc.walk().len());
        for pass in 1..=3 {
            let t = Instant::now();
            let out = g.refresh(doc.id.0, &doc, None, None);
            wait(&rs);
            let (res, fx) = g.compositor_bytes().unwrap_or((0, 0));
            let _ = writeln!(std::io::stdout(), "{pass},{},{},{:.1},{}", res >> 20, fx >> 20, t.elapsed().as_secs_f64() * 1000.0, out.kind);
            let _ = std::io::stdout().flush();
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
        // `--rect x,y,w,h --repeat n`: partial refreshes of that rect (a brush stroke's damage).
        if let Some(r) = arg(&args, "--rect") {
            let v: Vec<i32> = r.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            if let [x, y, w, h] = v[..] {
                let rect = Rect::from_xywh(x, y, w as u32, h as u32);
                let n: usize = arg(&args, "--repeat").and_then(|v| v.parse().ok()).unwrap_or(5);
                for i in 0..n {
                    let t = Instant::now();
                    let out = g.refresh(doc.id.0, &doc, Some(rect), None);
                    let _ = rs.device.poll(eframe::wgpu::PollType::Wait { submission_index: None, timeout: None });
                    let _ = writeln!(std::io::stdout(), "rect {i}: {:.1} ms {}", t.elapsed().as_secs_f64() * 1000.0, out.kind);
                }
            }
        }
        return;
    }
    let (w, h) = (4096, 4096);
    let mut doc = Document::new("probe", Size::new(w, h), ColorMode::Rgb, SampleType::U8);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let t0 = Instant::now();
    let mut stdout = std::io::stdout();
    let _ = writeln!(stdout, "layers,secs,resident_mb,fx_mb,refresh_ms,kind");
    for i in 1..=layers {
        let lw = (256.0 * 8f64.powf((rng.next() % 1000) as f64 / 1000.0)) as i32;
        let lh = (256.0 * 8f64.powf((rng.next() % 1000) as f64 / 1000.0)) as i32;
        let (x, y) = (rng.range(0, w as i32 - lw), rng.range(0, h as i32 - lh));
        let r = Rect::new(x, y, x + lw, y + lh);
        let mut l = noisy_layer(&format!("L{i}"), r, &mut rng);
        if mask_every > 0 && i % mask_every == 0 {
            // Like Layer › Layer Mask › Reveal Selection with an inset selection.
            let mut m = LayerMask::reveal_all();
            m.surface = photocraft_raster::Surface::with_default(PixelFormat { mode: ColorMode::Grayscale, sample: SampleType::U8, alpha: false }, &[0.0]);
            m.surface.fill_rect(Rect::new(x + lw / 8, y + lh / 8, x + lw * 7 / 8, y + lh * 7 / 8), &[1.0]);
            l.mask = Some(m);
        }
        if shadow_every > 0 && i % shadow_every == 0 {
            l.effects.items = vec![Effect::DropShadow(Shadow {
                common: FxCommon::new(BlendMode::Multiply, 0.75),
                color: Color::rgb(0.0, 0.0, 0.0),
                angle: 120.0,
                use_global_light: false,
                distance: 12.0,
                spread: 0.0,
                size: 16.0,
                contour: photocraft_doc::Contour::Linear,
                anti_alias: false,
                noise: 0.0,
                knocks_out: false,
            })];
        }
        doc.layers.push(l);
        let t = Instant::now();
        let mut out = g.refresh(doc.id.0, &doc, if full || i == 1 { None } else { Some(r) }, None);
        wait(&rs);
        for _ in 0..edits {
            history.push_back(doc.clone());
            if history.len() > 50 {
                history.pop_front();
            }
            let fresh = noisy_layer("edit", r, &mut rng);
            if let (Some(dst), Some(src)) = (doc.layers.last_mut().and_then(|l| l.surface_mut()), fresh.surface()) {
                dst.write_interleaved(r, &src.to_interleaved(r));
            }
            out = g.refresh(doc.id.0, &doc, Some(r), None);
            wait(&rs);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i % every == 0 {
            let (res, fx) = g.compositor_bytes().unwrap_or((0, 0));
            let _ = writeln!(stdout, "{i},{:.1},{},{},{ms:.1},{}", t0.elapsed().as_secs_f64(), res >> 20, fx >> 20, out.kind);
            let _ = stdout.flush();
            // Give the sampler a moment on this line.
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
    }
}

/// `layers` without the kinds named in `strip` (comma-separated): groups are replaced by their
/// children, the rest are dropped or cleared.
fn strip_layers(layers: Vec<Layer>, strip: &str) -> Vec<Layer> {
    let has = |k: &str| strip.split(',').any(|s| s.trim() == k);
    let mut out = Vec::new();
    for mut l in layers {
        if has("fx") {
            l.effects.items.clear();
        }
        if has("mask") {
            l.mask = None;
        }
        match l.content {
            photocraft_doc::LayerContent::Group(ref mut g) => {
                let children = strip_layers(std::mem::take(&mut g.children), strip);
                if has("groups") {
                    out.extend(children);
                } else {
                    g.children = children;
                    out.push(l);
                }
            }
            photocraft_doc::LayerContent::Adjustment(_) if has("adjust") => {}
            photocraft_doc::LayerContent::Smart(_) if has("smart") => {}
            _ => out.push(l),
        }
    }
    out
}
