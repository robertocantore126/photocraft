//! A pending full canvas refresh must preserve edits and converge to the CPU oracle.
#![cfg(not(target_arch = "wasm32"))]

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer};
use photocraft_geom::{Rect, Size};
use photocraft_ui_egui::gpu_canvas::GpuCanvas;

#[test]
fn progressive_canvas_merges_edits_and_converges() {
    let setup = photocraft_ui_egui::gpu_canvas::wgpu_setup();
    let state = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| egui_kittest::wgpu::create_render_state(setup, Default::default())));
    let Ok(rs) = state else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let gpu = GpuCanvas::with_tile(&rs, Some(256));
    for coarse in [false, true] {
        let mut doc = Document::new("progressive synthetic regression", Size::new(769, 513), ColorMode::Rgb, SampleType::U8);
        let mut layer = Layer::raster("pixels", PixelFormat::RGBA8);
        layer.surface_mut().unwrap().fill_rect(doc.bounds(), &[0.2, 0.6, 0.9, 0.7]);
        doc.layers.push(layer);
        let visible = Rect::new(512, 0, 769, 513);
        if coarse {
            gpu.set_preview_zoom(doc.id.0, [769, 513], 0.125);
        }
        let first = gpu.refresh_progressive(doc.id.0, &doc, None, visible, None, true);
        assert!(first.px < doc.size.area());
        assert!(gpu.pending(doc.id.0));
        assert_eq!(gpu.pending_pixels(doc.id.0), if coarse { doc.size.area() } else { doc.size.area() - first.px });
        assert_eq!(first.kind == "gpu-preview", coarse);
        let edit = Rect::new(530, 210, 730, 400);
        doc.layers[0].surface_mut().unwrap().fill_rect(edit, &[0.9, 0.1, 0.2, 1.0]);
        gpu.refresh_progressive(doc.id.0, &doc, Some(edit), visible, None, true);
        let mut frames = 0;
        while gpu.pending(doc.id.0) {
            gpu.refresh_progressive(doc.id.0, &doc, None, visible, None, false);
            frames += 1;
            assert!(frames < 1024, "canvas must converge");
        }
        let (_, _, actual) = gpu.read_texels(doc.id.0).expect("readback");
        let expected = photocraft_compose::flatten(&doc);
        let mut max_error = 0.0f32;
        for (a, e) in actual.iter().zip(&expected.px) {
            for channel in 0..4 {
                let value = if channel == 3 { e[channel] } else { e[channel] * e[3] };
                max_error = max_error.max((a[channel] - value).abs());
            }
        }
        assert!(max_error <= 1.0 / 255.0, "premultiplied pixel error {max_error}");
        gpu.retain(&[]);
        assert_eq!(gpu.pending_pixels(doc.id.0), 0);
    }
}
