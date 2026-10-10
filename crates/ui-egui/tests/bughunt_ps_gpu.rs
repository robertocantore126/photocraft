//! GPU probes using engine commands to construct every source document.
use eframe::wgpu;
use photocraft_engine::Session;
use photocraft_gpu::{
    Compositor,
    filters::{Filter, FilterEngine},
    render_to_vec_stats,
};
use serde_json::json;

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

fn gpu() -> (wgpu::Device, wgpu::Queue, Compositor) {
    let adapter = block_on(wgpu::Instance::default().request_adapter(&Default::default())).expect("GPU probe requires an adapter");
    eprintln!("adapter: {:?}", adapter.get_info());
    let format = Compositor::preferred_acc_format(&adapter);
    let (device, queue) = block_on(adapter.request_device(&Default::default())).unwrap();
    let comp = Compositor::try_new_with_format(&device, format).unwrap();
    (device, queue, comp)
}

fn scene(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width":101,"height":401,"depth":depth,"background":"transparent"})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("paint.stroke", json!({"points":[[3,7],[52,160],[96,395]],"size":23,"hardness":0.7,"color":"#e73491"})).unwrap();
    s
}

#[test]
fn gpu_filters_match_applied_cpu_all_channels() {
    filter_differentials(false);
}

/// BUG-05, reclassified: above radius 4 both sides run three running-sum box passes, but the
/// CPU restarts its sums at every tile window (`apply_tiled`) and the GPU once per view window,
/// so f32 rounding differs by a few ulps of the premultiplied sums. Unpremultiplying where alpha
/// is ~1e-7 amplifies that in the raw RGB of invisible pixels; the CPU is not bit-identical
/// across tilings either. What must hold is that nothing visible differs: every channel of
/// every pixel with alpha >= 1/255 within half an 8-bit level, at every depth.
#[test]
fn bug_05_large_gaussian_gpu_cpu() {
    let (device, queue, _) = gpu();
    let engine = FilterEngine::new(&device).unwrap();
    for depth in [8, 16, 32] {
        for radius in [8.0f32, 23.5] {
            let mut s = scene(depth);
            let st = s.active().unwrap();
            let source = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap();
            let params = json!({ "radius": radius });
            let fp = photocraft_engine::filters::params_for("filter.blur.gaussianBlur", &params).unwrap();
            let bounds = st.doc.bounds();
            let content = source.content_bounds();
            let area = photocraft_algo::output_area(&fp, content, bounds, None);
            let output = engine.run_surface(&device, &queue, source, Filter::Gaussian { radius }, area, bounds.union(&content), None).unwrap();
            s.execute("filter.blur.gaussianBlur", params).unwrap();
            let st = s.active().unwrap();
            let want = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().read_region(area);
            let got = output.read_region(area);
            assert_eq!(got.len(), want.len());
            let (mut visible, mut visible_worst, mut hidden_worst) = (0usize, 0.0f32, 0.0f32);
            for (a, b) in got.chunks_exact(4).zip(want.chunks_exact(4)) {
                let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                if a[3].max(b[3]) >= 1.0 / 255.0 {
                    visible += 1;
                    visible_worst = visible_worst.max(worst);
                } else {
                    hidden_worst = hidden_worst.max(worst);
                }
            }
            eprintln!("radius={radius} depth={depth}: {visible} visible pixels, worst {visible_worst}; below 1/255 alpha worst {hidden_worst}");
            assert!(visible > 1000);
            assert!(visible_worst <= 0.5 / 255.0, "radius={radius} depth={depth}: visible pixels differ by {visible_worst}");
        }
    }
}

fn filter_differentials(large_gaussian_only: bool) {
    let (device, queue, _) = gpu();
    let engine = FilterEngine::new(&device).unwrap();
    let mut failures = Vec::new();
    for depth in [8, 16, 32] {
        for (id, params, filter) in [
            ("filter.blur.gaussianBlur", json!({"radius":2.25}), Filter::Gaussian { radius: 2.25 }),
            ("filter.blur.gaussianBlur", json!({"radius":8.0}), Filter::Gaussian { radius: 8.0 }),
            ("filter.blur.boxBlur", json!({"radius":3.0}), Filter::Box { radius: 3.0 }),
            ("filter.blur.motionBlur", json!({"angle":33,"distance":17}), Filter::Motion { angle: 33.0, distance: 17.0 }),
            ("filter.sharpen.unsharpMask", json!({"radius":4,"amount":150,"threshold":2}), Filter::Unsharp { radius: 4.0, amount: 150.0, threshold: 2.0 }),
            ("filter.other.highPass", json!({"radius":4}), Filter::HighPass { radius: 4.0 }),
        ] {
            if matches!(filter, Filter::Gaussian { radius } if radius > 4.0) != large_gaussian_only {
                continue;
            }
            let mut s = scene(depth);
            let st = s.active().unwrap();
            let source = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap();
            let fp = photocraft_engine::filters::params_for(id, &params).unwrap();
            let bounds = st.doc.bounds();
            let content = source.content_bounds();
            let area = photocraft_algo::output_area(&fp, content, bounds, None);
            let output = engine.run_surface(&device, &queue, source, filter, area, bounds.union(&content), None).unwrap();
            s.execute(id, params).unwrap();
            let st = s.active().unwrap();
            let want = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().read_region(area);
            let got = output.read_region(area);
            assert_eq!(got.len(), want.len());
            let mut worst = 0.0f32;
            let mut different = 0;
            let mut channel_worst = [0.0f32; 4];
            for (index, (&a, &b)) in got.iter().zip(&want).enumerate() {
                assert!(a.is_finite() && b.is_finite());
                worst = worst.max((a - b).abs());
                different += usize::from(a != b);
                channel_worst[index % 4] = channel_worst[index % 4].max((a - b).abs());
            }
            eprintln!("{id} depth={depth}: differing samples={different}, worst={worst}, channel_worst={channel_worst:?}");
            // One stored code at integer depths, plus f32 subtraction rounding;
            // 1e-6 at float depth bounds ordinary shader arithmetic drift.
            // Every raw channel is included, even where alpha is zero.
            let tolerance = match depth {
                8 => 1.0 / 255.0 + f32::EPSILON,
                16 => 1.0 / 65535.0 + f32::EPSILON,
                _ => 1e-6,
            };
            let visible_worst = got
                .chunks_exact(4)
                .zip(want.chunks_exact(4))
                .filter(|(a, b)| a[3].max(b[3]) >= 1.0 / 255.0)
                .flat_map(|(a, b)| a.iter().zip(b))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("depth={depth} tolerance={tolerance} worst with alpha>=1/255: {visible_worst}");
            if worst > tolerance {
                failures.push(format!("{id} depth={depth} worst={worst}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

fn settle(comp: &mut Compositor, device: &wgpu::Device, queue: &wgpu::Queue, doc: &photocraft_doc::Document) -> (Vec<[f32; 4]>, u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut work = 0;
    loop {
        comp.take_effect_damage(doc.id);
        let (pixels, stats) = render_to_vec_stats(comp, device, queue, doc, doc.bounds()).unwrap();
        work += stats.fx_pixels;
        if !comp.effects_pending(doc.id) {
            return (pixels, work);
        }
        assert!(std::time::Instant::now() < deadline, "effects failed to settle");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

#[test]
fn async_effects_match_sync_when_settled() {
    let (device, queue, mut comp) = gpu();
    let mut s = scene(8);
    s.execute("layer.layerStyle.dropShadow", json!({"distance":17,"size":23})).unwrap();
    s.execute("layer.layerStyle.outerGlow", json!({"size":31,"color":"#23fa91"})).unwrap();
    let doc = &s.active().unwrap().doc;
    let (want, _) = settle(&mut comp, &device, &queue, doc);
    comp.forget_doc(doc.id);
    comp.set_async_effects(true);
    let (got, _) = settle(&mut comp, &device, &queue, doc);
    assert_eq!(got, want, "all RGBA samples, including transparent RGB");
    let (_, warm) = settle(&mut comp, &device, &queue, doc);
    assert_eq!(warm, 0, "unchanged warm frame must do no effect map work");
}

#[test]
fn bug_03_zoom_roundtrip_rebuilds_effects() {
    let (device, queue, mut comp) = gpu();
    let mut s = scene(8);
    s.execute("layer.layerStyle.dropShadow", json!({"distance":17,"size":23})).unwrap();
    let doc = &s.active().unwrap().doc;
    let half = photocraft_compose::mip_view::document(doc, 1).unwrap();
    let quarter = photocraft_compose::mip_view::document(doc, 2).unwrap();
    comp.set_async_effects(true);
    let (_, cold) = settle(&mut comp, &device, &queue, &half);
    assert!(cold > 0);
    let (_, warm) = settle(&mut comp, &device, &queue, &half);
    assert_eq!(warm, 0);
    settle(&mut comp, &device, &queue, &quarter);
    let (_, repeated) = settle(&mut comp, &device, &queue, &half);
    eprintln!("cold fx pixels={cold}, warm={warm}, return-to-zoom={repeated}");
    assert_eq!(repeated, 0, "BUG-03: no document edit occurred");
}

/// Like BUG-03 for layer pages: each display mip level keeps its own resident pages, so going
/// back to a zoom already shown uploads nothing (it re-uploaded every page before, a 1.3 s
/// first frame on a 479-layer PSB).
#[test]
fn zoom_roundtrip_reuses_layer_pages() {
    let (device, queue, mut comp) = gpu();
    let mut s = scene(8);
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("paint.stroke", json!({"points":[[90,20],[10,380]],"size":31,"color":"#3366cc"})).unwrap();
    let doc = &s.active().unwrap().doc;
    let half = photocraft_compose::mip_view::document(doc, 1).unwrap();
    let quarter = photocraft_compose::mip_view::document(doc, 2).unwrap();
    let uploads = |comp: &mut Compositor, d: &photocraft_doc::Document| render_to_vec_stats(comp, &device, &queue, d, d.bounds()).unwrap().1.tiles_uploaded;
    let cold = uploads(&mut comp, &half);
    uploads(&mut comp, &quarter);
    let back = uploads(&mut comp, &half);
    eprintln!("tiles uploaded: first visit {cold}, back to the same zoom {back}");
    assert!(cold > 0);
    assert_eq!(back, 0);
}
