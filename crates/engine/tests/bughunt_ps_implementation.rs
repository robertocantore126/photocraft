//! Command-door regressions for be4168c2..09faee61. No production fixes.
use photocraft_engine::{Session, jobs::Progress};
use serde_json::json;

fn scene(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width":101,"height":401,"depth":depth,"background":"transparent"})).unwrap();
    s.execute("layer.new.layer", json!({"name":"varied"})).unwrap();
    s.execute("paint.stroke", json!({"points":[[3,7],[52,160],[96,395]],"size":19,"hardness":0.7,"color":"#e73491"})).unwrap();
    s
}

#[test]
fn bug_01_repeat_filter_worker() {
    let mut s = scene(8);
    s.execute("filter.blur.gaussianBlur", json!({"radius":1.5})).unwrap();
    assert!(s.is_enabled("filter.lastFilter"));
    // Same user action and input, using each dispatch door.
    let mut sync = scene(8);
    sync.execute("filter.blur.gaussianBlur", json!({"radius":1.5})).unwrap();
    sync.execute("filter.lastFilter", json!({})).unwrap();
    let prepared = s.prepare_job("filter.lastFilter", json!({}));
    assert!(prepared.is_ok(), "BUG-01: {}", prepared.err().unwrap());
    let mut job = prepared.unwrap().unwrap();
    let result = job.run(&Progress::default());
    s.finish_job(job, result).unwrap();
    assert_eq!(photocraft_compose::flatten(&s.active().unwrap().doc).px, photocraft_compose::flatten(&sync.active().unwrap().doc).px);
}

#[test]
fn worker_lifecycle_and_tab_identity_200_iterations() {
    for iteration in 0..200 {
        let mut s = scene(8);
        let before = s.active().unwrap().doc.clone();
        let mut job = s.prepare_job("filter.blur.gaussianBlur", json!({"radius":0.8})).unwrap().unwrap();
        assert!(!s.undo());
        assert!(!s.redo());
        assert!(s.close(0).is_none());
        assert!(s.execute("edit.fill", json!({"color":"#123456"})).is_err());
        s.execute("file.new", json!({"width":7,"height":9})).unwrap();
        let other = s.active().unwrap().doc.clone();
        let progress = Progress::default();
        if iteration % 2 == 0 {
            progress.cancel();
        }
        let result = job.run(&progress);
        let committed = s.finish_job(job, result);
        assert_eq!(s.active().unwrap().doc, other);
        assert!(!s.documents()[0].busy);
        if iteration % 2 == 0 {
            assert!(committed.is_err());
            assert_eq!(s.documents()[0].doc, before);
        } else {
            assert!(committed.is_ok());
        }
    }
}

#[test]
fn worker_filter_matches_sync_at_all_depths() {
    for depth in [8, 16, 32] {
        let mut s = scene(depth);
        let mut sync = scene(depth);
        let params = json!({"radius":2.25});
        sync.execute("filter.blur.gaussianBlur", params.clone()).unwrap();
        let mut job = s.prepare_job("filter.blur.gaussianBlur", params).unwrap().unwrap();
        let result = job.run(&Progress::default());
        s.finish_job(job, result).unwrap();
        assert_eq!(photocraft_compose::flatten(&s.active().unwrap().doc).px, photocraft_compose::flatten(&sync.active().unwrap().doc).px);
    }
}

#[test]
fn prefix_cache_matches_oracle_after_property_mask_and_pixel_edits() {
    for depth in [8, 16, 32] {
        let mut s = scene(depth);
        let lower = s.active().unwrap().active_layer.unwrap();
        s.execute("layer.layerStyle.dropShadow", json!({"distance":11,"size":17})).unwrap();
        s.execute("layer.new.layer", json!({"name":"suffix"})).unwrap();
        let upper = s.active().unwrap().active_layer;
        s.execute("paint.stroke", json!({"points":[[11,211],[89,344]],"size":31,"color":"#18c931"})).unwrap();
        let mut cache = photocraft_compose::prefix::PrefixCache::default();
        for (id, params) in [
            ("layer.setProps", json!({"layer":lower.0,"opacity":0.73})),
            ("layer.setProps", json!({"layer":lower.0,"blend":"Multiply"})),
            ("layer.setProps", json!({"layer":lower.0,"visible":false})),
            ("layer.setProps", json!({"layer":lower.0,"visible":true})),
            ("layer.layerMask.hideAll", json!({"layer":lower.0})),
            ("layer.layerMask.delete", json!({"layer":lower.0})),
            ("layer.select", json!({"layer":lower.0})),
            ("paint.stroke", json!({"points":[[9,12],[71,300]],"size":29,"color":"#ac31f7"})),
            ("layer.setProps", json!({"layer":lower.0,"opacity":0.51})),
        ] {
            for _ in 0..2 {
                let doc = &s.active().unwrap().doc;
                let rect = photocraft_geom::Rect::new(3, 17, 97, 398);
                let got = cache.render(doc, rect, upper).unwrap();
                let want = photocraft_compose::render(doc, rect);
                let worst = got.px.iter().zip(&want.px).flat_map(|(a, b)| a.iter().zip(b)).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert_eq!(worst, 0.0, "depth={depth} before {id}");
            }
            s.execute(id, params).unwrap();
        }
    }
}

#[test]
fn pick_clicks_through_locked_layer_and_locked_group() {
    let mut s = scene(8);
    let lower = s.active().unwrap().active_layer.unwrap();
    s.execute("layer.new.layer", json!({"name":"locked overlay"})).unwrap();
    s.execute("edit.fill", json!({"color":"#123456"})).unwrap();
    s.execute("layer.setProps", json!({"locked":true})).unwrap();
    let pick = s.execute("layer.pickAt", json!({"x":52,"y":160,"select":false})).unwrap();
    assert_eq!(pick["layer"], json!(lower.0));
    s.execute("layer.setProps", json!({"locked":false})).unwrap();
    s.execute("layer.groupLayers", json!({})).unwrap();
    s.execute("layer.setProps", json!({"locked":true})).unwrap();
    let pick = s.execute("layer.pickAt", json!({"x":52,"y":160,"select":false})).unwrap();
    assert_eq!(pick["layer"], json!(lower.0));
}

#[test]
fn serial_parallel_solid_and_psd_switches_are_pixel_identical() {
    // Environment switches belong to fresh subprocesses: no unsafe set_var or races
    // with unrelated tests in this process.
    let root = std::env::temp_dir().join(format!("photocraft-bughunt-ab-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for depth in [8, 16, 32] {
        let paths = [root.join(format!("on-{depth}")), root.join(format!("off-{depth}"))];
        for (index, path) in paths.iter().enumerate() {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child.args(["--exact", "ab_fixture", "--ignored", "--nocapture"]).env("BUGHUNT_OUTPUT", path).env("BUGHUNT_DEPTH", depth.to_string());
            for key in [
                "PHOTOCRAFT_SOLID_TILES",
                "PHOTOCRAFT_PARALLEL_BRUSH",
                "PHOTOCRAFT_BRUSH_STAMP",
                "PHOTOCRAFT_PSD_PARALLEL",
                "PHOTOCRAFT_PSD_BANDS",
                "PHOTOCRAFT_PSD_PRIME_BOUNDS",
            ] {
                child.env(key, if index == 0 { "1" } else { "0" });
            }
            let result = child.output().unwrap();
            assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
        }
        assert_eq!(std::fs::read(&paths[0]).unwrap(), std::fs::read(&paths[1]).unwrap(), "depth={depth}, full raw samples differ");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "A/B fixture: invoked in isolated subprocesses by serial_parallel_solid_and_psd_switches_are_pixel_identical"]
fn ab_fixture() {
    let depth: u32 = std::env::var("BUGHUNT_DEPTH").unwrap().parse().unwrap();
    let mut s = Session::new();
    s.execute("file.new", json!({"width":769,"height":577,"depth":depth,"background":"transparent"})).unwrap();
    s.execute("layer.new.layer", json!({"name":"large varied dab"})).unwrap();
    s.execute(
        "paint.stroke",
        json!({"points":[[260,260],[510,310],[530,510]],"size":700,"hardness":0.43,"opacity":0.7,"flow":0.6,"spacing":0.3,"color":"#21ca85","seed":123}),
    )
    .unwrap();
    let before = s.active().unwrap().doc.clone();
    let before_pixels = photocraft_compose::flatten(&before).px;
    s.execute("layer.new.layer", json!({"name":"shared fills"})).unwrap();
    s.execute("edit.fill", json!({"color":"#812bd4"})).unwrap();
    let solid_snapshot = s.active().unwrap().doc.clone();
    let solid_pixels = photocraft_compose::flatten(&solid_snapshot).px;
    s.execute("select.rect", json!({"x":7,"y":17,"width":101,"height":401})).unwrap();
    s.execute("edit.fill", json!({"color":"#fa3147"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("layer.translate", json!({"dx":-17,"dy":29})).unwrap();
    assert_eq!(photocraft_compose::flatten(&before).px, before_pixels, "COW changed snapshot pixels");
    assert_eq!(photocraft_compose::flatten(&solid_snapshot).px, solid_pixels, "partial fill or translation mutated shared solid snapshot");
    let doc = &s.active().unwrap().doc;
    let file = photocraft_io::document_to_psd(doc);
    let bytes = file.to_bytes().unwrap();
    let parsed = photocraft_psd::PsdFile::from_bytes(&bytes).unwrap();
    let streamed = photocraft_psd::PsdFile::from_reader(&mut std::io::Cursor::new(&bytes)).unwrap();
    assert_eq!(parsed.to_bytes().unwrap(), streamed.to_bytes().unwrap(), "stream parser changed PSD records");
    let (imported, warnings) = photocraft_io::psd_to_document(&streamed);
    assert!(warnings.is_empty(), "{warnings:?}");
    let mut output = Vec::new();
    for document in [doc.as_ref(), &imported] {
        for layer in &document.layers {
            if let Some(surface) = layer.surface() {
                output.extend(surface.to_interleaved(document.bounds()));
            }
        }
    }
    std::fs::write(std::env::var_os("BUGHUNT_OUTPUT").unwrap(), output).unwrap();
}
