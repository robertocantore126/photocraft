//! Brush plug-ins (Fotox's ABI) painting through `paint.stroke`. Uses the plug-ins built for
//! Fotox in `%APPDATA%\Fotox\plugins` (or `PHOTOCRAFT_TEST_PLUGINS`); skipped when they are absent.

use photocraft_engine::Session;
use serde_json::json;

fn plugin_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("PHOTOCRAFT_TEST_PLUGINS")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("APPDATA").map(|a| std::path::PathBuf::from(a).join("Fotox").join("plugins")))
        .filter(|d| d.join("plain_eraser.wasm").is_file())
}

fn alpha(s: &Session, x: i32, y: i32) -> f32 {
    let st = s.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)[3]
}

#[test]
fn fotox_eraser_plugins_paint_through_paint_stroke() {
    let Some(dir) = plugin_dir() else {
        eprintln!("no Fotox plug-ins: skipped");
        return;
    };
    for (file, id) in [("plain_eraser.wasm", "plain-eraser"), ("blend_eraser.wasm", "blend-eraser"), ("feather_eraser.wasm", "feather-eraser")] {
        let bytes = std::fs::read(dir.join(file)).unwrap();
        if let Err(e) = photocraft_plugins::brush::BrushPlugin::load(&bytes, photocraft_plugins::brush::BrushPlugin::default_limits()) {
            panic!("{file}: {e}");
        }
        let name = photocraft_engine::brush_plugins::load_file(&dir.join(file)).unwrap();
        assert!(name.is_some(), "{file} is a brush plug-in");
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 80, "height": 40, "background": "transparent"})).unwrap();
        s.execute("edit.fill", json!({"contents": "foreground"})).ok();
        s.execute("paint.stroke", json!({"points": [[0, 20], [80, 20]], "size": 60, "hardness": 1.0, "color": "#808080"})).unwrap();
        assert!(alpha(&s, 40, 20) > 0.99);
        let undo = s.active().unwrap().history.past_len();
        let r = s.execute(
            "paint.stroke",
            json!({"points": [[10, 20], [70, 20]], "size": 20, "hardness": 1.0, "opacity": 1.0, "flow": 1.0, "plugin": id, "pluginParams": [0, 40, 0], "feather": 4}),
        );
        r.unwrap_or_else(|e| panic!("{id}: {e}"));
        assert!(alpha(&s, 40, 20) < 0.5, "{id} erased the middle: {}", alpha(&s, 40, 20));
        assert!(alpha(&s, 40, 2) > 0.99, "{id} left the outside alone");
        assert_eq!(s.active().unwrap().history.past_len(), undo + 1, "one undo step");
        // The live stroke paints the same pixels as the commit.
        let live = photocraft_engine::brush_cmds::LiveStroke::begin(
            &s,
            &json!({"points": [[10, 30], [70, 30]], "size": 10, "hardness": 1.0, "plugin": id, "pluginParams": [0, 40, 0], "feather": 4}),
        );
        assert!(live.is_ok(), "{id}: live stroke");
    }
}
