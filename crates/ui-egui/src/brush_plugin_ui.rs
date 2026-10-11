//! Brush plug-in tools (Fotox's brush plugins, D-096): the folder watch, the toolbar entries and
//! the option bar a plug-in's manifest describes.
//!
//! The folder is `%APPDATA%\Photocraft\plugins` (`PHOTOCRAFT_PLUGINS` overrides it; Fotox's
//! plug-ins are copied in the first time). PhotoCraft looks at it twice a second: a `.wasm` that
//! appears or changes is loaded, a `.rs` is built (`photocraft_engine::brush_script`) on a
//! background thread and then loaded, a deleted file's tool goes away. Each plug-in becomes a tool
//! in the flyout of its manifest's `slot` (`eraser`, `brush`, `dodge`…), painting through
//! `paint.stroke` with `"plugin"`; the option bar is the manifest's `options`: Size, Hardness,
//! Opacity, Flow, Smoothing, Spacing and the pressure toggles drive the brush as for the native
//! tools, every other field is a plug-in param.

use std::collections::HashMap;

use egui::Ui;
use photocraft_engine::BrushSettings;
use photocraft_plugins::brush::BrushManifest;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::state::{Tool, UiState};

/// The toolbar tool a manifest's `slot` puts the plug-in next to.
pub fn slot_tool(slot: &str) -> Tool {
    match slot {
        "eraser" => Tool::Eraser,
        "dodge" | "burn" | "sponge" => Tool::Dodge,
        "blur" | "sharpen" | "smudge" => Tool::Blur,
        "clone" | "clone-stamp" | "stamp" => Tool::CloneStamp,
        "history" | "history-brush" => Tool::HistoryBrush,
        "heal" | "spot-heal" | "healing" => Tool::SpotHealing,
        "gradient" => Tool::Gradient,
        _ => Tool::Brush,
    }
}

/// `(id, name)` of the plug-ins whose slot is the toolbar slot holding `tools`.
pub fn tools_for_slot(tools: &[Tool]) -> Vec<(String, String)> {
    photocraft_plugins::brush::list()
        .iter()
        .filter(|p| tools.contains(&slot_tool(&p.manifest().slot)))
        .map(|p| {
            let name = match p.stopped() {
                Some(_) => format!("{} (stopped)", p.manifest().name),
                None => p.manifest().name.clone(),
            };
            (p.id().to_string(), name)
        })
        .collect()
}

fn manifest(ui: &UiState) -> Option<BrushManifest> {
    let id = ui.plugin_tool.as_deref()?;
    photocraft_plugins::brush::get(id).map(|p| p.manifest().clone())
}

/// The active plug-in tool's display name.
pub fn current_name(ui: &UiState) -> Option<String> {
    manifest(ui).map(|m| m.name)
}

/// Pick plug-in `id` as the current tool.
pub fn select(app: &mut PhotocraftApp, id: &str) {
    app.ui.plugin_tool = Some(id.to_string());
    app.ui.tool = Tool::Plugin;
}

fn label_of(o: &Value) -> String {
    o.get("text").and_then(Value::as_str).unwrap_or_default().trim().trim_end_matches(':').trim().to_string()
}

fn kind_of(o: &Value) -> &str {
    o.get("type").and_then(Value::as_str).unwrap_or_default()
}

/// The field's default as a number: a `num`'s value (a string), a `range`'s, a `select`'s index,
/// a `toggle`'s 0/1.
fn default_of(o: &Value) -> f32 {
    match kind_of(o) {
        "select" => {
            let v = o.get("value").and_then(Value::as_str).unwrap_or_default();
            o.get("options").and_then(Value::as_array).and_then(|a| a.iter().position(|x| x.as_str() == Some(v))).unwrap_or(0) as f32
        }
        "toggle" => f32::from(u8::from(o.get("on").and_then(Value::as_bool).unwrap_or(false))),
        _ => o.get("value").and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))).unwrap_or(0.0) as f32,
    }
}

/// A field the brush engine owns (Size, Opacity…): read or write it on the brush.
fn native<'a>(b: &'a mut BrushSettings, label: &str) -> Option<&'a mut f32> {
    match label.to_ascii_lowercase().as_str() {
        "size" => Some(&mut b.size),
        "hardness" => Some(&mut b.hardness),
        "opacity" => Some(&mut b.opacity),
        "flow" => Some(&mut b.flow),
        "smoothing" => Some(&mut b.smoothing.amount),
        "spacing" => Some(&mut b.spacing),
        _ => None,
    }
}

/// Percent fields are stored as fractions on the brush.
fn native_scale(label: &str) -> f32 {
    if label.eq_ignore_ascii_case("size") { 1.0 } else { 100.0 }
}

fn native_toggle<'a>(b: &'a mut BrushSettings, label: &str) -> Option<&'a mut bool> {
    match label.to_ascii_lowercase().as_str() {
        "pressure for opacity" => Some(&mut b.pressure_opacity),
        "pressure for size" => Some(&mut b.pressure_size),
        _ => None,
    }
}

/// The first time a plug-in tool is used: its manifest's brush defaults (Size, Opacity, Flow…)
/// and its own params' defaults.
pub fn apply_defaults(ui: &mut UiState, id: &str, b: &mut BrushSettings) {
    let Some(p) = photocraft_plugins::brush::get(id) else { return };
    let m = p.manifest();
    let mut values = HashMap::new();
    for o in &m.options {
        let label = label_of(o);
        let v = default_of(o);
        if let Some(t) = native_toggle(b, &label) {
            *t = v != 0.0;
        } else if let Some(f) = native(b, &label) {
            *f = v / native_scale(&label);
        } else if !label.is_empty() {
            values.insert(label, v);
        }
    }
    b.tip = Default::default();
    ui.plugin_values.insert(id.to_string(), values);
}

fn value(ui: &UiState, id: &str, label: &str, o: Option<&Value>) -> f32 {
    ui.plugin_values.get(id).and_then(|v| v.get(label)).copied().unwrap_or_else(|| o.map_or(0.0, default_of))
}

/// The `paint.stroke` params a plug-in stroke adds: the plug-in, its params in manifest order,
/// and the Feather field's value for a `feather` tip.
pub fn stroke_params(ui: &UiState) -> Option<Value> {
    let m = manifest(ui)?;
    let id = m.id.clone();
    let field = |label: &str| m.options.iter().find(|o| label_of(o).eq_ignore_ascii_case(label));
    let params: Vec<f32> = m.params.iter().map(|k| value(ui, &id, k, field(k))).collect();
    let mut p = json!({"plugin": id, "pluginParams": params});
    if let Some(f) = m.brush.as_ref().and_then(|b| b.feather_from.clone()) {
        p["feather"] = json!(value(ui, &m.id, &f, field(&f)));
    }
    Some(p)
}

/// Does the plug-in's bar start with the brush preset chip?
pub fn has_preset_chip(ui: &UiState) -> bool {
    manifest(ui).is_some_and(|m| m.options.iter().any(|o| kind_of(o) == "brushpreset"))
}

fn opt_label(ui: &mut Ui, text: &str) {
    let t = crate::theme::Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(text).color(t.text_dim).size(12.0));
}

/// The option bar of the active plug-in tool, from its manifest.
pub fn options_bar(state: &mut UiState, ui: &mut Ui, b: &mut BrushSettings) {
    let Some(m) = manifest(state) else {
        opt_label(ui, "The plug-in is not loaded");
        return;
    };
    if let Some(why) = photocraft_plugins::brush::get(&m.id).and_then(|p| p.stopped()) {
        opt_label(ui, &format!("Stopped: {why} — save the file again to reload it"));
    }
    for (i, o) in m.options.iter().enumerate() {
        let label = label_of(o);
        let unit = o.get("unit").and_then(Value::as_str).unwrap_or_default();
        match kind_of(o) {
            "num" | "range" => {
                opt_label(ui, &format!("{label}:"));
                let (range, unit) = match kind_of(o) {
                    "range" => (0.0..=100.0, "%"),
                    _ if unit == "%" => (0.0..=100.0, "%"),
                    _ => (0.0..=5000.0, unit),
                };
                if let Some(f) = native(b, &label) {
                    let s = native_scale(&label);
                    let mut v = *f * s;
                    if crate::widgets::value_field(ui, &mut v, if label.eq_ignore_ascii_case("size") { 1.0..=5000.0 } else { range }, unit, 62.0).changed() {
                        *f = v / s;
                    }
                } else {
                    let mut v = value(state, &m.id, &label, Some(o));
                    if crate::widgets::value_field(ui, &mut v, range, unit, 62.0).changed() {
                        state.plugin_values.entry(m.id.clone()).or_default().insert(label, v);
                    }
                }
            }
            "select" => {
                opt_label(ui, &format!("{label}:"));
                let choices: Vec<(u32, String)> =
                    o.get("options").and_then(Value::as_array).map(|a| a.iter().enumerate().map(|(i, x)| (i as u32, x.as_str().unwrap_or_default().to_string())).collect()).unwrap_or_default();
                let refs: Vec<(u32, &str)> = choices.iter().map(|(i, s)| (*i, s.as_str())).collect();
                let mut v = value(state, &m.id, &label, Some(o)) as u32;
                if crate::widgets::dropdown(ui, &format!("plugin-select-{}-{i}", m.id), &mut v, &refs, 130.0) {
                    state.plugin_values.entry(m.id.clone()).or_default().insert(label, v as f32);
                }
            }
            "toggle" => {
                if let Some(t) = native_toggle(b, &label) {
                    crate::widgets::toggle(ui, t, &label);
                } else {
                    let mut on = value(state, &m.id, &label, Some(o)) != 0.0;
                    if crate::widgets::toggle(ui, &mut on, &label).clicked() {
                        state.plugin_values.entry(m.id.clone()).or_default().insert(label, f32::from(u8::from(on)));
                    }
                }
            }
            _ => {}
        }
    }
}

/// What the folder watch does on the UI thread.
#[derive(Default)]
pub struct Watch {
    next_ms: f64,
    started: bool,
    seen: HashMap<std::path::PathBuf, std::time::SystemTime>,
    #[cfg(not(target_arch = "wasm32"))]
    built: Option<(std::sync::mpsc::Sender<Built>, std::sync::mpsc::Receiver<Built>)>,
}

#[cfg(not(target_arch = "wasm32"))]
/// A `.rs` plug-in's build result.
type Built = (std::path::PathBuf, Result<Vec<u8>, String>);

fn say(app: &mut PhotocraftApp, text: String, error: bool) {
    app.ui.status = text;
    app.ui.status_error = error;
}

/// Look at the plug-in folder (twice a second) and load what changed. `PHOTOCRAFT_PLUGINS=0`
/// turns brush plug-ins off.
pub fn poll(app: &mut PhotocraftApp, ctx: &egui::Context) {
    #[cfg(target_arch = "wasm32")]
    let _ = (app, ctx);
    #[cfg(not(target_arch = "wasm32"))]
    {
        if std::env::var_os("PHOTOCRAFT_PLUGINS").is_some_and(|v| v == "0") {
            return;
        }
        let now = crate::gpu_canvas::now_ms();
        let (tx, rx) = app.plugin_watch.built.get_or_insert_with(std::sync::mpsc::channel);
        let tx = tx.clone();
        let results: Vec<_> = rx.try_iter().collect();
        for (path, r) in results {
            match r.and_then(|bytes| photocraft_plugins::brush::BrushPlugin::load(&bytes, photocraft_plugins::brush::BrushPlugin::default_limits()).map_err(|e| e.to_string())) {
                Ok(p) => {
                    let name = p.manifest().name.clone();
                    photocraft_plugins::brush::install(p.with_source(path.display().to_string()));
                    say(app, format!("Plug-in loaded: {name}"), false);
                }
                Err(e) => say(app, e, true),
            }
            ctx.request_repaint();
        }
        if now < app.plugin_watch.next_ms {
            return;
        }
        app.plugin_watch.next_ms = now + 500.0;
        let dir = if app.plugin_watch.started {
            photocraft_engine::brush_script::folder()
        } else {
            app.plugin_watch.started = true;
            photocraft_engine::brush_script::prepare()
        };
        let Some(dir) = dir else { return };
        let Ok(rd) = std::fs::read_dir(&dir) else { return };
        let mut present = Vec::new();
        for e in rd.flatten() {
            let path = e.path();
            let ext = path.extension().map(|x| x.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
            if ext != "wasm" && ext != "rs" {
                continue;
            }
            let Ok(modified) = e.metadata().and_then(|m| m.modified()) else { continue };
            present.push(path.clone());
            if app.plugin_watch.seen.get(&path) == Some(&modified) {
                continue;
            }
            app.plugin_watch.seen.insert(path.clone(), modified);
            if ext == "wasm" {
                match photocraft_engine::brush_plugins::load_file(&path) {
                    Ok(Some(name)) => say(app, format!("Plug-in loaded: {name}"), false),
                    Ok(None) => {}
                    Err(e) => say(app, format!("{}: {e}", path.display()), true),
                }
            } else {
                let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                say(app, format!("Building plug-in {file}…"), false);
                let tx = tx.clone();
                let _ = std::thread::Builder::new().name("photocraft-plugin-build".into()).spawn(move || {
                    let r = photocraft_engine::brush_script::build(&path);
                    let _ = tx.send((path, r));
                });
            }
        }
        let gone: Vec<_> = app.plugin_watch.seen.keys().filter(|p| !present.contains(p)).cloned().collect();
        for p in gone {
            app.plugin_watch.seen.remove(&p);
            if photocraft_plugins::brush::remove_source(&p.display().to_string()) {
                say(app, format!("Plug-in removed: {}", p.display()), false);
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(500));
    }
}
