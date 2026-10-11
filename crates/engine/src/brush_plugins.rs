//! Brush plug-ins in strokes (Fotox's brush plugins, `photocraft_plugins::brush`).
//!
//! `paint.stroke` with `"plugin": "<id>"` paints through that plug-in: the stroke engine still
//! does the tip, spacing, pressure, smoothing, selection, flow and opacity; the plug-in decides
//! each pixel from the pre-stroke pixels and the stroke's build-up. `"pluginParams"` holds the
//! values of the manifest's `params` in order (a number as shown, a toggle 0/1, a drop-down the
//! index). The manifest's `brush` hints shape the tip: `feather` (Size of core plus Feather of
//! fade on each side), `gaussian` (fully soft), `max` accumulation, a default spacing.

use std::sync::Arc;

use photocraft_paint::{BrushSettings, PixelOp};
use photocraft_plugins::brush::{BrushPlugin, HEADER_WORDS, MAX_PARAMS};
use serde_json::Value;

use crate::{EngineError, Result, Session};

/// A stroke's plug-in: the module and the header every rectangle call gets.
#[derive(Clone)]
pub struct StrokePlugin {
    pub plugin: Arc<BrushPlugin>,
    header: [f32; HEADER_WORDS],
    /// Fotox's `max` accumulation.
    pub max: bool,
}

impl StrokePlugin {
    /// The plug-in a stroke's params name, if any.
    pub fn from_params(s: &Session, p: &Value) -> Result<Option<StrokePlugin>> {
        let Some(id) = p.get("plugin").and_then(Value::as_str) else { return Ok(None) };
        let plugin = photocraft_plugins::brush::get(id).ok_or_else(|| EngineError::Other(format!("no brush plug-in {id:?} is loaded")))?;
        if let Some(why) = plugin.stopped() {
            return Err(EngineError::Other(format!("{} is stopped: {why} (save the file again to reload it)", plugin.manifest().name)));
        }
        let mut header = [0.0f32; HEADER_WORDS];
        if let Some(vals) = p.get("pluginParams").and_then(Value::as_array) {
            for (h, v) in header.iter_mut().zip(vals.iter().take(MAX_PARAMS)) {
                *h = v.as_f64().map(|v| v as f32).or_else(|| v.as_bool().map(|b| f32::from(u8::from(b)))).unwrap_or(0.0);
            }
        }
        let color = if plugin.manifest().color == "background" { s.tools.background } else { s.tools.foreground };
        header[16..20].copy_from_slice(&color);
        let max = plugin.manifest().brush.as_ref().and_then(|b| b.accumulate.as_deref()) == Some("max");
        Ok(Some(StrokePlugin { plugin, header, max }))
    }

    /// The brush as the manifest's hints shape it. `feather` is the Feather field's value (px).
    pub fn shape_brush(&self, mut b: BrushSettings, p: &Value) -> BrushSettings {
        b.erase = false;
        let Some(h) = self.plugin.manifest().brush.as_ref() else { return b };
        if p.get("spacing").is_none()
            && let Some(sp) = h.spacing
        {
            b.spacing = sp.clamp(0.01, 10.0);
            b.spacing_enabled = true;
        }
        match h.profile.as_deref() {
            Some("feather") => {
                let feather = p.get("feather").and_then(Value::as_f64).unwrap_or(0.0).max(0.0) as f32;
                let core = b.size.max(1.0);
                b.tip = Default::default();
                b.size = core + 2.0 * feather;
                b.hardness = (core / b.size).clamp(0.0, 1.0);
            }
            Some("gaussian") => {
                b.tip = Default::default();
                b.hardness = 0.0;
            }
            _ => {}
        }
        b
    }

    /// The per-rectangle operation for a layer whose transparency is `lock`ed or not.
    pub fn op(&self, lock: bool) -> PixelOp {
        let mut header = self.header;
        header[20] = f32::from(u8::from(lock));
        let plugin = self.plugin.clone();
        PixelOp(Arc::new(move |r, vals: &mut [f32], channels, k: &[f32]| {
            plugin.paint(&header, vals, channels, k, r.width(), r.height(), r.x0, r.y0).map_err(|e| e.to_string())
        }))
    }
}

/// Loads every brush `.wasm` in `dir` (not recursive). Returns `(loaded names, failures)`.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_folder(dir: &std::path::Path) -> (Vec<String>, Vec<(String, String)>) {
    let mut loaded = Vec::new();
    let mut failed = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return (loaded, failed) };
    let mut files: Vec<std::path::PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wasm"))).collect();
    files.sort();
    for f in files {
        match load_file(&f) {
            Ok(Some(name)) => loaded.push(name),
            Ok(None) => {}
            Err(e) => failed.push((f.display().to_string(), e)),
        }
    }
    (loaded, failed)
}

/// Loads one `.wasm` as a brush plug-in (`Ok(None)`: it is a filter, not a brush).
#[cfg(not(target_arch = "wasm32"))]
pub fn load_file(f: &std::path::Path) -> std::result::Result<Option<String>, String> {
    let bytes = std::fs::read(f).map_err(|e| e.to_string())?;
    if !photocraft_plugins::brush::is_brush_module(&bytes) {
        return Ok(None);
    }
    let p = BrushPlugin::load(&bytes, BrushPlugin::default_limits()).map_err(|e| e.to_string())?.with_source(f.display().to_string());
    let name = p.manifest().name.clone();
    photocraft_plugins::brush::install(p);
    Ok(Some(name))
}
