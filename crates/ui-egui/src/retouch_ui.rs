//! UI for the retouching tools (healing, clone, history brush, blur/sharpen/smudge, dodge/burn/
//! sponge) and the smart selection tools (Quick Selection, Object Selection): gesture → engine
//! command, options bars, and the clone-source marker.

use egui::{Color32, Stroke, vec2};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;
use crate::theme::Tokens;

/// Finish a stroke with a retouching tool. Returns false if `tool` isn't one.
pub fn finish_stroke(app: &mut PhotocraftApp, tool: Tool, points: &[[f64; 3]], mods: egui::Modifiers) -> bool {
    let o = app.ui.tool_options.clone();
    let pts = json!(points);
    let (cmd, mut p): (&str, Value) = match tool {
        Tool::SpotHealing => ("paint.spotHealing", json!({"type": o.spot_type})),
        Tool::Healing | Tool::CloneStamp => {
            let mut p = json!({"aligned": o.clone_aligned, "sampleLayer": o.clone_sample});
            // The Clone Source panel's active slot (set by ⌥-click) drives the stroke: the engine
            // keeps the aligned pairing and applies the slot's scale/rotation/flip.
            let slot = app.session.presets.clone.active().source.is_some();
            match (app.ui.clone_offset.filter(|_| o.clone_aligned && !slot), app.ui.clone_source.filter(|_| !slot)) {
                _ if slot => {}
                (Some(off), _) => p["offset"] = json!(off),
                (None, Some(src)) => p["source"] = json!(src),
                (None, None) => {
                    app.ui.status = "Option-click to define a source point to clone from".into();
                    app.ui.status_error = true;
                    return true;
                }
            }
            (if tool == Tool::Healing { "paint.healingBrush" } else { "paint.cloneStamp" }, p)
        }
        Tool::HistoryBrush => ("paint.historyBrush", json!({})),
        Tool::Blur => ("paint.blur", json!({"strength": o.strength})),
        Tool::Sharpen => ("paint.sharpen", json!({"strength": o.strength, "protectDetail": o.protect_detail})),
        Tool::Smudge => ("paint.smudge", json!({"strength": o.strength, "fingerPainting": o.finger_painting})),
        Tool::Dodge | Tool::Burn => (
            if tool == Tool::Dodge { "paint.dodge" } else { "paint.burn" },
            json!({"range": o.tone_range, "exposure": o.exposure, "protectTones": o.protect_tones}),
        ),
        Tool::Sponge => ("paint.sponge", json!({"mode": o.sponge_mode, "vibrance": o.vibrance})),
        Tool::QuickSelection => {
            let size = app.session.tools.brush.size;
            let mode = if mods.alt { "subtract" } else { "add" };
            let xy: Vec<[f64; 2]> = points.iter().map(|q| [q[0], q[1]]).collect();
            let _ = app
                .run("select.quick", json!({"points": xy, "size": size, "mode": mode, "enhanceEdge": o.enhance_edge, "sampleAllLayers": o.sample_all_layers}));
            return true;
        }
        _ => return false,
    };
    p["points"] = pts;
    match app.run(cmd, p) {
        Ok(r) if matches!(tool, Tool::Healing | Tool::CloneStamp) => {
            // Aligned: keep the offset for later strokes; non-aligned: every stroke restarts at the source.
            app.ui.clone_offset = r.get("offset").and_then(|v| serde_json::from_value(v.clone()).ok()).filter(|_| o.clone_aligned);
        }
        Ok(_) => {}
        Err(e) => {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
    true
}

/// ⌥-click with Clone Stamp / Healing Brush sets the source.
pub fn set_source(app: &mut PhotocraftApp, x: f64, y: f64) {
    app.ui.clone_source = Some([x.round(), y.round()]);
    app.ui.clone_offset = None;
    let _ = app.run("cloneSource.set", json!({"source": [x.round(), y.round()]}));
    app.ui.status = format!("Clone source set at {:.0}, {:.0}", x, y);
    app.ui.status_error = false;
}

/// Crosshair where the clone source is sampled from for the current pointer position.
pub fn draw_source_marker(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    if !matches!(app.ui.tool, Tool::CloneStamp | Tool::Healing) {
        return;
    }
    let src = match (crate::preset_panels::clone_sample_point(app, app.hover_doc), app.ui.clone_offset, app.hover_doc, app.ui.clone_source) {
        (Some(p), ..) => Some(p),
        (None, Some(off), Some(h), _) => Some([h[0] + off[0], h[1] + off[1]]),
        (None, None, _, Some(s)) => Some(s),
        _ => None,
    };
    let Some(s) = src else { return };
    let c = xf.to_screen(s[0] as f32, s[1] as f32);
    for (w, col) in [(3.0, Color32::from_black_alpha(160)), (1.0, Color32::WHITE)] {
        painter.line_segment([c - vec2(7.0, 0.0), c + vec2(7.0, 0.0)], Stroke::new(w, col));
        painter.line_segment([c - vec2(0.0, 7.0), c + vec2(0.0, 7.0)], Stroke::new(w, col));
    }
}

fn opt(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(text).color(t.text_dim).size(12.0));
}

fn pct(ui: &mut egui::Ui, label: &str, v: &mut f32) {
    opt(ui, label);
    crate::widgets::value_field(ui, v, 1.0..=100.0, "%", 58.0);
}

/// Options bar for the retouching and smart-selection tools. Returns false for other tools.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui, tool: Tool) -> bool {
    if !tool.is_brushlike() && !matches!(tool, Tool::QuickSelection | Tool::ObjectSelection) || matches!(tool, Tool::Brush | Tool::PixelBrush | Tool::SaiBrush | Tool::Eraser | Tool::Plugin) {
        return false;
    }
    let o = &mut app.ui.tool_options;
    match tool {
        Tool::SpotHealing => {
            opt(ui, "Type:");
            for (k, l) in [("contentAware", "Content-Aware"), ("createTexture", "Create Texture"), ("proximityMatch", "Proximity Match")] {
                let mut on = o.spot_type == k;
                if crate::widgets::checkbox(ui, &mut on, l).clicked() {
                    o.spot_type = k.into();
                }
            }
            crate::widgets::vline(ui, 22.0);
            crate::widgets::checkbox(ui, &mut o.sample_all_layers, "Sample All Layers");
        }
        Tool::Healing | Tool::CloneStamp => {
            crate::widgets::checkbox(ui, &mut o.clone_aligned, "Aligned");
            opt(ui, "Sample:");
            let opts = [("current".to_string(), "Current Layer"), ("currentAndBelow".to_string(), "Current & Below"), ("all".to_string(), "All Layers")];
            crate::widgets::dropdown(ui, "clone-sample", &mut o.clone_sample, &opts, 130.0);
            if app.ui.clone_source.is_none() {
                crate::widgets::vline(ui, 22.0);
                opt(ui, "⌥-click to set the source");
            }
        }
        Tool::Dodge | Tool::Burn => {
            opt(ui, "Range:");
            let opts = [("shadows".to_string(), "Shadows"), ("midtones".to_string(), "Midtones"), ("highlights".to_string(), "Highlights")];
            crate::widgets::dropdown(ui, "tone-range", &mut o.tone_range, &opts, 100.0);
            pct(ui, "Exposure:", &mut o.exposure);
            crate::widgets::checkbox(ui, &mut o.protect_tones, "Protect Tones");
        }
        Tool::Sponge => {
            opt(ui, "Mode:");
            let opts = [("desaturate".to_string(), "Desaturate"), ("saturate".to_string(), "Saturate")];
            crate::widgets::dropdown(ui, "sponge-mode", &mut o.sponge_mode, &opts, 110.0);
            crate::widgets::checkbox(ui, &mut o.vibrance, "Vibrance");
        }
        Tool::Blur | Tool::Sharpen | Tool::Smudge => {
            pct(ui, "Strength:", &mut o.strength);
            crate::widgets::checkbox(ui, &mut o.sample_all_layers, "Sample All Layers");
            if tool == Tool::Sharpen {
                crate::widgets::checkbox(ui, &mut o.protect_detail, "Protect Detail");
            }
            if tool == Tool::Smudge {
                crate::widgets::checkbox(ui, &mut o.finger_painting, "Finger Painting");
            }
        }
        Tool::HistoryBrush => opt(ui, "Paints from the document's opening state"),
        Tool::QuickSelection => {
            crate::widgets::checkbox(ui, &mut o.sample_all_layers, "Sample All Layers");
            crate::widgets::checkbox(ui, &mut o.enhance_edge, "Enhance Edge");
            opt(ui, "⌥ to subtract");
            crate::widgets::vline(ui, 22.0);
            if crate::widgets::secondary_button(ui, "Select Subject", 0.0).clicked() {
                let _ = app.run("select.subject", json!({}));
            }
        }
        Tool::ObjectSelection => {
            crate::widgets::checkbox(ui, &mut o.sample_all_layers, "Sample All Layers");
            opt(ui, "Drag a rectangle around the object");
            crate::widgets::vline(ui, 22.0);
            if crate::widgets::secondary_button(ui, "Select Subject", 0.0).clicked() {
                let _ = app.run("select.subject", json!({}));
            }
        }
        _ => {}
    }
    true
}
