//! Linework tool (PaintTool SAI's linework tools in one toolbar slot): Pen, Eraser, Curve, Line,
//! Edit, Pressure, Weight and Color, picked in the options bar. Each gesture becomes one engine
//! command (`linework.addStroke`, `linework.eraseAt`, `linework.editStroke`); this module only
//! turns pointer gestures into them and draws the in-progress stroke and the control points.

use egui::{Color32, Stroke, vec2};
use photocraft_doc::Linework;
use serde_json::json;

use crate::canvas::{ToolEvent, ViewXform};
use crate::state::Tool;
use crate::{PhotocraftApp, widgets};

/// The sub-tools, in SAI's order: (id, label).
pub const MODES: [(&str, &str); 8] = [
    ("pen", "Pen"),
    ("eraser", "Eraser"),
    ("curve", "Curve"),
    ("line", "Line"),
    ("edit", "Edit"),
    ("pressure", "Pressure"),
    ("weight", "Weight"),
    ("color", "Color"),
];

/// In-progress gesture (not saved).
#[derive(Clone, Debug, Default)]
pub struct LineworkUi {
    /// Pen / Eraser drag: x, y, width.
    drag: Vec<[f32; 3]>,
    /// Curve / Line clicks so far.
    clicks: Vec<[f32; 3]>,
    /// Edit / Pressure: (stroke, point, original point, current pointer).
    point: Option<(usize, usize, [f32; 3], [f32; 2])>,
    /// Edit with ⌘: (stroke, start, current pointer).
    moving: Option<(usize, [f32; 2], [f32; 2])>,
}

fn linework(app: &PhotocraftApp) -> Option<(u64, std::sync::Arc<Linework>)> {
    let st = app.session.active()?;
    let l = st.active_layer.and_then(|id| st.doc.layer(id))?;
    Some((l.id.0, l.linework.clone()?))
}

fn tol(app: &PhotocraftApp) -> f32 {
    8.0 / app.current_zoom().max(0.01)
}

fn width(app: &PhotocraftApp, pressure: f32) -> f32 {
    let o = &app.ui.tool_options;
    let w = o.linework_width.max(0.1);
    if o.linework_pressure { (w * pressure.clamp(0.05, 1.0)).max(0.3) } else { w }
}

pub(crate) fn commit_clicks(app: &mut PhotocraftApp) {
    let pts = std::mem::take(&mut app.linework.clicks);
    if pts.len() >= 2 {
        let kind = if app.ui.tool_options.linework_mode == "line" { "polyline" } else { "curve" };
        let label = if kind == "polyline" { "Linework Line" } else { "Linework Curve" };
        let _ = app.run("linework.addStroke", json!({"points": pts, "kind": kind, "label": label}));
    }
}

/// Pointer events for the Linework tool. `false` when another tool is active.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    if app.ui.tool != Tool::Linework {
        return false;
    }
    let mode = app.ui.tool_options.linework_mode.clone();
    match ev {
        ToolEvent::Down { x, y, pressure } => {
            let (x, y) = (x as f32, y as f32);
            // Linework tools draw on a Linework layer: make one when the active layer isn't.
            if linework(app).is_none() && app.run("layer.new.linework", json!({})).is_err() {
                return true;
            }
            let Some((_, lw)) = linework(app) else { return true };
            let t = tol(app);
            match mode.as_str() {
                "pen" => app.linework.drag = vec![[x, y, width(app, pressure)]],
                "eraser" => app.linework.drag = vec![[x, y, 0.0]],
                "curve" | "line" => {
                    let w = app.ui.tool_options.linework_width;
                    app.linework.clicks.push([x, y, w]);
                }
                "edit" if mods.command => {
                    if let Some(i) = photocraft_engine::linework_cmds::hit(&lw, x, y, t) {
                        app.linework.moving = Some((i, [x, y], [x, y]));
                    }
                }
                "edit" | "pressure" => {
                    if let Some((si, pi)) = photocraft_engine::linework_cmds::nearest_point(&lw, x, y, t) {
                        if mode == "edit" && mods.shift {
                            let _ = app.run("linework.editStroke", json!({"index": si, "point": pi, "deletePoint": true, "label": "Delete Linework Point"}));
                        } else if let Some(q) = lw.strokes.get(si).and_then(|s| s.points.get(pi)) {
                            app.linework.point = Some((si, pi, *q, [x, y]));
                        }
                    }
                }
                "weight" => {
                    if let Some(i) = photocraft_engine::linework_cmds::hit(&lw, x, y, t) {
                        let w = app.ui.tool_options.linework_width;
                        let _ = app.run("linework.editStroke", json!({"index": i, "width": w, "label": "Linework Weight"}));
                    }
                }
                "color" => {
                    if let Some(i) = photocraft_engine::linework_cmds::hit(&lw, x, y, t) {
                        let c = app.session.tools.foreground;
                        let _ = app.run("linework.editStroke", json!({"index": i, "color": [c[0], c[1], c[2], 1.0], "label": "Linework Color"}));
                    }
                }
                _ => {}
            }
        }
        ToolEvent::Move { x, y, pressure } => {
            let (x, y) = (x as f32, y as f32);
            let w = width(app, pressure);
            let l = &mut app.linework;
            if !l.drag.is_empty() {
                if l.drag.last().is_none_or(|p| (p[0] - x).abs() + (p[1] - y).abs() > 0.5) {
                    l.drag.push([x, y, w]);
                }
            } else if let Some(p) = l.point.as_mut() {
                p.3 = [x, y];
            } else if let Some(m) = l.moving.as_mut() {
                m.2 = [x, y];
            }
        }
        ToolEvent::Up { .. } => {
            let drag = std::mem::take(&mut app.linework.drag);
            if !drag.is_empty() {
                if mode == "eraser" {
                    let pts: Vec<[f32; 2]> = drag.iter().map(|p| [p[0], p[1]]).collect();
                    let r = tol(app) / 2.0;
                    let _ = app.run("linework.eraseAt", json!({"points": pts, "radius": r}));
                } else {
                    let simplify = app.ui.tool_options.linework_simplify / app.current_zoom().max(0.01);
                    let _ = app.run("linework.addStroke", json!({"points": drag, "simplify": simplify}));
                }
            }
            if let Some((si, pi, q, cur)) = app.linework.point.take() {
                if mode == "pressure" {
                    // Dragging up thickens, down thins: 1 px of width per 4 screen px.
                    let dy = (cur[1] - q[1]) * app.current_zoom() / 4.0;
                    let w = (q[2] - dy).max(0.0);
                    let _ = app.run("linework.editStroke", json!({"index": si, "point": pi, "pointWidth": w, "label": "Linework Pressure"}));
                } else if (cur[0] - q[0]).abs() + (cur[1] - q[1]).abs() > 0.0 {
                    let _ = app.run("linework.editStroke", json!({"index": si, "point": pi, "x": cur[0], "y": cur[1], "label": "Move Linework Point"}));
                }
            }
            if let Some((i, a, b)) = app.linework.moving.take()
                && (a[0] != b[0] || a[1] != b[1])
            {
                let _ = app.run("linework.editStroke", json!({"index": i, "translate": [b[0] - a[0], b[1] - a[1]], "label": "Move Linework Stroke"}));
            }
        }
    }
    true
}

/// Enter finishes a Curve / Line, Escape drops it.
pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.ui.tool != Tool::Linework || app.linework.clicks.is_empty() {
        return;
    }
    let (enter, esc) = ctx.input(|i| (i.key_pressed(egui::Key::Enter), i.key_pressed(egui::Key::Escape)));
    if esc {
        app.linework.clicks.clear();
    } else if enter {
        commit_clicks(app);
    }
}

/// The stroke being drawn, the pending Curve / Line and, for Edit / Pressure, the control points.
pub fn draw_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    if app.ui.tool != Tool::Linework {
        return;
    }
    let fg = app.session.tools.foreground;
    let col = Color32::from_rgb((fg[0] * 255.0) as u8, (fg[1] * 255.0) as u8, (fg[2] * 255.0) as u8);
    let accent = crate::theme::Tokens::get(painter.ctx()).accent;
    let l = &app.linework;
    let mode = app.ui.tool_options.linework_mode.as_str();
    if mode == "pen" {
        for w in l.drag.windows(2) {
            let (a, b) = (xf.to_screen(w[0][0], w[0][1]), xf.to_screen(w[1][0], w[1][1]));
            painter.line_segment([a, b], Stroke::new((w[1][2] * xf.zoom).max(1.0), col));
        }
    }
    if mode == "eraser" && l.drag.len() > 1 {
        let pts: Vec<_> = l.drag.iter().map(|p| xf.to_screen(p[0], p[1])).collect();
        painter.add(egui::Shape::line(pts, Stroke::new(1.0, Color32::from_rgb(220, 60, 60))));
    }
    if !l.clicks.is_empty() {
        let mut pts: Vec<_> = l.clicks.iter().map(|p| xf.to_screen(p[0], p[1])).collect();
        if let Some(h) = app.hover_doc {
            let next = xf.to_screen(h[0] as f32, h[1] as f32);
            if pts.last().is_none_or(|p| p.distance(next) > 0.01) {
                pts.push(next);
            }
        }
        let stroke = photocraft_doc::LineStroke {
            points: pts.iter().map(|p| [p.x, p.y, app.ui.tool_options.linework_width * xf.zoom]).collect(),
            color: fg,
            kind: if mode == "line" { photocraft_doc::LineKind::Polyline } else { photocraft_doc::LineKind::Curve },
            closed: false,
        };
        let curve = photocraft_engine::linework_cmds::sample(&stroke);
        painter.add(egui::Shape::line(
            curve.iter().map(|p| egui::pos2(p[0], p[1])).collect(),
            Stroke::new((app.ui.tool_options.linework_width * xf.zoom).max(1.0), col),
        ));
        for p in pts.iter().take(l.clicks.len()) {
            painter.rect_filled(egui::Rect::from_center_size(*p, vec2(5.0, 5.0)), 0.0, accent);
        }
    }
    if matches!(mode, "edit" | "pressure") {
        let Some((_, lw)) = linework(app) else { return };
        for (si, st) in lw.strokes.iter().enumerate() {
            let pts: Vec<_> = st
                .points
                .iter()
                .enumerate()
                .map(|(pi, q)| match l.point {
                    Some((s, p, _, cur)) if s == si && p == pi && mode == "edit" => xf.to_screen(cur[0], cur[1]),
                    _ => xf.to_screen(q[0], q[1]),
                })
                .collect();
            let off = match l.moving {
                Some((i, a, b)) if i == si => vec2((b[0] - a[0]) * xf.zoom, (b[1] - a[1]) * xf.zoom),
                _ => vec2(0.0, 0.0),
            };
            painter.add(egui::Shape::line(pts.iter().map(|p| *p + off).collect(), Stroke::new(1.0, accent.gamma_multiply(0.6))));
            for p in pts {
                painter.rect_stroke(egui::Rect::from_center_size(p + off, vec2(6.0, 6.0)), 0.0, Stroke::new(1.0, accent), egui::StrokeKind::Middle);
            }
        }
    }
}

/// Options bar: the sub-tool, width, pressure, stabilizer.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let modes: Vec<_> = MODES.iter().map(|(id, label)| (id.to_string(), *label)).collect();
    let mut mode = app.ui.tool_options.linework_mode.clone();
    if widgets::dropdown(ui, "linework-mode", &mut mode, &modes, 100.0) {
        commit_clicks(app);
        app.ui.tool_options.linework_mode = mode;
    }
    if widgets::secondary_button(ui, "SAI Kit", 0.0).clicked() {
        app.ui.panels.sai_kit = true;
    }
    let o = &mut app.ui.tool_options;
    widgets::vline(ui, 22.0);
    ui.label("Width");
    widgets::value_field(ui, &mut o.linework_width, 0.5..=200.0, "px", 62.0);
    widgets::checkbox(ui, &mut o.linework_pressure, "Pressure");
    ui.label("Simplify");
    widgets::value_field(ui, &mut o.linework_simplify, 0.0..=10.0, "", 46.0);
    if !app.linework.clicks.is_empty() {
        widgets::vline(ui, 22.0);
        if widgets::secondary_button(ui, "Finish (Enter)", 0.0).clicked() {
            commit_clicks(app);
        }
    }
}

/// The second press of a double-click has already added a control point.
pub(crate) fn finish_double_click(app: &mut PhotocraftApp) {
    if !matches!(app.ui.tool_options.linework_mode.as_str(), "curve" | "line") {
        return;
    }
    if app.linework.clicks.len() > 2 {
        app.linework.clicks.pop();
    }
    commit_clicks(app);
}

#[cfg(test)]
mod kit_tests {
    use super::*;

    #[test]
    fn double_click_finishes_curve_without_duplicate_endpoint() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.run("layer.new.linework", json!({})).unwrap();
        app.ui.tool = Tool::Linework;
        app.ui.tool_options.linework_mode = "curve".into();
        app.linework.clicks = vec![[5.0, 5.0, 2.0], [30.0, 10.0, 2.0], [30.0, 10.0, 2.0]];
        finish_double_click(&mut app);
        let (_, lw) = linework(&app).unwrap();
        assert_eq!(lw.strokes.len(), 1);
        assert_eq!(lw.strokes[0].points.len(), 2);
        assert!(app.linework.clicks.is_empty());
        app.run("edit.undo", json!({})).unwrap();
        assert!(linework(&app).unwrap().1.strokes.is_empty());
    }

    #[test]
    fn sai_and_pixel_options_are_not_swallowed_by_retouch_tools() {
        use egui_kittest::{Harness, kittest::Queryable};
        for (tool, label) in [(Tool::SaiBrush, "Size:"), (Tool::PixelBrush, "Size:"), (Tool::Linework, "Width")] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
            app.ui.tool = tool;
            let mut harness = Harness::builder().with_size(egui::vec2(1280.0, 720.0)).build_eframe(move |cc| {
                PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
                app
            });
            harness.run_steps(3);
            assert!(harness.query_by_label(label).is_some(), "missing options for {tool:?}");
        }
    }

    #[test]
    fn sai_panel_toggle_roundtrips_state_and_checkmark() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, "window.toggle.saiKit", json!({})).unwrap();
        assert_eq!(crate::menus::menu_items(&app).iter().find(|m| m.id == "window.toggle.saiKit").unwrap().checked, Some(true));
        let encoded = serde_json::to_value(&app.ui.panels).unwrap();
        let restored: crate::state::Panels = serde_json::from_value(encoded).unwrap();
        assert!(restored.sai_kit);
        crate::menus::invoke(&mut app, &ctx, "window.toggle.saiKit", json!({})).unwrap();
        assert!(!app.ui.panels.sai_kit);
    }
}
