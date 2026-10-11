//! The Magnetic Lasso (ported from Fotox): the path snaps to edges.
//!
//! A click starts the path; moving the pointer shows the live-wire from the last anchor to the
//! strongest edge within Width of the pointer (`photocraft_algo::livewire` on the composite's
//! luminance in a window around both). A click adds an anchor, anchors are also added
//! automatically when the live segment grows past a length set by Frequency, Backspace removes
//! the last anchor, Enter, a double-click or a click on the first anchor closes the path into one
//! `select.lasso` (a polygon), Escape cancels. The composite is the document as it was when the
//! path started.

use std::collections::HashMap;
use std::sync::Arc;

use egui::{Color32, Pos2, Rect, Stroke, vec2};
use photocraft_doc::Document;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::Tool;

const TILE: i64 = 256;
const DOUBLE_CLICK_MS: f64 = 400.0;

pub struct Magnetic {
    doc: Arc<Document>,
    anchors: Vec<[f64; 2]>,
    /// The fixed path: one segment per anchor after the first.
    segments: Vec<Vec<[f64; 2]>>,
    /// The live segment from the last anchor to the pointer.
    live: Vec<[f64; 2]>,
    mode: &'static str,
    lum: HashMap<(i64, i64), Arc<Vec<f32>>>,
    last_press: Option<(f64, [f64; 2])>,
    hover: Option<[f64; 2]>,
}

impl Magnetic {
    /// Luminance (× alpha) of a canvas tile of the composite, cached.
    fn lum_tile(&mut self, tx: i64, ty: i64) -> Arc<Vec<f32>> {
        if let Some(t) = self.lum.get(&(tx, ty)) {
            return t.clone();
        }
        let r = photocraft_geom::Rect::new((tx * TILE) as i32, (ty * TILE) as i32, ((tx + 1) * TILE) as i32, ((ty + 1) * TILE) as i32);
        let buf = photocraft_compose::render(&self.doc, r);
        let lum: Arc<Vec<f32>> = Arc::new(buf.px.iter().map(|p| (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) * p[3].clamp(0.0, 1.0)).collect());
        self.lum.insert((tx, ty), lum.clone());
        lum
    }

    /// The luminance of a canvas window sampled every `scale` pixels (edge-replicated).
    fn window(&mut self, x0: i64, y0: i64, w: usize, h: usize, scale: i64) -> Vec<f32> {
        let (dw, dh) = (i64::from(self.doc.size.width), i64::from(self.doc.size.height));
        let mut out = vec![0.0f32; w * h];
        for y in 0..h as i64 {
            let cy = (y0 + y * scale).clamp(0, dh - 1);
            for x in 0..w as i64 {
                let cx = (x0 + x * scale).clamp(0, dw - 1);
                let t = self.lum_tile(cx.div_euclid(TILE), cy.div_euclid(TILE));
                out[y as usize * w + x as usize] = t.get((cy.rem_euclid(TILE) * TILE + cx.rem_euclid(TILE)) as usize).copied().unwrap_or(0.0);
            }
        }
        out
    }

    /// The live-wire from the last anchor to the edge near `p`.
    fn trace(&mut self, p: [f64; 2], width: f64, contrast: f32) -> Vec<[f64; 2]> {
        let Some(&a) = self.anchors.last() else { return Vec::new() };
        let x0 = (a[0].min(p[0]) - width - 4.0).floor() as i64;
        let y0 = (a[1].min(p[1]) - width - 4.0).floor() as i64;
        let x1 = (a[0].max(p[0]) + width + 4.0).ceil() as i64;
        let y1 = (a[1].max(p[1]) + width + 4.0).ceil() as i64;
        let scale = ((x1 - x0).max(y1 - y0).max(1) + 1023) / 1024;
        let (w, h) = (((x1 - x0 + scale - 1) / scale).max(1) as usize, ((y1 - y0 + scale - 1) / scale).max(1) as usize);
        let lum = self.window(x0, y0, w, h, scale);
        let cost = photocraft_algo::livewire::edge_cost(&lum, w, h, contrast);
        // Snap the end to the cheapest (strongest-edge) pixel within Width.
        let (px, py) = (((p[0] - x0 as f64) / scale as f64) as i64, ((p[1] - y0 as f64) / scale as f64) as i64);
        let r = (width / scale as f64).ceil() as i64;
        let mut end = (px.clamp(0, w as i64 - 1) as usize, py.clamp(0, h as i64 - 1) as usize);
        let mut best = f32::INFINITY;
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (px + dx, py + dy);
                if dx * dx + dy * dy > r * r || x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                    continue;
                }
                let c = cost[y as usize * w + x as usize];
                if c < best {
                    best = c;
                    end = (x as usize, y as usize);
                }
            }
        }
        let start = (
            (((a[0] - x0 as f64) / scale as f64) as i64).clamp(0, w as i64 - 1) as usize,
            (((a[1] - y0 as f64) / scale as f64) as i64).clamp(0, h as i64 - 1) as usize,
        );
        let mut path: Vec<[f64; 2]> = photocraft_algo::livewire::live_wire(&cost, w, h, start, end)
            .into_iter()
            .map(|(x, y)| [x as f64 * scale as f64 + x0 as f64 + 0.5, y as f64 * scale as f64 + y0 as f64 + 0.5])
            .collect();
        if let Some(first) = path.first_mut() {
            *first = a;
        }
        path
    }

    fn anchor_live(&mut self) {
        if let Some(&end) = self.live.last() {
            self.segments.push(std::mem::take(&mut self.live));
            self.anchors.push(end);
        }
    }

    fn path(&self) -> Vec<[f64; 2]> {
        let mut points: Vec<[f64; 2]> = self.anchors.first().copied().into_iter().collect();
        for s in &self.segments {
            points.extend(s.iter().skip(1));
        }
        points
    }
}

fn options(app: &PhotocraftApp) -> (f64, f32, f64) {
    let o = &app.ui.tool_options;
    let width = f64::from(o.magnetic_width.clamp(1.0, 256.0));
    let contrast = o.magnetic_contrast.clamp(1.0, 100.0) / 100.0;
    // Anchor spacing from Frequency (0–100; Photoshop's default 57).
    let spacing = 20.0 + (100.0 - f64::from(o.magnetic_frequency.clamp(0.0, 100.0))) * 3.0;
    (width, contrast, spacing)
}

/// Close the path into a selection.
pub fn close(app: &mut PhotocraftApp) {
    let Some(mut m) = app.magnetic.take() else { return };
    m.anchor_live();
    let points = m.path();
    if points.len() >= 3 {
        let _ = app.run("select.lasso", json!({"points": points, "mode": m.mode, "antiAlias": app.ui.tool_options.anti_alias}));
    }
}

pub fn cancel(app: &mut PhotocraftApp) {
    app.magnetic = None;
}

/// Backspace: drop the last anchor (the whole path when only the first is left).
pub fn remove_last(app: &mut PhotocraftApp) {
    let Some(m) = app.magnetic.as_mut() else { return };
    m.live.clear();
    if m.segments.pop().is_some() {
        m.anchors.pop();
    } else {
        app.magnetic = None;
    }
}

pub fn active(app: &PhotocraftApp) -> bool {
    app.magnetic.is_some()
}

/// Pointer input with the Magnetic Lasso. `mode` is the selection mode the press chooses.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mode: &'static str) -> bool {
    if app.ui.tool != Tool::MagneticLasso {
        app.magnetic = None;
        return false;
    }
    let (width, contrast, spacing) = options(app);
    let zoom = f64::from(app.current_zoom().max(0.01));
    match ev {
        ToolEvent::Down { x, y, .. } => {
            let now = crate::gpu_canvas::now_ms();
            let p = [x, y];
            let Some(doc) = app.session.active().map(|st| st.doc.clone()) else { return true };
            let Some(m) = app.magnetic.as_mut() else {
                app.magnetic =
                    Some(Magnetic { doc, anchors: vec![p], segments: Vec::new(), live: Vec::new(), mode, lum: HashMap::new(), last_press: Some((now, p)), hover: Some(p) });
                return true;
            };
            let double = m.last_press.is_some_and(|(t, q)| now - t <= DOUBLE_CLICK_MS && (q[0] - x).hypot(q[1] - y) * zoom < 5.0);
            m.last_press = Some((now, p));
            let on_first = m.anchors.len() >= 2 && m.anchors.first().is_some_and(|f| (f[0] - x).hypot(f[1] - y) * zoom < 6.0);
            if double || on_first {
                if !on_first {
                    m.live = m.trace(p, width, contrast);
                }
                close(app);
                return true;
            }
            m.live = m.trace(p, width, contrast);
            m.anchor_live();
        }
        ToolEvent::Move { x, y, .. } => hover(app, [x, y], width, contrast, spacing),
        ToolEvent::Up { .. } => {}
    }
    true
}

fn hover(app: &mut PhotocraftApp, p: [f64; 2], width: f64, contrast: f32, spacing: f64) {
    let Some(m) = app.magnetic.as_mut() else { return };
    if m.hover == Some(p) {
        return;
    }
    m.hover = Some(p);
    m.live = m.trace(p, width, contrast);
    // Frequency: an automatic anchor once the segment is long.
    if let Some(&a) = m.anchors.last()
        && (p[0] - a[0]).hypot(p[1] - a[1]) > spacing
    {
        m.anchor_live();
    }
}

/// The pointer moved over the canvas (no button): follow it with the live-wire.
pub fn hover_at(app: &mut PhotocraftApp, p: Option<[f64; 2]>) {
    if app.ui.tool != Tool::MagneticLasso {
        app.magnetic = None;
        return;
    }
    if let Some(p) = p {
        let (width, contrast, spacing) = options(app);
        hover(app, p, width, contrast, spacing);
    }
}

/// The path so far, the live segment and the anchors.
pub fn draw(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(m) = &app.magnetic else { return };
    let mut pts: Vec<Pos2> = m.path().iter().map(|p| xf.to_screen(p[0] as f32, p[1] as f32)).collect();
    pts.extend(m.live.iter().skip(1).map(|p| xf.to_screen(p[0] as f32, p[1] as f32)));
    if pts.len() >= 2 {
        crate::tool_feedback::draw_ants(painter, &pts, false);
    }
    for a in &m.anchors {
        let c = xf.to_screen(a[0] as f32, a[1] as f32);
        let r = Rect::from_center_size(c, vec2(5.0, 5.0));
        painter.rect_filled(r, 0.0, Color32::WHITE);
        painter.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::BLACK), egui::StrokeKind::Outside);
    }
}
