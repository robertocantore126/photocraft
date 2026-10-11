//! SAI-style Linework layers: editable strokes on a pixel layer (`Layer::linework`), rendered into
//! the layer's own pixels after every edit. Linework tools edit strokes, never pixels.

use std::sync::Arc;

use photocraft_doc::{Document, LayerId, LineKind, LineStroke, Linework};
use photocraft_geom::Rect;
use photocraft_paint::{BrushSettings, StrokePoint, render_stroke};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn f(p: &Value, k: &str, d: f32) -> f32 {
    p.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32)
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn color(v: Option<&Value>, d: [f32; 4]) -> [f32; 4] {
    match v {
        Some(Value::Array(a)) if a.len() >= 3 => {
            let c: Vec<f32> = a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
            [c[0], c[1], c[2], c.get(3).copied().unwrap_or(1.0)]
        }
        Some(Value::String(h)) => {
            let h = h.trim_start_matches('#');
            let c = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map(|v| f32::from(v) / 255.0);
            match (c(0), c(2), c(4)) {
                (Some(r), Some(g), Some(b)) => [r, g, b, c(6).unwrap_or(1.0)],
                _ => d,
            }
        }
        _ => d,
    }
}

fn points3(p: &Value, k: &str, default_w: f32) -> Vec<[f32; 3]> {
    p.get(k)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|q| {
                    let q = q.as_array()?;
                    let x = q.first()?.as_f64()? as f32;
                    let y = q.get(1)?.as_f64()? as f32;
                    let w = q.get(2).and_then(Value::as_f64).map_or(default_w, |v| v as f32);
                    Some([x, y, w.max(0.0)])
                })
                .collect()
        })
        .unwrap_or_default()
}

fn kind_param(p: &Value) -> LineKind {
    match p.get("kind").and_then(Value::as_str) {
        Some("polyline" | "line") => LineKind::Polyline,
        _ => LineKind::Curve,
    }
}

fn layer_of(s: &Session, p: &Value) -> Result<LayerId> {
    match p.get("layer").and_then(Value::as_u64) {
        Some(id) => Ok(LayerId(id)),
        None => s.active().and_then(|d| d.active_layer).ok_or(EngineError::Other("no active layer".into())),
    }
}

fn has_linework(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let l = d.active_layer.and_then(|id| d.doc.layer(id)).ok_or("no active layer")?;
    if l.linework.is_some() { Ok(()) } else { Err("the active layer is not a Linework layer".into()) }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// Ramer–Douglas–Peucker on x, y (keeps the width of the points it keeps).
pub fn simplify(pts: &[[f32; 3]], tol: f32) -> Vec<[f32; 3]> {
    if pts.len() < 3 || tol <= 0.0 {
        return pts.to_vec();
    }
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    keep[pts.len() - 1] = true;
    let mut stack = vec![(0usize, pts.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (p, q) = (pts[a], pts[b]);
        let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        let mut best = (0.0f32, a);
        for (i, r) in pts.iter().enumerate().take(b).skip(a + 1) {
            let d = ((r[0] - p[0]) * dy - (r[1] - p[1]) * dx).abs() / len;
            // A width change is kept too, so pressure survives on straight runs.
            let dw = (r[2] - (p[2] + (q[2] - p[2]) * 0.5)).abs() * 0.5;
            let d = d.max(dw);
            if d > best.0 {
                best = (d, i);
            }
        }
        if best.0 > tol {
            keep[best.1] = true;
            stack.push((a, best.1));
            stack.push((best.1, b));
        }
    }
    pts.iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| *p).collect()
}

/// The stroke as dense points (about every half pixel): a Catmull-Rom curve through the control
/// points, or straight segments. The width is the third value.
pub fn sample(st: &LineStroke) -> Vec<[f32; 3]> {
    let mut pts = st.points.clone();
    if st.closed && pts.len() > 2 {
        pts.push(pts[0]);
    }
    let n = pts.len();
    if n < 2 {
        return pts;
    }
    let mut out = Vec::new();
    for i in 0..n - 1 {
        let (p1, p2) = (pts[i], pts[i + 1]);
        let p0 = if i > 0 { pts[i - 1] } else if st.closed && n > 3 { pts[n - 2] } else { p1 };
        let p3 = if i + 2 < n { pts[i + 2] } else if st.closed && n > 3 { pts[1] } else { p2 };
        let len = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
        let steps = ((len * 2.0).ceil() as usize).clamp(1, 4096);
        for k in 0..steps {
            let t = k as f32 / steps as f32;
            let w = p1[2] + (p2[2] - p1[2]) * t;
            let (x, y) = match st.kind {
                LineKind::Polyline => (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t),
                LineKind::Curve => {
                    let (t2, t3) = (t * t, t * t * t);
                    let c = |a: f32, b: f32, c: f32, d: f32| 0.5 * (2.0 * b + (-a + c) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2 + (-a + 3.0 * b - 3.0 * c + d) * t3);
                    (c(p0[0], p1[0], p2[0], p3[0]), c(p0[1], p1[1], p2[1], p3[1]))
                }
            };
            out.push([x, y, w]);
        }
    }
    if let Some(l) = pts.last() {
        out.push(*l);
    }
    out
}

/// Distance from (x, y) to the stroke's outline (negative inside).
pub fn distance(st: &LineStroke, x: f32, y: f32) -> f32 {
    let s = sample(st);
    let mut best = f32::MAX;
    if s.len() == 1 {
        return ((s[0][0] - x).powi(2) + (s[0][1] - y).powi(2)).sqrt() - s[0][2] / 2.0;
    }
    for w in s.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 1e-9 { (((x - a[0]) * dx + (y - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
        let (px, py) = (a[0] + dx * t, a[1] + dy * t);
        let half = (a[2] + (b[2] - a[2]) * t) / 2.0;
        best = best.min(((px - x).powi(2) + (py - y).powi(2)).sqrt() - half);
    }
    best
}

/// The topmost stroke within `tol` px of (x, y).
pub fn hit(lw: &Linework, x: f32, y: f32, tol: f32) -> Option<usize> {
    (0..lw.strokes.len()).rev().find(|&i| distance(&lw.strokes[i], x, y) <= tol)
}

/// The control point nearest (x, y) within `tol`: (stroke, point).
pub fn nearest_point(lw: &Linework, x: f32, y: f32, tol: f32) -> Option<(usize, usize)> {
    let mut best: Option<(f32, usize, usize)> = None;
    for (si, st) in lw.strokes.iter().enumerate() {
        for (pi, q) in st.points.iter().enumerate() {
            let d = ((q[0] - x).powi(2) + (q[1] - y).powi(2)).sqrt();
            if d <= tol && best.is_none_or(|b| d <= b.0) {
                best = Some((d, si, pi));
            }
        }
    }
    best.map(|b| (b.1, b.2))
}

fn stroke_rect(st: &LineStroke) -> Rect {
    st.bounds().map_or(Rect::EMPTY, |b| Rect::new(b[0].floor() as i32 - 2, b[1].floor() as i32 - 2, b[2].ceil() as i32 + 2, b[3].ceil() as i32 + 2))
}

/// Paint one stroke onto `surf` (limited by `clip`).
fn paint(surf: &mut Surface, st: &LineStroke, clip: Option<&Surface>) {
    let s = sample(st);
    let maxw = s.iter().map(|q| q[2]).fold(0.0f32, f32::max);
    if maxw <= 0.0 || s.is_empty() {
        return;
    }
    let brush = BrushSettings {
        size: maxw,
        hardness: 1.0,
        spacing: 0.05,
        pressure_size: true,
        pressure_opacity: false,
        opacity: st.color[3].clamp(0.0, 1.0),
        flow: 1.0,
        color: [st.color[0], st.color[1], st.color[2], 1.0],
        smoothing: Default::default(),
        ..Default::default()
    };
    let pts: Vec<StrokePoint> = s.iter().map(|q| StrokePoint::new(f64::from(q[0]), f64::from(q[1]), (q[2] / maxw).clamp(0.0, 1.0))).collect();
    render_stroke(surf, &brush, &pts, clip, false, 1.0);
}

/// Re-render the strokes over `area` of `surf` (cleared first); `None` = everything.
pub fn render_into(surf: &mut Surface, lw: &Linework, area: Option<Rect>) {
    match area {
        None => {
            *surf = Surface::new(surf.format());
            for st in &lw.strokes {
                paint(surf, st, None);
            }
        }
        Some(r) if !r.is_empty() => {
            let n = surf.format().channels();
            surf.write_region(r, &vec![0.0f32; r.width() as usize * r.height() as usize * n]);
            let mut clip = Surface::new(photocraft_color::PixelFormat::GRAY8);
            clip.fill_rect(r, &[1.0]);
            for st in lw.strokes.iter().filter(|st| !stroke_rect(st).intersect(&r).is_empty()) {
                paint(surf, st, Some(&clip));
            }
        }
        Some(_) => {}
    }
    surf.prune();
}

/// Apply `edit` to the layer's strokes and re-render the pixels it changed.
fn edit_linework<R>(s: &mut Session, label: &str, id: LayerId, edit: impl FnOnce(&mut Linework) -> Result<(R, Rect)>) -> Result<R> {
    s.edit(label, |doc: &mut Document, _| {
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let mut lw = l.linework.as_deref().cloned().ok_or_else(|| EngineError::Other("not a Linework layer".into()))?;
        let (r, dirty) = edit(&mut lw)?;
        let surf = l.surface_mut().ok_or_else(|| EngineError::Other("not a pixel layer".into()))?;
        render_into(surf, &lw, Some(dirty));
        l.linework = Some(Arc::new(lw));
        crate::sugimori_cmds::refresh_soft_lines(doc, id);
        Ok(r)
    })
}

fn new_linework(s: &mut Session, p: &Value) -> Result<Value> {
    let id = s.edit("New Linework Layer", |doc, active| {
        let name = p.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| doc.next_layer_name("Linework"));
        let mut l = photocraft_doc::Layer::raster(name, doc.pixel_format());
        l.linework = Some(Arc::new(Linework::default()));
        let id = doc.insert_above(*active, l);
        *active = Some(id);
        Ok(id)
    })?;
    Ok(json!({ "layer": id.0 }))
}

fn add_stroke(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "linework.addStroke";
    let id = layer_of(s, p)?;
    let w = f(p, "width", 3.0);
    let mut pts = points3(p, "points", w);
    if pts.is_empty() {
        return Err(bad(cmd, "no points"));
    }
    pts = simplify(&pts, f(p, "simplify", 0.0));
    let st = LineStroke { points: pts, color: color(p.get("color"), s.tools.foreground), kind: kind_param(p), closed: p.get("closed").and_then(Value::as_bool).unwrap_or(false) };
    let label = p.get("label").and_then(Value::as_str).unwrap_or("Linework Pen").to_string();
    let index = s.edit(&label, |doc, _| {
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let mut lw = l.linework.as_deref().cloned().ok_or_else(|| EngineError::Other("not a Linework layer".into()))?;
        let surf = l.surface_mut().ok_or_else(|| EngineError::Other("not a pixel layer".into()))?;
        // A new stroke is on top: painting it over the current pixels is exact.
        paint(surf, &st, None);
        surf.prune();
        lw.strokes.push(st);
        let n = lw.strokes.len() - 1;
        l.linework = Some(Arc::new(lw));
        crate::sugimori_cmds::refresh_soft_lines(doc, id);
        Ok(n)
    })?;
    Ok(json!({ "index": index }))
}

fn erase_at(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_of(s, p)?;
    let r = f(p, "radius", 2.0);
    let pts = points3(p, "points", 0.0);
    let removed = edit_linework(s, "Linework Eraser", id, |lw| {
        let mut dirty = Rect::EMPTY;
        let mut n = 0;
        let mut i = 0;
        while i < lw.strokes.len() {
            if pts.iter().any(|q| distance(&lw.strokes[i], q[0], q[1]) <= r) {
                dirty = dirty.union(&stroke_rect(&lw.strokes[i]));
                lw.strokes.remove(i);
                n += 1;
            } else {
                i += 1;
            }
        }
        Ok((n, dirty))
    })?;
    Ok(json!({ "removed": removed }))
}

fn edit_stroke(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "linework.editStroke";
    let id = layer_of(s, p)?;
    let index = p.get("index").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "missing index"))? as usize;
    let label = p.get("label").and_then(Value::as_str).unwrap_or("Edit Linework").to_string();
    let fg = s.tools.foreground;
    edit_linework(s, &label, id, |lw| {
        let st = lw.strokes.get_mut(index).ok_or_else(|| bad(cmd, format!("no stroke {index}")))?;
        let mut dirty = stroke_rect(st);
        if p.get("delete").and_then(Value::as_bool) == Some(true) {
            lw.strokes.remove(index);
            return Ok(((), dirty));
        }
        if p.get("points").is_some() {
            let pts = points3(p, "points", 3.0);
            if pts.is_empty() {
                return Err(bad(cmd, "no points"));
            }
            st.points = pts;
        }
        if let Some(c) = p.get("color") {
            st.color = color(Some(c), fg);
        }
        if let Some(w) = p.get("width").and_then(Value::as_f64) {
            st.points.iter_mut().for_each(|q| q[2] = w.max(0.0) as f32);
        }
        if let Some(t) = p.get("translate").and_then(Value::as_array) {
            let (dx, dy) = (t.first().and_then(Value::as_f64).unwrap_or(0.0) as f32, t.get(1).and_then(Value::as_f64).unwrap_or(0.0) as f32);
            st.points.iter_mut().for_each(|q| {
                q[0] += dx;
                q[1] += dy;
            });
        }
        if let Some(pt) = p.get("point").and_then(Value::as_u64) {
            let q = st.points.get_mut(pt as usize).ok_or_else(|| bad(cmd, format!("no point {pt}")))?;
            if let Some(v) = p.get("x").and_then(Value::as_f64) {
                q[0] = v as f32;
            }
            if let Some(v) = p.get("y").and_then(Value::as_f64) {
                q[1] = v as f32;
            }
            if let Some(v) = p.get("pointWidth").and_then(Value::as_f64) {
                q[2] = v.max(0.0) as f32;
            }
            if p.get("deletePoint").and_then(Value::as_bool) == Some(true) && st.points.len() > 1 {
                st.points.remove(pt as usize);
            }
        }
        dirty = dirty.union(&stroke_rect(st));
        Ok(((), dirty))
    })?;
    Ok(Value::Null)
}

fn rasterize(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_of(s, p)?;
    s.edit("Rasterize Linework", |doc, _| {
        doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.linework = None;
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Linework command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "layer.new.linework",
            label: "Linework Layer",
            menu: &["Layer", "New"],
            shortcut: None,
            params: r##"{"name":str?}"##,
            enabled: has_doc,
            run: new_linework,
            journal: true,
            job: None,
        },
        CommandSpec {
            id: "linework.addStroke",
            label: "Linework Stroke",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?,"points":[[x,y,width?],…],"width":px=3 (points without one),"color":"#rrggbb"=foreground,"kind":"curve|polyline"="curve","closed":bool=false,"simplify":px=0 (drop points within this distance of the curve)}"##,
            enabled: has_linework,
            run: add_stroke,
            journal: true,
            job: None,
        },
        CommandSpec {
            id: "linework.eraseAt",
            label: "Linework Eraser",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?,"points":[[x,y],…],"radius":px=2} (removes every stroke touched)"##,
            enabled: has_linework,
            run: erase_at,
            journal: true,
            job: None,
        },
        CommandSpec {
            id: "linework.editStroke",
            label: "Edit Linework",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?,"index":n,"delete":bool?,"points":[[x,y,w],…]?,"color":"#rrggbb"?,"width":px? (every point),"translate":[dx,dy]?,"point":n?,"x":px?,"y":px?,"pointWidth":px?,"deletePoint":bool?}"##,
            enabled: has_linework,
            run: edit_stroke,
            journal: true,
            job: None,
        },
        CommandSpec {
            id: "layer.rasterize.linework",
            label: "Rasterize Linework",
            menu: &["Layer", "Rasterize"],
            shortcut: None,
            params: r##"{"layer":id?}"##,
            enabled: has_linework,
            run: rasterize,
            journal: true,
            job: None,
        },
    ]
}
