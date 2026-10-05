//! Gate for the canvas damage rect: after any command, every pixel of the composite that changed
//! must lie inside the damage the canvas repaints (`DocState::damage_since`, used as is: it
//! already covers the effects of the changed layers and their groups). A pixel outside it would
//! stay stale on screen. Runs every registered command with default params on a small but varied document.

use std::panic::{AssertUnwindSafe, catch_unwind};

use photocraft_doc::Document;
use photocraft_engine::{Session, command_specs};
use photocraft_geom::Rect;
use serde_json::json;

fn fresh() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 600, "height": 400})).unwrap();
    // A painted layer with a mask and a drop shadow.
    s.execute("layer.new.layer", json!({"name": "a"})).unwrap();
    s.execute("select.rect", json!({"x": 40, "y": 30, "width": 200, "height": 120})).unwrap();
    s.execute("edit.fill", json!({"color": "#cc3311"})).unwrap();
    s.execute("select.rect", json!({"x": 60, "y": 40, "width": 120, "height": 90, "ellipse": true})).unwrap();
    s.execute("layer.layerMask.revealSelection", json!({})).unwrap();
    s.execute("layer.layerStyle.dropShadow", json!({"distance": 8, "size": 10})).unwrap();
    // Two layers in a group.
    s.execute("layer.new.layer", json!({"name": "b"})).unwrap();
    s.execute("select.rect", json!({"x": 330, "y": 200, "width": 150, "height": 100})).unwrap();
    s.execute("edit.fill", json!({"color": "#2255ee"})).unwrap();
    s.execute("layer.new.layer", json!({"name": "c"})).unwrap();
    s.execute("select.rect", json!({"x": 300, "y": 60, "width": 80, "height": 80})).unwrap();
    s.execute("edit.fill", json!({"color": "#22aa44"})).unwrap();
    s.execute("layer.groupLayers", json!({})).unwrap();
    // A shadow on the group: editing a child moves the group's shadow beyond the child's pixels.
    s.execute("layer.layerStyle.dropShadow", json!({"distance": 15, "size": 20})).unwrap();
    // An adjustment layer and a pixel layer above it, with a selection left active.
    s.execute("layer.newAdjustmentLayer.hueSaturation", json!({"hue": 20})).unwrap();
    s.execute("layer.new.layer", json!({"name": "top"})).unwrap();
    s.execute("select.rect", json!({"x": 400, "y": 20, "width": 120, "height": 60})).unwrap();
    s.execute("edit.fill", json!({"color": "#eeee22"})).unwrap();
    s.execute("select.rect", json!({"x": 100, "y": 100, "width": 300, "height": 200})).unwrap();
    s
}

/// The first pixel that differs between the composites of `a` and `b` outside `inside`.
fn changed_outside(a: &Document, b: &Document, inside: Rect) -> Option<(i32, i32)> {
    let r = a.bounds();
    let (pa, pb) = (photocraft_compose::render(a, r), photocraft_compose::render(b, r));
    let w = r.width() as usize;
    pa.px.iter().zip(&pb.px).enumerate().find_map(|(i, (x, y))| {
        let (px, py) = ((i % w) as i32 + r.x0, (i / w) as i32 + r.y0);
        let differs = x.iter().zip(y).any(|(u, v)| (u - v).abs() > 1e-6);
        (differs && !inside.contains(px, py)).then_some((px, py))
    })
}

#[test]
fn every_changed_pixel_is_inside_the_damage() {
    let mut bad = Vec::new();
    let mut checked = 0;
    for spec in command_specs() {
        // Commands that leave the document or need files are skipped by their errors.
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut s = fresh();
            let st = s.active().unwrap();
            let (rev, id, before) = (st.revision, st.doc.id, st.doc.clone());
            if s.execute(spec.id, json!({})).is_err() {
                return None;
            }
            let st = s.documents().iter().find(|d| d.doc.id == id)?;
            if st.revision == rev || st.doc.size != before.size {
                return None;
            }
            let damage = st.damage_since(rev)?;
            Some(changed_outside(&before, &st.doc, damage).map(|p| format!("{}: pixel {p:?} changed outside {damage:?}", spec.id)))
        }));
        match r {
            Ok(Some(Some(msg))) => bad.push(msg),
            Ok(Some(None)) => checked += 1,
            _ => {}
        }
    }
    eprintln!("{checked} commands changed the document with a known damage rect");
    assert!(checked > 50, "only {checked} commands were checked");
    assert!(bad.is_empty(), "{} commands leave stale pixels:\n{}", bad.len(), bad.join("\n"));
}
