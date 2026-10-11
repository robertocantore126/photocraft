//! Ken Sugimori watercolour set (after the noseclub "Sugimori watercolours" guide, our own
//! implementation): a group of role layers (`LayerRole`) — sketch, white base generated from the
//! line art, watercolour clipped to the base, paper texture, the line art blurred and the line art
//! itself as a Linework layer. The soft line is regenerated whenever the line changes.

use std::sync::Arc;

use photocraft_algo::paint::fill_coverage;
use photocraft_algo::selection as sel;
use photocraft_color::{BlendMode, Color, PixelFormat};
use photocraft_doc::{Document, Fill, Layer, LayerContent, LayerId, LayerRole, Linework};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn in_set(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    if role_sibling(&d.doc, id, |r| matches!(r, LayerRole::SugiLine)).is_some() { Ok(()) } else { Err("select a layer of a Sugimori set".into()) }
}

/// The layers next to `id` (same parent), `id` included.
fn siblings(doc: &Document, id: LayerId) -> Vec<LayerId> {
    let Some(path) = doc.path_of(id) else { return Vec::new() };
    let parent = &path[..path.len().saturating_sub(1)];
    let list: &[Layer] = if parent.is_empty() { &doc.layers } else { doc.layer_at(parent).and_then(|g| g.children()).unwrap_or(&[]) };
    list.iter().map(|l| l.id).collect()
}

fn role_sibling(doc: &Document, id: LayerId, want: impl Fn(&LayerRole) -> bool) -> Option<LayerId> {
    siblings(doc, id).into_iter().find(|s| doc.layer(*s).and_then(|l| l.role.as_ref()).is_some_and(&want))
}

/// `src` blurred by a Gaussian of `radius` px (premultiplied, so edges fade to transparent).
fn blur_surface(src: &Surface, bounds: photocraft_geom::Rect, radius: f32) -> Surface {
    let fmt = src.format();
    let mut out = Surface::new(fmt);
    let content = src.content_bounds();
    if content.is_empty() {
        return out;
    }
    let r = content.inflate((radius * 3.0).ceil() as i32 + 1).intersect(&bounds);
    let (w, h) = (r.width() as usize, r.height() as usize);
    let mut px = vec![[0.0f32; 4]; w * h];
    src.read_rgba_into(r, &mut px);
    let mut planes: Vec<Vec<f32>> = (0..4).map(|c| px.iter().map(|p| if c == 3 { p[3] } else { p[c] * p[3] }).collect()).collect();
    for p in planes.iter_mut() {
        *p = photocraft_algo::matting::gaussian_blur(p, w, h, radius.max(0.1));
    }
    let n = fmt.channels();
    let mut data = Vec::with_capacity(w * h * n);
    for (((r, g), b), a) in planes[0].iter().zip(&planes[1]).zip(&planes[2]).zip(&planes[3]) {
        let a = a.clamp(0.0, 1.0);
        let un = |v: f32| if a > 1e-6 { (v / a).clamp(0.0, 1.0) } else { 0.0 };
        let v = photocraft_raster::from_rgba(&fmt, [un(*r), un(*g), un(*b), a]);
        data.extend_from_slice(&v[..n]);
    }
    out.write_region(r, &data);
    out.prune();
    out
}

/// Regenerate every soft line next to the line layer `line` (inside an edit).
pub(crate) fn refresh_soft_lines(doc: &mut Document, line: LayerId) {
    if !doc.layer(line).is_some_and(|l| matches!(l.role, Some(LayerRole::SugiLine))) {
        return;
    }
    let Some(src) = doc.layer(line).and_then(|l| l.surface()).cloned() else { return };
    let bounds = doc.bounds();
    for id in siblings(doc, line) {
        let Some(l) = doc.layer_mut(id) else { continue };
        if let Some(LayerRole::SugiSoftLine { radius }) = l.role
            && let Some(surf) = l.surface_mut()
        {
            *surf = blur_surface(&src, bounds, radius);
        }
    }
}

/// The white base: everything the line art encloses (a fill from the canvas corner, sampling the
/// line, then inverted), grown by `expand` px so it runs under the line.
fn generate_base(doc: &mut Document, line: LayerId, base: LayerId, expand: f32) -> Result<()> {
    let area = doc.bounds();
    let src = doc.layer(line).and_then(|l| l.surface()).ok_or_else(|| EngineError::Other("the line layer has no pixels".into()))?;
    let img = sel::rgba8_image(src, area);
    // Seed at the first transparent corner.
    let corners = [(area.x0, area.y0), (area.x1 - 1, area.y0), (area.x0, area.y1 - 1), (area.x1 - 1, area.y1 - 1)];
    let w = area.width() as usize;
    let seed = corners.into_iter().find(|(x, y)| img[(y - area.y0) as usize * w + (x - area.x0) as usize][3] < 8).unwrap_or((area.x0, area.y0));
    let outside = sel::wand_region(&img, area, seed, 32.0, true, true).ok_or_else(|| EngineError::Other("nothing to fill".into()))?;
    drop(img);
    // Inside = not outside; grow it by shrinking the outside.
    let full = sel::Region { bbox: area, mask: vec![255; area.width() as usize * area.height() as usize] };
    let mut inside = full;
    for y in area.y0..area.y1 {
        for x in area.x0..area.x1 {
            let i = (y - area.y0) as usize * w + (x - area.x0) as usize;
            inside.mask[i] = (255.0 * (1.0 - outside.at(x, y))).round() as u8;
        }
    }
    let inside = sel::expand_region(&inside, area, expand);
    let l = doc.layer_mut(base).ok_or(EngineError::NoLayer(base))?;
    let surf = l.surface_mut().ok_or_else(|| EngineError::Other("the base is not a pixel layer".into()))?;
    *surf = Surface::new(surf.format());
    fill_coverage(surf, area, |x, y| inside.at(x, y), [1.0, 1.0, 1.0, 1.0], false);
    surf.prune();
    Ok(())
}

fn new_set(s: &mut Session, p: &Value) -> Result<Value> {
    let paper = p.get("paper").and_then(Value::as_str).unwrap_or("Acuarelle").to_string();
    let tile = s.tools.sai_textures.get(&paper).cloned();
    let soft_radius = p.get("softRadius").and_then(Value::as_f64).unwrap_or(4.0) as f32;
    let id = s.edit("New Sugimori Set", |doc, active| {
        let fmt = doc.pixel_format();
        let role = |mut l: Layer, r: LayerRole| {
            l.role = Some(r);
            l
        };
        let mut grey = Layer::new("Grey Background", LayerContent::Fill(Fill::Solid(Color::rgb(0.62, 0.62, 0.62))));
        grey.locks.all = false;
        let mut sketch = role(Layer::raster("Sketch", fmt), LayerRole::SugiSketch);
        sketch.opacity = 0.35;
        let base = role(Layer::raster("Base (white)", fmt), LayerRole::SugiBase);
        let mut colour = role(Layer::raster("Watercolour", fmt), LayerRole::SugiColour);
        colour.clipped = true;
        let mut layers = vec![grey, sketch, base, colour];
        if let Some(t) = &tile {
            // The paper map as an RGB pattern (grey replicated).
            let mut surf = Surface::new(PixelFormat::RGBA8);
            let v: Vec<f32> = t.to_f32().iter().flat_map(|g| [*g, *g, *g, 1.0]).collect();
            surf.write_region(photocraft_geom::Rect::new(0, 0, t.width as i32, t.height as i32), &v);
            let pat = photocraft_doc::Pattern::new(paper.clone(), surf, t.width, t.height);
            let fill = Fill::Pattern { name: pat.name.clone(), scale: 1.0, id: pat.id.clone(), angle: 0.0, link: false, phase: (0.0, 0.0) };
            if !doc.patterns.iter().any(|q| q.id == pat.id) {
                doc.patterns.push(pat);
            }
            let mut paper_l = role(Layer::new("Paper", LayerContent::Fill(fill)), LayerRole::SugiPaper);
            paper_l.blend = BlendMode::Multiply;
            paper_l.opacity = 0.25;
            paper_l.clipped = true;
            layers.push(paper_l);
        }
        let mut soft = role(Layer::raster("Soft Line", fmt), LayerRole::SugiSoftLine { radius: soft_radius });
        soft.opacity = 0.4;
        soft.clipped = true;
        layers.push(soft);
        let mut line = role(Layer::raster("Line", fmt), LayerRole::SugiLine);
        line.linework = Some(Arc::new(Linework::default()));
        line.reference = true;
        let line_id = line.id;
        layers.push(line);
        let group = Layer::group("Sugimori", layers);
        doc.insert_above(*active, group);
        *active = Some(line_id);
        Ok(line_id)
    })?;
    Ok(json!({ "line": id.0 }))
}

fn base_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = d.active_layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
    let line = role_sibling(&d.doc, id, |r| matches!(r, LayerRole::SugiLine)).ok_or_else(|| EngineError::Other("no Sugimori line layer here".into()))?;
    let base = role_sibling(&d.doc, id, |r| matches!(r, LayerRole::SugiBase)).ok_or_else(|| EngineError::Other("no Sugimori base layer here".into()))?;
    let expand = p.get("expand").and_then(Value::as_f64).unwrap_or(0.0) as f32;
    s.edit("Generate Base", |doc, _| generate_base(doc, line, base, expand))?;
    Ok(Value::Null)
}

fn refresh_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = d.active_layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
    let line = role_sibling(&d.doc, id, |r| matches!(r, LayerRole::SugiLine)).ok_or_else(|| EngineError::Other("no Sugimori line layer here".into()))?;
    let radius = p.get("radius").and_then(Value::as_f64).map(|v| v as f32);
    s.edit("Refresh Soft Line", |doc, _| {
        if let Some(r) = radius {
            for sid in siblings(doc, line) {
                if let Some(l) = doc.layer_mut(sid)
                    && matches!(l.role, Some(LayerRole::SugiSoftLine { .. }))
                {
                    l.role = Some(LayerRole::SugiSoftLine { radius: r.max(0.1) });
                }
            }
        }
        refresh_soft_lines(doc, line);
        Ok(())
    })?;
    Ok(Value::Null)
}

fn clear_role(s: &mut Session, p: &Value) -> Result<Value> {
    let id = match p.get("layer").and_then(Value::as_u64) {
        Some(v) => LayerId(v),
        None => s.active().and_then(|d| d.active_layer).ok_or_else(|| EngineError::Other("no active layer".into()))?,
    };
    s.edit("Make Normal Layer", |doc, _| {
        doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.role = None;
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Sugimori set command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "layer.new.sugimoriSet",
            label: "Sugimori Watercolour Set",
            menu: &["Layer", "New"],
            shortcut: None,
            params: r##"{"paper":texture name="Acuarelle" (from the SAI texture folder; none: no paper layer),"softRadius":px=4}"##,
            enabled: has_doc,
            run: new_set,
            journal: true,
            job: None,
        },
        CommandSpec {
            id: "sugimori.generateBase",
            label: "Generate Base from Line",
            menu: &["Layer", "Sugimori"],
            shortcut: None,
            params: r##"{"expand":px=0} (the active layer's Sugimori set)"##,
            enabled: in_set,
            run: base_cmd,
            journal: true,
            job: None,
        },
        CommandSpec {
            id: "sugimori.refreshSoftLine",
            label: "Refresh Soft Line",
            menu: &["Layer", "Sugimori"],
            shortcut: None,
            params: r##"{"radius":px? (new blur radius)}"##,
            enabled: in_set,
            run: refresh_cmd,
            journal: true,
            job: None,
        },
        CommandSpec {
            id: "layer.clearRole",
            label: "Make Normal Layer",
            menu: &["Layer", "Sugimori"],
            shortcut: None,
            params: r##"{"layer":id?}"##,
            enabled: has_doc,
            run: clear_role,
            journal: true,
            job: None,
        },
    ]
}
