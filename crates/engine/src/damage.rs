//! Which part of the canvas an edit changed, found by comparing the document before and after.
//!
//! Commands don't have to report what they touched: tiles are copy-on-write `Arc`s, so a tile
//! the edit didn't touch is the very same pointer in both documents, and comparing two
//! documents costs one pointer comparison per tile plus a field comparison per layer. The
//! canvas then re-composites only that area instead of the whole document.
//!
//! Conservative by design: anything that can change pixels everywhere (size, colour mode,
//! profile, Global Light, patterns, an adjustment or fill layer, a format change) gives `None`
//! or the whole canvas, and a layer whose rendering properties changed contributes all of its
//! extent (grown by its effects). Only changes that can't show on the canvas (the selection,
//! guides, paths, names, locks, invisible layers) give nothing.
//!
//! Every rect is complete: grown by the effects of the changed layer *and of the groups around
//! it* (a child's edit moves the group's shape and its shadow), so the canvas repaints it as is,
//! without growing it by the largest effect in the document.

use std::collections::HashMap;

use photocraft_doc::{Document, Layer, LayerContent, LayerId, LayerMask};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

/// The canvas area that may look different in `after` than in `before`: `None` = assume
/// everything, `Some(Rect::EMPTY)` = nothing visible changed.
pub fn between(before: &Document, after: &Document) -> Option<Rect> {
    if before.size != after.size
        || before.mode != after.mode
        || before.depth != after.depth
        || before.icc_profile != after.icc_profile
        || before.global_light != after.global_light
        || before.patterns != after.patterns
        || before.color_table != after.color_table
        || before.duotone != after.duotone
        || before.channels.len() != after.channels.len()
        || before.quick_mask.is_some() != after.quick_mask.is_some()
    {
        return None;
    }
    let full = after.bounds();
    let mut d = Rect::EMPTY;
    // Alpha channels and Quick Mask show on the canvas when their eye is on.
    for (a, b) in before.channels.iter().zip(&after.channels) {
        d = d.union(&a.surface.changed_rect(&b.surface)?);
    }
    if let (Some(a), Some(b)) = (&before.quick_mask, &after.quick_mask) {
        d = d.union(&a.surface.changed_rect(&b.surface)?);
    }

    let (old, new) = (flatten(before), flatten(after));
    let old_by_id: HashMap<LayerId, &Entry> = old.iter().map(|e| (e.layer.id, e)).collect();
    let new_by_id: HashMap<LayerId, &Entry> = new.iter().map(|e| (e.layer.id, e)).collect();

    // Added and removed layers.
    for e in &old {
        if !new_by_id.contains_key(&e.layer.id) && e.shown {
            d = d.union(&grow(extent(e.layer, full), e.up));
        }
    }
    for e in &new {
        if !old_by_id.contains_key(&e.layer.id) && e.shown {
            d = d.union(&grow(extent(e.layer, full), e.up));
        }
    }

    // Reordered or regrouped layers: everything between the first and last position where the
    // stacking order of the common layers differs.
    let common_old: Vec<(LayerId, Option<LayerId>)> = old.iter().filter(|e| new_by_id.contains_key(&e.layer.id)).map(|e| (e.layer.id, e.parent)).collect();
    let common_new: Vec<(LayerId, Option<LayerId>)> = new.iter().filter(|e| old_by_id.contains_key(&e.layer.id)).map(|e| (e.layer.id, e.parent)).collect();
    if common_old != common_new {
        let first = common_old.iter().zip(&common_new).position(|(a, b)| a != b).unwrap_or(0);
        let last = common_old.len().saturating_sub(1) - common_old.iter().rev().zip(common_new.iter().rev()).position(|(a, b)| a != b).unwrap_or(0);
        for (id, _) in
            common_old.iter().skip(first).take(last.saturating_sub(first) + 1).chain(common_new.iter().skip(first).take(last.saturating_sub(first) + 1))
        {
            for e in [old_by_id.get(id), new_by_id.get(id)].into_iter().flatten() {
                if e.shown {
                    d = d.union(&grow(extent(e.layer, full), e.up));
                }
            }
        }
    }

    // Layers in both: their own changes.
    for e in &new {
        let Some(o) = old_by_id.get(&e.layer.id) else { continue };
        if !o.shown && !e.shown {
            continue;
        }
        d = d.union(&grow(layer_change(o.layer, e.layer, o.shown != e.shown, full), o.up.max(e.up)));
        if d.contains_rect(&full) {
            return Some(full);
        }
    }
    Some(d.intersect(&full))
}

struct Entry<'a> {
    layer: &'a Layer,
    parent: Option<LayerId>,
    /// Visible together with every enclosing group.
    shown: bool,
    /// Effect reach of the enclosing groups.
    up: i32,
}

fn flatten(doc: &Document) -> Vec<Entry<'_>> {
    fn rec<'a>(layers: &'a [Layer], parent: Option<LayerId>, shown: bool, up: i32, out: &mut Vec<Entry<'a>>) {
        for l in layers {
            let s = shown && l.visible;
            out.push(Entry { layer: l, parent, shown: s, up });
            if let Some(ch) = l.children() {
                rec(ch, Some(l.id), s, up.saturating_add(effect_margin(l)), out);
            }
        }
    }
    let mut out = Vec::new();
    rec(&doc.layers, None, true, 0, &mut out);
    out
}

fn grow(r: Rect, by: i32) -> Rect {
    if r.is_empty() || by <= 0 { r } else { r.inflate(by) }
}

/// How far an edit of layer `id`'s pixels can change the canvas beyond the edited area: its own
/// effects plus those of every enclosing group.
pub fn chain_reach(doc: &Document, id: LayerId) -> i32 {
    flatten(doc).iter().find(|e| e.layer.id == id).map_or(0, |e| e.up.saturating_add(effect_margin(e.layer)))
}

/// [`chain_reach`] of every layer that differs between `before` and `after` (the largest).
/// Grows an area a command reports for its pixel edit (a brush stroke's dabs) into a complete one.
pub fn changed_reach(before: &Document, after: &Document) -> i32 {
    let old = flatten(before);
    let old_by_id: HashMap<LayerId, &Entry> = old.iter().map(|e| (e.layer.id, e)).collect();
    let full = after.bounds();
    flatten(after)
        .iter()
        .filter(|e| old_by_id.get(&e.layer.id).is_none_or(|o| !layer_change(o.layer, e.layer, false, full).is_empty()))
        .map(|e| e.up.saturating_add(effect_margin(e.layer)))
        .max()
        .unwrap_or(0)
}

/// What changing `a` into `b` (the same layer) can change on the canvas.
fn layer_change(a: &Layer, b: &Layer, visibility_changed: bool, full: Rect) -> Rect {
    let whole = || extent(a, full).union(&extent(b, full));
    if visibility_changed || !same_rendering(a, b) {
        return whole();
    }
    let reach = effect_margin(a).max(effect_margin(b));
    let grow = |r: Rect| if r.is_empty() || reach == 0 { r } else { r.inflate(reach) };
    let mut d = Rect::EMPTY;
    match (&a.content, &b.content) {
        (LayerContent::Raster(sa), LayerContent::Raster(sb)) => match sa.changed_rect(sb) {
            Some(r) => d = d.union(&grow(r)),
            None => return whole(),
        },
        // Children are compared as layers of their own; only the group's own fields count here.
        (LayerContent::Group(ga), LayerContent::Group(gb)) => {
            if ga.artboard != gb.artboard {
                return whole();
            }
        }
        (ca, cb) => {
            if ca != cb {
                return whole();
            }
        }
    }
    match (&a.mask, &b.mask) {
        (None, None) => {}
        (Some(ma), Some(mb)) if same_mask_settings(ma, mb) => match ma.surface.changed_rect(&mb.surface) {
            Some(r) => d = d.union(&grow(r)),
            None => return whole(),
        },
        _ => return whole(),
    }
    d
}

/// Every field that affects how the layer composites, except its content and mask pixels.
fn same_rendering(a: &Layer, b: &Layer) -> bool {
    a.blend == b.blend
        && a.opacity == b.opacity
        && a.fill_opacity == b.fill_opacity
        && a.clipped == b.clipped
        && a.vector_mask == b.vector_mask
        && a.effects == b.effects
        && a.fill_cache == b.fill_cache
        && a.excluded_channels == b.excluded_channels
        && a.blend_if == b.blend_if
        && a.video == b.video
        && std::mem::discriminant(&a.content) == std::mem::discriminant(&b.content)
}

fn same_mask_settings(a: &LayerMask, b: &LayerMask) -> bool {
    a.enabled == b.enabled && a.density == b.density && a.feather == b.feather
}

/// The canvas area a layer can paint, effects included: its allocated tiles for pixel and type
/// layers and groups (their children), the whole canvas for everything else.
fn extent(l: &Layer, full: Rect) -> Rect {
    let own = match &l.content {
        LayerContent::Raster(s) => surface_extent(s, full),
        LayerContent::Text(t) => t.cache.as_ref().map_or(full, |s| surface_extent(s, full)),
        // An artboard also paints its background over its rect, even when empty.
        LayerContent::Group(g) => g.children.iter().fold(g.artboard.as_ref().map_or(Rect::EMPTY, |a| a.rect), |r, c| r.union(&extent(c, full))),
        _ => full,
    };
    let m = effect_margin(l);
    if own.is_empty() || m == 0 { own } else { own.inflate(m) }
}

/// Allocated tiles, or everything when untouched pixels aren't transparent.
fn surface_extent(s: &Surface, full: Rect) -> Rect {
    let transparent = s.default_pixel().last().is_none_or(|a| *a == 0.0) && s.format().alpha;
    if transparent { s.tile_bounds() } else { full }
}

fn effect_margin(l: &Layer) -> i32 {
    if photocraft_compose::effects::has_effects(l) { photocraft_compose::effects::margin(l).max(0) } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{Color, ColorMode, PixelFormat, SampleType};
    use photocraft_geom::Size;

    fn doc() -> Document {
        let mut d = Document::with_background("d", Size::new(2048, 2048), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut a = Layer::raster("a", PixelFormat::RGBA8);
        a.surface_mut().unwrap().fill_rect(Rect::new(100, 100, 600, 400), &[1.0, 0.0, 0.0, 1.0]);
        let mut b = Layer::raster("b", PixelFormat::RGBA8);
        b.surface_mut().unwrap().fill_rect(Rect::new(1000, 1000, 1300, 1300), &[0.0, 0.0, 1.0, 1.0]);
        d.layers.push(a);
        d.layers.push(b);
        d
    }

    #[test]
    fn untouched_document_has_no_damage() {
        let d = doc();
        assert_eq!(between(&d, &d.clone()), Some(Rect::EMPTY));
    }

    #[test]
    fn painting_damages_only_the_touched_tiles() {
        let d = doc();
        let mut e = d.clone();
        e.layers[1].surface_mut().unwrap().fill_rect(Rect::new(110, 110, 120, 120), &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(between(&d, &e), Some(Rect::new(0, 0, 256, 256)));
    }

    #[test]
    fn an_empty_new_layer_and_the_selection_damage_nothing() {
        let d = doc();
        let mut e = d.clone();
        e.layers.push(Layer::raster("new", PixelFormat::RGBA8));
        let mut sel = Surface::new(PixelFormat { mode: ColorMode::Grayscale, sample: SampleType::U8, alpha: false });
        sel.fill_rect(Rect::new(0, 0, 500, 500), &[1.0]);
        e.selection = Some(sel);
        assert_eq!(between(&d, &e), Some(Rect::EMPTY));
    }

    #[test]
    fn property_changes_damage_the_layer_extent() {
        let d = doc();
        let mut e = d.clone();
        e.layers[2].opacity = 0.5;
        assert_eq!(between(&d, &e), Some(Rect::new(768, 768, 1536, 1536)));
        let mut f = d.clone();
        f.layers[2].visible = false;
        assert_eq!(between(&d, &f), Some(Rect::new(768, 768, 1536, 1536)));
    }

    #[test]
    fn removing_and_reordering_damage_the_layers_involved() {
        let d = doc();
        let mut e = d.clone();
        e.layers.remove(1);
        assert_eq!(between(&d, &e), Some(Rect::new(0, 0, 768, 512)));
        let mut f = d.clone();
        f.layers.swap(1, 2);
        assert_eq!(between(&d, &f), Some(Rect::new(0, 0, 1536, 1536)));
    }

    #[test]
    fn document_wide_changes_damage_everything() {
        let d = doc();
        let mut e = d.clone();
        e.global_light.angle += 10.0;
        assert_eq!(between(&d, &e), None);
        let mut f = d.clone();
        f.layers.push(Layer::new("inv", LayerContent::Adjustment(photocraft_doc::Adjustment::Invert)));
        assert_eq!(between(&d, &f), Some(d.bounds()));
    }

    #[test]
    fn a_childs_edit_reaches_as_far_as_its_groups_effects() {
        let mut d = doc();
        let child = d.layers.remove(2);
        let mut g = Layer::group("g", vec![child]);
        g.effects.items.push(photocraft_doc::Effect::default_drop_shadow());
        let m = effect_margin(&g);
        assert!(m > 0);
        d.layers.push(g);
        let mut e = d.clone();
        let id = match &mut e.layers[2].content {
            LayerContent::Group(gr) => {
                gr.children[0].surface_mut().unwrap().fill_rect(Rect::new(1100, 1100, 1110, 1110), &[0.0, 1.0, 0.0, 1.0]);
                gr.children[0].id
            }
            _ => panic!("group"),
        };
        assert_eq!(between(&d, &e), Some(Rect::new(1024, 1024, 1280, 1280).inflate(m)));
        assert_eq!(chain_reach(&e, id), m);
        assert_eq!(changed_reach(&d, &e), m);
    }

    #[test]
    fn invisible_layers_damage_nothing() {
        let mut d = doc();
        d.layers[1].visible = false;
        let mut e = d.clone();
        e.layers[1].surface_mut().unwrap().fill_rect(Rect::new(0, 0, 50, 50), &[0.0, 1.0, 0.0, 1.0]);
        e.layers[1].opacity = 0.2;
        assert_eq!(between(&d, &e), Some(Rect::EMPTY));
    }

    #[test]
    fn removed_effects_damage_their_old_reach() {
        let mut d = doc();
        d.layers[2].effects.items.push(photocraft_doc::Effect::DropShadow(photocraft_doc::Shadow {
            common: photocraft_doc::FxCommon::new(photocraft_color::BlendMode::Multiply, 0.75),
            color: Color::rgb(0.0, 0.0, 0.0),
            angle: 120.0,
            use_global_light: false,
            distance: 40.0,
            spread: 0.0,
            size: 60.0,
            contour: photocraft_doc::Contour::Linear,
            anti_alias: false,
            noise: 0.0,
            knocks_out: true,
        }));
        let mut e = d.clone();
        e.layers[2].effects.items.clear();
        let r = between(&d, &e).unwrap();
        assert!(r.contains_rect(&Rect::new(768 - 100, 768 - 100, 1536 + 100, 1536 + 100)), "{r:?}");
    }
}
