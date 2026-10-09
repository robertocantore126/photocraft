//! Display-only CPU mip documents. Source geometry is scaled alongside pixels so masks,
//! artboards, pattern phases and layer-effect distances retain their document-space meaning.
use photocraft_doc::{Document, Effect, Fill, FxPaint, Layer, LayerContent, Size};
use photocraft_geom::{Affine, Rect};
use photocraft_raster::Surface;

fn surface(source: &Surface, level: u32, visible: Rect) -> Result<Surface, String> {
    let bounds = crate::bounds::content_bounds(source);
    let mut proxy = source.mip_surface(level, visible.intersect(&bounds))?;
    proxy.set_content_bounds_hint(Some(Surface::mip_bounds(bounds, level)));
    Ok(proxy)
}
fn fill(fill: &mut Fill, k: f32) {
    if let Fill::Pattern { scale, phase, .. } = fill {
        *scale *= k;
        phase.0 *= k;
        phase.1 *= k;
    }
}
fn paint(paint: &mut FxPaint, k: f32) {
    if let FxPaint::Pattern { scale, .. } = paint {
        *scale *= k;
    }
}
fn layer(layer: &mut Layer, level: u32, depth: u32, visible: Rect) -> Result<(), String> {
    if depth >= 256 {
        return Err("display mip layer nesting exceeds 256".into());
    }
    let k = 1.0 / (1u32 << level) as f32;
    let a = Affine { m: [f64::from(k), 0.0, 0.0, f64::from(k), 0.0, 0.0] };
    if let Some(mask) = &mut layer.mask {
        mask.surface = surface(&mask.surface, level, visible)?;
        mask.feather *= k;
    }
    if let Some(mask) = &mut layer.vector_mask {
        mask.path = mask.path.transform(&a);
        mask.feather *= k;
    }
    if let Some(cache) = &mut layer.fill_cache {
        cache.surface = surface(&cache.surface, level, visible)?;
        fill(&mut cache.fill, k);
    }
    if let Some((x, y)) = &mut layer.effects.reference {
        *x *= f64::from(k);
        *y *= f64::from(k);
    }
    for fx in &mut layer.effects.items {
        match fx {
            Effect::DropShadow(s) | Effect::InnerShadow(s) => {
                s.distance *= k;
                s.size *= k;
            }
            Effect::OuterGlow(g) | Effect::InnerGlow(g) => {
                g.size *= k;
                paint(&mut g.paint, k);
            }
            Effect::Stroke(s) => {
                s.size *= k;
                paint(&mut s.paint, k);
            }
            Effect::Satin(s) => {
                s.distance *= k;
                s.size *= k;
            }
            Effect::BevelEmboss(b) => {
                b.size *= k;
                b.soften *= k;
                if let Some(t) = &mut b.texture {
                    t.scale *= k;
                    t.phase.0 *= k;
                    t.phase.1 *= k;
                }
            }
            Effect::PatternOverlay { scale, phase, .. } => {
                *scale *= k;
                phase.0 *= k;
                phase.1 *= k;
            }
            Effect::GradientOverlay { .. } | Effect::ColorOverlay { .. } => {}
        }
    }
    match &mut layer.content {
        LayerContent::Raster(s) => *s = surface(s, level, visible)?,
        LayerContent::Group(g) => {
            if let Some(artboard) = &mut g.artboard {
                artboard.rect = Surface::mip_bounds(artboard.rect, level);
            }
            for child in &mut g.children {
                self::layer(child, level, depth + 1, visible)?;
            }
        }
        LayerContent::Text(t) => {
            t.transform = a.mul(&t.transform);
            if let Some(cache) = &mut t.cache {
                *cache = surface(cache, level, visible)?;
            }
        }
        LayerContent::Shape(s) => {
            s.path = s.path.transform(&a);
            s.live = None;
            if let Some(f) = &mut s.fill {
                fill(f, k);
            }
            if let Some(stroke) = &mut s.stroke {
                stroke.width *= k;
                fill(&mut stroke.paint, k);
            }
            if let Some(cache) = &mut s.cache {
                *cache = surface(cache, level, visible)?;
            }
        }
        LayerContent::Smart(s) => {
            s.transform = a.mul(&s.transform);
            if let Some(cache) = &mut s.cache {
                *cache = surface(cache, level, visible)?;
            }
        }
        LayerContent::Fill(f) => fill(f, k),
        LayerContent::Adjustment(_) => {}
    }
    Ok(())
}

fn reach(layers: &[Layer], depth: u32) -> i32 {
    if depth >= 256 {
        return 0;
    }
    layers
        .iter()
        .map(|l| {
            let own = if crate::effects::has_effects(l) { crate::effects::margin(l) } else { 0 };
            let children = if let LayerContent::Group(g) = &l.content { reach(&g.children, depth + 1) } else { 0 };
            let mask_feather =
                l.mask.as_ref().filter(|m| m.enabled).map_or(0.0, |m| m.feather).max(l.vector_mask.as_ref().filter(|m| m.enabled).map_or(0.0, |m| m.feather));
            let mask_halo = (mask_feather.max(0.0).min(1_000_000.0) * 4.0).ceil() as i32;
            own.saturating_add(mask_halo).saturating_add(children)
        })
        .max()
        .unwrap_or(0)
}

/// Source pixels required around a display viewport by nested effects and mask feathering.
pub fn source_margin(doc: &Document) -> i32 {
    reach(&doc.layers, 0)
}

pub fn document(doc: &Document, level: u32) -> Result<Document, String> {
    document_region(doc, level, doc.bounds())
}

/// Display-only layer pyramid tiles covering the view and every surrounding effect's reach.
pub fn document_region(doc: &Document, level: u32, visible: Rect) -> Result<Document, String> {
    if level > 16 {
        return Err("display mip level exceeds 16".into());
    }
    let mut result = doc.clone();
    if level == 0 {
        return Ok(result);
    }
    let factor = 1u32 << level;
    result.size = Size::new(doc.size.width.div_ceil(factor), doc.size.height.div_ceil(factor));
    let visible = visible.inflate(source_margin(doc).saturating_add((factor as i32).saturating_mul(2)));
    for l in &mut result.layers {
        layer(l, level, 0, visible)?;
    }
    for channel in &mut result.channels {
        channel.surface = surface(&channel.surface, level, visible)?;
    }
    result.selection = None;
    Ok(result)
}
