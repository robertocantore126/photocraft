//! Exact CPU background-prefix cache for viewport rendering. The ordinary compositor remains
//! the oracle; this cache only changes where the same intermediate pixels are obtained.
use std::collections::HashMap;
use std::sync::Arc;

use photocraft_doc::{DocId, Document, Layer, LayerContent, LayerId, Pattern};
use photocraft_geom::{Rect, Size};

use crate::{Buffer, Ctx, composite_stack};

const BUDGET: usize = 64 << 20;
const MAX_PIXELS: u64 = 16 << 20;

struct Snapshot {
    id: DocId,
    size: Size,
    mode: photocraft_color::ColorMode,
    depth: photocraft_color::SampleType,
    light: photocraft_doc::GlobalLight,
    profile: Option<Arc<Vec<u8>>>,
    patterns: Vec<Pattern>,
    layers: Vec<Layer>,
    bounds: Vec<Option<Rect>>,
}

impl Snapshot {
    fn matches(&self, doc: &Document, prefix: &[Layer]) -> bool {
        self.id == doc.id
            && self.size == doc.size
            && self.mode == doc.mode
            && self.depth == doc.depth
            && self.light == doc.global_light
            && self.profile == doc.icc_profile
            && self.patterns == doc.patterns
            && self.layers == prefix
            && self.bounds == surface_bounds(prefix)
    }
    fn new(doc: &Document, prefix: &[Layer]) -> Self {
        Self {
            id: doc.id,
            size: doc.size,
            mode: doc.mode,
            depth: doc.depth,
            light: doc.global_light,
            profile: doc.icc_profile.clone(),
            patterns: doc.patterns.clone(),
            layers: prefix.to_vec(),
            bounds: surface_bounds(prefix),
        }
    }
}

// Surface equality deliberately excludes derived display metadata. Cropped mip proxies
// also carry global anchors: changing one must invalidate cached effects and gradients.
fn surface_bounds(layers: &[Layer]) -> Vec<Option<Rect>> {
    let mut result = Vec::new();
    let mut pending: Vec<_> = layers.iter().collect();
    while let Some(layer) = pending.pop() {
        if let Some(mask) = &layer.mask {
            result.push(mask.surface.content_bounds_hint());
        }
        if let Some(cache) = &layer.fill_cache {
            result.push(cache.surface.content_bounds_hint());
        }
        match &layer.content {
            LayerContent::Raster(surface) => result.push(surface.content_bounds_hint()),
            LayerContent::Group(group) => pending.extend(group.children.iter()),
            LayerContent::Text(text) => result.push(text.cache.as_ref().and_then(|s| s.content_bounds_hint())),
            LayerContent::Shape(shape) => result.push(shape.cache.as_ref().and_then(|s| s.content_bounds_hint())),
            LayerContent::Smart(smart) => result.push(smart.cache.as_ref().and_then(|s| s.content_bounds_hint())),
            _ => {}
        }
    }
    result
}

#[derive(Default)]
pub struct PrefixCache {
    snapshot: Option<Snapshot>,
    tiles: HashMap<Rect, Arc<Buffer>>,
    bytes: usize,
}

impl PrefixCache {
    pub fn clear(&mut self) {
        self.snapshot = None;
        self.tiles.clear();
        self.bytes = 0;
    }

    /// Cache the complete lower sibling stack at a top-level unclipped active layer.
    /// A split through a clipping group or inside a nested group uses the normal renderer.
    pub fn render(&mut self, doc: &Document, rect: Rect, active: Option<LayerId>) -> Result<Buffer, String> {
        let width = i64::from(rect.x1) - i64::from(rect.x0);
        let height = i64::from(rect.y1) - i64::from(rect.y0);
        if width <= 0 || height <= 0 {
            return Ok(Buffer::transparent(Rect::EMPTY));
        }
        if width.checked_mul(height).is_none_or(|pixels| pixels as u64 > MAX_PIXELS) {
            return Err("CPU prefix viewport exceeds the 16 MP output limit".into());
        }
        let split = active
            .and_then(|active| doc.layers.iter().position(|layer| layer.id == active))
            .filter(|index| *index > 0 && doc.layers.get(*index).is_some_and(|layer| !layer.clipped));
        let Some(split) = split
            .filter(|_| doc.mode != photocraft_color::ColorMode::Multichannel)
            .filter(|index| bounded(doc.layers.get(..*index).unwrap_or_default()))
            .filter(|_| std::env::var("PHOTOCRAFT_CPU_PREFIX_CACHE").ok().as_deref() != Some("0"))
        else {
            self.clear();
            return Ok(crate::render(doc, rect));
        };
        let prefix = doc.layers.get(..split).ok_or("invalid CPU prefix split")?;
        let suffix = doc.layers.get(split..).ok_or("invalid CPU suffix split")?;
        if self.snapshot.as_ref().is_none_or(|snapshot| !snapshot.matches(doc, prefix)) {
            self.clear();
            self.snapshot = Some(Snapshot::new(doc, prefix));
        }
        let patterns = crate::pattern::PreparedPatterns::new(&doc.patterns, crate::pattern::PREPARED_PATTERN_BYTES);
        let context = Ctx::for_doc(doc, &patterns);
        let cmyk = crate::cmyk_space(doc);
        let lab = doc.mode == photocraft_color::ColorMode::Lab;
        let requests: Vec<_> = rect
            .tiles()
            .map(|coord| {
                let tile = coord.rect();
                (tile, self.tiles.get(&tile).cloned())
            })
            .collect();
        let run = |(tile, cached): (Rect, Option<Arc<Buffer>>)| {
            photocraft_color::convert::with_cmyk_space(cmyk.as_ref(), || {
                struct Restore(bool);
                impl Drop for Restore {
                    fn drop(&mut self) {
                        crate::psblend::LAB_MIX.with(|lab| lab.set(self.0));
                    }
                }
                let previous = crate::psblend::LAB_MIX.with(|slot| slot.replace(lab));
                let _restore = Restore(previous);
                let built = cached.is_none();
                let lower = cached.unwrap_or_else(|| {
                    let mut lower = crate::multichannel::backdrop(doc, tile);
                    composite_stack(prefix, &mut lower, &context);
                    Arc::new(lower)
                });
                let mut result = (*lower).clone();
                composite_stack(suffix, &mut result, &context);
                (built, lower, result)
            })
        };
        #[cfg(not(target_arch = "wasm32"))]
        let completed: Vec<_> = {
            use rayon::prelude::*;
            requests.into_par_iter().map(run).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let completed: Vec<_> = requests.into_iter().map(run).collect();
        let mut output = Buffer::transparent(rect);
        let output_width = width as usize;
        for (built, lower, result) in completed {
            let visible = result.rect.intersect(&rect);
            let tile_width = result.rect.width() as usize;
            for y in visible.y0..visible.y1 {
                let source_offset = (y - result.rect.y0) as usize * tile_width + (visible.x0 - result.rect.x0) as usize;
                let length = visible.width() as usize;
                let samples = result.px.get(source_offset..source_offset + length).ok_or("CPU prefix source length mismatch")?;
                let offset = (y - rect.y0) as usize * output_width + (visible.x0 - rect.x0) as usize;
                output.px.get_mut(offset..offset + length).ok_or("CPU prefix output length mismatch")?.copy_from_slice(samples);
            }
            if built {
                let bytes = lower.px.len().saturating_mul(std::mem::size_of::<[f32; 4]>());
                if bytes <= BUDGET {
                    if self.bytes.saturating_add(bytes) > BUDGET {
                        self.tiles.clear();
                        self.bytes = 0;
                    }
                    self.bytes = self.bytes.saturating_add(bytes);
                    self.tiles.insert(lower.rect, lower);
                }
            }
        }
        Ok(output)
    }
}

fn bounded(layers: &[Layer]) -> bool {
    let mut pending: Vec<_> = layers.iter().map(|layer| (layer, 0u32)).collect();
    let mut count = 0usize;
    while let Some((layer, depth)) = pending.pop() {
        count += 1;
        if depth >= 64 || count > 20_000 {
            return false;
        }
        if let LayerContent::Group(group) = &layer.content {
            pending.extend(group.children.iter().map(|child| (child, depth + 1)));
        }
    }
    true
}
