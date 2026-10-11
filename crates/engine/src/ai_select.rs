//! Object Selection and Select Subject with local AI models (ported from Fotox's `fx-engine::ai`).
//!
//! BiRefNet by default (fine edges: hair, holes, thin parts); EfficientSAM when asked for
//! (`"model": "sam"`) or when BiRefNet finds nothing under a click. A box is read as a crop around
//! it at up to full resolution, so the model sees a small object with all its detail rather than
//! a whole document shrunk to 1024 px; what lies outside the box or the lasso outline is cut away.
//! A click finds the salient part connected to it on the whole document, then reads that object
//! again as a crop. The working-resolution mask is upsampled to the canvas and refined in the edge
//! band (Select and Mask's guided filter), so the outline follows the full-resolution edges.
//!
//! As in Photoshop, the tool reads the active layer's own content (the parts other layers cover
//! included) unless Sample All Layers is on; the composite for an adjustment layer.

use photocraft_algo::matting::{self, RefineParams};
use photocraft_algo::segment::Sampler;
use photocraft_algo::selection::Region;
use photocraft_doc::{Document, LayerContent, LayerId};
use photocraft_geom::Rect;
use photocraft_ml::models::{BIREFNET, EFFICIENT_SAM};
use photocraft_ml::{AiError, Input, ModelMask, Tensor32};

/// Long side of the image a model reads.
const WORKING: u32 = 1024;

/// Are the Object Selection models usable (runtime and at least one model)?
pub fn available() -> bool {
    std::env::var_os("PHOTOCRAFT_AI").is_none_or(|v| v != "0") && photocraft_ml::runtime::available() && (BIREFNET.installed() || EFFICIENT_SAM.installed())
}

/// The prompt of an Object Selection.
pub struct Prompt {
    /// Box `[x0, y0, x1, y1]` (canvas pixels), or a click when `None`.
    pub boxed: Option<[f64; 4]>,
    /// The click (or the lasso's centre).
    pub point: Option<(f64, f64)>,
    /// Lasso outline (canvas pixels); empty for a box or a click.
    pub lasso: Vec<(f64, f64)>,
    /// Use EfficientSAM even when BiRefNet is installed.
    pub sam: bool,
}

/// The working image: straight RGBA `0..=1` at mip `level`, its corner at canvas `origin`.
struct Working {
    pixels: Vec<[f32; 4]>,
    width: usize,
    height: usize,
    level: u32,
    origin: (i64, i64),
}

/// `doc` reduced to layer `id`'s own content (visible, opaque, Normal, no masks or styles).
/// `None` for an adjustment layer (no content of its own) or an unknown id.
pub fn layer_alone(doc: &Document, id: LayerId) -> Option<Document> {
    let l = doc.layer(id)?;
    if matches!(l.content, LayerContent::Adjustment(_)) {
        return None;
    }
    let mut l = l.clone();
    l.visible = true;
    l.opacity = 1.0;
    l.fill_opacity = 1.0;
    l.blend = photocraft_color::BlendMode::Normal;
    l.clipped = false;
    l.mask = None;
    l.vector_mask = None;
    l.effects = Default::default();
    l.blend_if = Default::default();
    l.excluded_channels = 0;
    let mut solo = doc.clone();
    solo.layers = vec![l];
    Some(solo)
}

/// The canvas rectangle `rect` of `doc` at the finest mip level whose long side is at most
/// `max_side`, flattened over neutral grey (a straight read of transparency would be black, where
/// a dark object would vanish).
fn working(doc: &Document, rect: Rect, max_side: u32) -> Result<Working, String> {
    let rect = rect.intersect(&doc.bounds());
    if rect.is_empty() {
        return Err("the area is outside the canvas".into());
    }
    let mut level = 0u32;
    while (rect.width().max(rect.height()) >> level) > max_side && level < 16 {
        level += 1;
    }
    let (proxy, area) = if level == 0 {
        (None, rect)
    } else {
        let proxy = photocraft_compose::mip_view::document_region(doc, level, rect)?;
        (Some(proxy), photocraft_raster::Surface::mip_bounds(rect, level))
    };
    let buf = photocraft_compose::render(proxy.as_ref().unwrap_or(doc), area);
    let pixels = buf
        .px
        .iter()
        .map(|p| {
            let a = p[3].clamp(0.0, 1.0);
            [p[0] * a + 0.5 * (1.0 - a), p[1] * a + 0.5 * (1.0 - a), p[2] * a + 0.5 * (1.0 - a), 1.0]
        })
        .collect();
    let step = 1i64 << level;
    Ok(Working {
        pixels,
        width: area.width() as usize,
        height: area.height() as usize,
        level,
        origin: (i64::from(area.x0) * step, i64::from(area.y0) * step),
    })
}

fn mask_over(work: &Working, mw: usize, mh: usize, values: Vec<u8>) -> ModelMask {
    let step = 1i64 << work.level;
    let (x0, y0) = work.origin;
    ModelMask { rect: [x0, y0, x0 + work.width as i64 * step, y0 + work.height as i64 * step], width: mw, height: mh, values }
}

fn subject_mask(work: &Working) -> Result<ModelMask, AiError> {
    let dir = BIREFNET.dir().ok_or_else(|| AiError::Missing(BIREFNET.name.into(), photocraft_ml::models::models_dir().display().to_string()))?;
    let model = photocraft_ml::runtime::model(&dir.join(BIREFNET.files[0].file))?;
    const SIDE: usize = 1024;
    let input =
        photocraft_ml::image::planar_rgb(&work.pixels, work.width, work.height, SIDE, SIDE, photocraft_ml::image::IMAGENET_MEAN, photocraft_ml::image::IMAGENET_STD);
    let out = model.run(vec![Tensor32::new(vec![1, 3, SIDE, SIDE], input).into()])?;
    // The finest output (exports may list several scales).
    let best = out.iter().rev().find(|t| t.data.len() == SIDE * SIDE).ok_or_else(|| AiError::Inference("BiRefNet gave no 1024² mask".into()))?;
    Ok(mask_over(work, SIDE, SIDE, photocraft_ml::image::to_u8(&best.data)))
}

struct Embedding {
    tensor: Tensor32,
    width: usize,
    height: usize,
    level: u32,
    origin: (i64, i64),
}

fn sam_embedding(work: &Working) -> Result<Embedding, AiError> {
    let dir = EFFICIENT_SAM.dir().ok_or_else(|| AiError::Missing(EFFICIENT_SAM.name.into(), photocraft_ml::models::models_dir().display().to_string()))?;
    let encoder = photocraft_ml::runtime::model(&dir.join(EFFICIENT_SAM.files[0].file))?;
    let (w, h) = (work.width, work.height);
    let input = photocraft_ml::image::planar_rgb(&work.pixels, w, h, w, h, [0.0; 3], [1.0; 3]);
    let out = encoder.run(vec![Tensor32::new(vec![1, 3, h, w], input).into()])?;
    let tensor = out.into_iter().next().ok_or_else(|| AiError::Inference("no embedding".into()))?;
    Ok(Embedding { tensor, width: w, height: h, level: work.level, origin: work.origin })
}

/// The EfficientSAM decoder: a box and points in canvas pixels.
fn sam_mask(e: &Embedding, boxed: Option<[f64; 4]>, points: &[((f64, f64), bool)]) -> Result<ModelMask, AiError> {
    let dir = EFFICIENT_SAM.dir().ok_or_else(|| AiError::Missing(EFFICIENT_SAM.name.into(), photocraft_ml::models::models_dir().display().to_string()))?;
    let decoder = photocraft_ml::runtime::model(&dir.join(EFFICIENT_SAM.files[1].file))?;
    let s = f64::from(1u32 << e.level);
    let (ox, oy) = (e.origin.0 as f64, e.origin.1 as f64);
    let boxed = boxed.map(|[x0, y0, x1, y1]| [((x0 - ox) / s) as f32, ((y0 - oy) / s) as f32, ((x1 - ox) / s) as f32, ((y1 - oy) / s) as f32]);
    let points: Vec<((f32, f32), bool)> = points.iter().map(|((x, y), p)| ((((x - ox) / s) as f32, ((y - oy) / s) as f32), *p)).collect();
    let (coords, labels, size) = photocraft_ml::sam::prompt(boxed, &points, (e.width, e.height));
    let inputs = photocraft_ml::runtime::ordered(
        &decoder,
        vec![
            ("image_embeddings", Input::F32(e.tensor.clone())),
            ("batched_point_coords", coords),
            ("batched_point_labels", labels),
            ("orig_im_size", size),
        ],
    )?;
    let out = decoder.run(inputs)?;
    let find = |name: &str| decoder.outputs.iter().position(|o| o == name).and_then(|i| out.get(i));
    let (masks, iou) = match (find("output_masks"), find("iou_predictions")) {
        (Some(m), Some(i)) => (m, i),
        _ if out.len() >= 2 => (&out[0], &out[1]),
        _ => return Err(AiError::Inference("EfficientSAM's decoder gave too few outputs".into())),
    };
    let best = photocraft_ml::sam::best_mask(masks, iou).ok_or_else(|| AiError::Inference("EfficientSAM gave no mask".into()))?;
    let (mh, mw) = match masks.shape.as_slice() {
        [.., h, w] => (*h, *w),
        _ => (e.height, e.width),
    };
    let step = 1i64 << e.level;
    let (x0, y0) = e.origin;
    Ok(ModelMask {
        rect: [x0, y0, x0 + e.width as i64 * step, y0 + e.height as i64 * step],
        width: mw,
        height: mh,
        values: photocraft_ml::image::to_u8(&best),
    })
}

/// The canvas area read for a box: the box and a fifth of its long side of context around it.
fn object_crop(boxed: [f64; 4], canvas: Rect) -> Rect {
    let (x0, x1) = (boxed[0].min(boxed[2]), boxed[0].max(boxed[2]));
    let (y0, y1) = (boxed[1].min(boxed[3]), boxed[1].max(boxed[3]));
    let margin = ((x1 - x0).max(y1 - y0) * 0.2).max(32.0);
    Rect::new((x0 - margin).floor() as i32, (y0 - margin).floor() as i32, (x1 + margin).ceil() as i32, (y1 + margin).ceil() as i32).intersect(&canvas)
}

fn cell_of(mask: &ModelMask, (x, y): (f64, f64)) -> (usize, usize) {
    let [rx0, ry0, rx1, ry1] = mask.rect;
    let (w, h) = (mask.width, mask.height);
    let cx = ((x - rx0 as f64) / (rx1 - rx0).max(1) as f64 * w as f64).floor();
    let cy = ((y - ry0 as f64) / (ry1 - ry0).max(1) as f64 * h as f64).floor();
    ((cx.max(0.0) as usize).min(w.saturating_sub(1)), (cy.max(0.0) as usize).min(h.saturating_sub(1)))
}

/// The selected cell nearest to `at` within a few percent of the mask, so a click just off a thin
/// object still finds it.
fn seed_near(mask: &ModelMask, at: (f64, f64)) -> Option<(usize, usize)> {
    let (w, h) = (mask.width, mask.height);
    let (sx, sy) = cell_of(mask, at);
    let radius = (w.max(h) / 40).max(2) as i64;
    let mut best: Option<((usize, usize), i64)> = None;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (x, y) = (sx as i64 + dx, sy as i64 + dy);
            if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                continue;
            }
            let d = dx * dx + dy * dy;
            if mask.values[y as usize * w + x as usize] >= 128 && best.is_none_or(|(_, b)| d < b) {
                best = Some(((x as usize, y as usize), d));
            }
        }
    }
    best.map(|(p, _)| p)
}

/// The cells connected to `seed` whose value is at least `floor`.
fn flood(mask: &ModelMask, seed: (usize, usize), floor: u8) -> Vec<bool> {
    let (w, h) = (mask.width, mask.height);
    let mut inside = vec![false; w * h];
    let mut stack = vec![seed];
    while let Some((x, y)) = stack.pop() {
        let i = y * w + x;
        if inside[i] || mask.values[i] < floor {
            continue;
        }
        inside[i] = true;
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < w {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }
    inside
}

fn component_bounds(mask: &ModelMask, at: (f64, f64)) -> Option<[f64; 4]> {
    let inside = flood(mask, seed_near(mask, at)?, 128);
    ModelMask { values: inside.iter().map(|&b| if b { 255 } else { 0 }).collect(), ..mask.clone() }.bounds()
}

/// Keep only the part connected to `at` (its soft edge included).
fn keep_component(mask: &mut ModelMask, at: (f64, f64)) {
    let Some(seed) = seed_near(mask, at) else {
        mask.values.fill(0);
        return;
    };
    let inside = flood(mask, seed, 8);
    for (v, keep) in mask.values.iter_mut().zip(inside) {
        if !keep {
            *v = 0;
        }
    }
}

fn point_in_polygon(poly: &[(f64, f64)], x: f64, y: f64) -> bool {
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let ((xi, yi), (xj, yj)) = (poly[i], poly[j]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn near_outline(poly: &[(f64, f64)], x: f64, y: f64, tol2: f64) -> bool {
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let ((ax, ay), (bx, by)) = (poly[j], poly[i]);
        let (dx, dy) = (bx - ax, by - ay);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 { (((x - ax) * dx + (y - ay) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
        let (px, py) = (ax + t * dx - x, ay + t * dy - y);
        if px * px + py * py <= tol2 {
            return true;
        }
        j = i;
    }
    false
}

/// Cut away what lies outside the box (a little tolerance) or the lasso outline (with the same
/// tolerance, so an outline drawn a bit tight does not slice the object).
fn clip_to_prompt(mask: &mut ModelMask, boxed: [f64; 4], lasso: &[(f64, f64)]) {
    let (x0, x1) = (boxed[0].min(boxed[2]), boxed[0].max(boxed[2]));
    let (y0, y1) = (boxed[1].min(boxed[3]), boxed[1].max(boxed[3]));
    let tol = ((x1 - x0).max(y1 - y0) * 0.03).max(4.0);
    let step = lasso.len().div_ceil(256).max(1);
    let poly: Vec<(f64, f64)> = lasso.iter().step_by(step).copied().collect();
    let [rx0, ry0, rx1, ry1] = mask.rect;
    let (w, h) = (mask.width, mask.height);
    let (sx, sy) = ((rx1 - rx0) as f64 / w as f64, (ry1 - ry0) as f64 / h as f64);
    let tol2 = tol * tol;
    for cy in 0..h {
        let y = ry0 as f64 + (cy as f64 + 0.5) * sy;
        for cx in 0..w {
            let i = cy * w + cx;
            if mask.values[i] == 0 {
                continue;
            }
            let x = rx0 as f64 + (cx as f64 + 0.5) * sx;
            let keep = if poly.len() >= 3 {
                point_in_polygon(&poly, x, y) || near_outline(&poly, x, y, tol2)
            } else {
                x >= x0 - tol && x <= x1 + tol && y >= y0 - tol && y <= y1 + tol
            };
            if !keep {
                mask.values[i] = 0;
            }
        }
    }
}

fn err(e: AiError) -> String {
    e.to_string()
}

/// BiRefNet on the area around a box: what stands out in it, clipped to the prompt.
fn birefnet_object(doc: &Document, boxed: [f64; 4], lasso: &[(f64, f64)]) -> Result<ModelMask, String> {
    let work = working(doc, object_crop(boxed, doc.bounds()), WORKING)?;
    let mut mask = subject_mask(&work).map_err(err)?;
    clip_to_prompt(&mut mask, boxed, lasso);
    if mask.values.iter().all(|&v| v < 128) {
        return Err("No object found there: draw the box or lasso closer around it".into());
    }
    Ok(mask)
}

/// A click with BiRefNet: the salient part connected to the click on the whole document, read
/// again as a crop at up to full resolution. `None` when nothing salient is near.
fn birefnet_click(doc: &Document, at: (f64, f64)) -> Result<Option<ModelMask>, String> {
    let coarse = subject_mask(&working(doc, doc.bounds(), WORKING)?).map_err(err)?;
    let Some([x0, y0, x1, y1]) = component_bounds(&coarse, at) else { return Ok(None) };
    let (cw, ch) = (f64::from(doc.size.width), f64::from(doc.size.height));
    let pad = (x1 - x0).max(y1 - y0) * 0.08;
    let boxed = [(x0 - pad).max(0.0), (y0 - pad).max(0.0), (x1 + pad).min(cw), (y1 + pad).min(ch)];
    let work = working(doc, object_crop(boxed, doc.bounds()), WORKING)?;
    let mut mask = subject_mask(&work).map_err(err)?;
    clip_to_prompt(&mut mask, boxed, &[]);
    keep_component(&mut mask, at);
    Ok(mask.values.iter().any(|&v| v >= 128).then_some(mask))
}

/// EfficientSAM: a box read as a crop; a click on the whole document, then again as a crop
/// around what it found.
fn sam_object(doc: &Document, boxed: Option<[f64; 4]>, points: &[((f64, f64), bool)]) -> Result<ModelMask, String> {
    if let Some(b) = boxed {
        let e = sam_embedding(&working(doc, object_crop(b, doc.bounds()), WORKING)?).map_err(err)?;
        return sam_mask(&e, Some(b), points).map_err(err);
    }
    let e = sam_embedding(&working(doc, doc.bounds(), WORKING)?).map_err(err)?;
    let coarse = sam_mask(&e, None, points).map_err(err)?;
    let Some([x0, y0, x1, y1]) = coarse.bounds() else { return Ok(coarse) };
    let (cw, ch) = (f64::from(doc.size.width), f64::from(doc.size.height));
    if (x1 - x0) * (y1 - y0) > 0.6 * cw * ch {
        return Ok(coarse);
    }
    let pad = (x1 - x0).max(y1 - y0) * 0.08;
    let b = [(x0 - pad).max(0.0), (y0 - pad).max(0.0), (x1 + pad).min(cw), (y1 + pad).min(ch)];
    let e = sam_embedding(&working(doc, object_crop(b, doc.bounds()), WORKING)?).map_err(err)?;
    sam_mask(&e, Some(b), points).map_err(err)
}

/// The Object Selection mask for `prompt` on `source` (the layer alone, or the document).
pub fn object_mask(source: &Document, prompt: &Prompt) -> Result<ModelMask, String> {
    let points: Vec<((f64, f64), bool)> = prompt.point.map(|p| vec![(p, true)]).unwrap_or_default();
    let use_sam = (prompt.sam || !BIREFNET.installed()) && EFFICIENT_SAM.installed();
    if use_sam {
        return sam_object(source, prompt.boxed, &points);
    }
    match prompt.boxed {
        Some(b) => birefnet_object(source, b, &prompt.lasso),
        None => {
            let at = prompt.point.ok_or("a click or a box is needed")?;
            match birefnet_click(source, at)? {
                Some(mask) => Ok(mask),
                // Nothing stands out under the click: EfficientSAM, whose point prompts find any region.
                None if EFFICIENT_SAM.installed() => sam_object(source, None, &points),
                None => Err("No object found under the click: drag a box around it".into()),
            }
        }
    }
}

/// Select Subject: BiRefNet on the whole document.
pub fn subject(source: &Document) -> Result<ModelMask, String> {
    subject_mask(&working(source, source.bounds(), WORKING)?).map_err(err)
}

/// A composite sampler for the edge refinement.
struct Composite<'a>(&'a Document);

impl Sampler for Composite<'_> {
    fn rgba(&self, r: Rect) -> Vec<[f32; 4]> {
        photocraft_compose::render(self.0, r).px
    }
}

/// The mask upsampled to the canvas and refined in the edge band.
pub fn to_region(source: &Document, mask: &ModelMask) -> Option<Region> {
    let [x0, y0, x1, y1] = mask.bounds()?;
    let canvas = source.bounds();
    let content = Rect::new(x0.floor() as i32, y0.floor() as i32, x1.ceil() as i32, y1.ceil() as i32).inflate(2).intersect(&canvas);
    let read = |r: Rect| -> Vec<f32> {
        let mut v = Vec::with_capacity(r.width() as usize * r.height() as usize);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                v.push(mask.at(f64::from(x) + 0.5, f64::from(y) + 0.5));
            }
        }
        v
    };
    // The guided filter's window: about one and a half mask cells.
    let cell = (mask.rect[2] - mask.rect[0]).max(mask.rect[3] - mask.rect[1]) as f32 / mask.width.max(mask.height).max(1) as f32;
    let params = RefineParams { radius: (cell * 1.5).clamp(2.0, 32.0), smart_radius: true, ..Default::default() };
    if std::env::var("PHOTOCRAFT_AI_REFINE").as_deref() == Ok("0") {
        return matting::refine_mask(&Composite(source), &read, content, canvas, &RefineParams::default());
    }
    matting::refine_mask(&Composite(source), &read, content, canvas, &params)
}

/// Load the models in the background, once, the first time the Object Selection tool is picked
/// (not at start: DirectML commits a lot of memory, and most sessions never use the AI).
pub fn warm() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if available() {
        ONCE.call_once(photocraft_ml::runtime::warm);
    }
}
