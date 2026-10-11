//! Dab rasterisation and stroke compositing.
//!
//! A stroke paints into a sparse, tiled *stroke buffer* (coverage plus, with Color Dynamics, a
//! premultiplied colour), exactly like a private stroke layer: dabs build up with flow up to their
//! opacity ceiling, stroke-level masks (Dual Brush, Texture) apply when the buffer is composited,
//! and the whole buffer composites onto the pre-stroke pixels at the stroke opacity. Because the
//! buffer is sparse and compositing always starts from the pre-stroke pixels, strokes can be fed in
//! chunks and re-composited incrementally for interactive painting (see [`StrokeRenderer`]).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
fn parallel_brush() -> bool {
    std::env::var("PHOTOCRAFT_PARALLEL_BRUSH").as_deref() != Ok("0")
}

use photocraft_color::{BlendMode, PixelFormat};
use photocraft_geom::Rect;
use photocraft_raster::{Surface, from_rgba_into, to_rgba};

use crate::brush::{BrushSettings, MaskMode, Pattern, TipShape};
use crate::dynamics::DabGenerator;
use crate::retouch::{Footprint, alpha_index, over_native};
use crate::rng::hash2;
use crate::tile::{Mips, PatternImage};
use crate::{Dab, StrokePoint, dab_coverage};

/// Edge of a stroke-buffer tile.
pub const COV_TILE: i32 = 64;

/// What a brush plug-in does to a stroke: called once per dirty rectangle (part of one
/// [`COV_TILE`] tile) with the **pre-stroke** pixels of `rect` (`channels` = 4: premultiplied
/// RGBA `0..=1`; 1: a mask's grey) and `k`, the stroke's opacity × coverage per pixel. It
/// overwrites the values with the result; an error keeps those pixels as they were.
pub type PixelOpFn = dyn Fn(Rect, &mut [f32], usize, &[f32]) -> Result<(), String> + Send + Sync;

/// A shared [`PixelOpFn`].
#[derive(Clone)]
pub struct PixelOp(pub Arc<PixelOpFn>);

impl std::fmt::Debug for PixelOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PixelOp")
    }
}
const NOISE_SALT: u64 = 0x006E_6F69_7365;

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Combine brush coverage `v` with a mask value `t` (texture/dual tip) at `depth`.
/// Every mode maps `v = 0` to 0 (a mask never adds paint where the brush has none).
pub fn mask_combine(mode: MaskMode, v: f32, t: f32, d: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    let lerp = |a: f32, b: f32| a + (b - a) * d;
    let r = match mode {
        MaskMode::Multiply => v * (1.0 - d + d * t),
        MaskMode::Subtract => (v - d * (1.0 - t)).max(0.0),
        MaskMode::Darken => v.min(1.0 - d + d * t),
        MaskMode::Overlay => {
            let o = if v < 0.5 { 2.0 * v * t } else { 1.0 - 2.0 * (1.0 - v) * (1.0 - t) };
            lerp(v, o)
        }
        MaskMode::ColorDodge => lerp(v, (v / (1.0 - t).max(1e-3)).min(1.0)),
        MaskMode::ColorBurn => lerp(v, 1.0 - ((1.0 - v) / t.max(1e-3)).min(1.0)),
        MaskMode::LinearBurn => lerp(v, (v + t - 1.0).max(0.0)),
        MaskMode::HardMix => lerp(v, if v + t > 1.0 { 1.0 } else { 0.0 }),
        MaskMode::LinearHeight => v * ((t - (1.0 - d * v)) * 4.0 + 0.5).clamp(0.0, 1.0),
        MaskMode::Height => {
            if t >= 1.0 - d * v {
                v
            } else {
                0.0
            }
        }
    };
    r.clamp(0.0, 1.0)
}

/// A stroke-ready brush: the settings plus prepared tip mips and texture.
#[derive(Clone, Debug)]
pub struct BrushContext {
    pub brush: BrushSettings,
    tip: Option<Arc<Mips>>,
    dual_tip: Option<Arc<Mips>>,
    texture: Option<Arc<PatternImage>>,
}

fn prepare_pattern(b: &BrushSettings) -> Option<Arc<PatternImage>> {
    let t = &b.texture;
    if !t.enabled {
        return None;
    }
    let mut img = match &t.pattern {
        Pattern::Procedural { style, size, seed } => crate::procedural::pattern(*style, *size, *seed),
        Pattern::Tile(g) if g.is_valid() => PatternImage { width: g.width as usize, height: g.height as usize, data: g.to_f32() },
        Pattern::Tile(_) => return None,
    };
    let (br, ct) = (t.brightness.clamp(-1.0, 1.0), t.contrast.clamp(-1.0, 1.0));
    for v in &mut img.data {
        let mut x = (*v - 0.5) * (1.0 + ct) + 0.5 + br;
        x = x.clamp(0.0, 1.0);
        *v = if t.invert { 1.0 - x } else { x };
    }
    Some(Arc::new(img))
}

impl BrushContext {
    pub fn new(brush: &BrushSettings) -> Self {
        let mips = |t: &TipShape| match t {
            TipShape::Sampled(g) if g.is_valid() => Some(Arc::new(Mips::new(g))),
            _ => None,
        };
        Self {
            tip: mips(&brush.tip),
            dual_tip: if brush.dual_brush.enabled { mips(&brush.dual_brush.tip) } else { None },
            texture: prepare_pattern(brush),
            brush: brush.clone(),
        }
    }

    /// Texture value at a document pixel (bilinear, tiled).
    #[inline]
    pub fn texture_at(&self, x: i32, y: i32) -> f32 {
        let Some(p) = &self.texture else { return 1.0 };
        let s = self.brush.texture.scale.max(0.01);
        if s == 1.0 {
            // Pixel centres land exactly on texels at 100 %.
            return p.data[y.rem_euclid(p.height as i32) as usize * p.width + x.rem_euclid(p.width as i32) as usize];
        }
        let (fx, fy) = ((x as f32 + 0.5) / s - 0.5, (y as f32 + 0.5) / s - 0.5);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let a = p.sample_wrap(x0, y0) + (p.sample_wrap(x0 + 1.0, y0) - p.sample_wrap(x0, y0)) * tx;
        let b = p.sample_wrap(x0, y0 + 1.0) + (p.sample_wrap(x0 + 1.0, y0 + 1.0) - p.sample_wrap(x0, y0 + 1.0)) * tx;
        a + (b - a) * ty
    }

    /// Bounding rectangle of a dab (rotated sampled tips need the corner reach).
    pub fn dab_rect(&self, d: &Dab, dual: bool) -> Rect {
        let sampled = if dual { self.dual_tip.is_some() } else { self.tip.is_some() };
        let reach = if sampled { d.radius * std::f32::consts::SQRT_2 } else { d.radius };
        let rr = reach.ceil() as i32 + 1;
        let (cx, cy) = (d.center.x.floor() as i32, d.center.y.floor() as i32);
        Rect::new(cx - rr, cy - rr, cx + rr + 1, cy + rr + 1)
    }

    /// Rasterise a dab over `rect` into `out` (tip shape × noise × per-tip texture × wet edges × flow).
    pub fn rasterize(&self, d: &Dab, dual: bool, rect: Rect, out: &mut Vec<f32>) {
        let (w, h) = (rect.width() as usize, rect.height() as usize);
        out.clear();
        if w == 0 || h == 0 {
            return;
        }
        let Some(len) = w.checked_mul(h) else { return };
        out.resize(len, 0.0);
        let b = &self.brush;
        let (hardness, mips) = if dual { (b.dual_brush.hardness, self.dual_tip.as_deref()) } else { (b.hardness, self.tip.as_deref()) };
        let aliased = b.aliased && !dual;
        let wet = b.wet_edges && !dual;
        let noise = b.noise && !dual;
        let tex_tip = !dual && b.texture.enabled && b.texture.each_tip && self.texture.is_some();
        let (cx, cy) = (d.center.x as f32, d.center.y as f32);
        let (sn, cs) = d.angle.sin_cos();
        let (fx, fy) = (if d.flip_x { -1.0 } else { 1.0 }, if d.flip_y { -1.0 } else { 1.0 });
        // Brush Projection: stretch the sampling coordinates along the tilt direction, which
        // foreshortens the tip there by `proj_scale`.
        let proj = (!dual && d.proj_scale < 0.999).then(|| {
            let (ps, pc) = d.proj_angle.sin_cos();
            (pc, ps, 1.0 / d.proj_scale.max(0.05) - 1.0)
        });
        let r = d.radius;
        let ro = d.roundness.max(0.5 / r).min(1.0);
        let rm = (r * ro).max(0.5);
        let reach2 = (rm + 0.5) * (rm + 0.5);
        // Sampled-tip mapping.
        let (level, inv_scale, tw, th) = match mips {
            Some(m) => {
                let (w0, h0, _) = &m.levels[0];
                let big = *w0.max(h0) as f32;
                (m.level_for(2.0 * r * ro.max(0.25)), big / (2.0 * r), *w0 as f32, *h0 as f32)
            }
            None => (0, 1.0, 1.0, 1.0),
        };
        let raster_row = |(yy, row): (usize, &mut [f32])| {
            let y = rect.y0 + yy as i32;
            let dy = y as f32 + 0.5 - cy;
            for xx in 0..w {
                let x = rect.x0 + xx as i32;
                let dx = x as f32 + 0.5 - cx;
                // To y-up, project, rotate by -angle, flip.
                let (mut ux, mut uy) = (dx, -dy);
                if let Some((ax, ay, k)) = proj {
                    let t = (ux * ax + uy * ay) * k;
                    ux += t * ax;
                    uy += t * ay;
                }
                let u = (ux * cs + uy * sn) * fx;
                let v = (-ux * sn + uy * cs) * fy;
                let (mut val, rn) = match mips {
                    None => {
                        let d2 = (u * ro).powi(2) + v * v;
                        if d2 >= reach2 {
                            continue;
                        }
                        let dd = d2.sqrt();
                        let val = if aliased { if dd <= rm { 1.0 } else { 0.0 } } else { dab_coverage(dd, rm, hardness) };
                        (val, dd / rm)
                    }
                    Some(m) => {
                        let (px, py) = (u * inv_scale + tw / 2.0, -(v / ro) * inv_scale + th / 2.0);
                        if px < -1.0 || py < -1.0 || px > tw + 1.0 || py > th + 1.0 {
                            continue;
                        }
                        let s = m.sample_clamped(level, px / tw, py / th);
                        let val = if aliased { if s >= 0.5 { 1.0 } else { 0.0 } } else { s };
                        (val, 1.0 - s)
                    }
                };
                if val <= 0.0 {
                    continue;
                }
                if noise && val < 1.0 {
                    let n = hash2(x, y, NOISE_SALT);
                    let hard = if n < val { 1.0 } else { 0.0 };
                    val += (hard - val) * 0.7;
                }
                if tex_tip {
                    val = mask_combine(b.texture.mode, val, self.texture_at(x, y), d.depth);
                }
                if wet {
                    val *= 0.5 + 0.5 * smoothstep(0.5, 1.0, rn);
                }
                if let Some(dst) = row.get_mut(xx) {
                    *dst = val * d.alpha;
                }
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        if len >= 65_536 && parallel_brush() {
            out.par_chunks_mut(w).enumerate().for_each(raster_row);
            return;
        }
        out.chunks_mut(w).enumerate().for_each(raster_row);
    }

    /// One dab's dense footprint (for the sequential retouch tools).
    pub fn footprint(&self, d: &Dab, index: usize) -> Footprint {
        let rect = self.dab_rect(d, false);
        let mut cov = Vec::new();
        self.rasterize(d, false, rect, &mut cov);
        Footprint { dab: *d, index, rect, cov }
    }

    /// Stroke-level masks applied to accumulated coverage `c` at a pixel.
    #[inline]
    pub fn stroke_mask(&self, c: f32, dual: Option<f32>, x: i32, y: i32) -> f32 {
        let b = &self.brush;
        let mut m = c;
        if let Some(dv) = dual {
            m = mask_combine(b.dual_brush.mode, m, dv, 1.0);
        }
        if self.texture.is_some() && !b.texture.each_tip {
            m = mask_combine(b.texture.mode, m, self.texture_at(x, y), b.texture.depth.clamp(0.0, 1.0));
        }
        m
    }
}

// ---------------------------------------------------------------------------
// Sparse stroke buffer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct CovTile {
    cov: Vec<f32>,
    /// Premultiplied native colour channels (`nc` per pixel), empty without per-dab colour.
    col: Vec<f32>,
}

/// A sparse tiled coverage (+ optional colour) buffer.
#[derive(Clone, Debug, Default)]
pub struct CoverageMap {
    tiles: HashMap<(i32, i32), CovTile>,
    nc: usize,
    bounds: Rect,
    dirty: HashSet<(i32, i32)>,
}

impl CoverageMap {
    pub fn new(color_channels: usize) -> Self {
        Self { nc: color_channels, ..Default::default() }
    }
    pub fn bounds(&self) -> Rect {
        self.bounds
    }
    #[inline]
    pub fn get(&self, x: i32, y: i32) -> f32 {
        let (tx, ty) = (x.div_euclid(COV_TILE), y.div_euclid(COV_TILE));
        self.tiles.get(&(tx, ty)).map_or(0.0, |t| t.cov[((y - ty * COV_TILE) * COV_TILE + (x - tx * COV_TILE)) as usize])
    }

    /// Accumulate dab values over `rect`: flow builds up towards the dab's opacity ceiling
    /// (`c ← c + v·(ceil − c)`), or with `max` (wet edges). `color` = native colour for per-dab colour.
    pub fn accumulate(&mut self, rect: Rect, vals: &[f32], ceil: f32, use_max: bool, color: Option<&[f32]>) {
        if rect.is_empty() {
            return;
        }
        self.bounds = self.bounds.union(&rect);
        let w = rect.width() as usize;
        let nc = self.nc;
        let (t0x, t0y) = (rect.x0.div_euclid(COV_TILE), rect.y0.div_euclid(COV_TILE));
        let (t1x, t1y) = ((rect.x1 - 1).div_euclid(COV_TILE), (rect.y1 - 1).div_euclid(COV_TILE));
        let want_col = color.is_some() && nc > 0;
        let mut work = Vec::new();
        for ty in t0y..=t1y {
            for tx in t0x..=t1x {
                let tr = Rect::new(tx * COV_TILE, ty * COV_TILE, (tx + 1) * COV_TILE, (ty + 1) * COV_TILE).intersect(&rect);
                let any = (tr.y0..tr.y1).any(|y| {
                    let row = (y - rect.y0) as usize * w;
                    vals.get(row + (tr.x0 - rect.x0) as usize..row + (tr.x1 - rect.x0) as usize).is_some_and(|v| v.iter().any(|&v| v > 0.0))
                });
                if !any {
                    continue;
                }
                let tile = self.tiles.remove(&(tx, ty)).unwrap_or_else(|| CovTile { cov: vec![0.0; (COV_TILE * COV_TILE) as usize], col: Vec::new() });
                work.push(((tx, ty), tr, tile));
            }
        }
        // A dab completes before the next one starts; only independent tiles run concurrently.
        let accumulate = |(key, tr, mut tile): ((i32, i32), Rect, CovTile)| {
            let (tx, ty) = key;
            if want_col && tile.col.is_empty() {
                tile.col = vec![0.0; (COV_TILE * COV_TILE) as usize * nc];
            }
            for y in tr.y0..tr.y1 {
                let row = (y - rect.y0) as usize * w;
                let trow = ((y - ty * COV_TILE) * COV_TILE) as usize;
                for x in tr.x0..tr.x1 {
                    let v = vals.get(row + (x - rect.x0) as usize).copied().unwrap_or(0.0);
                    if v <= 0.0 {
                        continue;
                    }
                    let ti = trow + (x - tx * COV_TILE) as usize;
                    let Some(c) = tile.cov.get(ti).copied() else { continue };
                    let nv = if use_max {
                        c.max(v * ceil)
                    } else if ceil > c {
                        c + v * (ceil - c)
                    } else {
                        c
                    };
                    if nv <= c {
                        continue;
                    }
                    if let (true, Some(col)) = (want_col, color) {
                        let k = if c < 1.0 { (nv - c) / (1.0 - c) } else { 0.0 };
                        if let Some(p) = tile.col.get_mut(ti * nc..(ti + 1) * nc) {
                            for (dst, src) in p.iter_mut().zip(col) {
                                *dst = *src * k + *dst * (1.0 - k);
                            }
                        }
                    }
                    if let Some(dst) = tile.cov.get_mut(ti) {
                        *dst = nv;
                    }
                }
            }
            (key, tile)
        };
        #[cfg(not(target_arch = "wasm32"))]
        let completed: Vec<_> =
            if work.len() >= 16 && parallel_brush() { work.into_par_iter().map(accumulate).collect() } else { work.into_iter().map(accumulate).collect() };
        #[cfg(target_arch = "wasm32")]
        let completed: Vec<_> = work.into_iter().map(accumulate).collect();
        for (key, tile) in completed {
            self.dirty.insert(key);
            self.tiles.insert(key, tile);
        }
    }

    fn take_dirty(&mut self) -> Vec<(i32, i32)> {
        let mut v: Vec<_> = self.dirty.drain().collect();
        v.sort_unstable();
        v
    }
}

// ---------------------------------------------------------------------------
// Stroke renderer
// ---------------------------------------------------------------------------

/// Incremental stroke rendering: feed points, composite dirty tiles from the pre-stroke pixels.
#[derive(Clone, Debug)]
pub struct StrokeRenderer {
    pub ctx: BrushContext,
    generator: DabGenerator,
    cov: CoverageMap,
    dual: Option<CoverageMap>,
    fmt: Option<PixelFormat>,
    per_dab_color: bool,
    dabs_done: usize,
    scratch: Vec<f32>,
    stamp: Option<([u32; 12], Arc<[f32]>)>,
    dab_buf: Vec<Dab>,
    dual_buf: Vec<Dab>,
    all_dabs: Option<Vec<Dab>>,
    /// A brush plug-in deciding the pixels instead of the brush's colour / erase.
    pixel_op: Option<PixelOp>,
    /// Each pixel keeps its strongest dab coverage instead of building up (Fotox's `max`).
    max_accumulate: bool,
}

impl StrokeRenderer {
    /// `fmt` = target pixel format (needed for per-dab colour); `zoom` for smoothing.
    pub fn new(brush: &BrushSettings, fmt: Option<PixelFormat>, zoom: f32) -> Self {
        let per_dab_color = brush.color_dynamics.enabled && !brush.erase && fmt.is_some();
        let nc = if per_dab_color { fmt.map_or(0, |f| f.mode.color_channels()) } else { 0 };
        Self {
            ctx: BrushContext::new(brush),
            generator: DabGenerator::new(brush, zoom),
            cov: CoverageMap::new(nc),
            dual: brush.dual_brush.enabled.then(|| CoverageMap::new(0)),
            fmt,
            per_dab_color,
            dabs_done: 0,
            scratch: Vec::new(),
            stamp: None,
            dab_buf: Vec::new(),
            dual_buf: Vec::new(),
            all_dabs: None,
            pixel_op: None,
            max_accumulate: false,
        }
    }

    /// Paint through a brush plug-in (see [`PixelOp`]).
    pub fn with_pixel_op(mut self, op: Option<PixelOp>) -> Self {
        self.pixel_op = op;
        self
    }

    /// Keep each pixel's strongest coverage (scrubbing over a stroke adds nothing).
    pub fn with_max_accumulate(mut self, max: bool) -> Self {
        self.max_accumulate = max;
        self
    }

    /// Keep a copy of every primary dab (for tests and sequential tools).
    pub fn record_dabs(mut self) -> Self {
        self.all_dabs = Some(Vec::new());
        self
    }
    pub fn dabs(&self) -> &[Dab] {
        self.all_dabs.as_deref().unwrap_or(&[])
    }
    pub fn dab_count(&self) -> usize {
        self.dabs_done
    }
    /// Union of the dab rectangles so far.
    pub fn bounds(&self) -> Rect {
        self.cov.bounds
    }

    fn raster_pending(&mut self) {
        let mut dabs = std::mem::take(&mut self.dab_buf);
        let mut duals = std::mem::take(&mut self.dual_buf);
        let wet = self.ctx.brush.wet_edges;
        let mut native = [0.0f32; 8];
        for d in &dabs {
            let rect = self.ctx.dab_rect(d, false);
            // Integer-position static tips share an exactly translation-invariant footprint.
            // Subpixel positions, pressure/shape dynamics, noise and per-tip texture bypass it.
            let b = &self.ctx.brush;
            let key = (std::env::var("PHOTOCRAFT_BRUSH_STAMP").as_deref() != Ok("0")
                && !b.pressure_size
                && !b.shape_dynamics.enabled
                && !b.noise
                && !(b.texture.enabled && b.texture.each_tip)
                && d.center.x.fract() == 0.0
                && d.center.y.fract() == 0.0
                && d.center.x.abs() < 1_000_000.0
                && d.center.y.abs() < 1_000_000.0)
                .then(|| {
                    [
                        d.radius.to_bits(),
                        d.alpha.to_bits(),
                        d.angle.to_bits(),
                        d.roundness.to_bits(),
                        d.depth.to_bits(),
                        d.proj_angle.to_bits(),
                        d.proj_scale.to_bits(),
                        u32::from(d.flip_x),
                        u32::from(d.flip_y),
                        rect.width(),
                        rect.height(),
                        (d.center.x as i32 - rect.x0) as u32,
                    ]
                });
            if let Some((_, stamp)) = self.stamp.as_ref().filter(|(cached, _)| Some(*cached) == key) {
                self.scratch.clear();
                self.scratch.extend_from_slice(stamp);
            } else {
                self.ctx.rasterize(d, false, rect, &mut self.scratch);
                // Do not keep a second unbounded copy of unusually large dabs.
                self.stamp = key.filter(|_| self.scratch.len() <= 4_194_304).map(|key| (key, Arc::from(self.scratch.as_slice())));
            }
            let col = match (self.per_dab_color, self.fmt) {
                (true, Some(f)) => {
                    from_rgba_into(&f, d.color, &mut native);
                    Some(&native[..f.mode.color_channels()])
                }
                _ => None,
            };
            self.cov.accumulate(rect, &self.scratch, d.opacity, wet || self.max_accumulate, col);
            // Bounds track the full dab rectangle (even fully transparent parts), like the damage.
            self.cov.bounds = self.cov.bounds.union(&rect);
        }
        if let Some(dm) = self.dual.as_mut() {
            for d in &duals {
                let rect = self.ctx.dab_rect(d, true);
                self.ctx.rasterize(d, true, rect, &mut self.scratch);
                dm.accumulate(rect, &self.scratch, 1.0, false, None);
            }
        }
        self.dabs_done += dabs.len();
        if let Some(all) = self.all_dabs.as_mut() {
            all.extend_from_slice(&dabs);
        }
        dabs.clear();
        duals.clear();
        self.dab_buf = dabs;
        self.dual_buf = duals;
    }

    /// Feed input points (any chunking gives the same result).
    pub fn push(&mut self, pts: &[StrokePoint]) {
        self.generator.push(pts, &mut self.dab_buf, &mut self.dual_buf);
        self.raster_pending();
    }

    /// End of stroke (smoothing catch-up, a lone first point).
    pub fn finish(&mut self) {
        self.generator.finish(&mut self.dab_buf, &mut self.dual_buf);
        self.raster_pending();
    }

    /// What finishing the stroke now would add (the smoothing catch-up tail to the last point, or
    /// a lone first dab), for live previews: a renderer holding copies of the coverage tiles the
    /// tail touches with the tail rendered in, so its `composite` draws exactly those tiles as
    /// [`finish`](Self::finish) would leave them. `None` when finishing adds nothing.
    pub fn tail_preview(&self) -> Option<StrokeRenderer> {
        let mut generator = self.generator.clone();
        let (mut dabs, mut duals) = (Vec::new(), Vec::new());
        generator.finish(&mut dabs, &mut duals);
        if dabs.is_empty() && duals.is_empty() {
            return None;
        }
        let mut keys = HashSet::new();
        for r in dabs.iter().map(|d| self.ctx.dab_rect(d, false)).chain(duals.iter().map(|d| self.ctx.dab_rect(d, true))) {
            if r.is_empty() {
                continue;
            }
            for ty in r.y0.div_euclid(COV_TILE)..=(r.y1 - 1).div_euclid(COV_TILE) {
                for tx in r.x0.div_euclid(COV_TILE)..=(r.x1 - 1).div_euclid(COV_TILE) {
                    keys.insert((tx, ty));
                }
            }
        }
        let subset = |m: &CoverageMap| CoverageMap {
            tiles: keys.iter().filter_map(|k| m.tiles.get(k).map(|t| (*k, t.clone()))).collect(),
            nc: m.nc,
            bounds: m.bounds,
            dirty: HashSet::new(),
        };
        let mut t = StrokeRenderer {
            ctx: self.ctx.clone(),
            generator,
            cov: subset(&self.cov),
            dual: self.dual.as_ref().map(subset),
            fmt: self.fmt,
            per_dab_color: self.per_dab_color,
            dabs_done: self.dabs_done,
            scratch: Vec::new(),
            stamp: None,
            dab_buf: dabs,
            dual_buf: duals,
            all_dabs: None,
            pixel_op: self.pixel_op.clone(),
            max_accumulate: self.max_accumulate,
        };
        t.raster_pending();
        // Tiles whose coverage the tail doesn't change still composite the same: redraw them all.
        t.cov.dirty.extend(t.cov.tiles.keys().copied());
        Some(t)
    }

    /// Mark the coverage tiles over `r` for the next `composite` (e.g. to redraw over a preview).
    pub fn mark_dirty(&mut self, r: Rect) {
        if r.is_empty() {
            return;
        }
        for ty in r.y0.div_euclid(COV_TILE)..=(r.y1 - 1).div_euclid(COV_TILE) {
            for tx in r.x0.div_euclid(COV_TILE)..=(r.x1 - 1).div_euclid(COV_TILE) {
                if self.cov.tiles.contains_key(&(tx, ty)) {
                    self.cov.dirty.insert((tx, ty));
                }
            }
        }
    }

    /// Final stroke coverage at a pixel (stroke-level masks applied, before opacity/selection).
    pub fn coverage_at(&self, x: i32, y: i32) -> f32 {
        let c = self.cov.get(x, y);
        if c <= 0.0 {
            return 0.0;
        }
        self.ctx.stroke_mask(c, self.dual.as_ref().map(|d| d.get(x, y)), x, y)
    }

    /// Dense final coverage over the stroke bounds.
    pub fn dense_coverage(&self) -> (Rect, Vec<f32>) {
        let b = self.bounds();
        let w = b.width() as usize;
        let mut out = vec![0.0f32; w * b.height() as usize];
        for (&(tx, ty), t) in &self.cov.tiles {
            let tr = Rect::new(tx * COV_TILE, ty * COV_TILE, (tx + 1) * COV_TILE, (ty + 1) * COV_TILE).intersect(&b);
            for y in tr.y0..tr.y1 {
                for x in tr.x0..tr.x1 {
                    let c = t.cov[((y - ty * COV_TILE) * COV_TILE + (x - tx * COV_TILE)) as usize];
                    if c > 0.0 {
                        out[(y - b.y0) as usize * w + (x - b.x0) as usize] = self.ctx.stroke_mask(c, self.dual.as_ref().map(|d| d.get(x, y)), x, y);
                    }
                }
            }
        }
        (b, out)
    }

    /// Composite the stroke buffer onto `target`, reading original pixels from `pre` (the
    /// pre-stroke surface; pass a clone of the target taken before the first `push`). With
    /// `all = false` only tiles touched since the last composite are processed (interactive use).
    /// Returns the damaged rectangle.
    pub fn composite(&mut self, pre: &Surface, target: &mut Surface, selection: Option<&Surface>, lock_transparency: bool, all: bool) -> Rect {
        let keys: Vec<(i32, i32)> = if all {
            self.cov.dirty.clear();
            let mut k: Vec<_> = self.cov.tiles.keys().copied().collect();
            k.sort_unstable();
            k
        } else {
            self.cov.take_dirty()
        };
        if let Some(d) = self.dual.as_mut() {
            d.dirty.clear();
        }
        let bounds = self.bounds();
        let fmt = target.format();
        let n = fmt.channels();
        let a_idx = alpha_index(&fmt);
        let nc = fmt.mode.color_channels();
        let b = &self.ctx.brush;
        let opacity = b.opacity.clamp(0.0, 1.0);
        let mut src = [0.0f32; 8];
        from_rgba_into(&fmt, b.color, &mut src);
        if let Some(a) = a_idx {
            src[a] = b.color[3];
        }
        let composite_tile = |&(tx, ty): &(i32, i32)| {
            let tile = self.cov.tiles.get(&(tx, ty))?;
            let tr = Rect::new(tx * COV_TILE, ty * COV_TILE, (tx + 1) * COV_TILE, (ty + 1) * COV_TILE).intersect(&bounds);
            if tr.is_empty() {
                return None;
            }
            let mut src = src;
            let mut region = Vec::new();
            pre.read_region_into(tr, &mut region);
            let sel = selection.map(|s| (s.channels(), s.read_region(tr)));
            let w = tr.width() as usize;
            if let Some(op) = &self.pixel_op {
                return self.composite_op(op, tile, tx, ty, tr, region, sel.as_ref(), opacity, lock_transparency, &fmt);
            }
            for y in tr.y0..tr.y1 {
                for x in tr.x0..tr.x1 {
                    let ti = ((y - ty * COV_TILE) * COV_TILE + (x - tx * COV_TILE)) as usize;
                    let c = tile.cov[ti];
                    if c <= 0.0 {
                        continue;
                    }
                    let i = (y - tr.y0) as usize * w + (x - tr.x0) as usize;
                    let m = self.ctx.stroke_mask(c, self.dual.as_ref().map(|d| d.get(x, y)), x, y);
                    let s = sel.as_ref().map_or(1.0, |(sc, v)| v[i * sc]);
                    let mut k = (m * opacity * s).min(1.0);
                    if k <= 0.0 {
                        continue;
                    }
                    if b.mode == BlendMode::Dissolve {
                        k = if hash2(x, y, b.seed) < k { 1.0 } else { 0.0 };
                        if k == 0.0 {
                            continue;
                        }
                    }
                    let px = &mut region[i * n..(i + 1) * n];
                    if b.erase {
                        if let (Some(a), false) = (a_idx, lock_transparency) {
                            px[a] *= 1.0 - k;
                        }
                        continue;
                    }
                    if lock_transparency && a_idx.is_some_and(|a| px[a] <= 0.0) {
                        continue;
                    }
                    if self.per_dab_color && !tile.col.is_empty() {
                        let p = &tile.col[ti * nc..(ti + 1) * nc];
                        for ch in 0..nc {
                            src[ch] = p[ch] / c;
                        }
                    }
                    if matches!(b.mode, BlendMode::Normal | BlendMode::Dissolve) {
                        over_native(&fmt, px, &src[..n], k, lock_transparency);
                    } else {
                        let d = to_rgba(&fmt, px);
                        let sr = to_rgba(&fmt, &src[..n]);
                        let mut o = photocraft_color::blend::composite(b.mode, d, sr, k);
                        if lock_transparency {
                            o[3] = d[3];
                        }
                        let mut enc = [0.0f32; 8];
                        from_rgba_into(&fmt, o, &mut enc);
                        px.copy_from_slice(&enc[..n]);
                    }
                }
            }
            Some((tr, region))
        };
        // Bound the temporary float buffers: at most sixteen coverage tiles per batch.
        let mut dmg = Rect::EMPTY;
        for batch in keys.chunks(16) {
            #[cfg(not(target_arch = "wasm32"))]
            let patches: Vec<_> = if batch.len() >= 8 && parallel_brush() {
                batch.par_iter().filter_map(composite_tile).collect()
            } else {
                batch.iter().filter_map(composite_tile).collect()
            };
            #[cfg(target_arch = "wasm32")]
            let patches: Vec<_> = batch.iter().filter_map(composite_tile).collect();
            for (tr, region) in patches {
                target.write_region(tr, &region);
                dmg = dmg.union(&tr);
            }
        }
        if all { bounds } else { dmg }
    }
}

impl StrokeRenderer {
    /// One coverage tile through the brush plug-in: `k` per pixel, the pre-stroke pixels in, the
    /// plug-in's result out (non-finite values keep the pixel; alpha and grey stay in `0..=1`).
    #[allow(clippy::too_many_arguments)]
    fn composite_op(
        &self,
        op: &PixelOp,
        tile: &CovTile,
        tx: i32,
        ty: i32,
        tr: Rect,
        mut region: Vec<f32>,
        sel: Option<&(usize, Vec<f32>)>,
        opacity: f32,
        lock_transparency: bool,
        fmt: &PixelFormat,
    ) -> Option<(Rect, Vec<f32>)> {
        let n = fmt.channels();
        let w = tr.width() as usize;
        let count = w * tr.height() as usize;
        let mut ks = vec![0.0f32; count];
        let mut any = false;
        for y in tr.y0..tr.y1 {
            for x in tr.x0..tr.x1 {
                let c = tile.cov[((y - ty * COV_TILE) * COV_TILE + (x - tx * COV_TILE)) as usize];
                if c <= 0.0 {
                    continue;
                }
                let i = (y - tr.y0) as usize * w + (x - tr.x0) as usize;
                let m = self.ctx.stroke_mask(c, self.dual.as_ref().map(|d| d.get(x, y)), x, y);
                let s = sel.map_or(1.0, |(sc, v)| v[i * sc]);
                ks[i] = (m * opacity * s).clamp(0.0, 1.0);
                any |= ks[i] > 0.0;
            }
        }
        if !any {
            return None;
        }
        let gray = alpha_index(fmt).is_none() && fmt.mode.color_channels() == 1;
        let channels = if gray { 1 } else { 4 };
        let mut vals = Vec::with_capacity(count * channels);
        for px in region.chunks_exact(n) {
            if gray {
                vals.push(px[0]);
            } else {
                let c = to_rgba(fmt, px);
                vals.extend([c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]]);
            }
        }
        (op.0)(tr, &mut vals, channels, &ks).ok()?;
        let mut enc = [0.0f32; 8];
        for (i, px) in region.chunks_exact_mut(n).enumerate() {
            if ks[i] <= 0.0 {
                continue;
            }
            let v = &vals[i * channels..(i + 1) * channels];
            if v.iter().any(|x| !x.is_finite()) {
                continue;
            }
            if gray {
                px[0] = v[0].clamp(0.0, 1.0);
                continue;
            }
            let old = to_rgba(fmt, px);
            let a = if lock_transparency { old[3] } else { v[3].clamp(0.0, 1.0) };
            let rgb = if v[3] > 1e-6 { [v[0] / v[3], v[1] / v[3], v[2] / v[3]] } else { [old[0], old[1], old[2]] };
            from_rgba_into(fmt, [rgb[0], rgb[1], rgb[2], a], &mut enc);
            px.copy_from_slice(&enc[..n]);
        }
        Some((tr, region))
    }
}

/// Render a whole stroke onto `target` (one-shot). Returns the damaged rectangle.
pub fn render_stroke(
    target: &mut Surface,
    brush: &BrushSettings,
    points: &[StrokePoint],
    selection: Option<&Surface>,
    lock_transparency: bool,
    zoom: f32,
) -> Rect {
    if points.is_empty() {
        return Rect::EMPTY;
    }
    let pre = target.clone();
    let mut r = StrokeRenderer::new(brush, Some(target.format()), zoom);
    r.push(points);
    r.finish();
    r.composite(&pre, target, selection, lock_transparency, true)
}
