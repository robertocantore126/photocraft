//! SAI-style painting brush: our own behavioural model of the Brush, Watercolor and Marker tools
//! of PaintTool SAI (no SAI code or data).
//!
//! Each dab moves the canvas under it towards a *carried* colour instead of stamping the brush
//! colour over it:
//! - the fresh paint is the brush colour with its alpha lowered by **Dilution**;
//! - the carried colour keeps **Persistence** of what it carried on the previous dab and takes the
//!   rest from the fresh paint;
//! - it then takes in **Blending** of the canvas colour averaged under the dab;
//! - the canvas is interpolated towards it (premultiplied) by the dab coverage × **Density** (the
//!   brush opacity). Interpolating, not compositing over, lets a diluted brush thin colour out:
//!   that is what makes the watercolor look.
//!
//! The dab edge can be broken up by a **shape** map (SAI's blotmap) and the coverage multiplied by
//! a **texture** (SAI's brush texture); both are tiled in document space, so they stay put on the
//! canvas. The painter works incrementally on the surface itself (each dab reads what the earlier
//! ones left), so a live preview fed point by point paints the same pixels as the whole stroke.

use std::sync::Arc;

use photocraft_geom::Rect;
use photocraft_raster::{Surface, from_rgba, to_rgba};
use serde::{Deserialize, Serialize};

use crate::dynamics::DabGenerator;
use crate::{BrushContext, BrushSettings, Dab, GrayTile, StrokePoint};

/// SAI brush options (fractions are 0..1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct SaiSettings {
    /// How much canvas colour each dab takes in.
    pub blending: f32,
    /// How much the brush colour is thinned (its alpha lowered).
    pub dilution: f32,
    /// How much of the colour carried so far survives each dab.
    pub persistence: f32,
    /// Never raise the layer's alpha (SAI "Keep Opacity").
    pub keep_opacity: bool,
    /// Strength of the shape map on the dab edge.
    pub shape_strength: f32,
    /// Shape map scale (1 = one texel per pixel).
    pub shape_scale: f32,
    /// Strength of the texture on the coverage.
    pub texture_strength: f32,
    /// Texture scale (1 = one texel per pixel).
    pub texture_scale: f32,
}

impl Default for SaiSettings {
    fn default() -> Self {
        Self {
            blending: 0.5,
            dilution: 0.0,
            persistence: 0.8,
            keep_opacity: false,
            shape_strength: 0.0,
            shape_scale: 1.0,
            texture_strength: 0.0,
            texture_scale: 1.0,
        }
    }
}

/// Grayscale maps for a SAI brush (looked up by name in the session's texture library).
#[derive(Clone, Debug, Default)]
pub struct SaiMaps {
    pub shape: Option<Arc<GrayTile>>,
    pub texture: Option<Arc<GrayTile>>,
}

fn tiled(t: &GrayTile, x: i32, y: i32, scale: f32) -> f32 {
    if t.width == 0 || t.height == 0 {
        return 1.0;
    }
    let s = scale.max(0.05);
    let (u, v) = (((x as f32 + 0.5) / s).floor() as i64, ((y as f32 + 0.5) / s).floor() as i64);
    t.get(u.rem_euclid(i64::from(t.width)) as u32, v.rem_euclid(i64::from(t.height)) as u32)
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t]
}

fn premul(c: [f32; 4]) -> [f32; 4] {
    [c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]]
}

/// Paints a SAI-style stroke incrementally onto a surface.
pub struct SaiPainter {
    ctx: BrushContext,
    generator: DabGenerator,
    settings: SaiSettings,
    maps: SaiMaps,
    /// Fresh paint (premultiplied, diluted).
    paint: [f32; 4],
    /// Carried colour (premultiplied); `None` before the first dab.
    carried: Option<[f32; 4]>,
    density: f32,
    index: usize,
    bounds: Rect,
    dabs: Vec<Dab>,
    dual: Vec<Dab>,
    buf: Vec<f32>,
}

impl SaiPainter {
    pub fn new(brush: &BrushSettings, settings: &SaiSettings, maps: SaiMaps, zoom: f32) -> Self {
        let dil = settings.dilution.clamp(0.0, 1.0);
        let c = brush.color;
        let paint = premul([c[0], c[1], c[2], c[3] * (1.0 - dil)]);
        // The density is applied per dab here, so the stroke-level opacity ceiling is off.
        let mut b = brush.clone();
        b.opacity = 1.0;
        Self {
            ctx: BrushContext::new(&b),
            generator: DabGenerator::new(&b, zoom),
            settings: settings.clone(),
            maps,
            paint,
            carried: None,
            density: brush.opacity.clamp(0.0, 1.0),
            index: 0,
            bounds: Rect::EMPTY,
            dabs: Vec::new(),
            dual: Vec::new(),
            buf: Vec::new(),
        }
    }

    /// Everything painted so far.
    pub fn bounds(&self) -> Rect {
        self.bounds
    }

    /// Paint more points; returns the rectangle that changed.
    pub fn push(&mut self, pts: &[StrokePoint], surf: &mut Surface, sel: Option<&Surface>, lock: bool) -> Rect {
        self.generator.push(pts, &mut self.dabs, &mut self.dual);
        self.paint_pending(surf, sel, lock)
    }

    /// End of stroke (smoothing catch-up, a lone first point).
    pub fn finish(&mut self, surf: &mut Surface, sel: Option<&Surface>, lock: bool) -> Rect {
        self.generator.finish(&mut self.dabs, &mut self.dual);
        self.paint_pending(surf, sel, lock)
    }

    fn paint_pending(&mut self, surf: &mut Surface, sel: Option<&Surface>, lock: bool) -> Rect {
        self.dual.clear();
        let dabs = std::mem::take(&mut self.dabs);
        let mut dmg = Rect::EMPTY;
        for d in &dabs {
            dmg = dmg.union(&self.dab(d, surf, sel, lock));
        }
        self.bounds = self.bounds.union(&dmg);
        dmg
    }

    fn dab(&mut self, d: &Dab, surf: &mut Surface, sel: Option<&Surface>, lock: bool) -> Rect {
        let fp = self.ctx.footprint(d, self.index);
        self.index += 1;
        let r = fp.rect;
        if r.is_empty() {
            return Rect::EMPTY;
        }
        let fmt = surf.format();
        let n = fmt.channels();
        let keep = lock || self.settings.keep_opacity || !fmt.alpha;
        let s = &self.settings;
        // Coverage per pixel: tip × shape map on the edge × texture × selection.
        let w = r.width() as usize;
        let mut cov = vec![0.0f32; fp.cov.len()];
        let mut any = false;
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let i = (y - r.y0) as usize * w + (x - r.x0) as usize;
                let mut c = fp.cov[i];
                if c <= 0.0 {
                    continue;
                }
                if let Some(m) = &self.maps.shape {
                    // Low tip coverage (the edge) survives only where the map is bright.
                    let k = s.shape_strength.clamp(0.0, 1.0);
                    let edge = (tiled(m, x, y, s.shape_scale) * 2.0 - (1.0 - c)).clamp(0.0, 1.0);
                    c *= 1.0 - k + k * edge;
                }
                if let Some(t) = &self.maps.texture {
                    let k = s.texture_strength.clamp(0.0, 1.0);
                    c *= 1.0 - k * (1.0 - tiled(t, x, y, s.texture_scale));
                }
                if let Some(m) = sel {
                    c *= m.sample_channel(x, y, 0);
                }
                cov[i] = c;
                any |= c > 0.0;
            }
        }
        if !any {
            return Rect::EMPTY;
        }
        surf.read_region_into(r, &mut self.buf);
        // RGBA layers (the usual case) are read and written in place, without conversions.
        let rgba = fmt.mode == photocraft_color::ColorMode::Rgb && fmt.alpha && n == 4;
        let read = |px: &[f32]| if rgba { [px[0], px[1], px[2], px[3]] } else { to_rgba(&fmt, px) };
        // Canvas colour under the dab (premultiplied, weighted by coverage).
        let (mut acc, mut wsum) = ([0.0f32; 4], 0.0f32);
        for (i, &c) in cov.iter().enumerate() {
            if c > 0.0 {
                let p = premul(read(&self.buf[i * n..i * n + n]));
                for k in 0..4 {
                    acc[k] += p[k] * c;
                }
                wsum += c;
            }
        }
        let canvas = if wsum > 0.0 { acc.map(|v| v / wsum) } else { [0.0; 4] };
        // Straight colour and alpha of the carried paint: the colour mixes with the canvas in
        // proportion to how much paint is there (empty canvas has no colour to give); the alpha
        // follows the canvas only for a diluted brush (a watercolor thins out over empty paper,
        // a plain brush keeps its density).
        let unpremul = |p: [f32; 4]| if p[3] > 1e-6 { [p[0] / p[3], p[1] / p[3], p[2] / p[3], p[3]] } else { [0.0, 0.0, 0.0, 0.0] };
        let fresh = unpremul(self.paint);
        let mut k = match self.carried {
            None => fresh,
            Some(prev) => {
                let prev = unpremul(prev);
                let t = s.persistence.clamp(0.0, 1.0);
                let rgb_t = if prev[3] > 1e-6 { t } else { 0.0 };
                [fresh[0] + (prev[0] - fresh[0]) * rgb_t, fresh[1] + (prev[1] - fresh[1]) * rgb_t, fresh[2] + (prev[2] - fresh[2]) * rgb_t, fresh[3] + (prev[3] - fresh[3]) * t]
            }
        };
        let cv = unpremul(canvas);
        let blend = s.blending.clamp(0.0, 1.0);
        let tc = blend * cv[3];
        for c in 0..3 {
            k[c] += (cv[c] - k[c]) * tc;
        }
        k[3] += (cv[3] - k[3]) * blend * s.dilution.clamp(0.0, 1.0);
        let carried = premul(k);
        self.carried = Some(carried);
        for (i, &c) in cov.iter().enumerate() {
            let a = (c * self.density).min(1.0);
            if a <= 0.0 {
                continue;
            }
            let px = &mut self.buf[i * n..i * n + n];
            let dst = read(px);
            let out = lerp4(premul(dst), carried, a);
            let alpha = if keep { dst[3] } else { out[3].clamp(0.0, 1.0) };
            let rgb = if out[3] > 1e-6 { [out[0] / out[3], out[1] / out[3], out[2] / out[3]] } else { [dst[0], dst[1], dst[2]] };
            let o = [rgb[0].clamp(0.0, 1.0), rgb[1].clamp(0.0, 1.0), rgb[2].clamp(0.0, 1.0), alpha];
            if rgba {
                px.copy_from_slice(&o);
            } else {
                px.copy_from_slice(&from_rgba(&fmt, o)[..n]);
            }
        }
        surf.write_region(r, &self.buf);
        r
    }
}
