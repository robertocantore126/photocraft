//! Feather Eraser: the softest transition between two images.
//!
//! Two controls, like a feathered selection: **Size** is the solid core
//! (erased fully at Flow 100 %), **Feather** is how far the fade runs on
//! each side, in pixels, independent of the core. The manifest's `brush`
//! asks the stroke engine for:
//!
//! * the **Feather** tip: a Gaussian fade whose tail runs on past the outline
//!   before it stops at 1/1024, so the eye cannot find where the erase
//!   begins (no corner, no Mach band);
//! * **Max** accumulation: the stroke is the tip swept along the path (each
//!   pixel takes the fade at its distance from the path), so scrubbing back
//!   and forth never erases more where you passed twice — no blotches, no
//!   ripple, the cross-section is always the clean fade. More strokes build
//!   up, each from what the last one left;
//! * 20 % spacing: dabs only mark the path's corners now.
//!
//! Flow is how much one stroke erases. **Breakup** roughens only the fade
//! band (never the fully erased or untouched parts) with soft cloud noise
//! scaled to the feather, for an organic edge instead of an airbrushed one.
//! Partial values get ±½ of an 8-bit step of dither, so the long fades do
//! not band.

use fotox_plugin::{Ctx, brush_plugin, noise};

brush_plugin! {
	manifest: include_str!("manifest.json"),
	rect: rect,
	gray: gray,
}

/// Smooth value noise in `0..1` with cells of `cell` pixels.
fn value_noise(x: f32, y: f32, cell: f32, seed: u32) -> f32 {
	let (fx, fy) = (x / cell, y / cell);
	let (ix, iy) = (fx.floor(), fy.floor());
	let (tx, ty) = (fx - ix, fy - iy);
	let ease = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
	let (sx, sy) = (ease(tx), ease(ty));
	let (ix, iy) = (ix as i32, iy as i32);
	let a = noise(ix, iy, seed);
	let b = noise(ix + 1, iy, seed);
	let c = noise(ix, iy + 1, seed);
	let d = noise(ix + 1, iy + 1, seed);
	let top = a + (b - a) * sx;
	let bottom = c + (d - c) * sx;
	top + (bottom - top) * sy
}

/// How much goes at build-up `k`, at canvas pixel `(x, y)`.
#[inline]
fn erased(k: f32, x: i32, y: i32, breakup: f32, cell: f32) -> f32 {
	let mut e = k;
	if breakup > 0.0 && e > 0.0 && e < 1.0 {
		// Two octaves of cloud noise, centred on 0.
		let n = 0.65 * value_noise(x as f32, y as f32, cell, 3) + 0.35 * value_noise(x as f32, y as f32, cell * 0.37, 4) - 0.5;
		// Only the band: e·(1 − e) is 0 where nothing or everything goes.
		e = (e + breakup * 4.0 * n * e * (1.0 - e) * 2.0).clamp(0.0, 1.0);
	}
	if e > 0.0 && e < 1.0 {
		e = (e + (noise(x, y, 2) - 0.5) / 255.0).clamp(0.0, 1.0);
	}
	e
}

fn settings(ctx: &Ctx) -> (f32, f32) {
	let breakup = (ctx.params[0] / 100.0).clamp(0.0, 1.0);
	// Clouds about half the feather across.
	let cell = (ctx.params[1] / 2.0).max(8.0);
	(breakup, cell)
}

fn rect(ctx: &Ctx, pixels: &mut [[f32; 4]], k: &[f32]) {
	let (breakup, cell) = settings(ctx);
	let [cr, cg, cb, _] = ctx.color;
	for (i, (p, &k)) in pixels.iter_mut().zip(k).enumerate() {
		if k <= 0.0 {
			continue;
		}
		let (x, y) = ctx.pos(i);
		let e = erased(k, x, y, breakup, cell);
		if ctx.lock_alpha {
			let a = p[3];
			p[0] += (cr * a - p[0]) * e;
			p[1] += (cg * a - p[1]) * e;
			p[2] += (cb * a - p[2]) * e;
		} else {
			let keep = 1.0 - e;
			p[0] *= keep;
			p[1] *= keep;
			p[2] *= keep;
			p[3] *= keep;
		}
	}
}

/// On a mask: toward the background colour's grey, with the same feather.
fn gray(ctx: &Ctx, values: &mut [f32], k: &[f32]) {
	let (breakup, cell) = settings(ctx);
	let c = ctx.color[0];
	for (i, (v, &k)) in values.iter_mut().zip(k).enumerate() {
		if k <= 0.0 {
			continue;
		}
		let (x, y) = ctx.pos(i);
		*v += (c - *v) * erased(k, x, y, breakup, cell);
	}
}
