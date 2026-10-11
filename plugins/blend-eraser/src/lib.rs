//! Blend Eraser: an eraser that dissolves the layer by its own tones.
//!
//! A soft round eraser takes the same opacity from every pixel under it, so
//! the transition between two images is a round fog that ignores both
//! images. This one gives every pixel a threshold from its luminance and
//! erases it as the stroke's build-up `k` passes that threshold:
//!
//! * **Shadows First**: dark pixels go at a light touch, mid-tones with more
//!   strokes, highlights last — the layer melts into what is below through
//!   its darks, its lights (hair, rays, glow) survive longest;
//! * **Highlights First**: the reverse;
//! * **Off**: no tone, only the S-curve (a cleaner edge than a linear fade).
//!
//! **Softness** is how wide the tonal transition is: low = a sharp tonal cut
//! (like Blend If with split sliders), high = close to a plain soft eraser.
//! **Grain** jitters each pixel's threshold with position-stable noise: a
//! dissolve that looks like film grain instead of a smooth gradient.
//! Opacity caps how far a stroke reaches into the tones; Flow sets how fast
//! repeated passes get there.
//!
//! Every partial value gets ±½ of an 8-bit step of position-stable dither,
//! so long soft fades do not band on 8-bit layers.

use fotox_plugin::{Ctx, brush_plugin, luma, noise, smoothstep};

brush_plugin! {
	manifest: include_str!("manifest.json"),
	rect: rect,
	gray: gray,
}

/// How much of the pixel goes, `0..=1`, at build-up `k` for a pixel whose
/// tonal threshold is `t` (`0` = goes first). At `k = 0` nothing goes; at
/// `k = 1` everything goes, whatever `t` and `softness`.
#[inline]
fn erased(k: f32, t: f32, softness: f32) -> f32 {
	smoothstep((k * (1.0 + softness) - t) / softness)
}

struct Settings {
	tone: u32,
	softness: f32,
	grain: f32,
}

fn settings(ctx: &Ctx) -> Settings {
	Settings {
		tone: ctx.params[0] as u32,
		softness: (ctx.params[1] / 100.0).clamp(0.02, 1.0),
		grain: (ctx.params[2] / 100.0).clamp(0.0, 1.0),
	}
}

/// The pixel's threshold from its tone, jittered by the grain.
#[inline]
fn threshold(s: &Settings, l: f32, x: i32, y: i32) -> f32 {
	let t = match s.tone {
		0 => l,
		1 => 1.0 - l,
		_ => 0.5,
	};
	if s.grain > 0.0 {
		(t + s.grain * (noise(x, y, 1) - 0.5)).clamp(0.0, 1.0)
	} else {
		t
	}
}

#[inline]
fn dither(e: f32, x: i32, y: i32) -> f32 {
	if e > 0.0 && e < 1.0 {
		(e + (noise(x, y, 2) - 0.5) / 255.0).clamp(0.0, 1.0)
	} else {
		e
	}
}

fn rect(ctx: &Ctx, pixels: &mut [[f32; 4]], k: &[f32]) {
	let s = settings(ctx);
	let [cr, cg, cb, _] = ctx.color;
	for (i, (p, &k)) in pixels.iter_mut().zip(k).enumerate() {
		let a = p[3];
		if k <= 0.0 || a <= 0.0 {
			continue;
		}
		let (x, y) = ctx.pos(i);
		let l = luma(p[0] / a, p[1] / a, p[2] / a);
		let e = dither(erased(k, threshold(&s, l, x, y), s.softness), x, y);
		if ctx.lock_alpha {
			// Lock Transparent Pixels: the colour comes in where the tones go.
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

/// On a mask there is no image to read tones from: the mask's own grey is
/// the tone (a mask painted dark goes first under Shadows First), toward the
/// background colour's grey.
fn gray(ctx: &Ctx, values: &mut [f32], k: &[f32]) {
	let s = settings(ctx);
	let c = ctx.color[0];
	for (i, (v, &k)) in values.iter_mut().zip(k).enumerate() {
		if k <= 0.0 {
			continue;
		}
		let (x, y) = ctx.pos(i);
		let e = dither(erased(k, threshold(&s, *v, x, y), s.softness), x, y);
		*v += (c - *v) * e;
	}
}
