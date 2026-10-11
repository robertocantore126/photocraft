//! The Eraser as a plugin: alpha down by `k`, exactly as the native
//! `fx_ops::brush::op::Erase`. Two jobs: the template to copy for a new brush
//! plugin, and the benchmark of what the plugin boundary costs
//! (`crates/fx-plugin/tests/boundary.rs` compares it with the native eraser).

use fotox_plugin::{Ctx, brush_plugin};

brush_plugin! {
	manifest: include_str!("manifest.json"),
	rect: rect,
	gray: gray,
}

fn rect(ctx: &Ctx, pixels: &mut [[f32; 4]], k: &[f32]) {
	let [cr, cg, cb, _] = ctx.color;
	for (p, &k) in pixels.iter_mut().zip(k) {
		if ctx.lock_alpha {
			// Lock Transparent Pixels: paint the colour, keep alpha.
			let a = p[3];
			p[0] += (cr * a - p[0]) * k;
			p[1] += (cg * a - p[1]) * k;
			p[2] += (cb * a - p[2]) * k;
		} else {
			let keep = 1.0 - k;
			p[0] *= keep;
			p[1] *= keep;
			p[2] *= keep;
			p[3] *= keep;
		}
	}
}

/// On a mask the eraser paints the background colour's grey.
fn gray(ctx: &Ctx, values: &mut [f32], k: &[f32]) {
	let c = ctx.color[0];
	for (v, &k) in values.iter_mut().zip(k) {
		*v += (c - *v) * k;
	}
}
