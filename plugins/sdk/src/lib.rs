//! The plugin side of Fotox's brush plugins (D-096).
//!
//! A brush plugin decides what a stroke does to the pixels under it. The
//! stroke engine keeps everything else: the tip, spacing, pressure, the
//! selection, the coverage buffer (opacity ceiling, flow build-up), undo, and
//! live = replay. It calls the plugin **once per rectangle** (a dirty part of
//! one layer tile), never once per pixel, with:
//!
//! * `pixels`: the layer **as it was when the stroke started**, premultiplied
//!   RGBA `0..=1`, row-major — the plugin overwrites them with the result;
//! * `k`: opacity × coverage `0..=1` per pixel (how much the stroke has
//!   built up there);
//! * [`Ctx`]: the option-bar params, the paint colour, the rectangle's canvas
//!   position (for position-stable noise).
//!
//! The result must depend on nothing else (no state between calls, no
//! randomness that is not a function of the position): a live stroke and its
//! replay then paint the same pixels, whatever the tiling.
//!
//! The host ABI (`crates/fx-plugin/src/lib.rs` is the other side):
//! * `fx_manifest() -> u64`: `ptr << 32 | len` of the manifest JSON;
//! * `fx_alloc(bytes) -> ptr`: a scratch buffer of at least `bytes`;
//! * `fx_rect(ptr, w, h, x0, y0) -> i32`: `ptr` holds the header
//!   ([`HEADER_WORDS`] f32), then `w·h` RGBA pixels, then `w·h` `k`;
//! * `fx_gray(ptr, w, h, x0, y0) -> i32` (optional): the same for a mask —
//!   `w·h` grey values instead of the pixels.

use std::cell::RefCell;

/// f32 words before the pixels: params `0..16`, colour `16..20`, flags `20`.
pub const HEADER_WORDS: usize = 32;
/// Option-bar params a plugin can read.
pub const MAX_PARAMS: usize = 16;

/// What a rectangle call knows besides its pixels.
pub struct Ctx<'a> {
	/// The manifest's `params`, in order: a number field's value as shown
	/// (a 40 % slider is `40.0`), a toggle `0`/`1`, a drop-down the index of
	/// the chosen entry.
	pub params: &'a [f32],
	/// The paint colour, straight RGBA `0..=1` (the manifest's `color`:
	/// foreground or background). For a mask, `color[0]` is its grey.
	pub color: [f32; 4],
	/// The layer's "Lock transparent pixels": keep alpha.
	pub lock_alpha: bool,
	/// Canvas pixel of the rectangle's first pixel.
	pub x0: i32,
	pub y0: i32,
	pub w: usize,
	pub h: usize,
}

impl Ctx<'_> {
	/// Canvas position of pixel `i` of the rectangle.
	#[inline]
	pub fn pos(&self, i: usize) -> (i32, i32) {
		(self.x0 + (i % self.w) as i32, self.y0 + (i / self.w) as i32)
	}
}

/// A hash of a canvas position, uniform in `0..1`: the same value for the
/// same pixel in every call, live or replayed.
#[inline]
pub fn noise(x: i32, y: i32, seed: u32) -> f32 {
	let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
	h ^= h >> 16;
	h = h.wrapping_mul(0x7feb_352d);
	h ^= h >> 15;
	h = h.wrapping_mul(0x846c_a68b);
	h ^= h >> 16;
	(h >> 8) as f32 / (1u32 << 24) as f32
}

/// Rec. 709 luma of straight RGB.
#[inline]
pub fn luma(r: f32, g: f32, b: f32) -> f32 {
	0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// `0` below 0, `1` above 1, an S-curve between.
#[inline]
pub fn smoothstep(x: f32) -> f32 {
	let x = x.clamp(0.0, 1.0);
	x * x * (3.0 - 2.0 * x)
}

thread_local! {
	static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[doc(hidden)]
pub fn alloc(bytes: u32) -> u32 {
	SCRATCH.with(|s| {
		let mut s = s.borrow_mut();
		let words = (bytes as usize).div_ceil(4);
		if s.len() < words {
			s.resize(words, 0.0);
		}
		s.as_mut_ptr() as u32
	})
}

#[doc(hidden)]
pub fn manifest(json: &'static str) -> u64 {
	(u64::from(json.as_ptr() as u32) << 32) | json.len() as u64
}

/// # Safety
/// `ptr` comes from [`alloc`] with room for the header, `w·h·channels` values
/// and `w·h` coverages.
#[doc(hidden)]
pub unsafe fn split<'a>(ptr: u32, w: u32, h: u32, x0: i32, y0: i32, channels: usize) -> (Ctx<'a>, &'a mut [f32], &'a [f32]) {
	let n = w as usize * h as usize;
	let all = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, HEADER_WORDS + n * channels + n) };
	let (header, rest) = all.split_at_mut(HEADER_WORDS);
	let (values, k) = rest.split_at_mut(n * channels);
	let ctx = Ctx {
		params: &header[..MAX_PARAMS],
		color: [header[16], header[17], header[18], header[19]],
		lock_alpha: header[20] != 0.0,
		x0,
		y0,
		w: w as usize,
		h: h as usize,
	};
	(ctx, values, k)
}

/// Export a brush plugin: its manifest JSON, the RGBA function
/// `fn(&Ctx, &mut [[f32; 4]], &[f32])` and, optionally, the mask function
/// `fn(&Ctx, &mut [f32], &[f32])`.
#[macro_export]
macro_rules! brush_plugin {
	(manifest: $manifest:expr, rect: $rect:path $(, gray: $gray:path)? $(,)?) => {
		#[unsafe(no_mangle)]
		pub extern "C" fn fx_manifest() -> u64 {
			$crate::manifest($manifest)
		}

		#[unsafe(no_mangle)]
		pub extern "C" fn fx_alloc(bytes: u32) -> u32 {
			$crate::alloc(bytes)
		}

		#[unsafe(no_mangle)]
		pub extern "C" fn fx_rect(ptr: u32, w: u32, h: u32, x0: i32, y0: i32) -> i32 {
			// SAFETY: the host lays the buffer out as `split` reads it.
			let (ctx, values, k) = unsafe { $crate::split(ptr, w, h, x0, y0, 4) };
			let (pixels, _) = values.as_chunks_mut::<4>();
			$rect(&ctx, pixels, k);
			0
		}

		$(
			#[unsafe(no_mangle)]
			pub extern "C" fn fx_gray(ptr: u32, w: u32, h: u32, x0: i32, y0: i32) -> i32 {
				// SAFETY: as `fx_rect`, one channel.
				let (ctx, values, k) = unsafe { $crate::split(ptr, w, h, x0, y0, 1) };
				$gray(&ctx, values, k);
				0
			}
		)?
	};
}
