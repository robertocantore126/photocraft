# Writing a PhotoCraft brush plugin

Paste this whole file to an AI, then describe the tool you want, for example
"an eraser that fades more in the bright areas" or "a brush that paints
with film grain". The AI answers with **one `.rs` file**; save it in
PhotoCraft's brush plug-in folder (Edit ▸ Brush Plug-ins ▸ Open Plug-ins Folder). PhotoCraft
builds it in a few seconds and the tool appears in its toolbar slot. If it
does not build, PhotoCraft writes `<name>.errors.txt` next to it: paste that file
back to the AI and ask it to fix the code. Save the fixed file over the old
one; PhotoCraft reloads it by itself.

---

## Instructions for the AI

You are writing a **brush plugin for PhotoCraft**, a Photoshop-like image editor.
A brush plugin decides what a brush stroke does to the pixels under it. PhotoCraft
does everything else: the brush tip, size, hardness, spacing, pen pressure,
smoothing, the selection, undo. It builds your file as Rust for
`wasm32-unknown-unknown` and runs it sandboxed.

### What to answer

Exactly **one complete Rust source file** in one code block, nothing to add
or install, plus a one-line suggested file name (lowercase, `-`, ending in
`.rs`, e.g. `grain-brush.rs`). The file is the whole crate (`src/lib.rs`).

### Hard rules (the build or the sandbox enforces them)

1. The only crate available is `fotox_plugin`. **No other crates.** `core`,
   `alloc` and the pure parts of `std` (math, `Vec`, iterators) are fine.
2. **No** files, network, threads, time, randomness from the system, `println!`,
   environment: there is no operating system inside the sandbox. Calling them
   crashes the plugin.
3. **No `unsafe`.** **Never panic**: no `unwrap()` on anything that can fail,
   no indexing that can go out of bounds (prefer iterators and `.get()`), no
   division by a value that can be 0 without a guard.
4. **No state between calls**: no `static mut`, no `thread_local`, no caches.
   The result must depend only on the inputs below, so that painting a stroke
   live and replaying it give the same pixels. For noise use
   `fotox_plugin::noise(x, y, seed)` with the pixel's canvas position.
5. **Be fast**: a call gets up to 65,536 pixels and must finish in well under
   0.5 s (aim for a few milliseconds). Simple per-pixel arithmetic is fine;
   no per-pixel loops over neighbours larger than a few pixels, no allocation
   per pixel.
6. Every value you write must be finite (no NaN or infinity) and in `0..=1`.

A plugin that crashes, hangs or is too slow is **stopped** by PhotoCraft until the
file is saved again; invalid pixels are ignored.

### The file's shape

```rust
use fotox_plugin::{Ctx, brush_plugin, luma, noise, smoothstep};

const MANIFEST: &str = r#"{
  "id": "my-tool",
  "name": "My Tool",
  "slot": "eraser",
  "icon": "i-eraser",
  "color": "background",
  "options": [
    { "type": "num", "text": "Size:", "value": "200", "unit": "px", "width": 40 },
    { "type": "num", "text": "Hardness:", "value": "0", "unit": "%", "width": 40 },
    { "type": "range", "text": "Opacity:", "value": 100 },
    { "type": "range", "text": "Flow:", "value": 50 },
    { "type": "range", "text": "Strength:", "value": 50 },
    { "type": "num", "text": "Smoothing:", "value": "10", "unit": "%", "width": 36 },
    { "type": "toggle", "text": "Pressure for size", "on": false }
  ],
  "params": ["Strength"]
}"#;

brush_plugin! {
    manifest: MANIFEST,
    rect: rect,
    gray: gray,
}

fn rect(ctx: &Ctx, pixels: &mut [[f32; 4]], k: &[f32]) {
    let strength = (ctx.params[0] / 100.0).clamp(0.0, 1.0);
    for (i, (p, &k)) in pixels.iter_mut().zip(k).enumerate() {
        if k <= 0.0 {
            continue;
        }
        let (x, y) = ctx.pos(i);
        // ... compute the new premultiplied pixel into *p ...
    }
}

fn gray(ctx: &Ctx, values: &mut [f32], k: &[f32]) {
    // The same tool on a layer mask (grey 0..=1). Optional: leave out
    // `gray: gray,` above and the tool does nothing on masks.
}
```

### The inputs

`rect(ctx, pixels, k)` is called for one rectangle of the layer at a time,
`ctx.w × ctx.h` pixels, row-major.

* `pixels[i]` is `[r, g, b, a]`, **premultiplied** (`r ≤ a`), `0..=1`: the
  layer **as it was when the stroke started**. Overwrite it with the result.
  Straight colour is `r / a` (guard `a > 0`).
* `k[i]` (`0..=1`) is how much the stroke has built up at that pixel: brush
  tip × opacity × flow × pen pressure × selection. `k = 0` means the stroke
  does not touch it (leave it unchanged); `k = 1` is full effect. A normal
  brush mixes by `k`: `new = old + (target − old) · k`.
* `ctx.params[n]` is the n-th option-bar field named in the manifest's
  `params`: a number field gives the number shown (a 40 % slider is `40.0`),
  a toggle `0.0`/`1.0`, a drop-down the index of the chosen entry (`0.0`,
  `1.0`…).
* `ctx.color` is the paint colour `[r, g, b, a]`, straight, `0..=1`: the
  foreground swatch, or the background one with `"color": "background"`
  (what erasers use).
* `ctx.lock_alpha` is the layer's Lock Transparent Pixels: when true, do not
  change `a` (scale r, g, b only).
* `ctx.pos(i)` is the canvas position `(x, y)` of pixel `i`; `ctx.x0`,
  `ctx.y0` the first pixel's.

Helpers: `noise(x, y, seed) -> f32` (`0..1`, the same for the same pixel
every time), `luma(r, g, b)` (Rec. 709, straight colour), `smoothstep(x)`.

Common operations, all premultiplied:
* erase by `e` (`0..=1`): multiply all four channels by `1 − e`;
* paint straight colour `c` over the pixel with amount `t` (normal blend):
  for each colour channel `p[ch] = p[ch] * (1.0 - t) + c[ch] * t;` and
  `p[3] = p[3] * (1.0 - t) + t;`
* change colour but keep alpha: work on straight colour `r / a`, then
  multiply back by `a`.

`gray(ctx, values, k)` is the same for a layer mask: `values[i]` is the mask
grey `0..=1` (1 = visible), `ctx.color[0]` the swatch's grey.

### The manifest

* `id`: unique, 1–64 characters of `a-z 0-9 - _`. Change it only to make a
  different tool.
* `name`: shown in the toolbar and the History panel, e.g. "Grain Brush Tool".
* `slot`: the toolbar group whose flyout gets the tool: `brush`, `eraser`,
  `clone`, `history-brush`, `blur`, `dodge`, `heal`, `gradient`.
* `icon`: one of `i-brush`, `i-eraser`, `i-clone-stamp`, `i-history-brush`,
  `i-blur`, `i-dodge`, `i-spot-heal`, `i-gradient`.
* `color`: `"foreground"` (default) or `"background"`.
* `options`: the option bar. Field types: `num` (`text`, `value` as a string,
  `unit`, `width`), `range` (a 0–100 slider; `text`, `value` number),
  `select` (`text`, `options` list, `value`), `toggle` (`text`, `on`). `text`
  ends with a colon except for toggles. These names drive the brush engine
  itself, include the ones you need: `Size:`, `Hardness:`, `Opacity:`,
  `Flow:`, `Spacing:`, `Smoothing:`, `Pressure for size`,
  `Pressure for opacity`. Add your own fields for your params.
* `params`: the field names (text without the colon) your code reads, in
  order, at most 16.
* `brush` (optional): `{"profile": "feather"}` for a long Gaussian tip
  (also `"classic"`, `"gaussian"`); `{"accumulate": "max"}` so going back
  over the same place within one stroke adds nothing (smooth, no blotches);
  `{"feather_from": "Feather"}` with a `Feather:` px field: `Size` becomes a
  solid core plus that many pixels of fade.

### Complete example: an eraser that removes dark tones first

```rust
use fotox_plugin::{Ctx, brush_plugin, luma, noise, smoothstep};

const MANIFEST: &str = r#"{
  "id": "shadow-eraser",
  "name": "Shadow Eraser Tool",
  "slot": "eraser",
  "icon": "i-eraser",
  "color": "background",
  "options": [
    { "type": "num", "text": "Size:", "value": "200", "unit": "px", "width": 40 },
    { "type": "num", "text": "Hardness:", "value": "0", "unit": "%", "width": 40 },
    { "type": "range", "text": "Opacity:", "value": 100 },
    { "type": "range", "text": "Flow:", "value": 30 },
    { "type": "range", "text": "Softness:", "value": 40 },
    { "type": "num", "text": "Smoothing:", "value": "10", "unit": "%", "width": 36 }
  ],
  "params": ["Softness"]
}"#;

brush_plugin! {
    manifest: MANIFEST,
    rect: rect,
    gray: gray,
}

fn amount(k: f32, tone: f32, softness: f32) -> f32 {
    smoothstep((k * (1.0 + softness) - tone) / softness)
}

fn rect(ctx: &Ctx, pixels: &mut [[f32; 4]], k: &[f32]) {
    let softness = (ctx.params[0] / 100.0).clamp(0.02, 1.0);
    for (i, (p, &k)) in pixels.iter_mut().zip(k).enumerate() {
        let a = p[3];
        if k <= 0.0 || a <= 0.0 {
            continue;
        }
        let (x, y) = ctx.pos(i);
        let tone = luma(p[0] / a, p[1] / a, p[2] / a);
        // Half a level of dither: no banding in long fades.
        let e = (amount(k, tone, softness) + (noise(x, y, 7) - 0.5) / 255.0).clamp(0.0, 1.0);
        let keep = 1.0 - e;
        if ctx.lock_alpha {
            for ch in 0..3 {
                p[ch] = p[ch] * keep + ctx.color[ch] * a * e;
            }
        } else {
            for v in p.iter_mut() {
                *v *= keep;
            }
        }
    }
}

fn gray(ctx: &Ctx, values: &mut [f32], k: &[f32]) {
    let softness = (ctx.params[0] / 100.0).clamp(0.02, 1.0);
    for (v, &k) in values.iter_mut().zip(k) {
        if k > 0.0 {
            *v += (ctx.color[0] - *v) * amount(k, *v, softness);
        }
    }
}
```

Before answering, check your code against the six hard rules.
