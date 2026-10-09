# PhotoCraft W3 implementation ? 2026-10-09

`ui-egui/src/thumbs.rs` provides a bounded latest-request queue with two dedicated native threads. Only painted layer/mask rows request work; the newest visible requests have priority. In-flight obsolete results are ignored. Each frame collects images and updates egui textures on the UI thread; the row keeps its previous texture or a transparent placeholder meanwhile. Worker pixel failures become status errors and do not crash the UI. A dropped queue wakes and stops its detached workers after the current tile work.

Wasm and thread creation failure use at most two queued thumbnails per frame. CPU mip selection keeps at least 64 pixels on the long side before weighted box averaging, including premultiplied alpha. Masks use their native grayscale values. Width, height, default pixels and COW tile pointers participate in revision identity. No shader, texture, native app or browser smoke test was run.

Files: `ui-egui/src/lib.rs`, new `ui-egui/src/thumbs.rs`. Requires lane B W4 raster APIs (9409269).

Switches:

- `PHOTOCRAFT_BACKGROUND_THUMBS=0`: original synchronous nearest-neighbor layer/mask thumbnail path; no thumbnail workers started.
- `PHOTOCRAFT_THUMB_WORKERS=0`: queue remains active but uses the bounded synchronous fallback.
- `PHOTOCRAFT_THUMB_MIPS=0`: box thumbnails read original pixels instead of CPU mips.
- `PHOTOCRAFT_CPU_MIPS=0`: also disables thumbnail mip use.

Compilation/layers/wasm pending serialized build slot. No tests, benchmarks, measurements or screenshots were executed, per the implementation document and user request. Vector mask/channel thumbnails retain their existing separate code paths; W3 specifically replaces layer surface/mask thumbnails.
