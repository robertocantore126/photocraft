# PhotoCraft W3 implementation ? 2026-10-09

`ui-egui/src/thumbs.rs` provides a bounded latest-request queue with two dedicated native threads. Requests are capped at 512 and completed replies use a bounded 32-image channel. Cache/queue keys include document identity, layer identity and thumbnail kind. Only painted layer/mask rows request work; the newest visible requests have priority. In-flight obsolete results are ignored. Each frame collects images and updates egui textures on the UI thread; the row keeps its previous texture or a transparent placeholder meanwhile. Worker pixel failures become status errors and do not crash the UI. A dropped queue wakes and stops its detached workers after the current tile work.

Wasm and thread creation failure use at most two queued thumbnails per frame. CPU mip selection keeps at least 64 pixels on the long side before weighted box averaging, including premultiplied alpha. Masks use their native grayscale values. Width, height, default pixels and COW tile pointers participate in revision identity. No shader, texture, native app or browser smoke test was run.

Files: `ui-egui/src/lib.rs`, new `ui-egui/src/thumbs.rs`, document-scoped vector thumbnail key in `mask_thumbs_ui.rs`. Existing thumbnail test key expressions were adjusted for the cache type only; no tests were added or run. Requires lane B W4 raster APIs (9409269).

Switches:

- `PHOTOCRAFT_BACKGROUND_THUMBS=0`: original synchronous nearest-neighbor layer/mask thumbnail path; no thumbnail workers started.
- `PHOTOCRAFT_THUMB_WORKERS=0`: queue remains active but uses the bounded synchronous fallback.
- `PHOTOCRAFT_THUMB_MIPS=0`: box thumbnails read original pixels instead of CPU mips.
- `PHOTOCRAFT_CPU_MIPS=0`: also disables thumbnail mip use.

`cargo check -p photocraft-raster -p photocraft-format -p photocraft-ui-egui` passed. Full build/layers/wasm pending unified integration validation. No tests, benchmarks, measurements or screenshots were executed, per the implementation document and user request. Vector mask/channel thumbnails retain their existing separate code paths; W3 specifically replaces layer surface/mask thumbnails.

Final unified integration (2026-10-09): native workspace build passed; layering passed for26 crates; wasm checks passed for20/20 crates with warnings. Both new WGSL shaders passed Naga30 static validation. Tests, runtime/UI validation, pixel comparisons and performance measurements were not run. See [integration report](2026-10-09-ps-integration.md) for the final tree, switches and remaining limits; earlier lane-local pending notes above describe their original commit checkpoints.
