# PhotoCraft W2, W4, W6 implementation ? 2026-10-09

Implemented in isolated lane B based on be4168c. No tests, benchmarks, measurements or UI smoke runs, as requested.

## W2

`raster/src/lib.rs`: `Tile::solid` and `Surface::fill_rect` share one Arc across fully covered tiles; default fills remove tiles, boundaries retain pixel writes. Identical fills preserve tile pointers, including decoded uniform tiles. Existing Arc::make_mut separates writes. `Surface::is_uniform_rect` classifies rectangular selections. `engine/src/pixels.rs` sends opaque unlocked fills and rectangular/unselected clears through the shared tile path; translucent and irregular selections retain their existing blend behavior. New filled layers, backgrounds and Canvas Size already call fill_rect. Existing pack Collect deduplicates by Arc pointer and spill Cells already account for shared storage once. Optional GPU solid upload sharing is not implemented.

Switch: `PHOTOCRAFT_SOLID_TILES=0` restores existing fill/clear paths.

## W4

`raster/src/mips.rs`: native-format CPU pyramid, 2x2 alpha-premultiplied box reduction, on-demand tiles, checked source coordinates, max level16, outward negative bounds. Cache is shared until a surface mutation; only affected ancestors are invalidated. Weak source keys do not retain old base tiles. Base pixels and surface equality remain independent of derived data. An optional non-persisted content bounds hint lets viewport proxies preserve original effect/gradient anchors; mutations clear it. Missing sparse quadrants share one default tile per cache. Viewport surface requests reject more than 65,536 tiles before allocation; source coordinates and mip levels are bounded. No mip is generated merely for saving.

`format/src/{manifest,convert,pack}.rs`: optional mip references reuse the pack's existing LZ4+CRC machinery and lazy tile backing. Old manifests deserialize empty pyramids. Native surfaces additionally persist exact non-default bounds, computed on the save path and validated on load without touching lazy pixels. Display-only bounds hints are excluded. Loaded bounds must be canonical and fit/intersect the allocated tile footprint; mutations clear restored bounds. `PHOTOCRAFT_PACK_BOUNDS=0` disables bounds persistence/restoration. Pack's two-pass collection freezes each pyramid snapshot so concurrent generation cannot create unresolved references. Raster mutations invalidate loaded mip tiles too.

`raster/src/spill.rs`: reconstructible tiles get additional bounded clock chances; they spill to scratch rather than being discarded.

Switches: `PHOTOCRAFT_CPU_MIPS=0` disables mip caching/loading/persistence; `PHOTOCRAFT_PACK_MIPS=0` omits saved pyramids; `PHOTOCRAFT_MIP_PRIORITY=0` restores ordinary clock priority. CPU mip access remains callable uncached when CPU_MIPS is disabled; consumers gate mip use themselves.

CPU mip thumbnail consumption is in W3. GPU upload, navigator and viewport compositing consumption belongs to lane C and requires integration; it is not claimed complete here.

## W6

`raster/src/spill.rs`: a dedicated native worker prepares compressed scratch slots above 80% budget, retaining resident pixels. It collects cells under the registry lock then performs IO outside it. Pixel read locking prevents concurrent mutation while installing a slot; existing writes invalidate slots. An active-load counter gives scratch reloads priority between speculative compressions/writes. Existing pressure eviction remains a fallback. Wasm spawns no workers and retains synchronous eviction.

Switch: `PHOTOCRAFT_WRITE_BEHIND=0` disables the writer. Thread creation failure leaves ordinary eviction available.

## Validation

`cargo check -p photocraft-raster -p photocraft-format -p photocraft-ui-egui` passed on this lane. Full build, layers and wasm remain pending unified integration validation. Static review additionally checked ancestor invalidation, negative-coordinate rounding, duplicate persisted mip rejection and pack load accounting; this is source inspection, not runtime verification. No runtime performance or pixel correctness claims are made.
