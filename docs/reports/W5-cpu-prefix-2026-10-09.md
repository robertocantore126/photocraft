# W5 step 4: CPU viewport prefix cache (2026-10-09)

Added app-owned `photocraft_compose::prefix::PrefixCache` for exact CPU viewport compositing. At a top-level unclipped active layer, the compositor caches the complete stack below it in globally aligned 256x256 float tiles. A brush change at or above that boundary reuses the lower composite. The suffix still runs through the ordinary CPU stack implementation, including blend modes, adjustment quantization, clipping, masks and effects.

Cache context includes document identity, size, depth, mode, profile, global light, patterns and lower layer snapshots. Any changed lower layer or context conservatively invalidates the cache. Old snapshots are released when invalidated/cleared. The UI owns and prunes caches when documents close. Pixel cache retention is capped at 64 MiB per cache, and an output rectangle at 16 MP. Raster source tiles are shared through existing COW storage.

Unsupported boundaries (an active clipped layer, an active layer nested inside a group, no lower prefix, excessive nesting or Multichannel) use the ordinary CPU renderer. No independent precomposition of arbitrary blend modes is attempted. This implements a lower-prefix cache; W11 is separate and optional.

Switch: `PHOTOCRAFT_CPU_PREFIX_CACHE=0` restores ordinary CPU rendering. Files: new `crates/compose/src/prefix.rs`, module registration in `crates/compose/src/lib.rs`, UI routing/pruning by lane C.

Validation: compilation pending. Tests, pixel comparisons, benchmarks and measurements deliberately skipped at the user's explicit request. No speedup or runtime parity is claimed.
