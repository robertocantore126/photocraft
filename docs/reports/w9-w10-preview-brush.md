# W9 persistent preview and W10 brush work — 2026-10-09

Based on be4168c, branch perf/w9-preview, isolated worktree E:\pc-ps-c.

W9 step 0: a dialog retains its immutable proxy across slider changes; document revision, scale and document identity invalidate it. Active-layer identity also invalidates the filtered result. PHOTOCRAFT_FILTER_PROXY=0 rebuilds the proxy each time.

W10: native large dabs rasterize independent rows using Rayon above 65,536 pixels. Coverage accumulates independent tiles in parallel, preserving the order of dabs. Stroke pixel compositing runs independent coverage tiles in batches of sixteen; insertion into the target Surface remains serial to preserve its ownership and COW semantics. Static integer-position tips cache one exact footprint (maximum 4,194,304 coverage samples); pressure size, shape dynamics, noise, per-tip texture and subpixel positioning bypass this cache. PHOTOCRAFT_PARALLEL_BRUSH=0 restores serial row/tile computation; PHOTOCRAFT_BRUSH_STAMP=0 disables the stamp cache. Wasm uses serial loops.

Files: crates/ui-egui/src/{canvas,filter_dialog}.rs; crates/paint/{Cargo.toml,src/render.rs}.

No tests, benchmarks, timing measurements or visual comparisons were added or run, as requested. Native cargo check -p photocraft-ui-egui passed, including photocraft-paint. W9 viewport/latest-wins and W5 work followed in W5-W9-viewport-2026-10-09.md; optional W11 remains deferred. Full build/wasm/layers are checked on the integration branch.

Final unified integration (2026-10-09): native workspace build passed; layering passed for26 crates; wasm checks passed for20/20 crates with warnings. Both new WGSL shaders passed Naga30 static validation. Tests, runtime/UI validation, pixel comparisons and performance measurements were not run. See [integration report](2026-10-09-ps-integration.md) for the final tree, switches and remaining limits; earlier lane-local pending notes above describe their original commit checkpoints.
