# SAI Kit follow-up - 2026-10-11

Working tree: `E:\pc-fix`, uncommitted. Separate native preview: `E:\pc-sai-preview\photocraft-sai.exe` (normal release profile). Existing UX/SAI work preserved.

## Confirmed findings

- `%APPDATA%\Photocraft\sai-textures` was absent at the start of this follow-up. The original report's statement that four maps were copied there is not confirmed by the current filesystem.
- `retouch_ui::options_bar` returned `true` for the new SAI Brush and Pixel Brush, preventing their own options from rendering. Confirmed in source and by an offscreen screenshot (empty options bar for SAI Brush).
- Pending Curve/Line gestures were shown as a thin straight guide, rather than the curve and brush width that would be committed.
- Curve/Line had Enter/Finish completion but no double-click completion.

## Changes

- Window > SAI Kit: a movable, resizable, scrollable panel with SAI Brush, Pixel Brush and Linework tool buttons; controls follow the selected tool.
- Brush/Watercolor/Marker settings; Shape/Texture selectors, strengths, scale, selected-texture thumbnail, map count, resolved folder and missing-folder/empty-folder feedback; Reload.
- Linework sub-tools, width, pressure, simplify, Finish, and New Linework Layer; Reference fill and Sugimori actions.
- Compact options bars link to SAI Kit. Retouch no longer intercepts SAI/Pixel Brush options.
- Double-click finishes Curve/Line without retaining the second press as a duplicate control point. Enter/Finish and Escape remain available.
- Curve/Line guide uses the engine's Catmull-Rom/polyline sampler and foreground colour/width. This is an egui overlay, not a pixel-identical raster preview.

## Validation

- Separate native release build passed; original executable was locked by the user's running app and was not replaced.
- Native smoke: panel and restored SAI options bar visible; synthetic texture loaded. The user started drawing in this preview session; it was left open without further automation.
- Offscreen visual QA: panel in all five themes at 900x700; brush/texture controls at 1280x900; all four original maps selected, previewed and painted successfully. Images: `log/sai-qa/`.
- UI library tests: 361 passed, 0 failed, 1 ignored. Clippy (all targets), layering (28 crates), wasm and parity passed. The first UI suite exposed an outdated B-key assumption after adding multiple brush tools; the test now selects the group from Move.
- Offscreen double-click smoke passed: last command `linework.addStroke`, committed curve visible, draft handles cleared (`log/sai-qa/curve-double-click-spaced.png`). Helper regression also verifies removal of the duplicated endpoint and undo.
- The snapshot example accepts `--step-dt` so scripted double-clicks can run within egui's click-time threshold (the harness default is 250 ms per simulated frame; 100 ms was used to keep distinct control-point clicks separate while making the final pair a double-click).
- No before/after performance benchmark or SAI pixel-parity claim.

## Still open

The four supplied BMPs are installed under `%APPDATA%\Photocraft\sai-textures`, byte-identical to Downloads: Acrylic and Bark2 256x256, Acuarelle and Crust 512x512 (8-bit uncompressed palette BMPs). All four load, have visible thumbnails, and produce visibly different textured strokes in offscreen UI runs at 100 px / Density 25% / texture strength approximately 100%. Hands-on equivalence with SAI remains open. The panel does not change the underlying watercolor model or make Linework feature-equivalent to SAI. Original report's rendering/filter/base-generation/preset-import limits still apply.

The still-open native QA session uses `PHOTOCRAFT_SAI_TEXTURES=.../log/sai-qa/maps` from the synthetic test. Save any drawing before closing it. A normal launch of the preview executable uses the installed four maps in the user folder.
