# W9 step 3: native GPU filters (2026-10-09)

Added `photocraft_gpu::filters::FilterEngine` and `Filter` with Gaussian Blur, Box Blur, Motion Blur, Unsharp Mask and High Pass. Compute pipelines are cheap to clone and intended to be cached once per device by the UI, then used on its filter preview worker. Results return a copy-on-write Surface with only the requested output rectangle replaced. Original format and bit depth are retained. No document-space colour conversion is involved.

Gaussian Blur uses the existing CPU algorithm's exact separable small-radius kernel and three running-sum box passes per axis above radius 4. Unsharp Mask and High Pass use their existing CPU algorithm's exact Gaussian definition at every radius, with original alpha and threshold behavior preserved. Motion Blur uses premultiplied bilinear samples in document coordinates. Selection coverage is applied to the native samples before encoding back to the original depth.

The implementation accepts up to four native channels, including RGB/Gray/Lab with alpha and CMYK without alpha. More than four channels, oversized windows, excessive dispatch work, failed pipelines, GPU errors and wasm readback use the caller's CPU fallback. Each input window is capped at 32 MiB of packed samples (three GPU working buffers plus staging and bounded CPU expansion). Long exact kernels and long motion sampling use a conservative 256 million sample work cap to avoid a single unbounded compute submission. Native waits have a 30 second timeout; the GPU dispatch itself is not preemptible by Esc, and the preview worker must discard cancelled results.

Set `PHOTOCRAFT_GPU_FILTERS=0` to select CPU filtering. The caller must route `Unsupported` back to `algo::apply_in`; it must never replace a failed filter with an unfiltered result.

Changed files: `crates/gpu/src/filters.rs`, `crates/gpu/src/filters.wgsl`, module registration in `crates/gpu/src/lib.rs`. Preview routing is implemented by lane C.

Validation: compilation pending. Tests, shader execution tests, benchmarks, measurements and visual comparisons deliberately skipped at the user's explicit request. The requested <=1/255 GPU/CPU agreement has not been measured or established.

Final unified integration (2026-10-09): native workspace build passed; layering passed for26 crates; wasm checks passed for20/20 crates with warnings. Both new WGSL shaders passed Naga30 static validation. Tests, runtime/UI validation, pixel comparisons and performance measurements were not run. See [integration report](2026-10-09-ps-integration.md) for the final tree, switches and remaining limits; earlier lane-local pending notes above describe their original commit checkpoints.
