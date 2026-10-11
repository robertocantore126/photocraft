//! Working-resolution buffers: planar model inputs from straight RGBA.

/// Straight RGBA `0..=1`, `w × h`, resampled bilinearly to `ow × oh` planar RGB, composited over
/// white (transparent areas read as white), then `(v − mean) / std` per channel.
pub fn planar_rgb(pixels: &[[f32; 4]], w: usize, h: usize, ow: usize, oh: usize, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * ow * oh];
    if w == 0 || h == 0 || pixels.len() < w * h {
        return out;
    }
    for y in 0..oh {
        let fy = ((y as f32 + 0.5) * h as f32 / oh as f32 - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..ow {
            let fx = ((x as f32 + 0.5) * w as f32 / ow as f32 - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(w - 1);
            for c in 0..3 {
                let at = |xx: usize, yy: usize| {
                    let p = pixels[yy * w + xx];
                    p[c] * p[3] + (1.0 - p[3])
                };
                let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
                let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
                let v = top + (bottom - top) * ty;
                out[c * ow * oh + y * ow + x] = (v - mean[c]) / std[c];
            }
        }
    }
    out
}

/// ImageNet normalisation (BiRefNet).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Logits → `0..=255`.
pub fn to_u8(logits: &[f32]) -> Vec<u8> {
    logits.iter().map(|v| (sigmoid(*v) * 255.0).round() as u8).collect()
}
