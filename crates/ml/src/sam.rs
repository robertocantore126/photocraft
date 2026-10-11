//! EfficientSAM prompts: a box (and optional points) in working image pixels, the way the decoder
//! takes them.

use crate::{Input, Tensor32};

/// Decoder inputs for a box `(x0, y0, x1, y1)` plus positive / negative points, all in the
/// working image's pixels, and the working size.
pub fn prompt(boxed: Option<[f32; 4]>, points: &[((f32, f32), bool)], size: (usize, usize)) -> (Input, Input, Input) {
    let mut coords = Vec::new();
    let mut labels = Vec::new();
    if let Some([x0, y0, x1, y1]) = boxed {
        coords.extend([x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)]);
        labels.extend([2.0, 3.0]);
    }
    for ((x, y), positive) in points {
        coords.extend([*x, *y]);
        labels.push(if *positive { 1.0 } else { 0.0 });
    }
    let n = labels.len();
    (
        Input::F32(Tensor32::new(vec![1, 1, n, 2], coords)),
        Input::F32(Tensor32::new(vec![1, 1, n], labels)),
        Input::I64 { shape: vec![2], data: vec![size.1 as i64, size.0 as i64] },
    )
}

/// The best of the decoder's masks (by predicted IoU), as logits `h × w`.
pub fn best_mask(masks: &Tensor32, iou: &Tensor32) -> Option<Vec<f32>> {
    let (k, h, w) = match masks.shape.as_slice() {
        [_, _, k, h, w] => (*k, *h, *w),
        _ => return None,
    };
    let best = (0..k.min(iou.data.len())).max_by(|a, b| iou.data[*a].total_cmp(&iou.data[*b]))?;
    masks.data.get(best * h * w..(best + 1) * h * w).map(<[f32]>::to_vec)
}
