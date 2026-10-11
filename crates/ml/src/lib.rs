//! Local AI models, ported from Fotox's `fx-ai`.
//!
//! * [`runtime`]: ONNX Runtime loaded at run time (`load-dynamic`), DirectML first on Windows,
//!   the CPU otherwise; each model file is loaded once per process and kept.
//! * [`models`]: the models PhotoCraft knows (BiRefNet, EfficientSAM) and the folder they live
//!   in. Fotox's folder is shared, so models installed for Fotox work here without a download.
//! * [`image`]: the working-resolution buffers a model reads.
//! * [`sam`]: EfficientSAM's prompt geometry.
//!
//! Nothing here touches the document: the engine reads the pixels at a bounded working
//! resolution and turns the masks into selections. The web build has no runtime: every call
//! says so.

pub mod image;
pub mod models;
pub mod runtime;
pub mod sam;

#[derive(Debug, thiserror::Error)]
pub enum AiError {
    #[error("ONNX Runtime is not available: {0}")]
    Runtime(String),
    #[error("the model {0} is not installed (put it in {1})")]
    Missing(String, String),
    #[error("inference failed: {0}")]
    Inference(String),
}

#[cfg(not(target_arch = "wasm32"))]
impl<T> From<ort::Error<T>> for AiError {
    fn from(e: ort::Error<T>) -> Self {
        AiError::Inference(e.to_string())
    }
}

/// A tensor: shape and row-major values.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor32 {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Self { shape, data }
    }
}

/// One model input.
#[derive(Clone, Debug)]
pub enum Input {
    F32(Tensor32),
    I64 { shape: Vec<usize>, data: Vec<i64> },
}

impl From<Tensor32> for Input {
    fn from(t: Tensor32) -> Self {
        Input::F32(t)
    }
}

/// A mask at a model's working resolution: `width × height` values `0..=255`, row-major,
/// stretched over the canvas rectangle `rect` (`x0, y0, x1, y1`, exclusive).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelMask {
    pub rect: [i64; 4],
    pub width: usize,
    pub height: usize,
    pub values: Vec<u8>,
}

impl ModelMask {
    /// The mask at canvas `(x, y)` (pixel centre), bilinear, `0..=1`; 0 outside `rect`.
    pub fn at(&self, x: f64, y: f64) -> f32 {
        let [x0, y0, x1, y1] = self.rect;
        if x < x0 as f64 || y < y0 as f64 || x >= x1 as f64 || y >= y1 as f64 || self.width == 0 || self.height == 0 {
            return 0.0;
        }
        let (w, h) = (self.width, self.height);
        let u = (x - x0 as f64) * w as f64 / (x1 - x0).max(1) as f64 - 0.5;
        let v = (y - y0 as f64) * h as f64 / (y1 - y0).max(1) as f64 - 0.5;
        let fx = u.clamp(0.0, (w - 1) as f64);
        let fy = v.clamp(0.0, (h - 1) as f64);
        let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
        let (jx, jy) = ((ix + 1).min(w - 1), (iy + 1).min(h - 1));
        let (tx, ty) = ((fx - ix as f64) as f32, (fy - iy as f64) as f32);
        let g = |xx: usize, yy: usize| f32::from(self.values.get(yy * w + xx).copied().unwrap_or(0)) / 255.0;
        let a = g(ix, iy) + (g(jx, iy) - g(ix, iy)) * tx;
        let b = g(ix, jy) + (g(jx, jy) - g(ix, jy)) * tx;
        a + (b - a) * ty
    }

    /// The canvas box around the selected part (values ≥ 128); `None` when nothing is.
    pub fn bounds(&self) -> Option<[f64; 4]> {
        let (w, h) = (self.width, self.height);
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for y in 0..h {
            for x in 0..w {
                if self.values[y * w + x] >= 128 {
                    (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
                }
            }
        }
        if x0 >= x1 {
            return None;
        }
        let [rx0, ry0, rx1, ry1] = self.rect;
        let (sx, sy) = ((rx1 - rx0) as f64 / w as f64, (ry1 - ry0) as f64 / h as f64);
        Some([rx0 as f64 + x0 as f64 * sx, ry0 as f64 + y0 as f64 * sy, rx0 as f64 + x1 as f64 * sx, ry0 as f64 + y1 as f64 * sy])
    }
}
