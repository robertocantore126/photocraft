//! The models PhotoCraft knows how to use and the folder they live in.
//!
//! The folder is Fotox's own models folder by default, so models already installed for Fotox work
//! here without a second copy: `PHOTOCRAFT_MODELS` overrides it; otherwise the first of
//! `%LOCALAPPDATA%\PhotoCraft\models` and `%LOCALAPPDATA%\Fotox\models` (`$XDG_DATA_HOME/…` or
//! `~/.local/share/…` elsewhere) that holds the model.

use std::path::PathBuf;

/// One file of a model.
#[derive(Clone, Copy, Debug)]
pub struct ModelFile {
    pub file: &'static str,
    pub url: &'static str,
    pub bytes: u64,
}

/// A model PhotoCraft knows how to use.
#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub licence: &'static str,
    pub files: &'static [ModelFile],
}

impl ModelSpec {
    /// The folder holding this model: the first candidate where every file is complete.
    pub fn dir(&self) -> Option<PathBuf> {
        model_dirs().into_iter().map(|d| d.join(self.id)).find(|d| self.files.iter().all(|f| complete(&d.join(f.file), f)))
    }

    /// Every file is there at its expected size.
    pub fn installed(&self) -> bool {
        self.dir().is_some()
    }

    /// The path of `file` (where it is installed, else where it would go).
    pub fn path(&self, file: &str) -> PathBuf {
        self.dir().unwrap_or_else(|| models_dir().join(self.id)).join(file)
    }
}

fn complete(path: &std::path::Path, f: &ModelFile) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && (f.bytes == 0 || m.len() == f.bytes))
}

/// BiRefNet (MIT), the general "lite" (Swin-T) export: input `input_image` [1, 3, 1024, 1024]
/// ImageNet-normalised RGB; output [1, 1, 1024, 1024] logits.
pub const BIREFNET: ModelSpec = ModelSpec {
    id: "birefnet-lite",
    name: "BiRefNet (general, lite)",
    licence: "MIT",
    files: &[ModelFile {
        file: "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        bytes: 224_005_088,
    }],
};

/// EfficientSAM ViT-T (Apache-2.0). Encoder `batched_images` [1, 3, H, W] RGB 0..1 →
/// `image_embeddings` [1, 256, 64, 64]; decoder: embeddings, `batched_point_coords` [1, 1, N, 2]
/// (image pixels), `batched_point_labels` [1, 1, N] (1 in, 0 out, 2 / 3 box corners),
/// `orig_im_size` [h, w] i64 → `output_masks` [1, 1, 3, h, w] logits and `iou_predictions`
/// [1, 1, 3].
pub const EFFICIENT_SAM: ModelSpec = ModelSpec {
    id: "efficientsam-vitt",
    name: "EfficientSAM (ViT-T)",
    licence: "Apache-2.0",
    files: &[
        ModelFile {
            file: "efficient_sam_vitt_encoder.onnx",
            url: "https://raw.githubusercontent.com/yformer/EfficientSAM/main/weights/efficient_sam_vitt_encoder.onnx",
            bytes: 24_799_761,
        },
        ModelFile {
            file: "efficient_sam_vitt_decoder.onnx",
            url: "https://raw.githubusercontent.com/yformer/EfficientSAM/main/weights/efficient_sam_vitt_decoder.onnx",
            bytes: 16_565_728,
        },
    ],
};

pub const ALL: [ModelSpec; 2] = [BIREFNET, EFFICIENT_SAM];

fn data_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
    }
}

/// Folders searched for models, in order.
pub fn model_dirs() -> Vec<PathBuf> {
    if let Some(p) = std::env::var_os("PHOTOCRAFT_MODELS") {
        return vec![PathBuf::from(p)];
    }
    let Some(base) = data_dir() else { return Vec::new() };
    let (own, fotox) = if cfg!(windows) { ("PhotoCraft", "Fotox") } else { ("photocraft", "fotox") };
    vec![base.join(own).join("models"), base.join(fotox).join("models")]
}

/// Where models are installed: the first folder that exists, else PhotoCraft's own.
pub fn models_dir() -> PathBuf {
    let dirs = model_dirs();
    dirs.iter().find(|d| d.is_dir()).cloned().or_else(|| dirs.first().cloned()).unwrap_or_else(|| PathBuf::from("models"))
}
