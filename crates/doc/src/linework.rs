//! SAI-style Linework: editable strokes kept on a pixel layer. The layer's raster content is the
//! strokes rendered (the engine re-renders it after every linework edit), so compositing, the GPU,
//! thumbnails and PSD export see an ordinary pixel layer.

use serde::{Deserialize, Serialize};

/// How a stroke passes through its control points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LineKind {
    /// A smooth curve through every point (Pen, Curve).
    #[default]
    Curve,
    /// Straight segments (Line).
    Polyline,
}

/// One editable stroke.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LineStroke {
    /// Control points: x, y (document px) and the stroke width there (px).
    pub points: Vec<[f32; 3]>,
    /// Straight RGBA.
    pub color: [f32; 4],
    #[serde(default)]
    pub kind: LineKind,
    #[serde(default)]
    pub closed: bool,
}

impl LineStroke {
    /// Bounding box (x0, y0, x1, y1) including the stroke width.
    pub fn bounds(&self) -> Option<[f32; 4]> {
        let mut it = self.points.iter();
        let f = it.next()?;
        let h = |p: &[f32; 3]| p[2] / 2.0 + 1.0;
        let mut b = [f[0] - h(f), f[1] - h(f), f[0] + h(f), f[1] + h(f)];
        for p in it {
            b = [b[0].min(p[0] - h(p)), b[1].min(p[1] - h(p)), b[2].max(p[0] + h(p)), b[3].max(p[1] + h(p))];
        }
        Some(b)
    }
}

/// The strokes of a Linework layer, bottom first.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Linework {
    pub strokes: Vec<LineStroke>,
}
