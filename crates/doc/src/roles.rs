//! Special layer roles: layers that belong to a painting recipe (the Ken Sugimori watercolour set)
//! and that the app treats specially (their own icon, regenerated content). A role never changes
//! how the layer composites; removing it leaves an ordinary layer.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum LayerRole {
    /// The sketch under everything.
    SugiSketch,
    /// The line art (a Linework layer, marked as reference).
    SugiLine,
    /// The white base inside the line art (generated from the line: `sugimori.generateBase`).
    SugiBase,
    /// The watercolour layer, clipped to the base.
    SugiColour,
    /// The paper texture over the colour (a pattern fill, Multiply, clipped).
    SugiPaper,
    /// The line art blurred (regenerated whenever the line changes), clipped over the colour.
    SugiSoftLine {
        /// Gaussian blur radius in pixels.
        radius: f32,
    },
}

impl LayerRole {
    /// Short name shown in the Layers panel.
    pub fn label(&self) -> &'static str {
        match self {
            LayerRole::SugiSketch => "Sugimori sketch",
            LayerRole::SugiLine => "Sugimori line",
            LayerRole::SugiBase => "Sugimori base",
            LayerRole::SugiColour => "Sugimori colour",
            LayerRole::SugiPaper => "Sugimori paper",
            LayerRole::SugiSoftLine { .. } => "Sugimori soft line",
        }
    }
}
