//! SAI Brush tool (PaintTool SAI's Brush / Watercolor / Marker): the options bar and the
//! `"sai"` part of its `paint.stroke` params. The painting model is `photocraft_paint::sai`.
//! Shape maps and textures come from the user's folder (`photocraft_engine::sai_textures`).

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{PhotocraftApp, widgets};

/// SAI Brush options (percentages 0..100, like SAI's sliders).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SaiOptions {
    /// "brush" | "watercolor" | "marker": the preset the sliders started from.
    pub kind: String,
    pub blending: f32,
    pub dilution: f32,
    pub persistence: f32,
    pub keep_opacity: bool,
    /// Shape map name ("" = none).
    pub shape: String,
    pub shape_strength: f32,
    /// Texture name ("" = none).
    pub texture: String,
    pub texture_strength: f32,
    pub texture_scale: f32,
}

impl Default for SaiOptions {
    fn default() -> Self {
        let mut o = Self {
            kind: String::new(),
            blending: 0.0,
            dilution: 0.0,
            persistence: 0.0,
            keep_opacity: false,
            shape: String::new(),
            shape_strength: 50.0,
            texture: String::new(),
            texture_strength: 50.0,
            texture_scale: 100.0,
        };
        o.apply_kind("brush");
        o
    }
}

impl SaiOptions {
    /// SAI's default slider positions for each tool.
    pub fn apply_kind(&mut self, kind: &str) {
        let (b, d, p) = match kind {
            "watercolor" => (50.0, 35.0, 80.0),
            "marker" => (10.0, 0.0, 30.0),
            _ => (50.0, 0.0, 80.0),
        };
        self.kind = kind.to_string();
        (self.blending, self.dilution, self.persistence) = (b, d, p);
    }

    /// The `"sai"` object of `paint.stroke`.
    pub fn params(&self) -> Value {
        json!({
            "blending": self.blending / 100.0,
            "dilution": self.dilution / 100.0,
            "persistence": self.persistence / 100.0,
            "keepOpacity": self.keep_opacity,
            "shape": self.shape,
            "shapeStrength": self.shape_strength / 100.0,
            "texture": self.texture,
            "textureStrength": self.texture_strength / 100.0,
            "textureScale": self.texture_scale / 100.0,
            "shapeScale": 1.0,
        })
    }
}

/// Selecting a Sugimori layer picks its tool once: the line → Linework, the watercolour → SAI
/// Brush as Watercolor.
pub fn follow_role(app: &mut PhotocraftApp) {
    let Some((id, role)) = app.session.active().and_then(|st| {
        let id = st.active_layer?;
        Some((id.0, st.doc.layer(id)?.role.clone()))
    }) else {
        return;
    };
    if app.sai_role_layer == Some(id) {
        return;
    }
    app.sai_role_layer = Some(id);
    match role {
        Some(photocraft_doc::LayerRole::SugiLine) => app.ui.tool = crate::state::Tool::Linework,
        Some(photocraft_doc::LayerRole::SugiColour) => {
            app.ui.tool = crate::state::Tool::SaiBrush;
            if app.ui.tool_options.sai.kind != "watercolor" {
                app.ui.tool_options.sai.apply_kind("watercolor");
            }
            ensure_textures(app, false);
        }
        _ => {}
    }
}

/// Loads the user's maps the first time the tool is used (and on Reload).
pub fn ensure_textures(app: &mut PhotocraftApp, force: bool) {
    if !force && (app.sai_textures_scanned || !app.session.tools.sai_textures.is_empty()) {
        return;
    }
    app.sai_textures_scanned = true;
    if let Some(dir) = photocraft_engine::sai_textures::default_dir() {
        app.session.tools.sai_textures = photocraft_engine::sai_textures::load_dir(&dir);
    }
}

fn map_dropdown(ui: &mut egui::Ui, id: &str, names: &[String], v: &mut String) {
    let mut opts: Vec<(String, &str)> = vec![(String::new(), "None")];
    opts.extend(names.iter().map(|n| (n.clone(), n.as_str())));
    widgets::dropdown(ui, id, v, &opts, 110.0);
}

/// The options bar after Size / Density: kind, Blending, Dilution, Persistence, Keep Opacity,
/// shape and texture.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    ensure_textures(app, false);
    if widgets::secondary_button(ui, "SAI Kit", 0.0).clicked() {
        app.ui.panels.sai_kit = true;
    }
}

/// Floating kit; all SAI controls fit at narrow window sizes and remain scrollable.
pub fn window(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if !app.ui.panels.sai_kit {
        return;
    }
    ensure_textures(app, false);
    let mut open = true;
    let t = crate::theme::Tokens::get(ctx);
    egui::Window::new("SAI Kit")
        .id(egui::Id::new("sai-kit"))
        .open(&mut open)
        .default_width(310.0)
        .default_height(680.0)
        .default_pos(egui::pos2(65.0, 105.0))
        .frame(
            egui::Frame::NONE
                .fill(t.card)
                .stroke(egui::Stroke::new(1.0, t.card_border))
                .corner_radius(egui::CornerRadius::same(t.radius_lg as u8))
                .inner_margin(egui::Margin::same(10)),
        )
        .vscroll(true)
        .show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (tool, name) in
                    [(crate::state::Tool::SaiBrush, "SAI Brush"), (crate::state::Tool::PixelBrush, "Pixel Brush"), (crate::state::Tool::Linework, "Linework")]
                {
                    if ui.selectable_label(app.ui.tool == tool, name).clicked() {
                        app.ui.tool = tool;
                        crate::paint_mouse::sync_tool_smoothing(app);
                    }
                }
            });
            if matches!(app.ui.tool, crate::state::Tool::SaiBrush | crate::state::Tool::PixelBrush) {
                let before = app.session.tools.brush.clone();
                let mut brush = before.clone();
                widgets::slider_row(ui, "Size", &mut brush.size, 1.0..=2500.0, " px", None);
                let mut density = brush.opacity * 100.0;
                widgets::slider_row(ui, "Density", &mut density, 0.0..=100.0, "%", None);
                brush.opacity = density / 100.0;
                widgets::checkbox(ui, &mut brush.pressure_size, "Pen pressure controls size");
                crate::brush_panel::commit_gesture(app, ctx, &before, &brush);
                if app.ui.tool == crate::state::Tool::PixelBrush {
                    widgets::checkbox(ui, &mut app.ui.tool_options.pixel_perfect, "Pixel Perfect (1 px)");
                }
            }
            if app.ui.tool == crate::state::Tool::SaiBrush {
                egui::CollapsingHeader::new("Brush / Watercolor / Marker").default_open(true).show(ui, |ui| {
                    let o = &mut app.ui.tool_options.sai;
                    ui.horizontal_wrapped(|ui| {
                        for (id, label) in [("brush", "Brush"), ("watercolor", "Watercolor"), ("marker", "Marker")] {
                            if ui.selectable_label(o.kind == id, label).clicked() {
                                o.apply_kind(id);
                                app.ui.tool = crate::state::Tool::SaiBrush;
                            }
                        }
                    });
                    for (label, value) in [("Blending", &mut o.blending), ("Dilution", &mut o.dilution), ("Persistence", &mut o.persistence)] {
                        widgets::slider_row(ui, label, value, 0.0..=100.0, "%", None);
                    }
                    widgets::checkbox(ui, &mut o.keep_opacity, "Keep Opacity");
                });
                egui::CollapsingHeader::new("Shape and texture").default_open(true).show(ui, |ui| {
                    let names: Vec<_> = app.session.tools.sai_textures.keys().cloned().collect();
                    ui.label(format!("{} maps loaded", names.len()));
                    if let Some(dir) = photocraft_engine::sai_textures::default_dir() {
                        ui.label(egui::RichText::new(dir.display().to_string()).small().color(t.text_dim));
                        if !dir.is_dir() {
                            ui.colored_label(t.text_dim, "Texture folder does not exist.");
                        } else if names.is_empty() {
                            ui.label("No readable BMP, PNG or JPEG maps found.");
                        }
                    } else {
                        ui.label("No texture folder configured on this platform.");
                    }
                    let o = &mut app.ui.tool_options.sai;
                    ui.horizontal(|ui| {
                        ui.label("Shape");
                        map_dropdown(ui, "kit-shape", &names, &mut o.shape);
                    });
                    widgets::slider_row(ui, "Shape strength", &mut o.shape_strength, 0.0..=100.0, "%", None);
                    ui.horizontal(|ui| {
                        ui.label("Texture");
                        map_dropdown(ui, "kit-texture", &names, &mut o.texture);
                    });
                    widgets::slider_row(ui, "Texture strength", &mut o.texture_strength, 0.0..=100.0, "%", None);
                    widgets::slider_row(ui, "Texture scale", &mut o.texture_scale, 10.0..=1000.0, "%", None);
                    if let Some(map) = app.session.tools.sai_textures.get(&o.texture) {
                        map_preview(ui, map);
                    } else if !o.texture.is_empty() {
                        ui.label(format!("Missing texture: {}", o.texture));
                    }
                    if widgets::secondary_button(ui, "Reload textures", 0.0).clicked() {
                        ensure_textures(app, true);
                    }
                });
            }
            if app.ui.tool == crate::state::Tool::Linework {
                egui::CollapsingHeader::new("Linework").default_open(true).show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        for (id, label) in crate::linework_ui::MODES {
                            if ui.selectable_label(app.ui.tool == crate::state::Tool::Linework && app.ui.tool_options.linework_mode == id, label).clicked() {
                                crate::linework_ui::commit_clicks(app);
                                app.ui.tool_options.linework_mode = id.into();
                                app.ui.tool = crate::state::Tool::Linework;
                            }
                        }
                    });
                    let o = &mut app.ui.tool_options;
                    widgets::slider_row(ui, "Width", &mut o.linework_width, 0.5..=200.0, " px", None);
                    widgets::checkbox(ui, &mut o.linework_pressure, "Pen pressure controls width");
                    widgets::slider_row(ui, "Simplify", &mut o.linework_simplify, 0.0..=10.0, " px", None);
                    ui.label("Curve / Line: click points; double-click or Enter to finish; Esc to cancel.");
                    if ui.button("Finish curve").clicked() {
                        crate::linework_ui::commit_clicks(app);
                    }
                    if ui.button("New Linework Layer").clicked() {
                        let _ = app.run("layer.new.linework", json!({}));
                    }
                });
            }
            ui.collapsing("Reference fill and Sugimori", |ui| {
                let opts = [("layer".to_string(), "Current Layer"), ("reference".to_string(), "Reference Layers"), ("all".to_string(), "All Layers")];
                widgets::dropdown(ui, "kit-fill-sample", &mut app.ui.tool_options.fill_sample, &opts, 160.0);
                widgets::slider_row(ui, "Expand", &mut app.ui.tool_options.bucket_expand, 0.0..=20.0, " px", None);
                for (id, label) in [
                    ("layer.reference", "Toggle Reference Layer"),
                    ("layer.new.sugimoriSet", "New Sugimori Watercolour Set"),
                    ("sugimori.generateBase", "Generate Base from Line"),
                ] {
                    if ui.button(label).clicked() {
                        let _ = crate::menus::invoke(app, ctx, id, json!({}));
                    }
                }
            });
        });
    app.ui.panels.sai_kit = open;
}

fn map_preview(ui: &mut egui::Ui, map: &std::sync::Arc<photocraft_engine::paint::GrayTile>) {
    let id = egui::Id::new("sai-kit-texture-preview");
    let cached = ui.ctx().data(|d| d.get_temp::<(std::sync::Arc<photocraft_engine::paint::GrayTile>, egui::TextureHandle)>(id));
    let handle = match cached {
        Some((previous, handle)) if std::sync::Arc::ptr_eq(&previous, map) => handle,
        _ => {
            let mut pixels = Vec::with_capacity(64 * 64);
            for y in 0..64u32 {
                for x in 0..64u32 {
                    let sx = x.saturating_mul(map.width) / 64;
                    let sy = y.saturating_mul(map.height) / 64;
                    let index = (sy as usize).saturating_mul(map.width as usize).saturating_add(sx as usize);
                    let gray = map.data.get(index).copied().unwrap_or(0) / 257;
                    pixels.push(egui::Color32::from_gray(gray as u8));
                }
            }
            let handle = ui.ctx().load_texture("SAI texture preview", egui::ColorImage::new([64, 64], pixels), egui::TextureOptions::LINEAR);
            ui.ctx().data_mut(|d| d.insert_temp(id, (map.clone(), handle.clone())));
            handle
        }
    };
    ui.image((handle.id(), egui::vec2(64.0, 64.0)));
}
