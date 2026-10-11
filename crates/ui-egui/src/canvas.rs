//! Document canvas: display texture cache, view transform, tool input, extra document windows.
//!
//! Rendering is CPU (`photocraft-compose`) for now, uploaded into an egui texture. Brush strokes
//! update only their damage rectangle (`set_partial`). The wgpu compositor (M5) will replace this
//! with direct GPU rendering behind the same `CanvasCache` interface.

use egui::{Color32, Pos2, Rect, Sense, Stroke, TextureOptions, Vec2, pos2, vec2};
use photocraft_doc::{Document, LayerContent};
use photocraft_geom::Rect as DRect;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::{Tool, View};

/// Largest texture side we upload; bigger documents display downsampled until the GPU path lands.
pub const MAX_TEXTURE: u32 = 4096;

pub struct CanvasCache {
    pub revision: u64,
    pub texture: Option<egui::TextureHandle>,
    /// Display pixels per document pixel of the cached image (≤ 1).
    pub scale: f32,
    /// Hash of the live-adjust preview used for this render (0 = none).
    pub preview_key: u64,
    /// The current image lives in the GPU canvas (`gpu_canvas`) rather than `texture`.
    pub on_gpu: bool,
    /// Revision and preview key `texture` was rendered at. Tracked apart from the GPU canvas's
    /// `revision`, so the Navigator never shows a stale CPU image after GPU-path edits.
    pub tex_revision: u64,
    pub tex_preview_key: u64,
    /// Revision of the document a live brush stroke was drawn over at the last refresh.
    pub live: Option<u64>,
}

/// An in-progress pointer gesture on the canvas.
#[derive(Clone, Debug)]
pub struct Drag {
    pub tool: Tool,
    pub start: [f64; 2],
    pub points: Vec<[f64; 3]>,
    pub modifiers: egui::Modifiers,
    /// A Brush/Eraser stroke that erases: the Eraser, or a right-button Brush drag with
    /// Preferences › Tools › Right-click with painting tools set to Erase.
    pub erase: bool,
    /// ⇧ constraint of a painting stroke (stroke_constraint.rs).
    pub constrain: Option<crate::stroke_constraint::Axis>,
    /// Lasso with ⌥ held after the press: the point the straight segment starts from
    /// (Photoshop's freehand ↔ polygon switch).
    pub straight_from: Option<usize>,
}

/// A Brush/Eraser stroke shown while it is drawn: the engine renders the real dabs onto a copy of
/// the document, and the canvas redraws only what each step changed.
pub(crate) struct LiveStroke {
    stroke: photocraft_engine::brush_cmds::LiveStroke,
    doc: photocraft_doc::DocId,
    revision: u64,
    /// Preview key of the stroke; step `n` displays as `key + n`.
    key: u64,
    /// Damage of each step (step 0 = the document before the stroke).
    damage: Vec<DRect>,
    /// Drag points rendered so far.
    fed: usize,
}

impl LiveStroke {
    fn display_key(&self) -> u64 {
        self.key + self.damage.len() as u64
    }

    /// What changed since the canvas showed preview `key` (0 = the document before the stroke).
    fn since(&self, key: u64) -> Option<DRect> {
        let seen = if key == 0 { 0 } else { usize::try_from(key.checked_sub(self.key)?).ok()? };
        Some(self.damage.get(seen..)?.iter().fold(DRect::EMPTY, |a, r| a.union(r)))
    }
}

/// The live stroke on document `idx`, while it is current.
fn live_stroke(app: &PhotocraftApp, idx: usize) -> Option<&LiveStroke> {
    let st = app.session.documents().get(idx)?;
    app.live_stroke.as_ref().filter(|l| app.drag.is_some() && l.doc == st.doc.id && l.revision == st.revision)
}

/// `paint.stroke` params for a Brush/Eraser drag (shared by the live preview and the commit). The
/// stroke smoothing is the session brush's (the options bar's Smoothing %).
fn stroke_params(app: &PhotocraftApp, erase: bool, points: &[Vec<f64>]) -> serde_json::Value {
    let mut p = json!({ "points": points, "erase": erase, "zoom": app.current_zoom(), "target": paint_target(app) });
    if app.ui.tool == Tool::PixelBrush {
        p["pixel"] = json!(true);
        p["pixelPerfect"] = json!(app.ui.tool_options.pixel_perfect);
    }
    if app.ui.tool == Tool::SaiBrush {
        p["sai"] = app.ui.tool_options.sai.params();
    }
    // A brush plug-in's tool paints through it (`brush_plugin_ui`).
    if app.ui.tool == Tool::Plugin
        && let (Some(o), Some(extra)) = (p.as_object_mut(), crate::brush_plugin_ui::stroke_params(&app.ui))
        && let Some(e) = extra.as_object()
    {
        o.extend(e.clone());
    }
    p
}

fn begin_live_stroke(app: &PhotocraftApp) -> Option<LiveStroke> {
    static STROKES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let st = app.session.active()?;
    let d = app.drag.as_ref()?;
    let stroke = photocraft_engine::brush_cmds::LiveStroke::begin(&app.session, &stroke_params(app, d.erase, &app.stylus.stroke_points(&d.points))).ok()?;
    let n = STROKES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) & 0xff_ffff;
    let damage = vec![stroke.bounds()];
    Some(LiveStroke { stroke, doc: st.doc.id, revision: st.revision, key: (1 << 44) | (n << 20), damage, fed: d.points.len() })
}

/// Render the drag points the live stroke hasn't seen yet, with the pen pressure, tilt and
/// rotation the commit's `paint.stroke` gets for them.
fn feed_live_stroke(app: &mut PhotocraftApp) {
    let (Some(l), Some(d)) = (app.live_stroke.as_mut(), app.drag.as_ref()) else { return };
    let pose = &app.stylus.stroke;
    let pts: Vec<_> = (l.fed..d.points.len())
        .filter_map(|i| {
            let p = d.points.get(i)?;
            let t = pose.get(i).or(pose.last()).copied().unwrap_or_default();
            let mut sp = photocraft_engine::paint::StrokePoint::new(p[0], p[1], p[2] as f32);
            (sp.tilt_x, sp.tilt_y, sp.rotation) = (t[0], t[1], t[2]);
            Some(sp)
        })
        .collect();
    if pts.is_empty() {
        return;
    }
    l.fed = d.points.len();
    match l.stroke.push(&pts) {
        Ok(r) => l.damage.push(r),
        Err(_) => app.live_stroke = None,
    }
}

/// Abstract tool event, produced by the mouse or by automation (`ui.pointer`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ToolEvent {
    Down { x: f64, y: f64, pressure: f32 },
    Move { x: f64, y: f64, pressure: f32 },
    Up { x: f64, y: f64 },
}

/// Document ↔ screen mapping for a canvas rect and a view.
#[derive(Clone, Copy, Debug)]
pub struct ViewXform {
    pub rect: Rect,
    pub zoom: f32,
    pub center: [f32; 2],
    /// View › Flip Horizontal: the view is mirrored left-right (the document is not).
    pub flip: bool,
}

impl ViewXform {
    pub fn to_screen(&self, x: f32, y: f32) -> Pos2 {
        let sx = if self.flip { -1.0 } else { 1.0 };
        self.rect.center() + vec2((x - self.center[0]) * self.zoom * sx, (y - self.center[1]) * self.zoom)
    }
    pub fn to_doc(&self, p: Pos2) -> [f64; 2] {
        let d = (p - self.rect.center()) / self.zoom;
        let dx = if self.flip { -d.x } else { d.x };
        [(dx + self.center[0]) as f64, (d.y + self.center[1]) as f64]
    }
    pub fn doc_rect(&self, r: DRect) -> Rect {
        Rect::from_two_pos(self.to_screen(r.x0 as f32, r.y0 as f32), self.to_screen(r.x1 as f32, r.y1 as f32))
    }
}

pub fn fit_view(view: &mut View, doc: &Document, area: Vec2) {
    let (w, h) = (doc.size.width as f32, doc.size.height as f32);
    let zoom = ((area.x - 40.0) / w).min((area.y - 40.0) / h).clamp(0.01, 1.0);
    view.zoom = zoom;
    view.center = [w / 2.0, h / 2.0];
    view.fit_pending = false;
}

/// Zoom steps like Photoshop's (⌘+ / ⌘−).
pub fn zoom_step(z: f32, dir: i32) -> f32 {
    const STEPS: [f32; 22] =
        [0.01, 0.02, 0.03, 0.05, 0.0667, 0.1, 0.125, 0.1667, 0.25, 0.333, 0.5, 0.6667, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 32.0];
    if dir > 0 { STEPS.iter().copied().find(|s| *s > z * 1.001).unwrap_or(32.0) } else { STEPS.iter().rev().copied().find(|s| *s < z * 0.999).unwrap_or(0.01) }
}

fn checker(app: &mut PhotocraftApp, ctx: &egui::Context) -> egui::TextureId {
    // Preferences › Transparency & Gamut colours (the texture is rebuilt when they change).
    let [ca, cb] = app.session.prefs().transparency_and_gamut.colors();
    if app.prefs_rt.checker_key != Some([ca, cb]) {
        app.prefs_rt.checker_key = Some([ca, cb]);
        app.checker = None;
    }
    app.checker
        .get_or_insert_with(|| {
            let (a, b) = (Color32::from_rgb(ca[0], ca[1], ca[2]), Color32::from_rgb(cb[0], cb[1], cb[2]));
            let img = egui::ColorImage::new([2, 2], vec![a, b, b, a]);
            let opts = TextureOptions {
                magnification: egui::TextureFilter::Nearest,
                minification: egui::TextureFilter::Nearest,
                wrap_mode: egui::TextureWrapMode::Repeat,
                mipmap_mode: None,
            };
            ctx.load_texture("checker", img, opts)
        })
        .id()
}

fn buffer_to_image(buf: &photocraft_compose::Buffer) -> egui::ColorImage {
    let img = buf.to_rgba8();
    egui::ColorImage::from_rgba_unmultiplied([img.width as usize, img.height as usize], &img.pixels)
}

/// A composite as monitor pixels: document profile → monitor profile (nothing to do in the
/// common sRGB-on-sRGB case).
fn display_image(display: Option<&photocraft_engine::display_color::CanvasDisplay>, buf: &photocraft_compose::Buffer) -> egui::ColorImage {
    match display {
        Some(d) if !d.is_identity() => {
            let img = d.to_rgba8(buf);
            egui::ColorImage::from_rgba_unmultiplied([img.width as usize, img.height as usize], &img.pixels)
        }
        _ => buffer_to_image(buf),
    }
}

/// The canvas display of `doc` and a key that changes with it (folded into the canvas caches'
/// preview keys, so a monitor or profile change re-renders).
fn canvas_display(app: &PhotocraftApp, doc: &Document) -> (Option<std::sync::Arc<photocraft_engine::display_color::CanvasDisplay>>, u64) {
    match app.session.color.canvas_display(doc) {
        Ok(d) => {
            let k = d.key;
            (Some(d), k)
        }
        Err(_) => (None, 0),
    }
}

/// The document to render: the committed one, or a clone with the live adjustment preview applied.
fn display_doc(app: &mut PhotocraftApp, idx: usize) -> (std::sync::Arc<Document>, u64) {
    // Puppet / Perspective Warp previews hide the layer they draw on a mesh.
    if let Some(shown) = crate::distort_ui::display_doc(app, idx) {
        return shown;
    }
    // Image › Adjustments dialog: a temporary adjustment layer clipped to the target.
    if let Some(shown) = crate::adjust_preview::display_doc(app, idx) {
        return shown;
    }
    let st = &app.session.documents()[idx];
    if let Some(l) = live_stroke(app, idx) {
        return (l.stroke.doc.clone(), l.display_key());
    }
    if let (Some(t), Some(_)) = (&app.ui.transform, &app.transform_preview)
        && app.session.active_index() == Some(idx)
        && st.doc.layer(photocraft_doc::LayerId(t.layer)).is_some()
    {
        if let Some(shown) = crate::transform_tool::in_place_doc(app, idx) {
            return shown;
        }
        let pv = app.transform_preview.as_ref().map(|pv| (pv.doc.clone(), (1 << 40) + pv.session));
        if let Some(shown) = pv {
            return shown;
        }
    }
    let st = &app.session.documents()[idx];
    // Layer Style dialog: show its effects live (Cancel just drops the preview).
    let style = app.ui.dialogs.iter().find(|d| d.kind == crate::state::DialogKind::LayerStyle);
    if style.is_none() {
        app.style_preview = None;
    }
    if let Some(d) = style
        && app.session.active_index() == Some(idx)
    {
        let key = (crate::layer_style::preview_hash(&d.fields) ^ st.revision.wrapping_mul(0x9e37_79b9_7f4a_7c15)) | 1 << 63;
        if app.style_preview.as_ref().map(|p| p.0) != Some(key) {
            let shown = crate::layer_style::preview_document(&st.doc, &app.session.patterns, &d.fields).map(std::sync::Arc::new);
            app.style_preview = Some((key, shown));
        }
        if let Some((_, Some(doc))) = &app.style_preview {
            return (doc.clone(), key);
        }
    }
    if let Some((layer, params)) = &app.live_adjust
        && let Some(l) = st.doc.layer(*layer)
        && let LayerContent::Adjustment(a) = &l.content
    {
        let kind = photocraft_engine::commands::adjustment_kind(a);
        let preview = photocraft_engine::adjust_params::from_params(kind, params, Some(a), st.doc.mode).unwrap_or_else(|_| a.clone());
        let mut doc = (*st.doc).clone();
        if let Some(lm) = doc.layer_mut(*layer) {
            lm.content = LayerContent::Adjustment(preview);
        }
        let key = 1 + params.to_string().bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
        return (std::sync::Arc::new(doc), key);
    }
    if let Some(preview) = &app.filter_preview
        && preview.doc == st.doc.id
        && preview.revision == st.revision
        && preview.active == st.active_layer
        && let Some(region) = &preview.region
        && let Some(result) = &region.result
        && app.ui.dialogs.iter().any(|d| d.fields.get("__preview").and_then(serde_json::Value::as_bool) == Some(true))
    {
        return (result.clone(), region.shown_hash | (1 << 59));
    }
    // Duotone documents display through their inks.
    if let Some(shown) = photocraft_engine::mode_cmds::display_document(&st.doc) {
        return (std::sync::Arc::new(shown), 1 << 41);
    }
    (st.doc.clone(), 0)
}

/// Longest side of the Navigator panel's image (about twice the panel's width, for HiDPI).
pub const NAVIGATOR_SIDE: u32 = 512;
/// How long adjustment-dialog settings must stay unchanged before the navigator shows them.
const NAVIGATOR_SETTLE_MS: f64 = 200.0;

/// The Navigator panel's image of document `idx`. With the CPU canvas that is the canvas texture
/// itself; with the GPU canvas a thumbnail cached per revision, so an edit to a huge document
/// doesn't also pay a full-resolution CPU composite for the navigator.
pub fn navigator_texture(app: &mut PhotocraftApp, ctx: &egui::Context, idx: usize) -> Option<egui::TextureId> {
    if app.gpu.is_none() && std::env::var("PHOTOCRAFT_VIEW_COMPOSE").as_deref() == Ok("0") {
        return ensure_texture(app, ctx, idx).map(|(t, _)| t);
    }
    let (revision, id) = app.session.documents().get(idx).map(|st| (st.revision, st.doc.id))?;
    let (doc, preview_key) = display_doc(app, idx);
    let (display, display_key) = canvas_display(app, &doc);
    let preview_key = preview_key ^ display_key;
    let key = egui::Id::new(("navigator", id.0));
    let cached: Option<(u64, u64, egui::TextureHandle)> = ctx.data(|d| d.get_temp(key));
    if let Some((r, p, t)) = &cached
        && (*r, *p) == (revision, preview_key)
    {
        return Some(t.id());
    }
    // While an adjustment dialog's settings are changing, the navigator keeps its image and
    // catches up once they settle (a thumbnail composite per change would cost more than the canvas).
    if let Some((r, _, t)) = &cached
        && *r == revision
        && crate::adjust_preview::settling(app, NAVIGATOR_SETTLE_MS)
    {
        ctx.request_repaint_after(std::time::Duration::from_millis(NAVIGATOR_SETTLE_MS as u64));
        return Some(t.id());
    }
    let t0 = crate::gpu_canvas::now_ms();
    let _span = photocraft_trace::span("navigator");
    let buffer = if std::env::var("PHOTOCRAFT_VIEW_COMPOSE").as_deref() != Ok("0") {
        // A viewport filter proxy intentionally omits pixels away from the view; the navigator
        // keeps the complete committed image until the filter is applied in that case.
        let source = if filter_display_factor(app, idx) > 1 { app.session.documents().get(idx)?.doc.clone() } else { doc.clone() };
        let level = photocraft_raster::Surface::mip_level_for_size(source.bounds(), NAVIGATOR_SIDE);
        let proxy = photocraft_compose::mip_view::document_region(&source, level, source.bounds()).ok()?;
        photocraft_compose::thumbnail_buffer(&proxy, NAVIGATOR_SIDE)
    } else {
        photocraft_compose::thumbnail_buffer(&doc, NAVIGATOR_SIDE)
    };
    let image = display_image(display.as_deref(), &buffer);
    let tex = match cached {
        Some((_, _, mut t)) => {
            t.set(image, TextureOptions::LINEAR);
            t
        }
        None => ctx.load_texture(format!("navigator-{}", id.0), image, TextureOptions::LINEAR),
    };
    app.perf.span("navigator", crate::gpu_canvas::now_ms() - t0);
    ctx.data_mut(|d| d.insert_temp(key, (revision, preview_key, tex.clone())));
    Some(tex.id())
}

/// Make sure the canvas texture for document `idx` is current; returns (texture id, scale).
pub fn ensure_texture(app: &mut PhotocraftApp, ctx: &egui::Context, idx: usize) -> Option<(egui::TextureId, f32)> {
    let _span = photocraft_trace::span("cpu_texture");
    let (revision, last_damage, id) = {
        let st = app.session.documents().get(idx)?;
        (st.revision, st.last_damage, st.doc.id)
    };
    let (mut doc, mut preview_key) = display_doc(app, idx);
    // A flipped view draws a GPU machine's canvas through here: an adjustment dialog's preview of
    // a large document (taken by the GPU path) would cost a full-size CPU composite per change.
    if crate::adjust_preview::shown_key(app) == Some(preview_key)
        && let Some(st) = app.session.documents().get(idx)
        && crate::proxy::factor(&st.doc) > 1
    {
        (doc, preview_key) = (st.doc.clone(), 0);
    }
    let (display, display_key) = canvas_display(app, &doc);
    let seen = app.canvases.get(&id).map(|c| (c.tex_revision, c.tex_preview_key));
    let damage = seen.and_then(|seen| damage_since(app, idx, seen, (revision, preview_key), display_key, last_damage));
    let preview_key = preview_key ^ display_key;
    let cache = app.canvases.entry(id).or_insert(CanvasCache {
        revision: 0,
        texture: None,
        scale: 1.0,
        preview_key: 0,
        on_gpu: false,
        tex_revision: 0,
        tex_preview_key: 0,
        live: None,
    });
    if cache.tex_revision != revision || cache.texture.is_none() || cache.tex_preview_key != preview_key {
        let t0 = crate::gpu_canvas::now_ms();
        let longest = doc.size.width.max(doc.size.height);
        let factor = longest.div_ceil(MAX_TEXTURE).max(1);
        let (w, h) = ((doc.size.width / factor).max(1), (doc.size.height / factor).max(1));
        let current = cache.texture.as_mut().filter(|t| t.size() == [w as usize, h as usize]);
        if let (Some(d), Some(t)) = (damage, current) {
            // Only what the edit or stroke touched: the reduced texture's pixels over it, with the
            // values the whole reduction gives them (factor 1 is the plain composite of `d`).
            let span = photocraft_trace::span("cpu_canvas_damage");
            let buf = photocraft_compose::render_reduced_damage(&doc, w, h, d);
            drop(span);
            let r = buf.rect;
            if !r.is_empty() {
                let t1 = crate::gpu_canvas::now_ms();
                t.set_partial([r.x0 as usize, r.y0 as usize], display_image(display.as_deref(), &buf), TextureOptions::LINEAR);
                let px = (r.width() as u64 * r.height() as u64).saturating_mul(u64::from(factor).pow(2));
                app.perf.record("rect", px, t1 - t0, crate::gpu_canvas::now_ms() - t1);
            }
        } else {
            // Reduced in bands straight from the compositor: no full-size composite in memory.
            let span = photocraft_trace::span("cpu_canvas_full");
            let full = photocraft_compose::render_reduced(&doc, w, h);
            drop(span);
            let t1 = crate::gpu_canvas::now_ms();
            let (img, scale) = (display_image(display.as_deref(), &full), 1.0 / factor as f32);
            match cache.texture.as_mut() {
                Some(t) if t.size() == img.size => t.set(img, TextureOptions::LINEAR),
                _ => cache.texture = Some(ctx.load_texture(format!("canvas-{}", id.0), img, TextureOptions::LINEAR)),
            }
            cache.scale = scale;
            app.perf.record("full", doc.size.width as u64 * doc.size.height as u64, t1 - t0, crate::gpu_canvas::now_ms() - t1);
        }
        cache.tex_revision = revision;
        cache.tex_preview_key = preview_key;
    }
    Some((cache.texture.as_ref()?.id(), cache.scale))
}

fn filter_display_factor(app: &PhotocraftApp, idx: usize) -> u32 {
    let Some(st) = app.session.documents().get(idx) else { return 1 };
    app.filter_preview
        .as_ref()
        .filter(|p| {
            p.doc == st.doc.id
                && p.revision == st.revision
                && p.active == st.active_layer
                && app.ui.dialogs.iter().any(|d| d.fields.get("__preview").and_then(serde_json::Value::as_bool) == Some(true))
        })
        .and_then(|p| p.region.as_ref())
        .filter(|r| r.result.is_some())
        .map_or(1, |r| r.factor)
}

fn viewport_level(visible: DRect, zoom: f32, input_level: u32) -> u32 {
    let mut level = if zoom.is_finite() && zoom > 0.0 && zoom < 0.5 { (1.0 / zoom).log2().floor().clamp(1.0, 16.0) as u32 } else { 0 }.max(input_level);
    while level < 16 && {
        let reduced = photocraft_raster::Surface::mip_bounds(visible, level);
        reduced.width() > MAX_TEXTURE - 64 || reduced.height() > MAX_TEXTURE - 64
    } {
        level += 1;
    }
    level
}

/// Display cache bounded by the viewport rather than the document's full-resolution canvas.
#[derive(Clone)]
pub(crate) struct ViewTexture {
    last_used: u64,
    revision: u64,
    preview: u64,
    level: u32,
    area: DRect,
    texture: egui::TextureHandle,
}

fn ensure_view_texture(app: &mut PhotocraftApp, ctx: &egui::Context, idx: usize, xf: &ViewXform, view_id: egui::Id) -> Option<(egui::TextureId, DRect)> {
    let _span = photocraft_trace::span("cpu_view_texture");
    if std::env::var("PHOTOCRAFT_VIEW_COMPOSE").as_deref() == Ok("0") {
        return None;
    }
    let st = app.session.documents().get(idx)?;
    let revision = st.revision;
    let (doc, preview) = display_doc(app, idx);
    let (display, display_key) = canvas_display(app, &doc);
    let preview = preview ^ display_key ^ if preview == 0 { std::sync::Arc::as_ptr(&doc) as usize as u64 } else { 0 };
    let original_bounds = app.session.documents().get(idx)?.doc.bounds();
    let original_visible = visible_doc_rect(xf).intersect(&original_bounds);
    let input_level = filter_display_factor(app, idx).trailing_zeros();
    let total_level = viewport_level(original_visible, xf.zoom * ctx.pixels_per_point(), input_level);
    let level = total_level.saturating_sub(input_level);
    let visible = photocraft_raster::Surface::mip_bounds(original_visible, input_level);
    if visible.is_empty() {
        return None;
    }
    let factor = 1i32 << total_level;
    let area = visible.inflate((1i32 << level).saturating_mul(32)).intersect(&doc.bounds());
    let reduced = photocraft_raster::Surface::mip_bounds(area, level);
    // Display textures are bounded even for hostile view/automation dimensions.
    if reduced.width() > MAX_TEXTURE || reduced.height() > MAX_TEXTURE {
        return None;
    }
    let key = egui::Id::new(("viewport-cpu", view_id, doc.id.0));
    let cache_id = (doc.id, key.value());
    let cached = app.view_textures.get_mut(&cache_id).map(|entry| {
        entry.last_used = app.frame;
        entry.clone()
    });
    if let Some(cached) = &cached
        && cached.revision == revision
        && cached.preview == preview
        && cached.level == total_level
        && cached.area.contains_rect(&original_visible)
    {
        return Some((cached.texture.id(), cached.area));
    }
    let proxy = photocraft_compose::mip_view::document_region(&doc, level, area).ok()?;
    let active = app.session.documents().get(idx)?.active_layer;
    let buffer = app.cpu_prefix.entry(doc.id).or_default().render(&proxy, reduced, active).unwrap_or_else(|_| photocraft_compose::render(&proxy, reduced));
    let image = display_image(display.as_deref(), &buffer);
    let texture = match cached {
        Some(mut cached) => {
            cached.texture.set(image, TextureOptions::LINEAR);
            cached.texture
        }
        None => ctx.load_texture(format!("viewport-{}", doc.id.0), image, TextureOptions::LINEAR),
    };
    let area =
        DRect::new(reduced.x0.saturating_mul(factor), reduced.y0.saturating_mul(factor), reduced.x1.saturating_mul(factor), reduced.y1.saturating_mul(factor));
    app.view_textures.insert(cache_id, ViewTexture { last_used: app.frame, revision, preview, level: total_level, area, texture: texture.clone() });
    Some((texture.id(), area))
}

#[derive(Clone)]
pub(crate) struct MipDocument {
    last_used: u64,
    revision: u64,
    preview: u64,
    level: u32,
    area: DRect,
    doc: std::sync::Arc<Document>,
}

struct GpuView {
    key: u64,
    factor: u32,
    area: DRect,
}

/// Composite only the visible rectangle into a texture of that rectangle's size. At low
/// zoom its layers are native CPU mip surfaces, so GPU upload never needs their level zero.
fn ensure_gpu_view(app: &mut PhotocraftApp, ctx: &egui::Context, idx: usize, xf: &ViewXform, view_id: egui::Id) -> Option<GpuView> {
    let _span = photocraft_trace::span("gpu_view");
    if std::env::var("PHOTOCRAFT_VIEW_COMPOSE").as_deref() == Ok("0") {
        return None;
    }
    let gpu = app.gpu.clone()?;
    gpu.begin_frame(app.frame);
    let (revision, last_damage) = app.session.documents().get(idx).map(|st| (st.revision, st.last_damage))?;
    let (doc, raw_key) = display_doc(app, idx);
    let (display, display_key) = canvas_display(app, &doc);
    // `damage_key` follows the full-document canvas cache's convention, so `damage_since` can
    // tell which rectangle changed since the texture was last composited.
    let damage_key = raw_key ^ display_key;
    let preview = damage_key ^ if raw_key == 0 { std::sync::Arc::as_ptr(&doc) as usize as u64 } else { 0 };
    let original_visible = visible_doc_rect(xf).intersect(&app.session.documents().get(idx)?.doc.bounds());
    let input_level = filter_display_factor(app, idx).trailing_zeros();
    let total_level = viewport_level(original_visible, xf.zoom * ctx.pixels_per_point(), input_level);
    let level = total_level.saturating_sub(input_level);
    let factor = 1u32 << total_level;
    gpu_budget(app, &gpu, idx, true, original_visible);
    let cache_key = egui::Id::new(("viewport-mip-document", view_id, doc.id.0));
    let source_visible = photocraft_raster::Surface::mip_bounds(original_visible, input_level);
    let cache_id = (doc.id, cache_key.value());
    let cached = app.view_mips.get_mut(&cache_id).map(|entry| {
        entry.last_used = app.frame;
        entry.clone()
    });
    let proxy = if level == 0 {
        doc.clone()
    } else {
        match cached {
            Some(cached) if (cached.revision, cached.preview, cached.level) == (revision, preview, level) && cached.area.contains_rect(&source_visible) => {
                cached.doc
            }
            _ => {
                let proxy_area = source_visible.inflate((1i32 << level).saturating_mul(64));
                let mut proxy = photocraft_compose::mip_view::document_region(&doc, level, proxy_area).ok()?;
                proxy.id = photocraft_doc::DocId(doc.id.0 ^ (1 << 56));
                let proxy = std::sync::Arc::new(proxy);
                app.view_mips.insert(cache_id, MipDocument { last_used: app.frame, revision, preview, level, area: proxy_area, doc: proxy.clone() });
                proxy
            }
        }
    };
    let visible = photocraft_raster::Surface::mip_bounds(source_visible, level);
    if visible.is_empty() {
        return None;
    }
    let wanted = visible.inflate(32).intersect(&proxy.bounds());
    if wanted.width() > MAX_TEXTURE || wanted.height() > MAX_TEXTURE {
        return None;
    }
    let output_key = view_id.with(("viewport-output", doc.id.0)).value();
    let uploaded_key = egui::Id::new(("viewport-gpu-upload", view_id, doc.id.0));
    let uploaded: Option<(u64, u64, u32, DRect, u64)> = ctx.data(|d| d.get_temp(uploaded_key));
    // A texture at this level that still covers the view can be updated in place.
    let reusable = uploaded.filter(|(_, _, l, a, _)| *l == level && a.contains_rect(&visible) && gpu.has(output_key, [a.width(), a.height()]));
    let existing = reusable.filter(|(r, p, ..)| (*r, *p) == (revision, preview));
    let area = reusable.map_or(wanted, |(_, _, _, a, _)| a);
    let pending = gpu.pending(output_key);
    // Only the changed rectangle (an edit's damage, the live stroke's dabs) is recomposited.
    // Through a filter preview's reduced input the damage is not in these coordinates.
    let region = match reusable {
        Some((r, _, _, _, k)) if existing.is_none() && !pending && input_level == 0 => {
            damage_since(app, idx, (r, k), (revision, raw_key), display_key, last_damage).map(|d| {
                let d = photocraft_raster::Surface::mip_bounds(d, level);
                if d.is_empty() { d } else { d.inflate(1).intersect(&area) }
            })
        }
        _ => None,
    }
    .unwrap_or(area);
    if existing.is_none() || pending {
        if !region.is_empty() {
            let owner = app.session.documents().get(idx)?.doc.id.0;
            match gpu.composite_view(&proxy, region, area, display.as_ref().is_some_and(|d| d.encode_srgb), output_key, owner) {
                Ok(stats) => {
                    app.perf.gpu_uploads = stats.tiles_uploaded as u64;
                    app.perf.gpu = true;
                }
                Err(error) => {
                    app.perf.gpu_fallback = Some(error.0);
                    return None;
                }
            }
        }
        ctx.data_mut(|d| d.insert_temp(uploaded_key, (revision, preview, level, area, damage_key)));
    }
    if gpu.pending(output_key) {
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
    Some(GpuView { key: output_key, factor, area })
}

/// What changed since a canvas cache showed (`revision`, `preview key`) `seen`, when only a
/// rectangle did: the last edit's damage, or the live stroke's dabs since then. Cached keys have
/// the colour display's key folded in (`^ display_key`); `now`'s is still raw.
fn damage_since(app: &PhotocraftApp, idx: usize, seen: (u64, u64), now: (u64, u64), display_key: u64, last_damage: Option<DRect>) -> Option<DRect> {
    if seen.1 == now.1 ^ display_key && seen.0 + 1 == now.0 {
        return last_damage;
    }
    if seen.0 == now.0
        && let Some(preview) = &app.filter_preview
        && let Some(region) = &preview.region
        && now.1 == (region.shown_hash | (1 << 59))
    {
        let reach = app.session.documents().get(idx).map_or(0, |st| effect_reach(&st.doc.layers));
        return Some(region.damage.inflate(reach));
    }
    // Several edits since that frame: the union of their damage, when every one is known.
    if seen.1 == now.1 ^ display_key
        && seen.0 < now.0
        && let Some(st) = app.session.documents().get(idx)
        && let Some(d) = st.damage_since(seen.0)
    {
        return Some(d);
    }
    // Between an adjustment dialog's previews (and the document): the target's area.
    if seen.0 == now.0
        && let Some(st) = app.session.documents().get(idx)
        && let Some(r) = crate::adjust_preview::switch_region(app, st.doc.id, now.0, seen.1 ^ display_key, now.1)
    {
        return Some(r);
    }
    // Free Transform's in-place preview: what moved between its updates.
    if seen.0 == now.0
        && let Some(r) = crate::transform_tool::in_place_damage(app, seen.1 ^ display_key, now.1)
    {
        return Some(r);
    }
    let l = live_stroke(app, idx).filter(|l| seen.0 == now.0 && l.display_key() == now.1)?;
    let r = l.since(seen.1 ^ display_key)?;
    let reach = l.stroke.layer().map_or_else(|| effect_reach(&l.stroke.doc.layers), |id| photocraft_engine::damage::chain_reach(&l.stroke.doc, id));
    Some(if r.is_empty() { r } else { r.inflate(reach) })
}

/// How far beyond an edit's damage rect the composite can change: layer effects (shadows, glows,
/// strokes, …) on the edited layer and on the groups around it reach that far.
pub(crate) fn effect_reach(layers: &[photocraft_doc::Layer]) -> i32 {
    layers
        .iter()
        .filter(|l| l.visible)
        .map(|l| {
            let own = if photocraft_compose::effects::has_effects(l) { photocraft_compose::effects::margin(l) } else { 0 };
            let inner = match &l.content {
                LayerContent::Group(g) => effect_reach(&g.children),
                _ => 0,
            };
            own + inner
        })
        .max()
        .unwrap_or(0)
}

/// GPU path: make sure document `idx` is current in the GPU canvas. Brush strokes re-composite
/// and upload only their damage rect; everything else re-composites the whole document.
/// Returns false if there is no GPU canvas.
fn ensure_gpu(app: &mut PhotocraftApp, idx: usize, visible: DRect, zoom: f32) -> bool {
    let _span = photocraft_trace::span("gpu_full_canvas");
    let Some(gpu) = app.gpu.clone() else { return false };
    gpu.begin_frame(app.frame);
    let Some((revision, last_damage, id)) = app.session.documents().get(idx).map(|st| (st.revision, st.last_damage, st.doc.id)) else {
        return false;
    };
    let (doc, raw_key) = display_doc(app, idx);
    let (display, display_key) = canvas_display(app, &doc);
    let seen = app.canvases.get(&id).map(|c| (c.revision, c.preview_key));
    let mut damage = seen.and_then(|seen| damage_since(app, idx, seen, (revision, raw_key), display_key, last_damage));
    // To an adjustment dialog's preview: only what the view shows now (the rest as it moves there).
    if damage.is_some()
        && let Some(r) = seen.filter(|s| s.0 == revision).and_then(|s| crate::adjust_preview::switch_region(app, id, revision, s.1 ^ display_key, raw_key))
    {
        damage = Some(crate::adjust_preview::switch_damage(app, raw_key, r, visible));
    }
    let preview_key = raw_key ^ display_key;
    let size = [doc.size.width, doc.size.height];
    gpu.set_preview_zoom(id.0, size, zoom);
    let cache = app.canvases.entry(id).or_insert(CanvasCache {
        revision: 0,
        texture: None,
        scale: 1.0,
        preview_key: 0,
        on_gpu: false,
        tex_revision: 0,
        tex_preview_key: 0,
        live: None,
    });
    let present = cache.on_gpu && gpu.has(id.0, size);
    if present && cache.revision == revision && cache.preview_key == preview_key {
        // An adjustment preview composited where the view was: catch up where it moved to.
        if let Some(r) = crate::adjust_preview::uncovered(app, id, raw_key, visible) {
            let r = gpu.refresh_progressive(id.0, &doc, Some(r), visible, display.as_deref(), true);
            if r.kind.starts_with("gpu") {
                app.perf.gpu_uploads = r.uploads;
            }
            app.perf.record(r.kind, r.px, r.composite_ms, r.upload_ms);
        } else if gpu.pending(id.0) {
            let r = gpu.refresh_progressive(id.0, &doc, None, visible, display.as_deref(), false);
            if r.kind.starts_with("gpu") {
                app.perf.gpu_uploads = r.uploads;
            }
            app.perf.record(r.kind, r.px, r.composite_ms, r.upload_ms);
        }
        return true;
    }
    let partial = present && damage.is_some();
    // From a live stroke's second frame on, only its layer's pixels change between refreshes.
    let live = live_stroke(app, idx).map(|l| l.revision);
    let pixels_only = partial && live.is_some() && app.canvases.get(&id).is_some_and(|c| c.live == live);
    gpu.set_pixels_only(pixels_only);
    gpu_budget(app, &gpu, idx, partial, visible);
    // Large refreshes (opening, a full re-render, an edit over most of a big document) go in
    // slices, the view first; strokes and other small damage are composited at once.
    let small = partial && damage.is_some_and(|r| u64::from(r.width()) * u64::from(r.height()) <= crate::gpu_canvas::SYNC_DAMAGE_PX);
    let r = if crate::gpu_canvas::progressive_doc(&doc) && !small {
        gpu.refresh_progressive(id.0, &doc, if partial { damage } else { None }, visible, display.as_deref(), true)
    } else {
        gpu.refresh(id.0, &doc, if partial { damage } else { None }, display.as_deref())
    };
    if let Some(e) = &r.fallback
        && app.perf.gpu_fallback.as_deref() != Some(e.as_str())
    {
        log::info!("{e}; using the CPU compositor");
    }
    app.perf.record(r.kind, r.px, r.composite_ms, r.upload_ms);
    if r.kind.starts_with("gpu") {
        app.perf.gpu_uploads = r.uploads;
    } else {
        app.perf.gpu_fallback = r.fallback;
    }
    let Some(cache) = app.canvases.get_mut(&id) else { return true };
    cache.revision = revision;
    cache.preview_key = preview_key;
    cache.on_gpu = true;
    cache.texture = None;
    cache.live = live;
    true
}

/// Point the wgpu compositor at what the view shows, and size its memory budget from Memory
/// Usage, the document's pixels and History, and physical memory: on full refreshes (pixels
/// counted once per structural change, not per brush dab), whenever the preference changes,
/// and with a scratch disk whenever the tiles in memory move by more than 64 MB.
fn gpu_budget(app: &mut PhotocraftApp, gpu: &crate::gpu_canvas::GpuCanvas, idx: usize, partial: bool, visible: DRect) {
    gpu.set_focus(Some(visible));
    let allowance = u64::from(app.session.prefs().performance.memory_usage_mb).saturating_mul(1 << 20);
    // With a scratch disk the tiles in memory are one cheap counter (at most the tile budget),
    // so partial refreshes follow it too: edits rarely refresh everything, and a budget sized
    // while the document was empty would let the GPU pages grow past Memory Usage.
    let spill = photocraft_raster::spill::enabled();
    let resident = if spill { photocraft_raster::spill::stats().resident_bytes as u64 } else { 0 };
    let moved = resident.abs_diff(app.perf.gpu_budget_pixels) > 64 << 20;
    if partial && app.perf.gpu_budget_allowance == allowance && gpu.memory_budget().is_some() && !(spill && moved) {
        return;
    }
    let pixels = if spill {
        resident
    } else {
        let Some(st) = app.session.documents().get(idx) else { return };
        st.history.pixel_bytes(&st.doc) as u64
    };
    let budget = crate::gpu_canvas::memory_budget(allowance, pixels, crate::gpu_canvas::physical_memory());
    gpu.set_memory_budget(budget);
    app.perf.gpu_budget_mb = budget >> 20;
    app.perf.gpu_budget_allowance = allowance;
    app.perf.gpu_budget_pixels = pixels;
}

/// Live preview for an open filter dialog: run the filter on the proxy and upload it.
fn ensure_filter_preview(app: &mut PhotocraftApp, idx: usize, visible: DRect, zoom: f32) -> Option<(u32, u64)> {
    let _span = photocraft_trace::span("filter_preview");
    if crate::adjust_preview::on_layer(app, idx) {
        return None;
    }
    let d = app.ui.dialogs.iter().find(|d| d.fields.contains_key("__filter"))?;
    if d.fields.get("__preview").and_then(serde_json::Value::as_bool) != Some(true) {
        return None;
    }
    let cmd = d.fields.get("__command")?.as_str()?.to_string();
    let params = crate::filter_dialog::params_of(&d.fields);
    let (doc_id, revision, doc, active, channel) = {
        let st = app.session.documents().get(idx)?;
        (st.doc.id, st.revision, st.doc.clone(), st.active_layer, st.channel_view.target)
    };
    if std::env::var("PHOTOCRAFT_FILTER_VIEW").as_deref() != Ok("0")
        && crate::filter_dialog::region::pixels_target(&channel, &params)
        && crate::filter_dialog::region::supported(&doc, active, &cmd)
    {
        crate::filter_dialog::region::update(app, doc, revision, active, cmd, params, visible, zoom);
        return None;
    }
    let k = crate::proxy::factor(&doc);
    let hash = format!("{cmd}{params}{channel:?}").bytes().fold(k as u64 ^ revision.wrapping_mul(0x9e37), |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
    let key = doc_id.0 ^ (1u64 << 61);
    let fresh = matches!(&app.filter_preview, Some(p) if p.doc == doc_id && p.hash == hash && p.active == active);
    if !fresh {
        let t0 = crate::gpu_canvas::now_ms();
        let proxy = app
            .filter_preview
            .as_ref()
            .filter(|p| std::env::var("PHOTOCRAFT_FILTER_PROXY").as_deref() != Ok("0") && p.doc == doc_id && p.revision == revision && p.k == k)
            .map(|p| p.proxy.clone())
            .unwrap_or_else(|| std::sync::Arc::new(crate::proxy::proxy_document(&doc, k)));
        let span = photocraft_trace::span("cpu_filter_preview");
        let result = crate::filter_dialog::preview_from_proxy_in(&proxy, active, channel, &cmd, &params, k).map(std::sync::Arc::new);
        if let Some(r) = &result {
            let buf = photocraft_compose::flatten(r);
            drop(span);
            let t1 = crate::gpu_canvas::now_ms();
            let (display, _) = canvas_display(app, &doc);
            app.gpu.as_ref()?.upload_buffer_full(key, &texture_buffer(display.as_deref(), &buf), doc.depth);
            app.perf.record("filter-preview", r.size.area(), t1 - t0, crate::gpu_canvas::now_ms() - t1);
        }
        app.filter_preview = Some(crate::filter_dialog::FilterPreview { doc: doc_id, revision, hash, proxy, active, k, result, region: None });
    }
    app.filter_preview.as_ref().filter(|p| p.result.is_some()).map(|p| (p.k, key))
}

/// The document pixels a (non-rotated) view shows, with a margin for filtering.
fn visible_doc_rect(xf: &ViewXform) -> DRect {
    let (a, b) = (xf.to_doc(xf.rect.min), xf.to_doc(xf.rect.max));
    let c = |v: f64| v.clamp(-1e9, 1e9) as i32;
    DRect::new(c(a[0].min(b[0]).floor()) - 2, c(a[1].min(b[1]).floor()) - 2, c(a[0].max(b[0]).ceil()) + 2, c(a[1].max(b[1]).ceil()) + 2)
}

/// Zoomed-out Image › Adjustments preview on a large document: the wgpu compositor renders the
/// reduced preview document (`adjust_preview::gpu_proxy`) into its own texture, over the target's
/// area after the first frame. Returns (factor, gpu key) to draw.
fn ensure_adjust_proxy(app: &mut PhotocraftApp, idx: usize, zoom: f32) -> Option<(u32, u64)> {
    let _span = photocraft_trace::span("adjust_proxy");
    let frame = crate::adjust_preview::gpu_proxy(app, idx, zoom)?;
    let key = frame.doc.id.0;
    let size = [frame.doc.size.width, frame.doc.size.height];
    let gpu = app.gpu.clone()?;
    if frame.stale || !gpu.has(key, size) {
        let doc = app.session.documents().get(idx)?.doc.clone();
        let (display, _) = canvas_display(app, &doc);
        let r = gpu.refresh(key, &frame.doc, frame.damage, display.as_deref());
        app.perf.record(if r.kind.starts_with("gpu") { "gpu-adjust-proxy" } else { "adjust-proxy" }, r.px, r.composite_ms, r.upload_ms);
        crate::adjust_preview::proxy_drawn(app, &frame);
    }
    Some((frame.k, key))
}

/// If a live adjustment preview is active on a large document, composite it on the proxy and upload
/// it under its own GPU key. Returns (factor, gpu key) when the proxy should be drawn.
fn ensure_proxy_preview(app: &mut PhotocraftApp, idx: usize) -> Option<(u32, u64)> {
    let _span = photocraft_trace::span("ensure_proxy_preview");
    let (layer, params) = app.live_adjust.clone()?;
    let (doc_id, revision, doc) = {
        let st = app.session.documents().get(idx)?;
        (st.doc.id, st.revision, st.doc.clone())
    };
    let k = crate::proxy::factor(&doc);
    if k <= 1 {
        return None;
    }
    // A display mip (effects and vector masks scaled with the pixels, layers reduced on all
    // cores from the cached pyramid) rather than the point-sampled proxy: that one left effect
    // sizes in document pixels (shadows several times too big in the preview) and took 13 s on
    // the UI thread for a 479-layer PSB.
    let level = k.next_power_of_two().trailing_zeros();
    let k = 1u32 << level;
    let fresh = matches!(&app.proxy, Some((d, r, kk, _)) if *d == doc_id && *r == revision && *kk == k);
    if !fresh {
        let proxy = photocraft_compose::mip_view::document(&doc, level).ok()?;
        app.proxy = Some((doc_id, revision, k, std::sync::Arc::new(proxy)));
    }
    let proxy = app.proxy.as_ref()?.3.clone();
    let key = doc_id.0 ^ (1u64 << 62);
    let hash = params.to_string().bytes().fold(k as u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
    if app.proxy_uploaded != Some((doc_id, hash)) {
        let l = proxy.layer(layer)?;
        let LayerContent::Adjustment(a) = &l.content else { return None };
        let kind = photocraft_engine::commands::adjustment_kind(a);
        let preview = photocraft_engine::adjust_params::from_params(kind, &params, Some(a), proxy.mode).unwrap_or_else(|_| a.clone());
        let mut p = (*proxy).clone();
        if let Some(lm) = p.layer_mut(layer) {
            lm.content = LayerContent::Adjustment(preview);
        }
        let t0 = crate::gpu_canvas::now_ms();
        let span = photocraft_trace::span("cpu_proxy_preview");
        let buf = photocraft_compose::flatten(&p);
        drop(span);
        let t1 = crate::gpu_canvas::now_ms();
        let (display, _) = canvas_display(app, &doc);
        app.gpu.as_ref()?.upload_buffer_full(key, &texture_buffer(display.as_deref(), &buf), doc.depth);
        app.perf.record("proxy", p.size.area(), t1 - t0, crate::gpu_canvas::now_ms() - t1);
        app.proxy_uploaded = Some((doc_id, hash));
    }
    Some((k, key))
}

/// A CPU composite as the GPU canvas texture stores it (sRGB-encoded for linear documents).
fn texture_buffer<'a>(
    display: Option<&photocraft_engine::display_color::CanvasDisplay>,
    buf: &'a photocraft_compose::Buffer,
) -> std::borrow::Cow<'a, photocraft_compose::Buffer> {
    match display {
        Some(d) => d.texture_buffer(buf),
        None => std::borrow::Cow::Borrowed(buf),
    }
}

/// Keep four saved views per document, plus any views already used this frame. Retiring a
/// cache referenced by an earlier canvas in this frame could free its egui texture before paint.
fn trim_view_cache<T>(cache: &mut std::collections::HashMap<(photocraft_doc::DocId, u64), T>, frame: u64, used: impl Fn(&T) -> u64) {
    let documents: std::collections::HashSet<_> = cache.keys().map(|(doc, _)| *doc).collect();
    for document in documents {
        let count = cache.keys().filter(|(doc, _)| *doc == document).count();
        let mut old: Vec<_> =
            cache.iter().filter(|((doc, _), entry)| *doc == document && used(entry) < frame).map(|(key, entry)| (used(entry), *key)).collect();
        old.sort_by_key(|(frame, (_, key))| (*frame, *key));
        for (_, key) in old.into_iter().take(count.saturating_sub(4)) {
            cache.remove(&key);
        }
    }
}

pub(crate) fn retain_gpu_documents(app: &mut PhotocraftApp) {
    // Remove a closed adjustment owner's proxy before collecting its live GPU keys.
    crate::adjust_preview::retain_documents(app);
    if let Some(gpu) = &app.gpu {
        // Keep each document's texture plus its preview textures (filter preview, adjustment proxy);
        // retaining only document ids freed the previews every frame (blank canvas while previewing).
        let mut live: Vec<u64> = app
            .session
            .documents()
            .iter()
            .flat_map(|st| [st.doc.id.0, st.doc.id.0 ^ (1u64 << 61), st.doc.id.0 ^ (1u64 << 62), st.doc.id.0 ^ (1u64 << 56)])
            .collect();
        live.extend(crate::adjust_preview::gpu_keys(app));
        gpu.retain(&live);
    }
    // Upload markers must not outlive the resources they describe: native reopen can reuse
    // both the document ID and revision before the next frame while a preview dialog stays open.
    let documents = app.session.documents();
    app.view_textures.retain(|(id, _), _| documents.iter().any(|st| st.doc.id == *id));
    app.view_mips.retain(|(id, _), _| documents.iter().any(|st| st.doc.id == *id));
    trim_view_cache(&mut app.view_textures, app.frame, |entry| entry.last_used);
    trim_view_cache(&mut app.view_mips, app.frame, |entry| entry.last_used);
    app.cpu_prefix.retain(|id, _| documents.iter().any(|st| st.doc.id == *id));
    if app.filter_preview.as_ref().is_some_and(|preview| !documents.iter().any(|st| st.doc.id == preview.doc)) {
        app.filter_preview = None;
    }
    if app.proxy_uploaded.is_some_and(|(doc, _)| !documents.iter().any(|st| st.doc.id == doc)) {
        app.proxy_uploaded = None;
        app.proxy = None;
    }
}

/// Tabs + canvas for the active document, or the start screen.
pub fn document_area(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    retain_gpu_documents(app);
    if app.ui.chrome.shows_home(app.session.documents().len()) {
        start_screen(app, ui);
        return;
    }
    if !app.ui.view.hides_tabs() {
        tabs(app, ui);
    }
    let Some(idx) = app.session.active_index() else { return };
    let rect = ui.available_rect_before_wrap();
    app.last_canvas_rect = rect;
    let n = app.session.documents().len();
    // Window › Arrange: tiled / n-up layouts show several documents side by side; the active one
    // takes input, a click elsewhere activates that document.
    if let Some(cells) = crate::view_cmds::cells(&app.ui.view.arrange, rect, n) {
        let order: Vec<usize> = (0..n).map(|k| (idx + k) % n).collect();
        let mut shown: Vec<(usize, Rect)> = order.into_iter().zip(cells).collect();
        shown.sort_by_key(|(d, _)| *d);
        let t = crate::theme::Tokens::get(ui.ctx());
        for (d, cell) in shown {
            let cell = cell.shrink(1.0);
            let view = app.ui.views[d].clone();
            let v = canvas_view(app, ui, d, cell, view, d == idx);
            if d != idx {
                app.ui.views[d] = v;
                if ui.input(|i| i.pointer.primary_pressed() && i.pointer.interact_pos().is_some_and(|p| cell.contains(p))) {
                    app.session.set_active(d);
                }
            }
            let stroke = if d == idx { Stroke::new(1.5, t.accent) } else { Stroke::new(1.0, Color32::from_black_alpha(160)) };
            ui.painter().rect_stroke(cell, 0, stroke, egui::StrokeKind::Inside);
        }
        return;
    }
    let view = app.ui.views[idx].clone();
    canvas_view(app, ui, idx, rect, view, true);
}

fn tabs(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = crate::theme::Tokens::get(ui.ctx());
    if t.pro {
        return pro_tabs(app, ui);
    }
    let active = app.session.active_index();
    let mut activate = None;
    let mut close = None;
    egui::Frame::NONE.fill(t.canvas).inner_margin(egui::Margin { left: 8, right: 8, top: 6, bottom: 4 }).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for (i, st) in app.session.documents().iter().enumerate() {
                let sel = Some(i) == active;
                let name = format!("{}{}", st.doc.name, if st.is_dirty() { " *" } else { "" });
                let meta = format!("{}/{}", mode_label(&st.doc), st.doc.depth.bits());
                let name_g = ui.painter().layout_no_wrap(name, crate::theme::medium(12.5), t.text);
                let meta_g = ui.painter().layout_no_wrap(meta, egui::FontId::proportional(10.5), t.text_faint);
                let w = name_g.size().x + meta_g.size().x + 44.0;
                let (r, resp) = ui.allocate_exact_size(egui::vec2(w, 26.0), Sense::click());
                if sel {
                    ui.painter().rect_filled(r, t.radius_sm, t.card);
                    ui.painter().rect_stroke(r, t.radius_sm, Stroke::new(1.0, t.card_border), egui::StrokeKind::Inside);
                } else if resp.hovered() {
                    ui.painter().rect_filled(r, t.radius_sm, t.hover.gamma_multiply(0.5));
                }
                let color = if sel { t.text } else { t.text_dim };
                let ny = r.center().y - name_g.size().y / 2.0;
                ui.painter().galley_with_override_text_color(egui::pos2(r.left() + 10.0, ny), name_g.clone(), color);
                ui.painter().galley(egui::pos2(r.left() + 16.0 + name_g.size().x, r.center().y - meta_g.size().y / 2.0), meta_g, t.text_faint);
                let xr = Rect::from_center_size(egui::pos2(r.right() - 12.0, r.center().y), egui::vec2(16.0, 16.0));
                let xresp = ui.interact(xr, ui.id().with(("tabx", i)), Sense::click());
                if xresp.hovered() {
                    ui.painter().rect_filled(xr, 4.0, t.hover);
                }
                crate::icons::paint(ui, xr, "x", 11.0, if xresp.hovered() { t.text } else { t.text_faint });
                if xresp.clicked() {
                    close = Some(i);
                } else if resp.clicked() {
                    activate = Some(i);
                }
            }
        });
    });
    if let Some(i) = activate {
        app.session.set_active(i);
    }
    if let Some(i) = close {
        let _ = crate::menus::invoke(app, ui.ctx(), "file.close", json!({"document": i}));
    }
}

/// Photoshop document tabs: "name @ 33.3% (RGB/8)" on a dark strip; active tab matches panels.
fn pro_tabs(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = crate::theme::Tokens::get(ui.ctx());
    let active = app.session.active_index();
    let (mut activate, mut close) = (None, None);
    let (strip, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 26.0), Sense::hover());
    ui.painter().rect_filled(strip, 0.0, t.tab_strip);
    let mut x = strip.left();
    for (i, st) in app.session.documents().iter().enumerate() {
        let zoom = app.ui.views.get(i).map_or(100.0, |v| v.zoom * 100.0);
        // Photoshop: "name @ 50% (Layer 1, RGB/8)", "(Layer 1, Layer Mask/8)" when the mask is targeted;
        // the Background layer's name is omitted.
        let active_layer = st.active_layer.and_then(|id| st.doc.layer(id)).filter(|l| !(l.name == "Background" && l.locks.transparency));
        let is_active_doc = app.session.active_index() == Some(i);
        let mask = is_active_doc && app.ui.mask_target && active_layer.is_some_and(|l| l.mask.is_some());
        let model = if mask { "Layer Mask".to_string() } else { mode_label(&st.doc).to_string() };
        let layer = active_layer.map(|l| format!("{}, ", l.name)).unwrap_or_default();
        let title = format!("{} @ {}% ({layer}{model}/{}){}", st.doc.name, fmt_zoom(zoom), st.doc.depth.bits(), if st.is_dirty() { "*" } else { "" });
        let g = ui.painter().layout_no_wrap(title, egui::FontId::proportional(11.5), t.text);
        let r = Rect::from_min_size(egui::pos2(x, strip.top()), egui::vec2(g.size().x + 42.0, strip.height()));
        let resp = ui.interact(r, ui.id().with(("ptab", i)), Sense::click());
        let sel = Some(i) == active;
        if sel {
            ui.painter().rect_filled(r, 0.0, t.chrome);
        } else if resp.hovered() {
            ui.painter().rect_filled(r, 0.0, t.hover.gamma_multiply(0.35));
        }
        ui.painter().line_segment([r.right_top(), r.right_bottom()], Stroke::new(1.0, t.separator));
        let xr = Rect::from_center_size(egui::pos2(r.left() + 13.0, r.center().y), egui::vec2(14.0, 14.0));
        let xresp = ui.interact(xr, ui.id().with(("ptabx", i)), Sense::click());
        crate::icons::paint(ui, xr, "x", 10.0, if xresp.hovered() { t.text } else { t.text_faint });
        ui.painter().galley_with_override_text_color(egui::pos2(r.left() + 26.0, r.center().y - g.size().y / 2.0), g, if sel { t.text } else { t.text_faint });
        if xresp.clicked() {
            close = Some(i);
        } else if resp.clicked() {
            activate = Some(i);
        }
        x = r.right();
    }
    if let Some(i) = activate {
        app.session.set_active(i);
    }
    if let Some(i) = close {
        let _ = crate::menus::invoke(app, ui.ctx(), "file.close", json!({"document": i}));
    }
}

/// Photoshop-style zoom label: "33.3", "100", "12.5".
pub fn fmt_zoom(pct: f32) -> String {
    if (pct - pct.round()).abs() < 0.05 { format!("{}", pct.round() as i64) } else { format!("{pct:.1}") }
}

/// Repeating dot-grid texture for the canvas surround.
fn dots(ctx: &egui::Context, t: &crate::theme::Tokens) -> Option<egui::TextureId> {
    if t.bevel {
        return None;
    }
    let key = egui::Id::new(("canvas-dots", format!("{:?}", t.kind)));
    if let Some(tex) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(key)) {
        return Some(tex.id());
    }
    let n = 22usize;
    let mut px = vec![Color32::TRANSPARENT; n * n];
    for (dx, dy, a) in [(0usize, 0usize, 255u8), (1, 0, 110), (0, 1, 110), (1, 1, 60)] {
        let c = t.canvas_dot;
        px[(n / 2 + dy) * n + n / 2 + dx] = Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a);
    }
    let opts = TextureOptions {
        magnification: egui::TextureFilter::Linear,
        minification: egui::TextureFilter::Linear,
        wrap_mode: egui::TextureWrapMode::Repeat,
        mipmap_mode: None,
    };
    let tex = ctx.load_texture("canvas-dots", egui::ColorImage::new([n, n], px), opts);
    let id = tex.id();
    ctx.data_mut(|d| d.insert_temp(key, tex));
    Some(id)
}

fn paint_dots(ui: &egui::Ui, rect: Rect) {
    let t = crate::theme::Tokens::get(ui.ctx());
    if let Some(id) = dots(ui.ctx(), &t) {
        let uv = Rect::from_min_max(Pos2::ZERO, pos2(rect.width() / 22.0, rect.height() / 22.0));
        ui.painter_at(rect).image(id, rect, uv, Color32::WHITE);
    }
}

pub fn mode_label(doc: &Document) -> &'static str {
    match doc.mode {
        photocraft_doc::ColorMode::Rgb => "RGB",
        photocraft_doc::ColorMode::Grayscale => "Gray",
        photocraft_doc::ColorMode::Cmyk => "CMYK",
        photocraft_doc::ColorMode::Lab => "Lab",
        photocraft_doc::ColorMode::Indexed => "Indexed",
        photocraft_doc::ColorMode::Bitmap => "Bitmap",
        photocraft_doc::ColorMode::Duotone => "Duotone",
        photocraft_doc::ColorMode::Multichannel => "Multichannel",
    }
}

fn start_screen(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = crate::theme::Tokens::get(ui.ctx());
    let area = ui.available_rect_before_wrap();
    paint_dots(ui, area);
    let card = Rect::from_center_size(area.center(), egui::vec2(460.0, 330.0));
    ui.scope_builder(egui::UiBuilder::new().max_rect(card), |ui| {
        ui.vertical_centered(|ui| {
            ui.horizontal(|ui| {
                let title = ui.painter().layout_no_wrap("PhotoCraft".into(), crate::theme::semibold(38.0), t.text);
                let by = ui.painter().layout_no_wrap("open source".into(), egui::FontId::proportional(13.0), t.text_faint);
                let total = title.size().x + by.size().x + 10.0;
                ui.add_space(((card.width() - total) / 2.0).max(0.0));
                let (r, _) = ui.allocate_exact_size(title.size(), Sense::hover());
                ui.painter().galley(r.min, title, t.text);
                let (r2, _) = ui.allocate_exact_size(egui::vec2(by.size().x + 10.0, r.height()), Sense::hover());
                ui.painter().galley(egui::pos2(r2.left() + 10.0, r.bottom() - by.size().y - 8.0), by, t.text_faint);
            });
            ui.add_space(6.0);
            ui.label(egui::RichText::new("Create a new document or open an existing file.").color(t.text_dim).size(14.0));
            ui.add_space(22.0);
            ui.horizontal(|ui| {
                ui.add_space(((card.width() - 2.0 * 190.0 - 12.0) / 2.0).max(0.0));
                ui.spacing_mut().item_spacing.x = 12.0;
                if crate::widgets::primary_button(ui, "New document…     ⌘N", 190.0).clicked() {
                    app.ui.open_dialog(crate::state::DialogKind::NewDocument, crate::state::UiState::new_document_fields());
                }
                if crate::widgets::secondary_button(ui, "Open…     ⌘O", 190.0).clicked() {
                    app.open_dialog_file();
                }
            });
            ui.add_space(22.0);
            ui.horizontal(|ui| {
                let msg = "Drop an image or PSD anywhere to open it.";
                let g = ui.painter().layout_no_wrap(msg.into(), egui::FontId::proportional(12.5), t.text_faint);
                ui.add_space(((card.width() - g.size().x - 24.0) / 2.0).max(0.0));
                let (r, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), Sense::hover());
                crate::icons::paint(ui, r, "image", 15.0, t.text_faint);
                ui.label(egui::RichText::new(msg).color(t.text_faint));
            });
            ui.add_space(26.0);
            crate::links::discord_button(app, ui, 190.0);
            ui.add_space(10.0);
            crate::links::link_row(app, ui);
        });
    });
}

/// Lattice size of the canvas display LUT (colour management, Proof Colors, Gamut Warning).
const DISPLAY_LUT: usize = 33;

/// Keep the GPU display LUT under `key` (the document's texture or a preview of it) in step with
/// `doc`'s colour management: document → monitor profile and View › Proof Colors / Gamut
/// Warning (the 32-bit preview is applied by the canvas shader, see [`hdr_preview`]). Returns the canvas `display` mode (0 none — the identity, e.g. sRGB on
/// an sRGB monitor —, 1 LUT, 2 LUT + gamut warning).
fn sync_display_lut(app: &mut PhotocraftApp, doc: &photocraft_doc::Document, key: u64) -> u8 {
    let Some(gpu) = app.gpu.clone() else { return 0 };
    // Rebuild only when anything feeding the LUT changes.
    let sig = app.session.color.display_signature(doc);
    if let Some((s, mode)) = gpu.display_lut_signature(key)
        && s == sig
    {
        return mode;
    }
    let gamut = app.session.color.proof(doc.id).gamut_warning;
    let mode = match app.session.color.gpu_canvas_lut(doc, DISPLAY_LUT) {
        Ok(Some(bytes)) => {
            gpu.set_display_lut(key, DISPLAY_LUT as u32, Some(&bytes));
            if gamut { 2 } else { 1 }
        }
        Ok(None) => {
            gpu.set_display_lut(key, DISPLAY_LUT as u32, None);
            0
        }
        Err(e) => {
            app.ui.status = format!("Color management: {e}");
            gpu.set_display_lut(key, DISPLAY_LUT as u32, None);
            0
        }
    };
    gpu.cache_display_lut_signature(key, sig, mode);
    mode
}

/// View › 32-bit Preview Options for the GPU canvas shader: (exposure, gamma) when active.
fn hdr_preview(app: &PhotocraftApp, doc: &photocraft_doc::Document) -> Option<[f32; 2]> {
    app.session.color.hdr_preview(doc).map(|h| [h.exposure, h.gamma])
}

/// Draw one canvas view and handle its input. `primary` = main window (tools active).
pub fn canvas_view(app: &mut PhotocraftApp, ui: &mut egui::Ui, idx: usize, rect: Rect, mut view: View, primary: bool) -> View {
    let ctx = ui.ctx().clone();
    let full = rect;
    let rect = if primary { crate::rulers::content_rect(app, rect) } else { rect };
    let doc = app.session.documents()[idx].doc.clone();
    let size = [doc.size.width, doc.size.height];
    if view.doc_size != size {
        // Like Photoshop: keep the zoom level, re-centre the resized document.
        if view.doc_size != [0, 0] {
            view.center = [size[0] as f32 / 2.0, size[1] as f32 / 2.0];
        }
        view.doc_size = size;
    }
    if view.fit_pending && rect.width() > 50.0 {
        fit_view(&mut view, &doc, rect.size());
    }
    let flip = app.ui.view.flip_horizontal;
    let xf = ViewXform { rect, zoom: view.zoom, center: view.center, flip };
    let pixel_grid = app.ui.view.shows(app.ui.view.show.pixel_grid);
    let response = ui.allocate_rect(rect, Sense::click_and_drag());
    let painter = ui.painter_at(rect);

    match crate::prefs_ui::pasteboard_color(app) {
        Some(c) => {
            ui.painter_at(rect).rect_filled(rect, 0.0, c);
        }
        None => paint_dots(ui, rect),
    }
    let border = app.session.prefs().interface.canvas_border;
    let drop_shadow = border == photocraft_engine::prefs::CanvasBorder::DropShadow;
    // Drop shadow, checkerboard, document image.
    let img_rect = xf.doc_rect(doc.bounds());
    // Live adjustment previews on big documents use a downsampled proxy (see proxy.rs).
    let mut on_gpu = false;
    let filter_frame = ensure_filter_preview(app, idx, visible_doc_rect(&xf), view.zoom * ctx.pixels_per_point());
    if app.filter_preview.as_ref().is_some_and(|p| p.region.as_ref().is_some_and(|r| r.pending())) {
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
    // A flipped view draws through the CPU path (the GPU canvas shader has no mirroring).
    if app.gpu.is_some()
        && !flip
        && let Some((k, key)) = ensure_adjust_proxy(app, idx, view.zoom * ctx.pixels_per_point()).or(filter_frame).or_else(|| ensure_proxy_preview(app, idx))
    {
        on_gpu = true;
        let params = crate::gpu_canvas::ViewParams {
            doc: key,
            doc_size: [doc.size.width.div_ceil(k), doc.size.height.div_ceil(k)],
            zoom: view.zoom * k as f32,
            center: [view.center[0] / k as f32, view.center[1] / k as f32],
            shadow: {
                let t = crate::theme::Tokens::get(&ctx);
                !t.bevel && !t.pro && drop_shadow
            },
            pixel_grid: false,
            view_key: egui::Id::new(("pc-canvas-proxy", ctx.viewport_id(), idx)).value(),
            display: sync_display_lut(app, &doc, key),
            hdr: hdr_preview(app, &doc),
        };
        crate::gpu_canvas::GpuCanvas::paint(&painter, rect, params);
    } else if !flip && let Some(frame) = ensure_gpu_view(app, &ctx, idx, &xf, ui.id().with(ctx.viewport_id())) {
        on_gpu = true;
        if !crate::theme::Tokens::get(&ctx).bevel && drop_shadow {
            painter.add(egui::Shadow { offset: [0, 8], blur: 28, spread: 0, color: Color32::from_black_alpha(150) }.as_shape(img_rect, 0));
        }
        let k = frame.factor as f32;
        let params = crate::gpu_canvas::ViewParams {
            doc: frame.key,
            doc_size: [frame.area.width(), frame.area.height()],
            zoom: view.zoom * k,
            center: [view.center[0] / k - frame.area.x0 as f32, view.center[1] / k - frame.area.y0 as f32],
            shadow: false,
            pixel_grid,
            view_key: egui::Id::new(("pc-canvas-viewport", ctx.viewport_id(), idx)).value(),
            display: sync_display_lut(app, &doc, frame.key),
            hdr: hdr_preview(app, &doc),
        };
        crate::gpu_canvas::GpuCanvas::paint(&painter, rect, params);
    } else if std::env::var("PHOTOCRAFT_VIEW_COMPOSE").as_deref() == Ok("0")
        && !flip
        && ensure_gpu(app, idx, visible_doc_rect(&xf), view.zoom * ctx.pixels_per_point())
    {
        if app.gpu.as_ref().is_some_and(|gpu| gpu.pending(doc.id.0)) {
            ctx.request_repaint();
        }
        on_gpu = true;
        app.perf.gpu = true;
        // Shadow, checkerboard, document and pixel grid in one custom shader (gpu_canvas.rs).
        let params = crate::gpu_canvas::ViewParams {
            doc: doc.id.0,
            doc_size: [doc.size.width, doc.size.height],
            zoom: view.zoom,
            center: view.center,
            shadow: {
                let t = crate::theme::Tokens::get(&ctx);
                !t.bevel && !t.pro && drop_shadow
            },
            pixel_grid,
            view_key: egui::Id::new(("pc-canvas", ctx.viewport_id(), idx)).value(),
            display: sync_display_lut(app, &doc, doc.id.0),
            hdr: hdr_preview(app, &doc),
        };
        crate::gpu_canvas::GpuCanvas::paint(&painter, rect, params);
    } else {
        if !crate::theme::Tokens::get(&ctx).bevel && drop_shadow {
            painter.add(egui::Shadow { offset: [0, 8], blur: 28, spread: 0, color: Color32::from_black_alpha(150) }.as_shape(img_rect, 0));
        }
        match app.session.prefs().transparency_and_gamut.square() {
            Some(square) => {
                let checker_id = checker(app, &ctx);
                let tiles = img_rect.size() / (2.0 * square);
                painter.image(checker_id, img_rect, Rect::from_min_max(Pos2::ZERO, pos2(tiles.x, tiles.y)), Color32::WHITE);
            }
            None => {
                painter.rect_filled(img_rect, 0.0, Color32::WHITE);
            }
        }
        if let Some((tex, area)) = ensure_view_texture(app, &ctx, idx, &xf, ui.id().with(ctx.viewport_id())) {
            let uv = if flip { Rect::from_min_max(pos2(1.0, 0.0), pos2(0.0, 1.0)) } else { Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)) };
            painter.image(tex, xf.doc_rect(area), uv, Color32::WHITE);
        } else if std::env::var("PHOTOCRAFT_VIEW_COMPOSE").as_deref() == Ok("0")
            && let Some((tex, _scale)) = ensure_texture(app, &ctx, idx)
        {
            let uv = if flip { Rect::from_min_max(pos2(1.0, 0.0), pos2(0.0, 1.0)) } else { Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)) };
            painter.image(tex, img_rect, uv, Color32::WHITE);
        }
    }
    // Channels panel: alpha / Quick Mask overlays and single-channel views (channel_view.rs).
    if let Some(tex) = crate::channel_view::ensure(app, &ctx, idx) {
        let uv = if flip { Rect::from_min_max(pos2(1.0, 0.0), pos2(0.0, 1.0)) } else { Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)) };
        painter.image(tex, img_rect, uv, Color32::WHITE);
    }
    // Artboards: pasteboard between the boards, outlines and names (artboard_ui.rs).
    if doc.has_artboards() {
        let t = crate::theme::Tokens::get(&ctx);
        let pasteboard = crate::prefs_ui::pasteboard_color(app);
        let dot_tex = dots(&ctx, &t);
        for r in crate::artboard_ui::pasteboard_rects(&xf, &doc) {
            let r = r.intersect(rect);
            if !r.is_positive() {
                continue;
            }
            match pasteboard {
                Some(c) => {
                    painter.rect_filled(r, 0.0, c);
                }
                None => {
                    painter.rect_filled(r, 0.0, t.canvas);
                    if let Some(id) = dot_tex {
                        // Same phase as the dots around the document.
                        let uv = Rect::from_min_max(((r.min - rect.min) / 22.0).to_pos2(), ((r.max - rect.min) / 22.0).to_pos2());
                        painter.image(id, r, uv, Color32::WHITE);
                    }
                }
            }
        }
        if primary {
            crate::artboard_ui::draw_frames(app, &painter, &xf);
        }
    }

    // Pixel grid at high zoom (the GPU path draws its own).
    if !on_gpu && pixel_grid && view.zoom >= 12.0 {
        let vis = img_rect.intersect(rect);
        let a = xf.to_doc(vis.min);
        let b = xf.to_doc(vis.max);
        let grid = Stroke::new(1.0, Color32::from_white_alpha(40));
        for x in (a[0].floor() as i32)..=(b[0].ceil() as i32) {
            let sx = xf.to_screen(x as f32, 0.0).x;
            painter.line_segment([pos2(sx, vis.top()), pos2(sx, vis.bottom())], grid);
        }
        for y in (a[1].floor() as i32)..=(b[1].ceil() as i32) {
            let sy = xf.to_screen(0.0, y as f32).y;
            painter.line_segment([pos2(vis.left(), sy), pos2(vis.right(), sy)], grid);
        }
    }

    // View › Show › Layer Edges: the active layer's content bounds.
    if app.ui.view.shows(app.ui.view.show.layer_edges)
        && let Some(st) = app.session.documents().get(idx)
        && let Some(b) = st.active_layer.and_then(|id| st.doc.layer(id)).and_then(|l| l.surface()).map(photocraft_compose::bounds::content_bounds)
        && !b.is_empty()
    {
        painter.rect_stroke(xf.doc_rect(b), 0, Stroke::new(1.0, Color32::from_rgb(0x2d, 0x8c, 0xeb)), egui::StrokeKind::Outside);
    }

    // Selection outline: true boundary, animated marching ants (cached per revision).
    if let Some(sel) = doc.selection.as_ref().filter(|_| app.ui.view.shows(app.ui.view.show.selection_edges)) {
        // Trace at display resolution over the visible part only; key by the mask's tile identity
        // (not the document revision) so unrelated edits don't re-trace it.
        let step = (1.0 / view.zoom.max(1e-3)).log2().floor().exp2().clamp(1.0, 64.0) as u32;
        let tl = xf.to_doc(rect.min);
        let br = xf.to_doc(rect.max);
        let q = 256 * step as i32; // quantise the region so small pans reuse the cache
        let vis = photocraft_geom::Rect::new(
            (tl[0].floor() as i32).div_euclid(q) * q - q,
            (tl[1].floor() as i32).div_euclid(q) * q - q,
            ((br[0].ceil() as i32).div_euclid(q) + 2) * q,
            ((br[1].ceil() as i32).div_euclid(q) + 2) * q,
        );
        let key = crate::surface_fingerprint(sel)
            ^ (step as u64) << 56
            ^ doc.id.0.rotate_left(17)
            ^ (vis.x0 as u64) << 8
            ^ (vis.y0 as u64) << 24
            ^ (vis.x1 as u64) << 36
            ^ (vis.y1 as u64) << 48;
        let fresh = matches!(&app.outline_cache, Some((d, k, _)) if *d == doc.id && *k == key);
        if !fresh {
            let t0 = crate::gpu_canvas::now_ms();
            let b = app.cached_bounds(u64::MAX - doc.id.0, sel).intersect(&vis);
            let segs = crate::outline::outline_scaled(sel, b, step);
            app.perf.span("outline", crate::gpu_canvas::now_ms() - t0);
            app.outline_cache = Some((doc.id, key, std::sync::Arc::new(segs)));
        }
        if let Some((_, _, segs)) = &app.outline_cache {
            let time = ui.input(|i| i.time);
            marching_ants_segments(&painter, &xf, segs, time);
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    // Under an open dialog the canvas widget is inert, but the image still pans and zooms.
    let under_dialog = !app.ui.dialogs.is_empty();
    let free_hover = under_dialog && crate::dialogs::free_pointer_over(&ctx, rect).is_some();
    // Navigation: scroll pans, pinch / ⌘-scroll zooms around the pointer.
    if response.hovered() || free_hover {
        let (scroll, zoom_delta, pointer) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta(), i.pointer.hover_pos()));
        if zoom_delta != 1.0
            && let Some(p) = pointer
        {
            let nz = (view.zoom * zoom_delta).clamp(0.01, 64.0);
            zoom_about(&mut view, &xf, p, nz);
        } else if scroll.y != 0.0
            && app.session.prefs().general.zoom_with_scroll_wheel
            && let Some(p) = pointer
        {
            // Preferences › General › Zoom with Scroll Wheel.
            let nz = (view.zoom * (scroll.y / 200.0).exp()).clamp(0.01, 64.0);
            zoom_about(&mut view, &xf, p, nz);
        } else if scroll != Vec2::ZERO {
            view.center[0] -= scroll.x / view.zoom * if flip { -1.0 } else { 1.0 };
            view.center[1] -= scroll.y / view.zoom;
        }
    }

    // Pen pressure/tilt for this frame's tool events (mouse = 1.0).
    app.stylus.update(&ui.input(|i| i.events.clone()));
    let space_down = ui.input(|i| i.key_down(egui::Key::Space));
    // Space while drawing a crop frame moves it instead of panning.
    crate::crop_ui::set_space(app, space_down);
    let space_pan = space_down && !crate::crop_ui::active(app);
    let middle = ui.input(|i| i.pointer.middle_down());
    let tool = if space_pan || middle { Tool::Hand } else { app.ui.tool };

    if under_dialog {
        if let Some(d) = crate::dialogs::pan_delta(&ctx, rect, app.ui.tool == Tool::Hand) {
            view.center[0] -= d.x / view.zoom * if flip { -1.0 } else { 1.0 };
            view.center[1] -= d.y / view.zoom;
            ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
        } else if free_hover && (space_pan || app.ui.tool == Tool::Hand) {
            ctx.set_cursor_icon(egui::CursorIcon::Grab);
        }
    }
    if tool == Tool::Hand && response.dragged() {
        let d = response.drag_delta();
        view.center[0] -= d.x / view.zoom * if flip { -1.0 } else { 1.0 };
        view.center[1] -= d.y / view.zoom;
    } else if primary {
        let mods = ui.input(|i| i.modifiers);
        // Tools follow the left button; the right one opens the Brush Preset picker or erases
        // (Preferences › Tools, `paint_mouse`).
        crate::paint_mouse::sync_tool_smoothing(app);
        let mut buttons = crate::paint_mouse::canvas_buttons(app, &response, tool);
        // Zoom tool drags: scrubby zoom or a zoom rectangle (zoom_tool.rs); clicks step below.
        if tool == Tool::Zoom && crate::zoom_tool::drag(app, &ctx, &mut view, &xf, &buttons, response.interact_pointer_pos()) {
            (buttons.started, buttons.dragged, buttons.stopped) = (false, false, false);
        }
        // A drag is only recognised once the pointer has moved past egui's click distance: the
        // gesture starts where the button went down, not where it is now (#123).
        if buttons.started
            && let Some(p) = ui.input(|i| i.pointer.press_origin()).filter(|p| rect.contains(*p)).or(response.interact_pointer_pos())
        {
            if tool == Tool::Move && app.ui.transform.is_none() {
                begin_transform_controls_at(app, &ctx, &xf, p);
            }
            let d = xf.to_doc(p);
            tool_event(app, ToolEvent::Down { x: d[0], y: d[1], pressure: app.stylus.pressure() }, mods);
        }
        if buttons.dragged
            && let Some(p) = response.interact_pointer_pos()
        {
            let d = xf.to_doc(p);
            tool_event(app, ToolEvent::Move { x: d[0], y: d[1], pressure: app.stylus.pressure() }, mods);
        }
        if buttons.stopped {
            let p = response.interact_pointer_pos().map(|p| xf.to_doc(p)).or_else(|| app.drag.as_ref().and_then(|d| d.points.last().map(|q| [q[0], q[1]])));
            if let Some(d) = p {
                tool_event(app, ToolEvent::Up { x: d[0], y: d[1] }, mods);
            }
        }
        if buttons.clicked
            && let Some(p) = response.interact_pointer_pos()
        {
            let d = xf.to_doc(p);
            match tool {
                Tool::Zoom => {
                    let nz = zoom_step(view.zoom, if mods.alt { -1 } else { 1 });
                    zoom_about(&mut view, &xf, p, nz);
                }
                _ => {
                    if tool == Tool::Move && app.ui.transform.is_none() {
                        begin_transform_controls_at(app, &ctx, &xf, p);
                    }
                    tool_event(app, ToolEvent::Down { x: d[0], y: d[1], pressure: 1.0 }, mods);
                    tool_event(app, ToolEvent::Up { x: d[0], y: d[1] }, mods);
                }
            }
        }
        if app.ui.transform.is_some() && response.double_clicked() {
            crate::transform_tool::commit(app);
        }
        // A double-click closes the polygonal lasso (its second press added a point there).
        if tool == Tool::PolygonLasso && response.double_clicked() && app.ui.polygon.len() >= 4 {
            app.ui.polygon.pop();
            commit_polygon(app, ui.input(|i| i.modifiers));
        }
        if tool == Tool::Linework && response.double_clicked() {
            crate::linework_ui::finish_double_click(app);
        }
        if tool == Tool::Type && response.double_clicked() {
            crate::type_tool::select_word(app);
        }
        if app.ui.extras.grid && app.ui.view.extras {
            crate::rulers::draw_grid(app, &painter, &xf, &doc);
        }
        if app.ui.view.shows(app.ui.view.show.canvas_guides) {
            crate::rulers::draw_guides(app, &painter, &xf, &doc);
        }
        draw_drag_preview(app, &painter, &xf);
        crate::zoom_tool::draw(&ctx, &painter);
        draw_transform_controls(app, &painter, &xf);
        crate::paint_mouse::show_picker(app, &ctx);
        crate::snap_ui::draw(app, &painter, &xf);
        if border == photocraft_engine::prefs::CanvasBorder::Line {
            painter.rect_stroke(img_rect, 0.0, Stroke::new(1.0, Color32::from_gray(20)), egui::StrokeKind::Outside);
        }
        crate::type_tool::draw_overlay(app, &painter, &xf);
        crate::transform_tool::draw_overlay(app, &painter, &xf);
        crate::distort_ui::draw_overlay(app, &painter, &xf);
        crate::retouch_ui::draw_source_marker(app, &painter, &xf);
        crate::vector_ui::draw_overlay(app, &painter, &xf, &doc);
        crate::analysis_ui::draw_overlay(app, &painter, &xf);
        crate::slice_ui::draw_overlay(app, &painter, &xf);
        crate::linework_ui::draw_overlay(app, &painter, &xf);
        crate::linework_ui::keys(app, &ctx);
        crate::sai_ui::follow_role(app);
        // Tool cursors (Photoshop-style).
        let guide_hover = response.hover_pos().filter(|_| tool == Tool::Move).and_then(|p| {
            let d = xf.to_doc(p);
            crate::rulers::guide_at(app, d[0], d[1])
        });
        if let Some((vertical, _)) = guide_hover {
            ui.ctx().set_cursor_icon(if vertical { egui::CursorIcon::ResizeHorizontal } else { egui::CursorIcon::ResizeVertical });
        } else if let Some(c) = response.hover_pos().and_then(|p| crate::transform_tool::cursor(app, xf.to_doc(p))) {
            ui.ctx().set_cursor_icon(c);
        } else if let Some(c) = response.hover_pos().filter(|_| tool == Tool::Crop).and_then(|p| crate::crop_ui::cursor(app, xf.to_doc(p))) {
            ui.ctx().set_cursor_icon(c);
        } else if let Some(p) = response.hover_pos() {
            let alt = ui.input(|i| i.modifiers.alt);
            let icon = match tool {
                t if t.is_brushlike() || t == Tool::QuickSelection => {
                    // Preferences › Cursors: brush tip outline (normal = the 50% contour, or
                    // full size), precise crosshair, or the standard pointer.
                    use photocraft_engine::prefs::PaintingCursor;
                    let cur = app.session.prefs().cursors.clone();
                    let painting = app.drag.is_some();
                    let brush = &app.session.tools.brush;
                    let full = (brush.size / 2.0 * view.zoom).max(1.0);
                    let r = if cur.painting == PaintingCursor::NormalTip { (full * (0.5 + 0.5 * brush.hardness.clamp(0.0, 1.0))).max(1.0) } else { full };
                    let crosshair = |len: f32| {
                        for (w, c) in [(2.5, Color32::from_black_alpha(140)), (1.0, Color32::from_white_alpha(220))] {
                            painter.line_segment([p - vec2(len, 0.0), p + vec2(len, 0.0)], Stroke::new(w, c));
                            painter.line_segment([p - vec2(0.0, len), p + vec2(0.0, len)], Stroke::new(w, c));
                        }
                    };
                    match cur.painting {
                        PaintingCursor::Standard => egui::CursorIcon::Default,
                        PaintingCursor::Precise => {
                            crosshair(6.0);
                            egui::CursorIcon::None
                        }
                        _ if painting && cur.show_only_crosshair_while_painting => {
                            crosshair(5.0);
                            egui::CursorIcon::None
                        }
                        _ => {
                            painter.circle_stroke(p, r + 0.5, Stroke::new(1.0, Color32::from_black_alpha(140)));
                            painter.circle_stroke(p, r, Stroke::new(1.0, Color32::from_white_alpha(220)));
                            // The Background Eraser always shows its sampling hotspot (Photoshop).
                            // Quick Selection shows its +/− badge there instead.
                            if (cur.show_crosshair_in_brush_tip || r > 6.0 || tool == Tool::BackgroundEraser) && tool != Tool::QuickSelection {
                                crosshair(3.0);
                            }
                            egui::CursorIcon::None
                        }
                    }
                }
                // Preferences › Cursors › Other Cursors: Precise shows a crosshair for every tool.
                Tool::Move | Tool::Type | Tool::Eyedropper if app.session.prefs().cursors.other == photocraft_engine::prefs::OtherCursor::Precise => {
                    egui::CursorIcon::Crosshair
                }
                Tool::Move => egui::CursorIcon::Move,
                Tool::Hand => {
                    if response.dragged() {
                        egui::CursorIcon::Grabbing
                    } else {
                        egui::CursorIcon::Grab
                    }
                }
                Tool::Zoom => {
                    if alt {
                        egui::CursorIcon::ZoomOut
                    } else {
                        egui::CursorIcon::ZoomIn
                    }
                }
                Tool::Type => egui::CursorIcon::Text,
                _ => egui::CursorIcon::Crosshair,
            };
            ui.ctx().set_cursor_icon(icon);
            // Selection tools: + / − / × badge for the effective mode (#170). A gesture keeps the
            // mode it started with (⇧ then constrains the marquee instead of adding).
            let held = crate::workspace_ui::sticky_mods(app, ui.input(|i| i.modifiers));
            let mods = app.drag.as_ref().map_or(held, |d| d.modifiers);
            if let Some(b) = crate::tool_feedback::badge(app, tool, mods) {
                crate::tool_feedback::draw_badge(&painter, p, b, tool == Tool::QuickSelection);
            }
        }
    }
    if primary {
        app.hover_doc = response.hover_pos().map(|p| xf.to_doc(p));
        if app.ui.tool == Tool::MagneticLasso && app.drag.is_none() {
            let p = app.hover_doc;
            crate::magnetic_ui::hover_at(app, p);
        } else {
            app.magnetic = None;
        }
    }
    if primary && app.ui.extras.rulers {
        crate::rulers::draw_rulers(app, ui, full, &xf);
    }
    if primary {
        app.ui.views[idx] = view.clone();
    }
    view
}

fn zoom_about(view: &mut View, xf: &ViewXform, p: Pos2, new_zoom: f32) {
    let before = xf.to_doc(p);
    view.zoom = new_zoom;
    let d = (p - xf.rect.center()) / new_zoom;
    let dx = if xf.flip { -d.x } else { d.x };
    view.center = [before[0] as f32 - dx, before[1] as f32 - d.y];
}

/// Draw boundary segments as marching ants: white base, black dashes phased along x + y.
fn marching_ants_segments(painter: &egui::Painter, xf: &ViewXform, segs: &[crate::outline::Segment], time: f64) {
    let dash = 4.0f32;
    let phase = ((time * 10.0) % (dash as f64 * 2.0)) as f32;
    let white = Stroke::new(1.0, Color32::WHITE);
    let black = Stroke::new(1.0, Color32::BLACK);
    let clip = painter.clip_rect();
    for (a, b) in segs {
        let pa = xf.to_screen(a[0] as f32, a[1] as f32);
        let pb = xf.to_screen(b[0] as f32, b[1] as f32);
        if !clip.intersects(Rect::from_two_pos(pa, pb).expand(1.0)) {
            continue;
        }
        let pa = pos2(pa.x.round() + 0.5, pa.y.round() + 0.5);
        let pb = pos2(pb.x.round() + 0.5, pb.y.round() + 0.5);
        painter.line_segment([pa, pb], white);
        let len = pa.distance(pb);
        if len < 0.5 {
            continue;
        }
        let dir = (pb - pa) / len;
        // Phase by screen position so dashes line up across joined segments.
        let start = (pa.x + pa.y + phase).rem_euclid(dash * 2.0);
        let mut t = -start;
        while t < len {
            let s0 = t.max(0.0);
            let s1 = (t + dash).min(len);
            if s1 > s0 {
                painter.line_segment([pa + dir * s0, pa + dir * s1], black);
            }
            t += dash * 2.0;
        }
    }
}

#[allow(dead_code)]
fn marching_ants(painter: &egui::Painter, r: Rect, time: f64) {
    let dash = 4.0;
    let offset = ((time * 8.0) % (dash as f64 * 2.0)) as f32;
    painter.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::WHITE), egui::StrokeKind::Middle);
    let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = a.distance(b);
        let dir = (b - a) / len.max(1e-3);
        let mut t = -offset;
        while t < len {
            let s = t.max(0.0);
            let e = (t + dash).min(len);
            if e > s {
                painter.line_segment([a + dir * s, a + dir * e], Stroke::new(1.0, Color32::BLACK));
            }
            t += dash * 2.0;
        }
    }
}

/// Photoshop crop overlay: dimmed outside, bright frame, rule-of-thirds grid, corner and edge handles.
fn crop_overlay(painter: &egui::Painter, r: Rect) {
    let clip = painter.clip_rect();
    let dim = Color32::from_black_alpha(130);
    for band in [
        Rect::from_min_max(clip.min, egui::pos2(clip.max.x, r.min.y)),
        Rect::from_min_max(egui::pos2(clip.min.x, r.max.y), clip.max),
        Rect::from_min_max(egui::pos2(clip.min.x, r.min.y), egui::pos2(r.min.x, r.max.y)),
        Rect::from_min_max(egui::pos2(r.max.x, r.min.y), egui::pos2(clip.max.x, r.max.y)),
    ] {
        painter.rect_filled(band, 0.0, dim);
    }
    painter.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::WHITE), egui::StrokeKind::Middle);
    let thin = Stroke::new(1.0, Color32::from_white_alpha(90));
    for i in 1..3 {
        let fx = r.left() + r.width() * i as f32 / 3.0;
        let fy = r.top() + r.height() * i as f32 / 3.0;
        painter.line_segment([egui::pos2(fx, r.top()), egui::pos2(fx, r.bottom())], thin);
        painter.line_segment([egui::pos2(r.left(), fy), egui::pos2(r.right(), fy)], thin);
    }
    let h = Stroke::new(3.0, Color32::WHITE);
    let l = 14.0f32.min(r.width() / 3.0).min(r.height() / 3.0);
    for (c, dx, dy) in [(r.left_top(), 1.0, 1.0), (r.right_top(), -1.0, 1.0), (r.left_bottom(), 1.0, -1.0), (r.right_bottom(), -1.0, -1.0)] {
        painter.line_segment([c, c + vec2(l * dx, 0.0)], h);
        painter.line_segment([c, c + vec2(0.0, l * dy)], h);
    }
    // Edge handles: short bars at the middle of each side.
    let (lx, ly) = (l.min(r.width() / 4.0) / 2.0, l.min(r.height() / 4.0) / 2.0);
    for (c, horizontal) in [(r.center_top(), true), (r.center_bottom(), true), (r.left_center(), false), (r.right_center(), false)] {
        let e = if horizontal { vec2(lx, 0.0) } else { vec2(0.0, ly) };
        painter.line_segment([c - e, c + e], h);
    }
}

/// Overlays that persist between gestures: polygonal lasso in progress, pending crop box.
fn draw_tool_state(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, hover: Option<Pos2>) {
    if !app.ui.polygon.is_empty() {
        let mut pts: Vec<Pos2> = app.ui.polygon.iter().map(|p| xf.to_screen(p[0] as f32, p[1] as f32)).collect();
        if let Some(h) = hover {
            pts.push(h);
        }
        crate::tool_feedback::draw_ants(painter, &pts, false);
        for p in pts.iter().take(app.ui.polygon.len()) {
            painter.rect_filled(Rect::from_center_size(*p, vec2(5.0, 5.0)), 0.0, Color32::WHITE);
            painter.rect_stroke(Rect::from_center_size(*p, vec2(5.0, 5.0)), 0.0, Stroke::new(1.0, Color32::BLACK), egui::StrokeKind::Outside);
        }
    }
    if let Some(c) = app.ui.crop_rect {
        let r = Rect::from_two_pos(xf.to_screen(c[0] as f32, c[1] as f32), xf.to_screen(c[2] as f32, c[3] as f32));
        crop_overlay(painter, r);
    }
    crate::magnetic_ui::draw(app, painter, xf);
}

/// Move tool › Show Transform Controls: the active layer's transform bounds.
fn transform_controls_rect(app: &PhotocraftApp, xf: &ViewXform) -> Option<Rect> {
    if app.ui.tool != Tool::Move || !app.ui.tool_options.move_show_transform || app.ui.transform.is_some() || app.drag.is_some() {
        return None;
    }
    let st = app.session.active()?;
    let l = st.active_layer.and_then(|id| st.doc.layer(id))?;
    if crate::doc_props_ui::is_background(&st.doc, l) {
        return None;
    }
    let b = photocraft_engine::transform_cmds::transform_bounds(&st.doc, l);
    if b.is_empty() {
        return None;
    }
    Some(Rect::from_two_pos(xf.to_screen(b.x0 as f32, b.y0 as f32), xf.to_screen(b.x1 as f32, b.y1 as f32)))
}

/// A visible handle starts scaling; the narrow band just outside the box starts rotation.
/// The interior stays the normal Move-tool drag target.
fn transform_controls_hit(r: Rect, p: Pos2) -> bool {
    let handles = [r.left_top(), r.center_top(), r.right_top(), r.right_center(), r.right_bottom(), r.center_bottom(), r.left_bottom(), r.left_center()];
    handles.iter().any(|h| h.distance(p) <= 8.0) || (r.expand(18.0).contains(p) && !r.expand(5.0).contains(p))
}

/// Enter the existing Free Transform session when a Move-tool transform control is pressed.
fn begin_transform_controls_at(app: &mut PhotocraftApp, ctx: &egui::Context, xf: &ViewXform, p: Pos2) -> bool {
    let Some(r) = transform_controls_rect(app, xf) else { return false };
    if !transform_controls_hit(r, p) {
        return false;
    }
    match crate::transform_tool::begin(app, ctx) {
        Ok(()) => true,
        Err(e) => {
            app.ui.status = e;
            false
        }
    }
}

/// Move tool › Show Transform Controls: the active layer's bounding box with its eight handles.
fn draw_transform_controls(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(r) = transform_controls_rect(app, xf) else { return };
    let accent = crate::theme::Tokens::get(painter.ctx()).accent;
    painter.rect_stroke(r, 0.0, Stroke::new(1.0, accent), egui::StrokeKind::Middle);
    for p in [r.left_top(), r.center_top(), r.right_top(), r.right_center(), r.right_bottom(), r.center_bottom(), r.left_bottom(), r.left_center()] {
        let h = Rect::from_center_size(p, vec2(7.0, 7.0));
        painter.rect_filled(h, 0.0, Color32::WHITE);
        painter.rect_stroke(h, 0.0, Stroke::new(1.0, accent), egui::StrokeKind::Inside);
    }
}

fn draw_drag_preview(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    draw_tool_state(app, painter, xf, painter.ctx().input(|i| i.pointer.hover_pos()));
    let Some(d) = &app.drag else { return };
    let mut last = d.points.last().map(|p| [p[0], p[1]]).unwrap_or(d.start);
    if matches!(d.tool, Tool::RectMarquee | Tool::EllipseMarquee) {
        let o = &app.ui.tool_options;
        last = crate::chrome_ui::marquee_end(&o.marquee_style, o.marquee_width as f64, o.marquee_height as f64, false, d.start, last);
    }
    match d.tool {
        // The canvas shows the live stroke itself (`LiveStroke`).
        Tool::Brush | Tool::PixelBrush | Tool::SaiBrush | Tool::Eraser | Tool::Plugin => {}
        t if t.is_brushlike() || t == Tool::QuickSelection => {
            // Retouching strokes preview as a translucent trail of the brush footprint.
            let pts: Vec<Pos2> = d.points.iter().map(|p| xf.to_screen(p[0] as f32, p[1] as f32)).collect();
            let w = (app.session.tools.brush.size * xf.zoom).max(1.0);
            let col = Color32::from_white_alpha(if t == Tool::QuickSelection { 40 } else { 60 });
            if pts.len() == 1 {
                painter.circle_filled(pts[0], w / 2.0, col);
            } else {
                painter.add(egui::Shape::line(pts, Stroke::new(w, col)));
            }
        }
        t if crate::vector_ui::is_shape_tool(t) => crate::vector_ui::draw_shape_preview(app, painter, xf, t, d.start, last, d.modifiers),
        Tool::ObjectSelection if app.ui.tool_options.object_lasso => {
            let pts: Vec<Pos2> = std::iter::once(d.start).chain(d.points.iter().map(|p| [p[0], p[1]])).map(|p| xf.to_screen(p[0] as f32, p[1] as f32)).collect();
            crate::tool_feedback::draw_ants(painter, &pts, true);
        }
        Tool::RectMarquee | Tool::EllipseMarquee | Tool::ObjectSelection => {
            // Marching ants, visible on any pixels (#172).
            let r = Rect::from_two_pos(xf.to_screen(d.start[0] as f32, d.start[1] as f32), xf.to_screen(last[0] as f32, last[1] as f32));
            let r = Rect::from_min_max(r.min.round() + vec2(0.5, 0.5), r.max.round() + vec2(0.5, 0.5));
            let pts = if d.tool == Tool::EllipseMarquee {
                crate::tool_feedback::ellipse_points(r)
            } else {
                vec![r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom()]
            };
            crate::tool_feedback::draw_ants(painter, &pts, true);
        }
        Tool::Lasso => {
            let pts: Vec<Pos2> = d.points.iter().map(|p| xf.to_screen(p[0] as f32, p[1] as f32)).collect();
            crate::tool_feedback::draw_ants(painter, &pts, false);
        }
        Tool::Gradient => {
            let a = xf.to_screen(d.start[0] as f32, d.start[1] as f32);
            let b = xf.to_screen(last[0] as f32, last[1] as f32);
            painter.line_segment([a, b], Stroke::new(3.0, Color32::from_black_alpha(140)));
            painter.line_segment([a, b], Stroke::new(1.0, Color32::WHITE));
            painter.circle_filled(a, 3.0, Color32::WHITE);
            painter.circle_filled(b, 3.0, Color32::WHITE);
        }
        Tool::Type => {
            let r = Rect::from_two_pos(xf.to_screen(d.start[0] as f32, d.start[1] as f32), xf.to_screen(last[0] as f32, last[1] as f32));
            let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
            painter.add(egui::Shape::line(pts.to_vec(), Stroke::new(1.0, Color32::WHITE)));
            painter.add(egui::Shape::dashed_line(&pts, Stroke::new(1.0, Color32::BLACK), 3.0, 3.0));
        }
        Tool::Move => {
            let off = vec2(((last[0] - d.start[0]) as f32) * xf.zoom, ((last[1] - d.start[1]) as f32) * xf.zoom);
            painter.arrow(xf.to_screen(d.start[0] as f32, d.start[1] as f32), off, Stroke::new(2.0, crate::theme::Tokens::get(painter.ctx()).accent));
        }
        _ => {}
    }
}

fn sample_eyedropper(app: &mut PhotocraftApp, x: f64, y: f64, mods: egui::Modifiers) {
    if let Ok(v) = app.run("document.pixel", json!({"x": x.floor(), "y": y.floor()})) {
        let color: Vec<f32> = serde_json::from_value(v).unwrap_or_default();
        if color.len() == 4 && color[3] > 0.0 {
            let key = if mods.alt { "background" } else { "foreground" };
            let _ = app.run("tools.setColors", json!({ key: [color[0], color[1], color[2], 1.0] }));
        }
    }
}

/// Tool state machine. Shared by mouse input and automation.
pub fn tool_event(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) {
    if app.session.active().is_some_and(|st| st.busy) && !matches!(app.ui.tool, Tool::Hand | Tool::Zoom | Tool::Eyedropper) {
        app.drag = None;
        app.live_stroke = None;
        app.ui.status = "Document is busy; pan and zoom remain available".into();
        return;
    }
    // View › Snap / Snap To and smart guides (snap_ui.rs).
    let ev = crate::snap_ui::filter_event(app, ev, mods);
    if crate::transform_tool::pointer(app, ev, mods) {
        return;
    }
    if crate::distort_ui::pointer(app, ev, mods) {
        return;
    }
    // Window › Modifier Keys: sticky Shift/⌘/⌥ act as held keys.
    let mods = crate::workspace_ui::sticky_mods(app, mods);
    // Move tool: ⇧ locks the axis, ⌥ duplicates (move_mods.rs).
    let ev = crate::move_mods::filter_event(app, ev, mods);
    // Ruler, Count and Note tools.
    if crate::analysis_ui::pointer(app, ev, mods) {
        return;
    }
    // Slice and Slice Select tools.
    if crate::linework_ui::pointer(app, ev, mods) {
        return;
    }
    if crate::slice_ui::pointer(app, ev, mods) {
        return;
    }
    // Crop tool: draw, move and resize the frame.
    if crate::crop_ui::pointer(app, ev, mods) {
        return;
    }
    let tool = app.ui.tool;
    if tool == Tool::MagneticLasso {
        let mode = selection_mode(app, mods);
        crate::magnetic_ui::pointer(app, ev, mode);
        return;
    }
    if tool == Tool::Eyedropper {
        match ev {
            ToolEvent::Down { x, y, .. } | ToolEvent::Move { x, y, .. } => {
                sample_eyedropper(app, x, y, mods);
                return;
            }
            ToolEvent::Up { .. } => return,
        }
    }
    // Move tool over a guide drags the guide (off the canvas deletes it).
    match ev {
        ToolEvent::Down { x, y, .. } if tool == Tool::Move => {
            if let Some((vertical, i)) = crate::rulers::guide_at(app, x, y) {
                app.guide_drag = Some(crate::rulers::GuideDrag { vertical, index: Some(i), pos: if vertical { x } else { y } });
                return;
            }
            // Auto-Select (or ⌘-click while it is off) picks the layer under the pointer first.
            if app.ui.tool_options.move_auto_select != mods.command {
                let target = app.ui.tool_options.move_target.clone();
                let mode = if mods.shift { "add" } else { "replace" };
                let _ = app.run("layer.pickAt", json!({"x": x, "y": y, "target": target, "mode": mode}));
            }
        }
        ToolEvent::Move { x, y, .. } => {
            if let Some(d) = app.guide_drag.as_mut().filter(|d| d.index.is_some()) {
                d.pos = if d.vertical { x } else { y };
                return;
            }
        }
        ToolEvent::Up { x, y } => {
            if let Some(mut d) = app.guide_drag.filter(|d| d.index.is_some()) {
                app.guide_drag = None;
                d.pos = if d.vertical { x } else { y };
                crate::rulers::finish_drag(app, d);
                return;
            }
        }
        _ => {}
    }
    match ev {
        ToolEvent::Down { x, y, pressure } => {
            match tool {
                Tool::Pen => {
                    crate::vector_ui::pen_down(app, x, y);
                    return;
                }
                Tool::CloneStamp | Tool::Healing if mods.alt => {
                    crate::retouch_ui::set_source(app, x, y);
                    return;
                }
                Tool::MagicWand => {
                    let o = app.ui.tool_options.clone();
                    let mode = selection_mode(app, mods);
                    let _ = app.run("select.magicWand", json!({"x": x.floor(), "y": y.floor(), "tolerance": o.tolerance, "contiguous": o.contiguous, "antiAlias": o.anti_alias, "sample": o.fill_sample, "mode": mode}));
                    return;
                }
                Tool::MagicEraser => {
                    crate::eraser_ui::click(app, tool, x, y);
                    return;
                }
                Tool::PaintBucket => {
                    let o = app.ui.tool_options.clone();
                    let contents = if o.bucket_fill_pattern { "pattern" } else { "foreground" };
                    let _ = app.run("paint.bucket", json!({"x": x.floor(), "y": y.floor(), "tolerance": o.tolerance, "contiguous": o.contiguous, "antiAlias": o.anti_alias, "opacity": o.fill_opacity, "contents": contents, "sample": o.fill_sample, "expand": o.bucket_expand, "target": paint_target(app)}));
                    return;
                }
                Tool::PolygonLasso => {
                    // Click adds a vertex; clicking near the first vertex closes the polygon.
                    let close = app.ui.polygon.first().is_some_and(|p0| {
                        app.ui.polygon.len() >= 3 && ((p0[0] - x).powi(2) + (p0[1] - y).powi(2)).sqrt() < 8.0 / app.current_zoom().max(0.01) as f64
                    });
                    if close {
                        commit_polygon(app, mods);
                    } else {
                        app.ui.polygon.push([x, y]);
                    }
                    return;
                }
                Tool::Type if crate::type_tool::pointer_down(app, x, y, mods.shift) => return,
                _ => {}
            }
            if ask_rasterize(app, tool) {
                return;
            }
            crate::paint_mouse::sync_tool_smoothing(app);
            let erase = tool == Tool::Eraser || std::mem::take(&mut app.secondary_erase);
            // ⇧-click after a stroke: a straight line from where it ended (stroke_constraint.rs).
            let active = app.session.active().map(|st| st.doc.id);
            let from = app.last_stroke_end.filter(|(doc, _)| mods.shift && crate::stroke_constraint::connects(tool) && Some(*doc) == active).map(|(_, p)| p);
            let mut points = vec![[x, y, pressure as f64]];
            if let Some(p) = from {
                points.insert(0, [p[0], p[1], pressure as f64]);
            }
            app.drag = Some(Drag { tool, start: from.unwrap_or([x, y]), points, modifiers: mods, erase, constrain: None, straight_from: None });
            app.stylus.begin_stroke();
            if from.is_some() {
                app.stylus.record_point();
            }
            app.live_stroke = if matches!(tool, Tool::Brush | Tool::PixelBrush | Tool::SaiBrush | Tool::Eraser | Tool::Plugin) { begin_live_stroke(app) } else { None };
        }
        ToolEvent::Move { x, y, pressure } => {
            if tool == Tool::Type && app.drag.is_none() {
                crate::type_tool::pointer_move(app, x, y);
            }
            if tool == Tool::Pen {
                crate::vector_ui::pen_move(app, x, y);
            }
            let zoom = app.current_zoom();
            // Lasso: ⌥ pressed during the drag draws a straight segment from where it was pressed
            // (the subtract mode is chosen by ⌥ at the press only).
            if let Some(d) = app.drag.as_mut().filter(|d| d.tool == Tool::Lasso && !d.modifiers.alt) {
                if mods.alt {
                    let from = *d.straight_from.get_or_insert(d.points.len());
                    d.points.truncate(from);
                    d.points.push([x, y, pressure as f64]);
                    return;
                }
                if let Some(from) = d.straight_from.take() {
                    // Keep the segment's end as a vertex and go on freehand from it.
                    d.points.truncate(from + 1);
                }
            }
            if let Some(d) = &mut app.drag {
                // ⇧: straight 0/45/90° strokes, 45° gradient angles (stroke_constraint.rs).
                let last = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
                let [x, y] = crate::stroke_constraint::constrain(d.tool, &mut d.constrain, d.start, last, [x, y], mods.shift, zoom);
                if d.points.last().is_none_or(|p| (p[0] - x).abs() + (p[1] - y).abs() > 0.25) {
                    d.points.push([x, y, pressure as f64]);
                    app.stylus.record_point();
                }
            }
            feed_live_stroke(app);
        }
        ToolEvent::Up { x, y } => {
            if tool == Tool::Type
                && let Some(e) = app.ui.text_edit.as_mut()
            {
                e.dragging = false;
            }
            if tool == Tool::Pen {
                crate::vector_ui::pen_up(app);
            }
            let zoom = app.current_zoom();
            let Some(mut d) = app.drag.take() else { return };
            let last = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
            let [x, y] = crate::stroke_constraint::constrain(d.tool, &mut d.constrain, d.start, last, [x, y], mods.shift, zoom);
            if d.points.last().is_none_or(|p| p[0] != x || p[1] != y) {
                d.points.push([x, y, d.points.last().map_or(1.0, |p| p[2])]);
                app.stylus.record_point();
            }
            finish_gesture(app, d);
            crate::move_mods::finish(app);
        }
    }
}

fn finish_gesture(app: &mut PhotocraftApp, d: Drag) {
    let end = d.points.last().copied().unwrap_or([d.start[0], d.start[1], 1.0]);
    // Where the next ⇧-click line starts.
    if crate::stroke_constraint::connects(d.tool)
        && let Some(st) = app.session.active()
    {
        app.last_stroke_end = Some((st.doc.id, [end[0], end[1]]));
    }
    if crate::eraser_ui::finish_stroke(app, d.tool, &d.points) || crate::retouch_ui::finish_stroke(app, d.tool, &d.points, d.modifiers) {
        return;
    }
    match d.tool {
        Tool::ObjectSelection => finish_object_selection(app, &d),
        t if crate::vector_ui::is_shape_tool(t) => crate::vector_ui::finish_shape(app, t, d.start, [end[0], end[1]], d.modifiers),
        Tool::PathSelection => crate::vector_ui::path_selection_finish(app, d.start, [end[0], end[1]]),
        Tool::Type => crate::type_tool::pointer_up(app, d.start, [end[0], end[1]]),
        Tool::Brush | Tool::PixelBrush | Tool::SaiBrush | Tool::Eraser | Tool::Plugin => {
            let live = app.live_stroke.take();
            let mut p = stroke_params(app, d.erase, &app.stylus.stroke_points(&d.points));
            if let Some(l) = &live {
                p["seed"] = json!(l.stroke.seed);
            }
            // The canvas already shows the stroke: let the commit's damage rect refresh it rather
            // than recompositing the whole document.
            if app.run("paint.stroke", p).is_ok()
                && let Some(l) = live
            {
                let display_key = app.session.active().map_or(0, |st| canvas_display(app, &st.doc).1);
                if let Some(c) = app.canvases.get_mut(&l.doc) {
                    // Raw preview key 0 = the document itself (its colour display folded in).
                    if l.since(c.preview_key ^ display_key).is_some() {
                        c.preview_key = display_key;
                    }
                    if l.since(c.tex_preview_key ^ display_key).is_some() {
                        c.tex_preview_key = display_key;
                    }
                }
            }
        }
        Tool::RectMarquee | Tool::EllipseMarquee => {
            let o = &app.ui.tool_options;
            let e = crate::chrome_ui::marquee_end(&o.marquee_style, o.marquee_width as f64, o.marquee_height as f64, false, d.start, [end[0], end[1]]);
            let end = [e[0], e[1], 1.0];
            let (x0, y0) = (d.start[0].min(end[0]).floor(), d.start[1].min(end[1]).floor());
            let (x1, y1) = (d.start[0].max(end[0]).ceil(), d.start[1].max(end[1]).ceil());
            if x1 - x0 < 2.0 || y1 - y0 < 2.0 {
                if app.session.is_enabled("select.deselect") {
                    let _ = app.run("select.deselect", json!({}));
                }
                return;
            }
            let mode = selection_mode(app, d.modifiers);
            let (aa, feather) = (app.ui.tool_options.anti_alias, app.ui.tool_options.feather);
            let _ = app.run("select.rect", json!({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "mode": mode, "ellipse": d.tool == Tool::EllipseMarquee, "antiAlias": aa, "feather": feather}));
        }
        Tool::Lasso => {
            let pts: Vec<[f64; 2]> = d.points.iter().map(|p| [p[0], p[1]]).collect();
            if pts.len() >= 3 {
                let mode = selection_mode(app, d.modifiers);
                let _ = app.run("select.lasso", json!({"points": pts, "mode": mode, "antiAlias": app.ui.tool_options.anti_alias}));
            } else if app.session.is_enabled("select.deselect") {
                let _ = app.run("select.deselect", json!({}));
            }
        }
        Tool::Gradient => {
            if (end[0] - d.start[0]).abs() + (end[1] - d.start[1]).abs() >= 2.0 {
                let o = app.ui.tool_options.clone();
                let fg = app.session.tools.foreground;
                let bg = app.session.tools.background;
                let _ = app.run(
                    "paint.gradient",
                    json!({"from": [d.start[0], d.start[1]], "to": [end[0], end[1]], "style": o.gradient_style, "reverse": o.gradient_reverse, "dither": o.gradient_dither, "colors": [hex(fg), hex(bg)], "opacity": o.fill_opacity, "target": paint_target(app)}),
                );
            }
        }
        Tool::Move => {
            let (dx, dy) = ((end[0] - d.start[0]).round(), (end[1] - d.start[1]).round());
            if dx != 0.0 || dy != 0.0 {
                let _ = app.run("layer.translate", json!({"dx": dx, "dy": dy}));
            }
        }
        _ => {}
    }
}

/// Extra OS windows showing documents (multi-window / multi-monitor).
pub fn extra_windows(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let wins = app.ui.windows.clone();
    for w in wins.into_iter().filter(|w| w.open) {
        let Some(st) = app.session.documents().get(w.document) else { continue };
        let title = format!("{} — window {}", st.doc.name, w.id);
        let vid = egui::ViewportId::from_hash_of(("docwin", w.id));
        let builder = egui::ViewportBuilder::default().with_title(title).with_inner_size([800.0, 600.0]);
        let mut view = w.view.clone();
        let mut close = false;
        ctx.show_viewport_immediate(vid, builder, |ui, _class| {
            if ui.ctx().input(|i| i.viewport().close_requested()) {
                close = true;
            }
            egui::CentralPanel::default().frame(egui::Frame::NONE.fill(crate::theme::Tokens::get(ui.ctx()).canvas)).show(ui, |ui| {
                let rect = ui.available_rect_before_wrap();
                view = canvas_view(app, ui, w.document, rect, view.clone(), false);
            });
        });
        if let Some(win) = app.ui.windows.iter_mut().find(|x| x.id == w.id) {
            win.view = view;
            if close {
                win.open = false;
            }
        }
    }
    app.ui.windows.retain(|w| w.open);
}

/// Object Selection: a box (or, in Lasso mode, the outline drawn) around the object, or a click
/// on it. With the AI models installed (`ai_select`) every prompt works; without them only a box
/// does (GrabCut).
fn finish_object_selection(app: &mut PhotocraftApp, d: &Drag) {
    let end = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
    let mode = selection_mode(app, d.modifiers);
    let o = app.ui.tool_options.clone();
    let zoom = f64::from(app.current_zoom().max(0.01));
    let (mut x0, mut y0, mut x1, mut y1) = (d.start[0], d.start[1], d.start[0], d.start[1]);
    for p in &d.points {
        (x0, y0, x1, y1) = (x0.min(p[0]), y0.min(p[1]), x1.max(p[0]), y1.max(p[1]));
    }
    // A drag shorter than 3 screen pixels is a click.
    let click = (x1 - x0).max(y1 - y0) * zoom < 3.0;
    let mut p = json!({"mode": mode, "sampleAllLayers": o.sample_all_layers, "model": o.object_model});
    if click {
        p["point"] = json!([d.start[0], d.start[1]]);
    } else if o.object_lasso && d.points.len() >= 3 {
        p["lasso"] = json!(std::iter::once(d.start).chain(d.points.iter().map(|q| [q[0], q[1]])).collect::<Vec<_>>());
    } else {
        let (x, y) = (d.start[0].min(end[0]), d.start[1].min(end[1]));
        p["rect"] = json!([x.round(), y.round(), (end[0] - d.start[0]).abs().round(), (end[1] - d.start[1]).abs().round()]);
    }
    if !click || photocraft_engine::ai_select::available() {
        app.ui.status = "Object Selection…".into();
        let _ = app.run("select.object", p);
    }
}

/// Selection mode from the options bar, overridden by modifier keys (⇧ add, ⌥ subtract, ⇧⌥ intersect).
/// The cursor badge announces the same mode (`tool_feedback`).
fn selection_mode(app: &PhotocraftApp, m: egui::Modifiers) -> &'static str {
    crate::tool_feedback::selection_mode(Tool::Lasso, app.ui.selection_mode, m)
}

/// Painting, filling or a gradient on the pixels of a layer that has none of its own (a Smart
/// Object, type, shape or fill layer) asks to rasterize it first, as Photoshop does, instead of
/// silently doing nothing. OK rasterizes the layer; the next stroke then paints.
fn ask_rasterize(app: &mut PhotocraftApp, tool: Tool) -> bool {
    if !(tool.is_brushlike() || matches!(tool, Tool::Gradient | Tool::PaintBucket | Tool::MagicEraser)) || paint_target(app) != json!("pixels") {
        return false;
    }
    let Some(l) = app.session.active().and_then(|st| st.active_layer.and_then(|id| st.doc.layer(id))) else { return false };
    if l.linework.is_some() {
        let message = format!("This Linework layer must be rasterized before painting on it. Its strokes will no longer be editable.\n\nRasterize \"{}\"?", l.name);
        let mut fields = serde_json::Map::new();
        fields.insert("__command".into(), json!("layer.rasterize.linework"));
        fields.insert("__label".into(), json!("Rasterize Linework"));
        fields.insert("__message".into(), json!(message));
        fields.insert("layer".into(), json!(l.id.0));
        app.ui.open_dialog(crate::state::DialogKind::Command, fields);
        return true;
    }
    let what = match &l.content {
        LayerContent::Smart(_) => "This Smart Object",
        LayerContent::Text(_) => "This type layer",
        LayerContent::Shape(_) => "This shape layer",
        LayerContent::Fill(_) => "This fill layer",
        _ => return false,
    };
    let message = format!(
        "{what} must be rasterized before proceeding. Its contents will no longer be editable.\n\nRasterize \"{}\"?",
        l.name
    );
    let mut fields = serde_json::Map::new();
    fields.insert("__command".into(), json!("layer.rasterize.layer"));
    fields.insert("__label".into(), json!("Rasterize Layer"));
    fields.insert("__message".into(), json!(message));
    fields.insert("layer".into(), json!(l.id.0));
    app.ui.open_dialog(crate::state::DialogKind::Command, fields);
    true
}

/// Close the polygonal lasso and make the selection.
pub fn commit_polygon(app: &mut PhotocraftApp, mods: egui::Modifiers) {
    let pts = std::mem::take(&mut app.ui.polygon);
    if pts.len() >= 3 {
        let mode = selection_mode(app, mods);
        let _ = app.run("select.lasso", json!({"points": pts, "mode": mode, "antiAlias": app.ui.tool_options.anti_alias}));
    }
}

/// Apply the crop tool's rectangle.
pub fn commit_crop(app: &mut PhotocraftApp) {
    let Some(r) = app.ui.crop_rect.take() else { return };
    let (x, y) = (r[0].round(), r[1].round());
    let (w, h) = ((r[2] - r[0]).round().max(1.0), (r[3] - r[1]).round().max(1.0));
    let delete = app.ui.tool_options.crop_delete;
    if app.run("image.crop", json!({"x": x, "y": y, "width": w, "height": h, "deleteCroppedPixels": delete})).is_ok()
        && let Some(i) = app.session.active_index()
    {
        app.ui.views[i].fit_pending = true;
    }
}

/// "mask" when the Layers panel targets the active layer's mask, else "pixels".
pub fn paint_target(app: &PhotocraftApp) -> serde_json::Value {
    use photocraft_engine::channel_cmds::ChannelTarget;
    let Some(st) = app.session.active() else { return json!("pixels") };
    // A targeted alpha channel (Channels panel) or Quick Mask mode wins over the layer.
    match st.channel_view.target {
        ChannelTarget::Alpha(i) if i < st.doc.channels.len() => return json!({ "channel": i }),
        ChannelTarget::Composite if st.doc.quick_mask.is_some() => return json!("quickMask"),
        _ => {}
    }
    let has_mask = st.active_layer.and_then(|id| st.doc.layer(id)).is_some_and(|l| l.mask.is_some());
    json!(if app.ui.mask_target && has_mask { "mask" } else { "pixels" })
}

/// `#rrggbb` for an sRGB colour (the engine's colour parameter notation).
fn hex(c: [f32; 4]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_transform_roundtrip() {
        for flip in [false, true] {
            let xf = ViewXform { rect: Rect::from_min_size(pos2(100.0, 50.0), vec2(800.0, 600.0)), zoom: 2.5, center: [320.0, 240.0], flip };
            let s = xf.to_screen(10.0, 20.0);
            let d = xf.to_doc(s);
            assert!((d[0] - 10.0).abs() < 1e-3 && (d[1] - 20.0).abs() < 1e-3);
            assert_eq!(xf.to_screen(320.0, 240.0), xf.rect.center());
            // Flipped, document x grows to the left.
            assert_eq!(xf.to_screen(330.0, 240.0).x > xf.rect.center().x, !flip);
            let r = xf.doc_rect(DRect::new(0, 0, 10, 10));
            assert!(r.width() > 0.0 && r.height() > 0.0);
        }
    }

    #[test]
    fn eyedropper_drag_updates_the_sampled_colour() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 40, "height": 20, "background": "transparent"})).unwrap();
        app.run("shape.create", json!({"kind": "rect", "rect": [0, 0, 20, 20], "fill": "#ff0000"})).unwrap();
        app.run("shape.create", json!({"kind": "rect", "rect": [20, 0, 20, 20], "fill": "#00ff00"})).unwrap();
        app.ui.tool = Tool::Eyedropper;

        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, egui::Modifiers::NONE);
        assert!(app.session.tools.foreground[0] > 0.99 && app.session.tools.foreground[1] < 0.01);

        tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 10.0, pressure: 1.0 }, egui::Modifiers::NONE);
        assert!(app.session.tools.foreground[1] > 0.99 && app.session.tools.foreground[0] < 0.01);
    }

    #[test]
    fn brush_drag_shows_the_real_stroke_and_commits_it() {
        // The drag used to draw a hard, flat stand-in and only showed the soft brush on release.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 120, "height": 60, "background": "transparent"})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 20, "hardness": 0.0}})).unwrap();
        app.ui.tool = Tool::Brush;
        let m = egui::Modifiers::NONE;
        let alpha = |d: &Document, x, y| d.layers[0].surface().unwrap().rgba(x, y)[3];
        let rev = app.session.documents()[0].revision;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 30.0, pressure: 1.0 }, m);
        for x in [40.0, 70.0, 100.0] {
            tool_event(&mut app, ToolEvent::Move { x, y: 30.0, pressure: 1.0 }, m);
        }
        let (shown, key) = display_doc(&mut app, 0);
        assert_ne!(key, 0);
        assert!(alpha(&shown, 40, 30) > 0.5 && (0.01..0.5).contains(&alpha(&shown, 40, 38)), "soft stroke while drawing");
        assert_eq!(alpha(&app.session.documents()[0].doc, 40, 30), 0.0, "not committed yet");
        // The canvas redraws only what the stroke touched.
        let dk = canvas_display(&app, &app.session.documents()[0].doc).1;
        let d = damage_since(&app, 0, (rev, dk), (rev, key), dk, None).unwrap();
        assert!(d.contains(40, 30) && !d.contains(40, 2) && d.width() < 120, "{d:?}");
        tool_event(&mut app, ToolEvent::Up { x: 100.0, y: 30.0 }, m);
        let doc = app.session.documents()[0].doc.clone();
        assert!(app.live_stroke.is_none() && display_doc(&mut app, 0).1 == 0);
        assert_eq!((alpha(&doc, 40, 30), alpha(&doc, 40, 38)), (alpha(&shown, 40, 30), alpha(&shown, 40, 38)), "commit matches the preview");
    }

    #[test]
    fn live_stroke_uses_pen_tilt_and_updates_a_reduced_texture_partially() {
        // Pen tilt drives the size; the preview must use the tilt the commit gets. The document
        // is wider than MAX_TEXTURE, so the CPU texture is reduced (factor 2) and each step
        // must update only the reduced pixels it touched.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 4097, "height": 90, "background": "transparent"})).unwrap();
        let tilt = json!({"size": 30, "hardness": 1.0, "spacing": 0.05, "shapeDynamics": {"enabled": true, "size": {"control": "penTilt"}}});
        app.run("tools.setBrush", json!({ "brush": tilt })).unwrap();
        app.ui.tool = Tool::Brush;
        assert_eq!(ensure_texture(&mut app, &ctx, 0).map(|t| t.1), Some(0.5));
        let m = egui::Modifiers::NONE;
        app.stylus.feed.set(Some(crate::stylus::PenSample { pressure: 1.0, tilt_x: 60.0, tilt_y: 0.0, rotation: 0.0 }));
        tool_event(&mut app, ToolEvent::Down { x: 100.0, y: 45.0, pressure: 1.0 }, m);
        for x in [300.0, 600.0, 900.0] {
            tool_event(&mut app, ToolEvent::Move { x, y: 45.0, pressure: 1.0 }, m);
            ensure_texture(&mut app, &ctx, 0);
            assert_eq!(app.perf.last_refresh, "rect");
            assert!(app.perf.last_refresh_px < 4097 * 90 / 4, "{}", app.perf.last_refresh_px);
        }
        let shown = display_doc(&mut app, 0).0;
        tool_event(&mut app, ToolEvent::Up { x: 900.0, y: 45.0 }, m);
        ensure_texture(&mut app, &ctx, 0);
        assert_eq!(app.perf.last_refresh, "rect", "the commit refreshes only the stroke");
        let doc = app.session.documents()[0].doc.clone();
        let (a, b) = (doc.layers[0].surface().unwrap(), shown.layers[0].surface().unwrap());
        // 60° tilt shrinks the dab well below 30 px: the preview shows that size, as committed.
        assert!(a.rgba(500, 45)[3] > 0.5 && a.rgba(500, 45 + 12)[3] == 0.0);
        // (The smoothed tail catches up to the end point only when the stroke finishes.)
        assert!((0..90).all(|y| (0..700).all(|x| a.rgba(x, y) == b.rgba(x, y))), "commit matches the preview");
    }

    /// Brush drag along y = 40 with one canvas frame per pointer move; returns the document the
    /// canvas showed at the last move, the committed one, and whether the release refreshed only
    /// the stroke's rectangle.
    fn drag_frames(smoothing: f32, xs: &[f64]) -> (std::sync::Arc<Document>, std::sync::Arc<Document>, bool, PhotocraftApp) {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 200, "height": 80, "background": "transparent"})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 16, "hardness": 0.5}})).unwrap();
        // What the options bar's Smoothing field writes.
        app.session.tools.brush.smoothing.amount = smoothing;
        app.ui.tool = Tool::Brush;
        ensure_texture(&mut app, &ctx, 0);
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 40.0, pressure: 1.0 }, m);
        for &x in xs {
            tool_event(&mut app, ToolEvent::Move { x, y: 40.0 + (x / 7.0).sin() * 8.0, pressure: 1.0 }, m);
            ensure_texture(&mut app, &ctx, 0);
        }
        let live = display_doc(&mut app, 0).0;
        let last = *xs.last().unwrap();
        tool_event(&mut app, ToolEvent::Up { x: last, y: 40.0 + (last / 7.0).sin() * 8.0 }, m);
        let (next, key) = display_doc(&mut app, 0);
        assert_eq!(key, 0, "the frame after release shows the committed document");
        ensure_texture(&mut app, &ctx, 0);
        let partial = app.perf.last_refresh == "rect";
        (live, next, partial, app)
    }

    fn same_pixels(a: &Document, b: &Document) -> bool {
        let (a, b) = (a.layers[0].surface().unwrap(), b.layers[0].surface().unwrap());
        (0..80).all(|y| (0..200).all(|x| a.rgba(x, y) == b.rgba(x, y)))
    }

    #[test]
    fn release_shows_nothing_new_without_smoothing() {
        // #73: at 0 % the stroke follows the pointer; the last frame drawn while dragging is
        // exactly the committed stroke, so release changes nothing on screen.
        let (live, done, partial, app) = drag_frames(0.0, &[40.0, 70.0, 100.0, 130.0, 160.0]);
        assert!(same_pixels(&live, &done), "preview at the last move = committed stroke");
        assert!(partial, "the handover refreshes only the stroke");
        let p = app.session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").map(|(_, p)| p.clone()).unwrap();
        assert!(p.get("smoothing").is_none(), "the commit uses the session brush's smoothing, not a hard-coded one");
    }

    #[test]
    fn smoothed_stroke_shows_its_catch_up_tail_while_drawing() {
        // #73: with smoothing the brush lags behind the pointer and catches up at the end; the
        // preview draws that tail live, so no frame after release is missing the end.
        let xs = [30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0];
        let (live, done, partial, _) = drag_frames(0.5, &xs);
        assert!(same_pixels(&live, &done), "preview at the last move = committed stroke, tail included");
        assert!(partial);
        let end = |d: &Document| (0..200).rev().find(|&x| (0..80).any(|y| d.layers[0].surface().unwrap().rgba(x, y)[3] > 0.0)).unwrap();
        assert!(end(&live) >= 170, "the end reaches the pointer while drawing ({})", end(&live));
        // The options-bar value reaches the stroke: 50 % smooths the wiggle, 0 % doesn't.
        let (_, rough, _, _) = drag_frames(0.0, &xs);
        assert!(!same_pixels(&rough, &done), "smoothing changes the stroke");
        // And the commit is what `paint.stroke` gives with that smoothing.
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 200, "height": 80, "background": "transparent"})).unwrap();
        s.execute("tools.setBrush", json!({"brush": {"size": 16, "hardness": 0.5, "smoothing": {"amount": 0.5}}})).unwrap();
        let mut pts = vec![json!([10.0, 40.0, 1.0])];
        pts.extend(xs.iter().map(|&x| json!([x, 40.0 + (x / 7.0).sin() * 8.0, 1.0])));
        // (Round tips without dynamics draw the same for any seed.)
        s.execute("paint.stroke", json!({"points": pts, "seed": 0})).unwrap();
        assert!(same_pixels(&s.active().unwrap().doc, &done));
    }

    #[test]
    fn smoothing_preview_redraws_the_tail_each_step() {
        // The tail drawn at one step must not linger once the brush moves on: a sharp turn would
        // leave a stale tail behind if it weren't restored.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 200, "height": 120, "background": "transparent"})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 10, "hardness": 1.0, "smoothing": {"amount": 0.6}}})).unwrap();
        app.ui.tool = Tool::Brush;
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 20.0, pressure: 1.0 }, m);
        for (x, y) in [(60.0, 20.0), (110.0, 20.0), (110.0, 70.0), (110.0, 110.0), (60.0, 110.0)] {
            tool_event(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, m);
        }
        let live = display_doc(&mut app, 0).0;
        tool_event(&mut app, ToolEvent::Up { x: 60.0, y: 110.0 }, m);
        let done = app.session.documents()[0].doc.clone();
        let (a, b) = (live.layers[0].surface().unwrap(), done.layers[0].surface().unwrap());
        assert!((0..120).all(|y| (0..200).all(|x| a.rgba(x, y) == b.rgba(x, y))), "no stale tails in the preview");
    }

    #[test]
    fn layer_style_dialog_previews_live_and_cancel_restores() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        let fx = |d: &Document| d.layers.iter().map(|l| l.effects.items.len()).sum::<usize>();
        let id = crate::layer_style::open(&mut app, Some("colorOverlay")).unwrap();
        let (shown, key) = display_doc(&mut app, 0);
        assert_eq!((fx(&shown), fx(&app.session.documents()[0].doc)), (1, 0), "previewed, not committed");
        app.ui.dialog_mut(id).unwrap().fields.insert("on:stroke".into(), json!(true));
        let (shown, key2) = display_doc(&mut app, 0);
        assert_eq!(fx(&shown), 2);
        assert_ne!(key, key2, "an edit re-renders the canvas");
        app.ui.close_dialog(id);
        let (shown, key) = display_doc(&mut app, 0);
        assert_eq!((fx(&shown), key), (0, 0));
        assert!(app.style_preview.is_none());
    }

    #[test]
    fn zoom_steps_monotone() {
        assert_eq!(zoom_step(1.0, 1), 2.0);
        assert_eq!(zoom_step(1.0, -1), 0.6667);
        assert_eq!(zoom_step(0.4, 1), 0.5);
        assert_eq!(zoom_step(32.0, 1), 32.0);
    }

    #[test]
    fn damage_grows_by_nested_effect_reach() {
        use photocraft_doc::{Effect, Layer};
        let fmt = photocraft_color::PixelFormat::RGBA8;
        assert_eq!(effect_reach(&[Layer::raster("plain", fmt)]), 0);
        let mut inner = Layer::raster("inner", fmt);
        inner.effects.items.push(Effect::default_drop_shadow());
        let m = photocraft_compose::effects::margin(&inner);
        let mut group = Layer::group("g", vec![inner.clone()]);
        group.effects.items.push(Effect::default_drop_shadow());
        assert_eq!(effect_reach(std::slice::from_ref(&inner)), m);
        // A child's edit moves the group's shape, whose effects reach further.
        assert_eq!(effect_reach(&[group.clone()]), 2 * m);
        group.visible = false;
        assert_eq!(effect_reach(&[group]), 0);
    }
}

#[cfg(test)]
mod transform_controls_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn move_transform_controls_start_free_transform_for_vector_shapes() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 200, "height": 200})).unwrap();
        app.session.execute("shape.create", json!({"kind": "rect", "rect": [20, 30, 80, 40], "fill": "#ff0000"})).unwrap();
        app.sync_views();
        app.ui.tool = Tool::Move;
        app.ui.tool_options.move_show_transform = true;

        let xf = ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(200.0, 200.0)), zoom: 1.0, center: [100.0, 100.0], flip: false };
        let r = transform_controls_rect(&app, &xf).expect("shape layers have transform bounds");
        assert!(transform_controls_hit(r, r.right_bottom()));

        let ctx = egui::Context::default();
        assert!(begin_transform_controls_at(&mut app, &ctx, &xf, r.right_bottom()));
        let t = app.ui.transform.as_ref().expect("control press starts Free Transform");
        assert_eq!(t.rect, [20.0, 30.0, 100.0, 70.0]);
    }

    #[test]
    fn transform_control_hit_keeps_move_interior_free_and_has_rotation_band() {
        let r = Rect::from_min_max(pos2(20.0, 30.0), pos2(100.0, 70.0));
        assert!(transform_controls_hit(r, r.left_top()));
        assert!(transform_controls_hit(r, pos2(60.0, 18.0)));
        assert!(!transform_controls_hit(r, r.center()));
        assert!(!transform_controls_hit(r, pos2(60.0, 4.0)));
    }
}
