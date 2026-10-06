//! Same-process GPU compositor A/B: identity culling and mip compositing on a real document.
use photocraft_geom::Rect;
use photocraft_gpu::{Compositor, FAST_ACC_FORMAT};
use std::time::Instant;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: region_performance_probe FILE [samples=15]")?;
    let samples: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15).clamp(3, 100);
    // `PHOTOCRAFT_PROBE_SCRATCH=dir` (+ `PHOTOCRAFT_TILE_BUDGET_MB`, default 2048): spill tiles
    // like the app does, so a document larger than free RAM doesn't page.
    if let Ok(dir) = std::env::var("PHOTOCRAFT_PROBE_SCRATCH") {
        let mb: usize = std::env::var("PHOTOCRAFT_TILE_BUDGET_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
        photocraft_raster::spill::configure(Some(std::path::Path::new(&dir)), mb << 20)?;
    }
    let start = Instant::now();
    let mut doc = if path == "--synthetic" {
        use photocraft_color::{ColorMode, PixelFormat, SampleType};
        use photocraft_doc::{Document, Layer};
        use photocraft_geom::Size;
        let mut doc = Document::new("24MP sparse compositor", Size::new(6000, 4000), ColorMode::Rgb, SampleType::U8);
        for i in 0..120i32 {
            let mut layer = Layer::raster(format!("L{i}"), PixelFormat::RGBA8);
            let x = i * 331 % 5000;
            let y = i * 557 % 3000;
            let rect = Rect::new(x, y, x + 256 + i * 37 % 768, y + 256 + i * 53 % 768);
            if let Some(surface) = layer.surface_mut() {
                surface.fill_rect(rect, &[(i % 17) as f32 / 16.0, (i % 23) as f32 / 22.0, 0.3, 0.7]);
            }
            doc.layers.push(layer);
        }
        doc
    } else if path.ends_with(".pcraft") {
        photocraft_format::load_path(std::path::Path::new(path)).map_err(|e| e.to_string())?
    } else {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        photocraft_io::import_owned(path, data).map_err(|e| e.to_string())?.document
    };
    eprintln!("open {:.3}s {}x{} layers {}", start.elapsed().as_secs_f64(), doc.size.width, doc.size.height, doc.walk().len());
    let setup = photocraft_ui_egui::gpu_canvas::wgpu_setup();
    let rs = egui_kittest::wgpu::create_render_state(setup, Default::default());
    eprintln!("adapter {:?}", rs.adapter.get_info());
    let (device, queue) = (&rs.device, &rs.queue);
    let mut comp = Compositor::try_new_with_format(device, FAST_ACC_FORMAT).map_err(|e| e.to_string())?;
    let output = device.create_texture(&eframe::wgpu::TextureDescriptor {
        label: Some("perf_probe_output"),
        size: eframe::wgpu::Extent3d { width: doc.size.width, height: doc.size.height, depth_or_array_layers: 1 },
        mip_level_count: 1 + doc.size.width.max(doc.size.height).ilog2(),
        sample_count: 1,
        dimension: eframe::wgpu::TextureDimension::D2,
        format: FAST_ACC_FORMAT,
        usage: eframe::wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let partial = Rect::new(1000, 1000, 2100, 1500).intersect(&doc.bounds());
    println!("kind,culling,mip,samples,p50_ms,p95_ms,worst_ms,drawn,culled,uploads");
    // Alternation prevents one setting always receiving the warmer cache.
    // `edit`: before each render, paint a small square into a raster layer inside a group with
    // layer effects (a brush stroke in such a layer), then refresh the partial region.
    let edit_path = path_in_fx_group(&doc.layers, partial, false);
    eprintln!("edit target: {edit_path:?}");
    let mut dabs = 0i32;
    for (kind, rect, mip, count) in [("full", doc.bounds(), 0, 5), ("partial", partial, 0, samples), ("edit", partial, 0, samples), ("fit", doc.bounds(), 2, 5)]
    {
        if kind == "edit" && edit_path.is_none() {
            continue;
        }
        for cull in [false, true, false, true] {
            comp.set_pass_culling(cull);
            let mut times = Vec::new();
            let mut last = Default::default();
            for iteration in 0..count + 2 {
                if kind == "edit"
                    && let Some(p) = &edit_path
                    && let Some(surface) = layer_at(&mut doc.layers, p).and_then(|l| l.surface_mut())
                {
                    dabs += 1;
                    let (x, y) = (partial.x0 + 40 + dabs * 23 % (partial.width() as i32 - 120), partial.y0 + 40 + dabs * 17 % (partial.height() as i32 - 120));
                    surface.fill_rect(Rect::new(x, y, x + 48, y + 48), &[0.9, 0.2, 0.1, 1.0]);
                }
                let t = Instant::now();
                last = comp
                    .render_at_mip(device, queue, &doc, rect, mip, |encoder, out| {
                        encoder.copy_texture_to_texture(
                            out.texture.as_image_copy(),
                            eframe::wgpu::TexelCopyTextureInfo {
                                texture: &output,
                                mip_level: mip,
                                origin: eframe::wgpu::Origin3d { x: out.rect.x0 as u32 >> mip, y: out.rect.y0 as u32 >> mip, z: 0 },
                                aspect: eframe::wgpu::TextureAspect::All,
                            },
                            eframe::wgpu::Extent3d { width: out.pixel_size[0], height: out.pixel_size[1], depth_or_array_layers: 1 },
                        );
                    })
                    .map_err(|e| e.to_string())?;
                device.poll(eframe::wgpu::PollType::Wait { submission_index: None, timeout: None }).map_err(|e| e.to_string())?;
                if iteration >= 2 {
                    times.push(t.elapsed().as_secs_f64() * 1000.0);
                }
            }
            times.sort_by(f64::total_cmp);
            let percentile = |p: f64| times.get(((times.len() - 1) as f64 * p).ceil() as usize).copied().unwrap_or(0.0);
            println!(
                "{kind},{cull},{mip},{count},{:.3},{:.3},{:.3},{},{},{}",
                percentile(0.5),
                percentile(0.95),
                percentile(1.0),
                last.passes_drawn,
                last.passes_culled,
                last.tiles_uploaded
            );
        }
    }
    Ok(())
}

/// Index path of the top-most visible raster layer inside a group with layer effects whose tiles
/// overlap `r`.
fn path_in_fx_group(layers: &[photocraft_doc::Layer], r: Rect, in_fx: bool) -> Option<Vec<usize>> {
    use photocraft_doc::LayerContent;
    for (i, l) in layers.iter().enumerate().rev() {
        match &l.content {
            LayerContent::Group(g) => {
                let fx = in_fx || l.effects.items.iter().any(|e| e.enabled());
                if let Some(mut p) = path_in_fx_group(&g.children, r, fx) {
                    p.insert(0, i);
                    return Some(p);
                }
            }
            LayerContent::Raster(_) if in_fx && l.visible && l.surface().is_some_and(|s| !s.tile_bounds().intersect(&r).is_empty()) => return Some(vec![i]),
            _ => {}
        }
    }
    None
}

fn layer_at<'a>(layers: &'a mut [photocraft_doc::Layer], path: &[usize]) -> Option<&'a mut photocraft_doc::Layer> {
    let (first, rest) = path.split_first()?;
    let l = layers.get_mut(*first)?;
    if rest.is_empty() {
        return Some(l);
    }
    match &mut l.content {
        photocraft_doc::LayerContent::Group(g) => layer_at(&mut g.children, rest),
        _ => None,
    }
}
