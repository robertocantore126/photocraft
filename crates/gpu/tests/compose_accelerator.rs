//! The GPU compositor registered as `photocraft_compose::accel` (as the app does) must give the
//! whole-document entry points (`render`, `render_bands`, thumbnails) the CPU compositor's
//! result within the parity tolerance, with layer effects. Its own binary: the accelerator is
//! process-wide state. Skips without a GPU adapter that renders Rgba32Float.
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Effect, Layer};
use photocraft_geom::{Rect, Size};
use photocraft_gpu::Compositor;
use std::sync::{Arc, Mutex};

const TOL: f32 = 2.0 / 255.0;

fn layer(name: &str, rect: Rect, seed: u32) -> Layer {
    let mut l = Layer::raster(name, PixelFormat::RGBA8);
    let data: Vec<f32> = (0..rect.width() * rect.height() * 4)
        .map(|i| {
            let mut h = seed.wrapping_mul(0x9e37_79b9) ^ i.wrapping_mul(0x85eb_ca6b);
            h ^= h >> 15;
            h = h.wrapping_mul(0x2c1b_3c6d);
            if i % 4 == 3 { 0.5 + (h & 0xff) as f32 / 512.0 } else { (h & 0xffff) as f32 / 65535.0 }
        })
        .collect();
    l.surface_mut().unwrap().write_region(rect, &data);
    l
}

fn document() -> Document {
    let mut d = Document::new("accel", Size::new(1100, 900), ColorMode::Rgb, SampleType::U8);
    d.layers.push(layer("bg", d.bounds(), 1));
    let mut shadowed = layer("shadowed", Rect::new(200, 150, 700, 600), 2);
    shadowed.effects.items.push(Effect::default_drop_shadow());
    d.layers.push(shadowed);
    d.layers.push(layer("top", Rect::new(500, 400, 1000, 850), 3));
    d
}

fn worst(cpu: &[[f32; 4]], gpu: &[[f32; 4]]) -> f32 {
    cpu.iter()
        .zip(gpu)
        .flat_map(|(c, g)| {
            let (pc, pg) = ([c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]], [g[0] * g[3], g[1] * g[3], g[2] * g[3], g[3]]);
            (0..4).map(move |k| (pc[k] - pg[k]).abs())
        })
        .fold(0.0, f32::max)
}

#[test]
fn accelerated_whole_document_renders_match_the_cpu() {
    let Ok(adapter) = pollster::block_on(wgpu::Instance::default().request_adapter(&wgpu::RequestAdapterOptions::default())) else {
        eprintln!("skipping: no adapter");
        return;
    };
    if Compositor::preferred_acc_format(&adapter) != wgpu::TextureFormat::Rgba32Float {
        eprintln!("skipping: no Rgba32Float targets");
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let doc = document();
    let cpu = photocraft_compose::render(&doc, doc.bounds());
    let mut cpu_bands = Vec::new();
    photocraft_compose::render_bands(&doc, doc.bounds(), 256, |b| -> Result<(), ()> {
        cpu_bands.extend(b.px);
        Ok(())
    })
    .unwrap();
    let cpu_thumb = photocraft_compose::thumbnail_buffer(&doc, 384);

    let calls = Arc::new(Mutex::new(0usize));
    let counted = calls.clone();
    let comp = Mutex::new(Compositor::try_new_with_format(&device, wgpu::TextureFormat::Rgba32Float).unwrap());
    photocraft_compose::accel::set(Some(Arc::new(move |d: &Document, r: Rect| {
        *counted.lock().unwrap() += 1;
        photocraft_gpu::render_to_vec(&mut comp.lock().unwrap(), &device, &queue, d, r).ok()
    })));
    let gpu = photocraft_compose::render(&doc, doc.bounds());
    let mut gpu_bands = Vec::new();
    photocraft_compose::render_bands(&doc, doc.bounds(), 512, |b| -> Result<(), ()> {
        gpu_bands.extend(b.px);
        Ok(())
    })
    .unwrap();
    let gpu_thumb = photocraft_compose::thumbnail_buffer(&doc, 384);
    photocraft_compose::accel::set(None);

    assert!(*calls.lock().unwrap() >= 3, "the accelerator was not used");
    let (whole, bands, thumb) = (worst(&cpu.px, &gpu.px), worst(&cpu_bands, &gpu_bands), worst(&cpu_thumb.px, &gpu_thumb.px));
    eprintln!("worst premultiplied difference: whole {:.2}/255, bands {:.2}/255, thumbnail {:.2}/255", whole * 255.0, bands * 255.0, thumb * 255.0);
    assert_eq!(gpu.px.len(), cpu.px.len());
    assert_eq!(gpu_bands.len(), cpu_bands.len());
    assert!(whole <= TOL && bands <= TOL && thumb <= TOL);
}
