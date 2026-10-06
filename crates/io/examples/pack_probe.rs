//! Native pack timing and lossless roundtrip probe (release, no GPU).
use std::{path::PathBuf, time::Instant};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    // `--convert IN OUT`: import (with the scratch disk) and save a fresh pack, nothing else.
    if args.first().is_some_and(|a| a == "--convert") {
        let (input, output) = (PathBuf::from(args.get(1).ok_or("input path required")?), PathBuf::from(args.get(2).ok_or("output path required")?));
        photocraft_raster::spill::configure(Some(std::path::Path::new("E:/PhotoCraftScratch")), 2048 << 20)?;
        let t = Instant::now();
        let doc = photocraft_io::import_path(&input)?.document;
        println!("import_ms={:.0}", t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        photocraft_format::pack::save_compact_path(&doc, &output, &Default::default())?;
        println!("save_ms={:.0} bytes={}", t.elapsed().as_secs_f64() * 1000.0, std::fs::metadata(&output)?.len());
        return Ok(());
    }
    if args.first().is_some_and(|a| a == "--open-only") {
        let input = PathBuf::from(args.get(1).ok_or("pack path required")?);
        let t = Instant::now();
        let doc = photocraft_format::load_path(&input)?;
        println!(
            "fresh_lazy_open_ms={:.3} layers={} resident_bytes={}",
            t.elapsed().as_secs_f64() * 1000.0,
            doc.layers.len(),
            photocraft_raster::spill::stats().resident_bytes
        );
        std::thread::sleep(std::time::Duration::from_secs(2));
        return Ok(());
    }
    if args.first().is_some_and(|a| a == "--synthetic") {
        let dir = PathBuf::from(args.get(1).ok_or("output directory required")?);
        std::fs::create_dir_all(&dir)?;
        let mut doc = photocraft_doc::Document::new(
            "25MP noise",
            photocraft_doc::Size::new(6144, 4096),
            photocraft_color::ColorMode::Rgb,
            photocraft_color::SampleType::U8,
        );
        let mut layer = photocraft_doc::Layer::raster("noise", doc.pixel_format());
        let surface = layer.surface_mut().ok_or("missing pixels")?;
        let mut rng = 42u64;
        for ty in 0..16 {
            for tx in 0..24 {
                let mut bytes = surface.tile_mut(photocraft_geom::TileCoord::new(tx, ty)).bytes_mut();
                for px in bytes.chunks_exact_mut(4) {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    if let [r, g, b, a] = px {
                        *r = rng as u8;
                        *g = (rng >> 8) as u8;
                        *b = (rng >> 16) as u8;
                        *a = 255;
                    }
                }
            }
        }
        doc.layers.push(layer);
        let mut old = photocraft_format::PcraftWriter::new();
        let t = Instant::now();
        let (bytes, stats) = old.save_zip(&doc, &Default::default())?;
        std::fs::write(dir.join("old.zip.pcraft"), &bytes)?;
        println!("synthetic_25mp_old_save_ms={:.3} bytes={} stats={stats:?}", t.elapsed().as_secs_f64() * 1000.0, bytes.len());
        let t = Instant::now();
        let olddoc = photocraft_format::load_from_bytes(&bytes)?;
        println!("synthetic_25mp_old_open_ms={:.3}", t.elapsed().as_secs_f64() * 1000.0);
        drop(olddoc);
        drop(bytes);
        let target = dir.join("new.pcraft");
        let t = Instant::now();
        let stats = photocraft_format::pack::save_compact_path(&doc, &target, &Default::default())?;
        println!("synthetic_25mp_pack_save_ms={:.3} stats={stats:?}", t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        let loaded = photocraft_format::load_path(&target)?;
        println!("synthetic_25mp_pack_open_ms={:.3}", t.elapsed().as_secs_f64() * 1000.0);
        if doc != loaded {
            return Err("synthetic pixel roundtrip mismatch".into());
        }
        drop(loaded);
        doc.layers.first_mut().and_then(|l| l.surface_mut()).ok_or("missing pixels")?.write_pixel(10, 10, &[1.0, 0.0, 0.0, 1.0]);
        let t = Instant::now();
        let (bytes, _) = old.save_zip(&doc, &Default::default())?;
        std::fs::write(dir.join("old.zip.pcraft"), bytes)?;
        println!("synthetic_25mp_old_edit_save_ms={:.3}", t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        let stats = photocraft_format::PcraftWriter::new().save_path(&doc, &target, &Default::default())?;
        println!("synthetic_25mp_pack_edit_save_ms={:.3} stats={stats:?}", t.elapsed().as_secs_f64() * 1000.0);
        return Ok(());
    }
    let input = PathBuf::from(args.first().ok_or("input path required")?);
    let output = PathBuf::from(args.get(1).ok_or("output path required")?);
    photocraft_raster::spill::configure(Some(std::path::Path::new("E:/PhotoCraftScratch")), 512 << 20)?;
    let t = Instant::now();
    let result = photocraft_io::import_path(&input)?;
    let mut doc = result.document;
    println!(
        "import_ms={:.3} layers={} resident_bytes={}",
        t.elapsed().as_secs_f64() * 1000.0,
        doc.layers.len(),
        photocraft_raster::spill::stats().resident_bytes
    );
    let t = Instant::now();
    let stats = photocraft_format::PcraftWriter::new().save_path(&doc, &output, &Default::default())?;
    println!("save_as_ms={:.3} bytes={} stats={stats:?}", t.elapsed().as_secs_f64() * 1000.0, std::fs::metadata(&output)?.len());
    let t = Instant::now();
    let back = photocraft_format::load_path(&output)?;
    println!("lazy_open_ms={:.3} resident_bytes={}", t.elapsed().as_secs_f64() * 1000.0, photocraft_raster::spill::stats().resident_bytes);
    let t = Instant::now();
    let equal = doc == back;
    println!("pixel_and_model_equal={equal} equality_ms={:.3}", t.elapsed().as_secs_f64() * 1000.0);
    if !equal {
        report_differences(&doc, &back);
        return Err("lossless roundtrip failed".into());
    }
    let t = Instant::now();
    let stats = photocraft_format::PcraftWriter::new().save_path(&doc, &output, &Default::default())?;
    println!("unchanged_save_ms={:.3} stats={stats:?}", t.elapsed().as_secs_f64() * 1000.0);
    if let Some(surface) = doc.layers.iter_mut().find_map(|l| l.surface_mut()) {
        surface.write_pixel(300, 300, &[0.2, 0.4, 0.6, 1.0]);
    }
    let t = Instant::now();
    let stats = photocraft_format::PcraftWriter::new().save_path(&doc, &output, &Default::default())?;
    println!("one_pixel_save_ms={:.3} stats={stats:?}", t.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

/// Names of the document and layer fields that differ (the first 40).
fn report_differences(a: &photocraft_doc::Document, b: &photocraft_doc::Document) {
    let mut out = Vec::new();
    macro_rules! doc_fields { ($($f:ident),*) => { $( if a.$f != b.$f { out.push(format!("document.{}", stringify!($f))); } )* } }
    doc_fields!(
        id,
        name,
        size,
        resolution_dpi,
        mode,
        depth,
        icc_profile,
        channels,
        guides,
        selection,
        metadata,
        global_light,
        paths,
        work_path,
        clipping_path,
        quick_mask,
        patterns,
        color_table,
        duotone,
        layer_comps,
        last_applied_comp,
        last_document_state,
        measurement,
        notes,
        text_styles,
        slices,
        variables,
        timeline
    );
    for (n, x, y) in [
        ("slices", format!("{:?}", a.slices), format!("{:?}", b.slices)),
        ("variables", format!("{:?}", a.variables), format!("{:?}", b.variables)),
        ("timeline", format!("{:?}", a.timeline), format!("{:?}", b.timeline)),
    ] {
        if x != y {
            let i = x.bytes().zip(y.bytes()).position(|(p, q)| p != q).unwrap_or(0);
            out.push(format!(
                "{n}: orig ..{}.. back ..{}..",
                &x[i.saturating_sub(120)..(i + 120).min(x.len())],
                &y[i.saturating_sub(120)..(i + 120).min(y.len())]
            ));
        }
    }
    fn layers(x: &[photocraft_doc::Layer], y: &[photocraft_doc::Layer], path: &str, out: &mut Vec<String>) {
        if x.len() != y.len() {
            out.push(format!("{path}: {} vs {} layers", x.len(), y.len()));
        }
        for (l, m) in x.iter().zip(y) {
            let here = format!("{path}/{}", l.name);
            macro_rules! layer_fields { ($($f:ident),*) => { $( if l.$f != m.$f { out.push(format!("{here}: {}", stringify!($f))); } )* } }
            if l.effects != m.effects && !out.iter().any(|d| d.ends_with("[effects detail]")) {
                let (x, y) = (format!("{:?}", l.effects), format!("{:?}", m.effects));
                let i = x.bytes().zip(y.bytes()).position(|(p, q)| p != q).unwrap_or(0);
                out.push(format!(
                    "{here}: orig ..{}..
    back ..{}.. [effects detail]",
                    &x[i.saturating_sub(150)..(i + 80).min(x.len())],
                    &y[i.saturating_sub(150)..(i + 80).min(y.len())]
                ));
            }
            if l.content != m.content && !out.iter().any(|d| d.ends_with("[content detail]")) {
                let (x, y) = (format!("{:?}", l.content), format!("{:?}", m.content));
                let i = x.bytes().zip(y.bytes()).position(|(p, q)| p != q).unwrap_or(0);
                out.push(format!(
                    "{here}: orig ..{}..
    back ..{}.. [content detail] (lens {} {})",
                    &x[i.saturating_sub(150)..(i + 80).min(x.len())],
                    &y[i.saturating_sub(150)..(i + 80).min(y.len())],
                    x.len(),
                    y.len()
                ));
            }
            layer_fields!(
                id,
                name,
                visible,
                locks,
                blend,
                opacity,
                fill_opacity,
                clipped,
                mask,
                vector_mask,
                effects,
                label,
                psd_blocks,
                psd_id,
                fill_cache,
                link_group,
                excluded_channels,
                blend_if,
                video
            );
            match (&l.content, &m.content) {
                (photocraft_doc::LayerContent::Group(g), photocraft_doc::LayerContent::Group(h)) => {
                    if g.expanded != h.expanded || g.artboard != h.artboard {
                        out.push(format!("{here}: group settings"));
                    }
                    layers(&g.children, &h.children, &here, out);
                }
                (c, d) if c != d => out.push(format!("{here}: content {:?}", std::mem::discriminant(c))),
                _ => {}
            }
        }
    }
    layers(&a.layers, &b.layers, "", &mut out);
    println!("{} differences", out.len());
    for d in out.iter().take(40) {
        println!("  {d}");
    }
}
