//! Lazy native packs: all document fields, incremental commits, corruption and torn-write recovery.
mod common;
use common::*;
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, Layer, Size};
use photocraft_format::{LoadOptions, PcraftWriter, SaveOptions, load_path, pack};
use photocraft_geom::TileCoord;
use std::io::{Seek, SeekFrom, Write};

#[test]
fn pack_roundtrips_all_models_and_depths_and_old_bundles() {
    for mode in [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab] {
        for depth in SampleType::ALL {
            let dir = temp_dir("pack-models");
            let path = dir.join("doc.pcraft");
            let doc = rich_doc(mode, depth);
            let stats = PcraftWriter::new().save_path(&doc, &path, &SaveOptions::default()).unwrap();
            assert!(stats.tiles_written > 0);
            assert_eq!(load_path(&path).unwrap(), doc);
            let old = photocraft_format::save_to_bytes(&doc, &SaveOptions::default()).unwrap();
            assert_eq!(photocraft_format::load_from_bytes(&old).unwrap(), doc);
            let legacy = dir.join("old.pcraft");
            std::fs::write(&legacy, old).unwrap();
            assert_eq!(load_path(&legacy).unwrap(), doc);
            PcraftWriter::new().save_path(&doc, &legacy, &SaveOptions::default()).unwrap();
            assert!(std::fs::read(&legacy).unwrap().starts_with(pack::MAGIC));
            assert_eq!(load_path(&legacy).unwrap(), doc);
        }
    }
}
#[test]
fn lazy_tiles_stay_nonresident_and_incremental_save_reuses_them() {
    let dir = temp_dir("pack-lazy");
    let path = dir.join("doc.pcraft");
    let doc = rich_doc(ColorMode::Rgb, SampleType::U16);
    let first = PcraftWriter::new().save_path(&doc, &path, &Default::default()).unwrap();
    let back = load_path(&path).unwrap();
    let tiles: Vec<_> = back.layers.iter().filter_map(|l| l.surface()).flat_map(|s| s.tiles()).collect();
    assert!(!tiles.is_empty());
    assert!(tiles.iter().all(|(_, tile)| !tile.is_resident()));
    // Saving the just-opened document must copy neither decoded nor compressed tile payloads.
    let second = PcraftWriter::new().save_path(&back, &path, &Default::default()).unwrap();
    assert!(tiles.iter().all(|(_, tile)| !tile.is_resident()));
    assert_eq!(second.tiles_written, 0);
    assert_eq!(second.blobs_written, 0);
    assert_eq!(second.tiles_reused, first.tiles_total);
    assert_eq!(back, doc);
    let other = dir.join("compact.pcraft");
    let compact = PcraftWriter::new().save_path(&back, &other, &Default::default()).unwrap();
    assert_eq!(compact.tiles_written, first.tiles_total);
    assert_eq!(load_path(&other).unwrap(), doc);
    let original_again = PcraftWriter::new().save_path(&back, &path, &Default::default()).unwrap();
    assert_eq!(original_again.tiles_written, 0, "Save As/another pack must preserve the original unchanged extents");
}
fn small_doc() -> Document {
    let mut doc = Document::new("old", Size::new(16, 16), ColorMode::Rgb, SampleType::U8);
    let mut layer = Layer::raster("pixels", doc.pixel_format());
    layer.surface_mut().unwrap().write_pixel(0, 0, &[0.3, 0.4, 0.5, 1.0]);
    doc.layers.push(layer);
    doc
}
#[test]
fn torn_write_at_random_points_recovers_last_complete_footer() {
    let dir = temp_dir("pack-torn");
    let path = dir.join("doc.pcraft");
    let old = small_doc();
    PcraftWriter::new().save_path(&old, &path, &Default::default()).unwrap();
    let committed = std::fs::metadata(&path).unwrap().len() as usize;
    let mut new = old.clone();
    new.name = "new".into();
    new.layers[0].surface_mut().unwrap().write_pixel(2, 3, &[0.8, 0.1, 0.2, 1.0]);
    PcraftWriter::new().save_path(&new, &path, &Default::default()).unwrap();
    let complete = std::fs::read(&path).unwrap();
    let mut rng = Rng::new(8102);
    let mut cuts = vec![committed, complete.len()];
    for _ in 0..80 {
        cuts.push(committed + rng.next() as usize % (complete.len() - committed));
    }
    for i in 1..=40 {
        cuts.push(complete.len() - i);
    }
    for cut in cuts {
        let broken = dir.join("broken.pcraft");
        std::fs::write(&broken, &complete[..cut]).unwrap();
        let result = load_path(&broken).unwrap();
        assert_eq!(result, if cut == complete.len() { new.clone() } else { old.clone() }, "truncated at {cut}");
    }
    for cut in [0, 8, 31, 32, committed - 1] {
        let broken = dir.join("no-commit.pcraft");
        std::fs::write(&broken, &complete[..cut]).unwrap();
        assert!(load_path(&broken).is_err());
    }
}
#[test]
fn corrupt_lazy_payload_is_an_error_and_cannot_be_resaved_as_zero_pixels() {
    let dir = temp_dir("pack-corrupt");
    let path = dir.join("doc.pcraft");
    let doc = small_doc();
    PcraftWriter::new().save_path(&doc, &path, &Default::default()).unwrap();
    let loaded = load_path(&path).unwrap();
    let tile = loaded.layers[0].surface().unwrap().tile(TileCoord::new(0, 0)).unwrap();
    let mut header = std::fs::read(&path).unwrap();
    let mut id = [0; 16];
    id.copy_from_slice(&header[8..24]);
    let extent = tile.pack_extent(id).unwrap();
    header[extent.offset as usize] ^= 1;
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(extent.offset)).unwrap();
    file.write_all(&[header[extent.offset as usize]]).unwrap();
    file.sync_all().unwrap();
    assert!(tile.try_bytes().is_err());
    assert!(tile.packed_bytes().is_err());
    assert!(PcraftWriter::new().save_path(&loaded, &dir.join("bad-save.pcraft"), &Default::default()).is_err());
    // A COW copy must preserve the read failure instead of laundering the zero fallback.
    let clone = tile.as_ref().clone();
    assert!(clone.try_bytes().is_err());
}
#[test]
fn manifest_limits_apply_before_any_pixels_are_read() {
    let dir = temp_dir("pack-limits");
    let path = dir.join("doc.pcraft");
    let doc = small_doc();
    PcraftWriter::new().save_path(&doc, &path, &Default::default()).unwrap();
    let opts = LoadOptions { max_total_bytes: 1, ..Default::default() };
    assert!(photocraft_format::load_path_with(&path, &opts).is_err());
    let opts = LoadOptions { max_manifest_bytes: 8, ..Default::default() };
    assert!(photocraft_format::load_path_with(&path, &opts).is_err());
}

#[test]
fn divergent_copies_of_a_pack_do_not_reuse_foreign_offsets() {
    let dir = temp_dir("pack-fork");
    let a = dir.join("a.pcraft");
    let b = dir.join("b.pcraft");
    let mut first = small_doc();
    PcraftWriter::new().save_path(&first, &a, &Default::default()).unwrap();
    std::fs::copy(&a, &b).unwrap();
    let mut second = load_path(&b).unwrap();
    second.name = "branch B".into();
    second.layers[0].surface_mut().unwrap().write_pixel(3, 2, &[0.1, 0.2, 0.3, 1.0]);
    PcraftWriter::new().save_path(&second, &b, &Default::default()).unwrap();
    first.name = "branch A".into();
    first.layers[0].surface_mut().unwrap().write_pixel(3, 2, &[0.9, 0.8, 0.7, 1.0]);
    PcraftWriter::new().save_path(&first, &a, &Default::default()).unwrap();
    let stats = PcraftWriter::new().save_path(&first, &b, &Default::default()).unwrap();
    assert!(stats.tiles_written > 0);
    assert_eq!(load_path(&b).unwrap(), first);
}

#[test]
fn optional_pack_previews_and_manifest_remain_available() {
    let dir = temp_dir("pack-preview");
    let path = dir.join("doc.pcraft");
    let doc = small_doc();
    let thumbnail = photocraft_raster::Rgba8Image { width: 1, height: 1, pixels: vec![1, 2, 3, 255] };
    PcraftWriter::new().save_path(&doc, &path, &SaveOptions { thumbnail: Some(thumbnail), composite: None }).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let png = photocraft_format::read_thumbnail(&bytes).unwrap().unwrap();
    assert!(png.starts_with(b"\x89PNG"));
    assert_eq!(photocraft_format::read_manifest(&bytes).unwrap().document.name, doc.name);
    assert_eq!(photocraft_format::load_from_bytes(&bytes).unwrap(), doc);
}

#[test]
fn closing_a_document_releases_its_pack_file() {
    // Tiles and blobs (raw PSD blocks) read lazily from the pack: once every document from it is
    // gone, no cache may keep the file open (on Windows the directory then can't be removed).
    let dir = temp_dir("pack-release");
    let path = dir.join("doc.pcraft");
    let mut doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    if let Some(l) = doc.layers.first_mut() {
        l.psd_blocks = vec![(*b"tst1", std::sync::Arc::new(vec![7u8; 4096]))];
    }
    PcraftWriter::new().save_path(&doc, &path, &SaveOptions::default()).unwrap();
    let loaded = load_path(&path).unwrap();
    assert_eq!(loaded, doc);
    // A second save of the loaded document reuses its blobs from the pack.
    PcraftWriter::new().save_path(&loaded, &path, &SaveOptions::default()).unwrap();
    drop(loaded);
    drop(doc);
    std::fs::remove_dir_all(&dir).unwrap();
}
