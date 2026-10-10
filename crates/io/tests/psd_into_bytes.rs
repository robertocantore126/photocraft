//! `PsdFile::into_bytes` (the save path, which frees channels as it writes) must produce the
//! same bytes as `to_bytes`, at every depth (16 and 32 bits keep the layers in a global block).
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer};
use photocraft_geom::{Rect, Size};
use photocraft_io::document_to_psd_with;

fn layer(name: &str, fmt: PixelFormat, rect: Rect, seed: u32) -> Layer {
    let mut l = Layer::raster(name, fmt);
    let data: Vec<f32> =
        (0..rect.width() * rect.height() * fmt.channels() as u32).map(|i| ((i.wrapping_mul(2654435761) ^ seed) % 1000) as f32 / 999.0).collect();
    l.surface_mut().unwrap().write_region(rect, &data);
    l
}

#[test]
fn into_bytes_matches_to_bytes() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        for psb in [false, true] {
            let mut d = Document::new("t", Size::new(97, 61), ColorMode::Rgb, depth);
            let fmt = d.pixel_format();
            d.layers.push(layer("a", fmt, Rect::new(0, 0, 97, 61), 1));
            d.layers.push(layer("b", fmt, Rect::new(10, 7, 60, 50), 2));
            d.layers.push(Layer::raster("empty", fmt));
            let opts = photocraft_io::PsdExportOptions { force_psb: psb };
            let (file, _) = document_to_psd_with(&d, &opts);
            let expected = file.to_bytes().unwrap();
            let (file, _) = document_to_psd_with(&d, &opts);
            assert_eq!(file.into_bytes().unwrap(), expected, "{depth:?}, psb {psb}");
        }
    }
}
