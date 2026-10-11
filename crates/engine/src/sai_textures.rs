//! SAI brush maps: grayscale shape maps (SAI "blotmap"), brush textures ("brushtex") and paper
//! textures ("papertex"), read from the user's own folder of `.bmp` files (or any image the codecs
//! decode). Nothing is bundled: the maps are the user's.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use photocraft_paint::GrayTile;

/// The default folder: `%APPDATA%\Photocraft\sai-textures` (`PHOTOCRAFT_SAI_TEXTURES` overrides).
pub fn default_dir() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("PHOTOCRAFT_SAI_TEXTURES") {
        return Some(PathBuf::from(p));
    }
    std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("Photocraft").join("sai-textures"))
}

/// Uncompressed BMP (1/4/8-bit palette, 24/32-bit) to luminance 0..1.
pub fn bmp_gray(bytes: &[u8]) -> Option<GrayTile> {
    let u16le = |o: usize| -> Option<u32> { Some(u32::from(u16::from_le_bytes(bytes.get(o..o + 2)?.try_into().ok()?))) };
    let u32le = |o: usize| -> Option<u32> { Some(u32::from_le_bytes(bytes.get(o..o + 4)?.try_into().ok()?)) };
    if bytes.get(0..2)? != b"BM" {
        return None;
    }
    let data_off = u32le(10)? as usize;
    let hdr = u32le(14)? as usize;
    let w = i32::from_le_bytes(bytes.get(18..22)?.try_into().ok()?);
    let h = i32::from_le_bytes(bytes.get(22..26)?.try_into().ok()?);
    let bpp = u16le(28)?;
    let compression = u32le(30)?;
    if compression != 0 && compression != 3 || w <= 0 || h == 0 || w > 8192 || h.unsigned_abs() > 8192 {
        return None;
    }
    let (wu, hu) = (w as usize, h.unsigned_abs() as usize);
    let top_down = h < 0;
    let colors = match u32le(46)? {
        0 if bpp <= 8 => 1usize << bpp,
        n => n as usize,
    };
    let pal_off = 14 + hdr;
    let palette: Vec<f32> = (0..if bpp <= 8 { colors } else { 0 })
        .map(|i| {
            let o = pal_off + i * 4;
            match bytes.get(o..o + 3) {
                Some(c) => (0.114 * f32::from(c[0]) + 0.587 * f32::from(c[1]) + 0.299 * f32::from(c[2])) / 255.0,
                None => 0.0,
            }
        })
        .collect();
    let row = (wu * bpp as usize).div_ceil(32) * 4;
    let mut out = vec![0.0f32; wu * hu];
    for y in 0..hu {
        let src_y = if top_down { y } else { hu - 1 - y };
        let line = bytes.get(data_off + src_y * row..data_off + src_y * row + row)?;
        for x in 0..wu {
            let v = match bpp {
                1 | 4 | 8 => {
                    let bit = x * bpp as usize;
                    let byte = line[bit / 8];
                    let idx = (byte >> (8 - bpp as usize - bit % 8)) & ((1u16 << bpp) - 1) as u8;
                    palette.get(idx as usize).copied().unwrap_or(0.0)
                }
                24 | 32 => {
                    let o = x * (bpp as usize / 8);
                    (0.114 * f32::from(line[o]) + 0.587 * f32::from(line[o + 1]) + 0.299 * f32::from(line[o + 2])) / 255.0
                }
                _ => return None,
            };
            out[y * wu + x] = v;
        }
    }
    Some(GrayTile::from_f32(wu as u32, hu as u32, &out))
}

/// Any image the codecs decode (PNG, JPEG…) to luminance, else [`bmp_gray`].
pub fn decode_gray(bytes: &[u8]) -> Option<GrayTile> {
    if let Some(t) = bmp_gray(bytes) {
        return Some(t);
    }
    let img = photocraft_codecs::decode(bytes).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = (img.width(), img.height());
    let v: Vec<f32> = rgba.chunks_exact(4).map(|p| (0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2])) / 255.0).collect();
    Some(GrayTile::from_f32(w, h, &v))
}

/// Every map in `dir` and its sub-folders, by file stem (a later duplicate name is skipped).
pub fn load_dir(dir: &Path) -> BTreeMap<String, Arc<GrayTile>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let ext = p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
            if !matches!(ext.as_str(), "bmp" | "png" | "jpg" | "jpeg") {
                continue;
            }
            let Some(name) = p.file_stem().and_then(|s| s.to_str()).map(str::to_string) else { continue };
            if out.contains_key(&name) {
                continue;
            }
            if let Some(t) = std::fs::read(&p).ok().and_then(|b| decode_gray(&b)) {
                out.insert(name, Arc::new(t));
            }
        }
    }
    out
}
