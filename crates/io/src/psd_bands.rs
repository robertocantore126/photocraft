//! Raw/PackBits layer pixels decoded in tile-row bands, without whole channel planes.
use crate::pixels::{interleave, max_sample, zero_sample};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_geom::{Rect, TILE_SIZE};
use photocraft_psd::{Compression, LayerRecord, PsdFile};
use photocraft_raster::Surface;
use std::ops::Range;

struct Plane<'a> {
    data: &'a [u8],
    rows: Vec<Range<usize>>,
    rle: bool,
}
struct Buffers {
    planes: Vec<Vec<u8>>,
}

pub(crate) fn decode(file: &PsdFile, rec: &LayerRecord, fmt: PixelFormat) -> Option<Surface> {
    let (width, height) = rec.rect.size().ok()?;
    if width == 0 || height == 0 {
        return Some(Surface::new(fmt));
    }
    let row_bytes = width.checked_mul(fmt.sample.bytes())?;
    // Keep the same per-channel allocation guard as the standalone PSD decoder.
    if row_bytes.checked_mul(height)? as u64 > photocraft_psd::compression::MAX_DECODED_BYTES {
        return None;
    }
    let colors = fmt.mode.color_channels();
    let channels = colors.checked_add(1)?;
    let mut planes = Vec::with_capacity(channels);
    for id in (0..colors).map(|c| c as i16).chain(std::iter::once(-1)) {
        let Some(channel) = rec.channel(id) else {
            planes.push(None);
            continue;
        };
        let mut rows = Vec::with_capacity(height);
        let rle = match channel.compression {
            Some(Compression::Raw) => {
                for row in 0..height {
                    let start = row.checked_mul(row_bytes)?;
                    let end = start.checked_add(row_bytes)?;
                    channel.data.get(start..end)?;
                    rows.push(start..end);
                }
                false
            }
            Some(Compression::Rle) => {
                let stride = if file.header.version.is_psb() { 4 } else { 2 };
                let mut offset = height.checked_mul(stride)?;
                channel.data.get(..offset)?;
                for row in 0..height {
                    let start = row.checked_mul(stride)?;
                    let bytes = channel.data.get(start..start.checked_add(stride)?)?;
                    let length =
                        if stride == 4 { u32::from_be_bytes(bytes.try_into().ok()?) as usize } else { usize::from(u16::from_be_bytes(bytes.try_into().ok()?)) };
                    let end = offset.checked_add(length)?;
                    channel.data.get(offset..end)?;
                    rows.push(offset..end);
                    offset = end;
                }
                true
            }
            _ => return None,
        };
        planes.push(Some(Plane { data: &channel.data, rows, rle }));
    }
    let mut bands = Vec::new();
    let mut top = rec.rect.top;
    while top < rec.rect.bottom {
        let bottom = top.div_euclid(TILE_SIZE).saturating_add(1).saturating_mul(TILE_SIZE).min(rec.rect.bottom);
        if bottom <= top {
            return None;
        }
        bands.push(Rect::new(rec.rect.left, top, rec.rect.right, bottom));
        top = bottom;
    }
    let mut fill = vec![zero_sample(fmt.sample); colors];
    fill.push(max_sample(fmt.sample));
    let mut invert = vec![fmt.mode == ColorMode::Cmyk; colors];
    invert.push(false);
    let decode_band = |buffers: &mut Buffers, band: &Rect| -> Option<Surface> {
        let first = usize::try_from(i64::from(band.y0) - i64::from(rec.rect.top)).ok()?;
        let rows = usize::try_from(band.height()).ok()?;
        for (index, plane) in planes.iter().enumerate() {
            let target = buffers.planes.get_mut(index)?;
            target.clear();
            if let Some(plane) = plane {
                for row in first..first.checked_add(rows)? {
                    let range = plane.rows.get(row)?.clone();
                    let data = plane.data.get(range)?;
                    if plane.rle {
                        photocraft_psd::compression::packbits::decode_into(data, row_bytes, target).ok()?;
                    } else {
                        target.extend_from_slice(data);
                    }
                }
            }
        }
        let refs: Vec<Option<&[u8]>> = planes.iter().zip(&buffers.planes).map(|(plane, buffer)| plane.as_ref().map(|_| buffer.as_slice())).collect();
        let mut bytes = interleave(&refs, &fill, width.checked_mul(rows)?, fmt.sample, &invert);
        if fmt.mode == ColorMode::Lab && fmt.sample == SampleType::U16 {
            crate::pixels::lab16_chroma(&mut bytes, channels, true);
        }
        Some(Surface::from_interleaved(fmt, *band, &bytes))
    };
    // map_init retains planar scratch buffers for each worker, rather than reallocating per row.
    #[cfg(not(target_arch = "wasm32"))]
    let decoded: Vec<Option<Surface>> = {
        use rayon::prelude::*;
        bands.par_iter().map_init(|| Buffers { planes: vec![Vec::new(); channels] }, decode_band).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let decoded: Vec<Option<Surface>> = {
        let mut buffers = Buffers { planes: vec![Vec::new(); channels] };
        bands.iter().map(|band| decode_band(&mut buffers, band)).collect()
    };
    let mut surface = Surface::new(fmt);
    for band in decoded {
        for (coord, tile) in band?.tiles() {
            surface.insert_tile(*coord, tile.clone()).ok()?;
        }
    }
    surface.prune();
    Some(surface)
}
