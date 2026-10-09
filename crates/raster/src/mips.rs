//! Disposable CPU pyramids. Clones share cache storage until a base tile is changed.
use crate::{Surface, Tile};
use photocraft_color::{read_sample, write_sample};
use photocraft_geom::{Rect, TILE_SIZE, TileCoord};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError, Weak};

const MAX_LEVEL: u32 = 16;
#[derive(Clone, Debug)]
struct Entry {
    tile: Arc<Tile>,
    // Weak keys never keep old source pixels alive merely for a derived cache.
    _sources: [Option<Weak<Tile>>; 4],
}
#[derive(Debug, Default)]
pub struct MipChain {
    tiles: Mutex<BTreeMap<(u32, TileCoord), Entry>>,
    default_tile: Mutex<Option<Arc<Tile>>>,
}
impl Clone for MipChain {
    fn clone(&self) -> Self {
        Self {
            tiles: Mutex::new(self.tiles.lock().unwrap_or_else(PoisonError::into_inner).clone()),
            default_tile: Mutex::new(self.default_tile.lock().unwrap_or_else(PoisonError::into_inner).clone()),
        }
    }
}
impl MipChain {
    pub(crate) fn invalidate(&mut self, coord: TileCoord) {
        let tiles = self.tiles.get_mut().unwrap_or_else(PoisonError::into_inner);
        for level in 1..=MAX_LEVEL {
            let scale = 1i32 << level;
            tiles.remove(&(level, TileCoord::new(coord.tx.div_euclid(scale), coord.ty.div_euclid(scale))));
        }
    }
}
impl Surface {
    pub(crate) fn invalidate_mips(&mut self, coord: TileCoord) {
        Arc::make_mut(&mut self.mips).invalidate(coord);
    }
    /// Largest mip whose long side still has at least `target` pixels.
    pub fn mip_level_for_size(bounds: Rect, target: u32) -> u32 {
        let mut size = (i64::from(bounds.x1) - i64::from(bounds.x0)).max(i64::from(bounds.y1) - i64::from(bounds.y0)).max(0) as u64;
        let mut level = 0;
        while level < MAX_LEVEL && size / 2 >= u64::from(target.max(1)) {
            size /= 2;
            level += 1;
        }
        level
    }
    /// Mip coordinates of a base-coordinate rectangle, rounding outward (also for negatives).
    pub fn mip_bounds(bounds: Rect, level: u32) -> Rect {
        let scale = 1i64 << level.min(MAX_LEVEL);
        let floor = |v: i32| (i64::from(v).div_euclid(scale)) as i32;
        let ceil = |v: i32| ((i64::from(v) + scale - 1).div_euclid(scale)) as i32;
        if bounds.is_empty() { Rect::EMPTY } else { Rect::new(floor(bounds.x0), floor(bounds.y0), ceil(bounds.x1), ceil(bounds.y1)) }
    }
    fn default_mip_tile(&self) -> Arc<Tile> {
        self.mips
            .default_tile
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert_with(|| Arc::new(Tile::solid(&self.format, &self.default_pixel)))
            .clone()
    }
    /// A lazily computed native-format tile. Level zero reads the original pixels.
    pub fn mip_tile(&self, level: u32, coord: TileCoord) -> Result<Arc<Tile>, String> {
        if level > MAX_LEVEL {
            return Err("CPU mip level exceeds 16".into());
        }
        if level == 0 {
            return Ok(self.tile(coord).cloned().unwrap_or_else(|| self.default_mip_tile()));
        }
        let cache = std::env::var("PHOTOCRAFT_CPU_MIPS").as_deref() != Ok("0");
        if cache && let Some(entry) = self.mips.tiles.lock().unwrap_or_else(PoisonError::into_inner).get(&(level, coord)) {
            return Ok(entry.tile.clone());
        }
        let scale = 1i64 << level;
        let x0 = i64::from(coord.tx) * scale;
        let y0 = i64::from(coord.ty) * scale;
        let max = i64::from(i32::MAX);
        let min = i64::from(i32::MIN);
        let present = x0 <= max
            && y0 <= max
            && x0 + scale > min
            && y0 + scale > min
            && self
                .tiles
                .range(TileCoord::new(x0.max(min) as i32, i32::MIN)..=TileCoord::new((x0 + scale - 1).min(max) as i32, i32::MAX))
                .any(|(c, _)| i64::from(c.ty) >= y0 && i64::from(c.ty) < y0 + scale);
        if !present {
            let tile = self.default_mip_tile();
            tile.cell.set_rebuild_cost(level);
            if cache {
                self.mips
                    .tiles
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert((level, coord), Entry { tile: tile.clone(), _sources: std::array::from_fn(|_| None) });
            }
            return Ok(tile);
        }
        let sx = coord.tx.checked_mul(2).ok_or("mip source coordinate overflow")?;
        let sy = coord.ty.checked_mul(2).ok_or("mip source coordinate overflow")?;
        let mut sources = Vec::with_capacity(4);
        for dy in 0..2 {
            for dx in 0..2 {
                sources.push(self.mip_tile(
                    level - 1,
                    TileCoord::new(sx.checked_add(dx).ok_or("mip coordinate overflow")?, sy.checked_add(dy).ok_or("mip coordinate overflow")?),
                )?);
            }
        }
        let n = self.channels();
        let ts = TILE_SIZE as usize;
        let mut bytes = vec![0; ts * ts * self.format.bytes_per_pixel()];
        // Each parent quadrant comes from a single source. Hold one guard per source,
        // never per pixel, and release it before allocating the derived spill cell.
        for (quadrant, source) in sources.iter().enumerate() {
            let data = source.try_bytes()?;
            let ox = quadrant % 2 * (ts / 2);
            let oy = quadrant / 2 * (ts / 2);
            let mut sum = vec![0.0f32; n];
            for y in 0..ts / 2 {
                for x in 0..ts / 2 {
                    sum.fill(0.0);
                    let mut alpha_sum = 0.0;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let index = ((y * 2 + dy) * ts + x * 2 + dx) * n;
                            let alpha = if self.format.alpha { read_sample(&data, self.format.sample, index + n - 1) } else { 1.0 };
                            alpha_sum += alpha;
                            for (c, value) in sum.iter_mut().enumerate() {
                                let sample = read_sample(&data, self.format.sample, index + c);
                                *value += if self.format.alpha && c + 1 == n { sample } else { sample * alpha };
                            }
                        }
                    }
                    let out = ((y + oy) * ts + x + ox) * n;
                    for (c, sum) in sum.iter().copied().enumerate() {
                        let value = if self.format.alpha && c + 1 == n {
                            sum * 0.25
                        } else if alpha_sum > 0.0 {
                            sum / alpha_sum
                        } else {
                            0.0
                        };
                        write_sample(&mut bytes, self.format.sample, out + c, value);
                    }
                }
            }
        }
        let tile = Arc::new(Tile::from_vec(bytes));
        tile.cell.set_rebuild_cost(level);
        if cache {
            let entry = Entry { tile: tile.clone(), _sources: std::array::from_fn(|i| sources.get(i).map(Arc::downgrade)) };
            self.mips.tiles.lock().unwrap_or_else(PoisonError::into_inner).insert((level, coord), entry);
        }
        Ok(tile)
    }
    /// Construct only the mip tiles intersecting `bounds`, expressed in base coordinates.
    pub fn mip_surface(&self, level: u32, bounds: Rect) -> Result<Surface, String> {
        if level > MAX_LEVEL {
            return Err("CPU mip level exceeds 16".into());
        }
        let region = Self::mip_bounds(bounds, level);
        if region.is_empty() {
            return Ok(Surface::with_default(self.format, &self.default_pixel()));
        }
        let first = TileCoord::containing(region.x0, region.y0);
        let last = TileCoord::containing(region.x1.saturating_sub(1), region.y1.saturating_sub(1));
        let columns = (i64::from(last.tx) - i64::from(first.tx) + 1).max(0) as u64;
        let rows = (i64::from(last.ty) - i64::from(first.ty) + 1).max(0) as u64;
        if columns.checked_mul(rows).is_none_or(|tiles| tiles > 65_536) {
            return Err("CPU mip region exceeds 65536 tiles; request a viewport or a coarser mip".into());
        }
        let mut surface = Surface::with_default(self.format, &self.default_pixel());
        for coord in region.tiles() {
            surface.insert_tile(coord, self.mip_tile(level, coord)?)?;
        }
        Ok(surface)
    }
    /// Snapshot of valid derived tiles; saving does not generate missing mip levels.
    pub fn cached_mips(&self) -> Vec<(u32, TileCoord, Arc<Tile>)> {
        if std::env::var("PHOTOCRAFT_CPU_MIPS").as_deref() == Ok("0") {
            return Vec::new();
        }
        self.mips.tiles.lock().unwrap_or_else(PoisonError::into_inner).iter().map(|((level, coord), entry)| (*level, *coord, entry.tile.clone())).collect()
    }
    /// Install a persisted derived tile after base tiles have been loaded.
    pub fn install_mip(&mut self, level: u32, coord: TileCoord, tile: Arc<Tile>) -> Result<(), String> {
        if level == 0 || level > MAX_LEVEL || tile.len() != TILE_SIZE as usize * TILE_SIZE as usize * self.format.bytes_per_pixel() {
            return Err("invalid persisted mip tile".into());
        }
        if std::env::var("PHOTOCRAFT_CPU_MIPS").as_deref() == Ok("0") {
            return Ok(());
        }
        tile.cell.set_rebuild_cost(level);
        let chain = Arc::make_mut(&mut self.mips);
        let entries = chain.tiles.get_mut().unwrap_or_else(PoisonError::into_inner);
        if entries.insert((level, coord), Entry { tile, _sources: std::array::from_fn(|_| None) }).is_some() {
            return Err("duplicate persisted mip tile".into());
        }
        Ok(())
    }
}
