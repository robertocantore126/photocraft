//! Scratch disk: tile pixels beyond a memory budget move to a file and come back on access.
//!
//! Every [`Tile`](crate::Tile) keeps its bytes in a [`Cell`]: resident (`Some`) or on disk
//! (`None`, with its [`Extent`] in the scratch file). Readers take the cell's read lock, so a
//! tile in use is never evicted; a spilled tile is read back (LZ4 + CRC-checked) by whoever
//! touches it first.
//!
//! - **Accounting.** [`stats`] counts the bytes of every live tile that is resident, whether or
//!   not a scratch disk is configured.
//! - **Eviction** is the clock ("second chance") algorithm over a registry of weak cell
//!   pointers: a tile touched since the hand last passed keeps its place once. A background
//!   thread evicts down to 90 % of the budget; a thread that pushes memory past 125 % helps.
//! - **Clean tiles are free to drop.** A tile written once and not changed since keeps its
//!   extent, so evicting it again only frees memory; changing it frees the extent.
//! - **Off by default.** Until [`configure`] sets a directory and a budget nothing is evicted
//!   (tests, the web build).
//!
//! Failures never panic: a scratch write that fails keeps the tile in memory and turns spilling
//! off (reported in [`SpillStats::last_error`]); a read that fails gives a tile of zeros and is
//! reported the same way.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};

use crate::scratch_file::{Extent, ScratchFile};

/// Persistent compressed tile extent. Bytes use LZ4 block encoding and CRC-32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PackedExtent {
    pub offset: u64,
    pub len: u32,
    pub crc32: u32,
}

/// Read-only document backing; retained by every lazy tile and never freed by eviction.
pub trait TileBacking: Send + Sync + std::fmt::Debug {
    fn identity(&self) -> [u8; 16];
    /// Durable commit marker used to distinguish divergent copies of one pack.
    fn anchor(&self) -> Option<(u64, [u8; 40])> {
        None
    }
    fn read_packed(&self, extent: PackedExtent) -> Result<Vec<u8>, String>;
}

#[derive(Clone)]
enum Slot {
    Scratch(Arc<ScratchFile>, Extent),
    Pack(Arc<dyn TileBacking>, PackedExtent),
}
impl Slot {
    fn read(&self) -> Result<Vec<u8>, String> {
        match self {
            Self::Scratch(file, extent) => {
                let mut bytes = Vec::new();
                file.read_into(*extent, &mut bytes).map_err(|e| e.to_string())?;
                Ok(bytes)
            }
            Self::Pack(file, extent) => file.read_packed(*extent),
        }
    }
    fn release(self) {
        if let Self::Scratch(file, extent) = self {
            file.free(extent);
        }
    }
}

/// One tile's bytes and where they live.
pub(crate) struct Cell {
    data: RwLock<Option<Vec<u8>>>,
    /// Byte length of the tile (fixed for its lifetime).
    len: usize,
    error: Mutex<Option<String>>,
    /// A copy on disk that matches `data` (kept while the tile is unchanged).
    slot: Mutex<Option<Slot>>,
    // Keep a few immutable pack aliases so autosave and Save As do not invalidate normal saves.
    packs: Mutex<Vec<(Arc<dyn TileBacking>, PackedExtent)>>,
    /// Touched since the clock hand last passed.
    referenced: AtomicBool,
    rebuild_cost: AtomicUsize,
    chances: AtomicUsize,
}

impl Cell {
    pub(crate) fn new(bytes: Vec<u8>) -> Arc<Cell> {
        let len = bytes.len();
        let cell = Arc::new(Cell {
            data: RwLock::new(Some(bytes)),
            len,
            error: Mutex::new(None),
            slot: Mutex::new(None),
            packs: Mutex::new(Vec::new()),
            referenced: AtomicBool::new(true),
            rebuild_cost: AtomicUsize::new(0),
            chances: AtomicUsize::new(0),
        });
        M.live.fetch_add(1, Ordering::Relaxed);
        M.resident.fetch_add(len, Ordering::Relaxed);
        M.register(&cell);
        maybe_evict();
        cell
    }

    pub(crate) fn lazy(len: usize, backing: Arc<dyn TileBacking>, extent: PackedExtent) -> Arc<Cell> {
        let cell = Arc::new(Cell {
            data: RwLock::new(None),
            len,
            error: Mutex::new(None),
            slot: Mutex::new(Some(Slot::Pack(backing, extent))),
            packs: Mutex::new(Vec::new()),
            referenced: AtomicBool::new(false),
            rebuild_cost: AtomicUsize::new(0),
            chances: AtomicUsize::new(0),
        });
        M.live.fetch_add(1, Ordering::Relaxed);
        M.register(&cell);
        cell
    }

    pub(crate) fn set_rebuild_cost(&self, level: u32) {
        let cost = 1usize << level.saturating_mul(2).min(8);
        self.rebuild_cost.fetch_max(cost, Ordering::Relaxed);
        self.chances.store(cost, Ordering::Relaxed);
    }

    pub(crate) fn is_resident(&self) -> bool {
        self.data.read().unwrap_or_else(PoisonError::into_inner).is_some()
    }
    pub(crate) fn read_error(&self) -> Option<String> {
        self.error.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }
    pub(crate) fn set_read_error(&self, error: Option<String>) {
        *self.error.lock().unwrap_or_else(PoisonError::into_inner) = error;
    }
    pub(crate) fn pack_backing(&self, id: [u8; 16]) -> Option<(Arc<dyn TileBacking>, PackedExtent)> {
        if self.read_error().is_some() {
            return None;
        }
        let slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        match slot.as_ref() {
            Some(Slot::Pack(file, extent)) if file.identity() == id => Some((file.clone(), *extent)),
            _ => self.packs.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(file, _)| file.identity() == id).cloned(),
        }
    }
    pub(crate) fn pack_extent(&self, id: [u8; 16]) -> Option<PackedExtent> {
        self.pack_backing(id).map(|(_, extent)| extent)
    }

    pub(crate) fn packed(&self) -> Result<Vec<u8>, String> {
        // Hold the pixel read lock before the slot lock, as in eviction and mutation.
        let data = self.data.read().unwrap_or_else(PoisonError::into_inner);
        if let Some(error) = self.read_error() {
            return Err(error);
        }
        let slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        match slot.as_ref() {
            Some(slot) => slot.read(),
            None => data.as_ref().map(|bytes| lz4_flex::block::compress(bytes)).ok_or_else(|| "tile has no readable backing".into()),
        }
    }

    pub(crate) fn attach(&self, backing: Arc<dyn TileBacking>, extent: PackedExtent) {
        let _data = self.data.read().unwrap_or_else(PoisonError::into_inner);
        let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(old) = slot.replace(Slot::Pack(backing.clone(), extent)) {
            match old {
                Slot::Pack(file, old_extent) if file.identity() != backing.identity() => {
                    let mut packs = self.packs.lock().unwrap_or_else(PoisonError::into_inner);
                    packs.retain(|(old, _)| old.identity() != file.identity() && old.identity() != backing.identity());
                    if packs.len() >= 4 {
                        packs.remove(0);
                    }
                    packs.push((file, old_extent));
                }
                old => old.release(),
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn touch(&self) {
        if !self.referenced.load(Ordering::Relaxed) {
            self.referenced.store(true, Ordering::Relaxed);
        }
    }

    /// Read access; reads the tile back from the scratch disk if it was evicted.
    pub(crate) fn read(&self) -> RwLockReadGuard<'_, Option<Vec<u8>>> {
        self.touch();
        loop {
            let g = self.data.read().unwrap_or_else(PoisonError::into_inner);
            if g.is_some() {
                return g;
            }
            drop(g);
            let mut w = self.data.write().unwrap_or_else(PoisonError::into_inner);
            self.load(&mut w);
            drop(w);
            maybe_evict();
        }
    }

    /// Write access: resident, and any disk copy is released (it no longer matches).
    pub(crate) fn write(&self) -> RwLockWriteGuard<'_, Option<Vec<u8>>> {
        self.touch();
        let mut w = self.data.write().unwrap_or_else(PoisonError::into_inner);
        let loaded = w.is_none();
        self.load(&mut w);
        if let Some(slot) = self.slot.lock().unwrap_or_else(PoisonError::into_inner).take() {
            slot.release();
        }
        self.packs.lock().unwrap_or_else(PoisonError::into_inner).clear();
        if loaded {
            // Evicting here could only pick other tiles (this one is locked), so it is safe, and
            // a long run of writes to spilled tiles must not outgrow the budget.
            maybe_evict();
        }
        w
    }

    /// Fill `w` from the scratch disk if it is not resident (`w` is this cell's write guard).
    fn load(&self, w: &mut Option<Vec<u8>>) {
        if w.is_some() {
            return;
        }
        let slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let _reading = ActiveRead::new();
        let bytes = match slot {
            Some(slot) => match slot.read().and_then(|packed| unpack(&packed, self.len)) {
                Ok(b) => b,
                Err(e) => {
                    self.set_read_error(Some(e.clone()));
                    M.error(format!("tile backing read failed ({e}); a tile was lost"));
                    vec![0; self.len]
                }
            },
            None => {
                self.set_read_error(Some("tile has no readable backing".into()));
                M.error("a spilled tile had no scratch copy; it was lost".into());
                vec![0; self.len]
            }
        };
        *w = Some(bytes);
        M.resident.fetch_add(self.len, Ordering::Relaxed);
        M.reloads.fetch_add(1, Ordering::Relaxed);
    }

    /// Prepare a durable copy without evicting the resident pixels. Mutation holds
    /// the exclusive pixel lock, so it cannot invalidate a slot while this is written.
    #[cfg(not(target_arch = "wasm32"))]
    fn prewrite(&self, file: &Arc<ScratchFile>) -> Result<(), String> {
        if M.reading.load(Ordering::Acquire) != 0 {
            return Ok(());
        }
        let Ok(data) = self.data.try_read() else { return Ok(()) };
        let Some(bytes) = data.as_ref() else { return Ok(()) };
        let Ok(mut slot) = self.slot.try_lock() else { return Ok(()) };
        if slot.is_some() || self.read_error().is_some() {
            return Ok(());
        }
        let packed = lz4_flex::block::compress(bytes);
        if M.reading.load(Ordering::Acquire) != 0 {
            return Ok(());
        }
        let extent = file.write(&packed).map_err(|e| format!("scratch write-behind failed: {e}"))?;
        M.written.fetch_add(packed.len() as u64, Ordering::Relaxed);
        *slot = Some(Slot::Scratch(file.clone(), extent));
        Ok(())
    }

    /// Move this tile to disk if nobody is using it. Returns the bytes freed.
    fn evict(&self, file: &Arc<ScratchFile>) -> Result<usize, String> {
        let Ok(mut w) = self.data.try_write() else { return Ok(0) };
        let Some(bytes) = w.as_ref() else { return Ok(0) };
        let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            let packed = lz4_flex::block::compress(bytes);
            let extent = file.write(&packed).map_err(|e| format!("scratch write to {} failed: {e}", file.path().display()))?;
            M.written.fetch_add(packed.len() as u64, Ordering::Relaxed);
            *slot = Some(Slot::Scratch(file.clone(), extent));
        }
        drop(slot);
        *w = None;
        M.resident.fetch_sub(self.len, Ordering::Relaxed);
        M.evictions.fetch_add(1, Ordering::Relaxed);
        Ok(self.len)
    }
}

impl Drop for Cell {
    fn drop(&mut self) {
        let resident = self.data.get_mut().unwrap_or_else(PoisonError::into_inner).is_some();
        if resident {
            M.resident.fetch_sub(self.len, Ordering::Relaxed);
        }
        if let Some(slot) = self.slot.get_mut().unwrap_or_else(PoisonError::into_inner).take() {
            slot.release();
        }
        M.live.fetch_sub(1, Ordering::Relaxed);
    }
}

fn unpack(packed: &[u8], len: usize) -> Result<Vec<u8>, String> {
    let bytes = lz4_flex::block::decompress(packed, len).map_err(|e| e.to_string())?;
    if bytes.len() == len { Ok(bytes) } else { Err(format!("tile decoded to {} bytes, expected {len}", bytes.len())) }
}

/// Process-wide state. Plain statics: everything here is const-constructible.
struct Manager {
    resident: AtomicUsize,
    live: AtomicUsize,
    /// Pixel bytes to keep resident; 0 = spilling off.
    budget: AtomicUsize,
    registry: Mutex<Registry>,
    config: Mutex<Config>,
    wake: Condvar,
    evictions: AtomicU64,
    reloads: AtomicU64,
    written: AtomicU64,
    reading: AtomicUsize,
}

struct Registry {
    cells: Vec<Weak<Cell>>,
    hand: usize,
}

struct Config {
    dir: Option<PathBuf>,
    /// Created on the first eviction, so configuring costs nothing until memory runs short.
    file: Option<Arc<ScratchFile>>,
    worker: bool,
    writer: bool,
    pending: bool,
    last_error: Option<String>,
}

static M: Manager = Manager {
    resident: AtomicUsize::new(0),
    live: AtomicUsize::new(0),
    budget: AtomicUsize::new(0),
    registry: Mutex::new(Registry { cells: Vec::new(), hand: 0 }),
    config: Mutex::new(Config { dir: None, file: None, worker: false, writer: false, pending: false, last_error: None }),
    wake: Condvar::new(),
    evictions: AtomicU64::new(0),
    reloads: AtomicU64::new(0),
    written: AtomicU64::new(0),
    reading: AtomicUsize::new(0),
};

impl Manager {
    fn register(&self, cell: &Arc<Cell>) {
        let mut r = self.registry.lock().unwrap_or_else(PoisonError::into_inner);
        r.cells.push(Arc::downgrade(cell));
        // Drop the pointers of freed tiles once they outnumber the live ones.
        if r.cells.len() > 2 * self.live.load(Ordering::Relaxed) + 4096 {
            r.cells.retain(|w| w.strong_count() > 0);
            r.hand = 0;
        }
    }

    fn config(&self) -> std::sync::MutexGuard<'_, Config> {
        self.config.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn error(&self, msg: String) {
        log_error(&msg);
        self.config().last_error = Some(msg);
    }

    /// The scratch file, created on first use; `None` (and spilling off) if it can't be.
    fn file(&self) -> Option<Arc<ScratchFile>> {
        let mut c = self.config();
        if let Some(f) = &c.file {
            return Some(f.clone());
        }
        let dir = c.dir.clone()?;
        match ScratchFile::create(&dir) {
            Ok(f) => {
                let f = Arc::new(f);
                c.file = Some(f.clone());
                Some(f)
            }
            Err(e) => {
                let msg = format!("could not create a scratch file in {}: {e}; spilling is off", dir.display());
                log_error(&msg);
                c.last_error = Some(msg);
                c.dir = None;
                self.budget.store(0, Ordering::Relaxed);
                None
            }
        }
    }

    /// Evict until resident memory is at most `target` or `max_scan` cells were looked at.
    /// Returns the bytes freed.
    fn sweep(&self, target: usize, max_scan: usize) -> usize {
        let Some(file) = self.file() else { return 0 };
        let mut freed = 0;
        let mut scanned = 0;
        while self.resident.load(Ordering::Relaxed) > target && scanned < max_scan {
            // Pick victims under the registry lock (cheap), evict them after releasing it.
            let mut victims = Vec::with_capacity(64);
            {
                let mut r = self.registry.lock().unwrap_or_else(PoisonError::into_inner);
                let mut i = 0;
                while victims.len() < 64 && i < 4096 && !r.cells.is_empty() {
                    if r.hand >= r.cells.len() {
                        r.hand = 0;
                    }
                    let h = r.hand;
                    i += 1;
                    match r.cells.get(h).and_then(Weak::upgrade) {
                        None => {
                            r.cells.swap_remove(h);
                            continue;
                        }
                        Some(cell) => {
                            r.hand += 1;
                            if cell.referenced.swap(false, Ordering::Relaxed) {
                                cell.chances.store(cell.rebuild_cost.load(Ordering::Relaxed), Ordering::Relaxed);
                                continue;
                            }
                            if std::env::var("PHOTOCRAFT_MIP_PRIORITY").as_deref() != Ok("0")
                                && cell.chances.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| value.checked_sub(1)).is_ok()
                            {
                                continue;
                            }
                            if cell.data.try_read().is_ok_and(|d| d.is_some()) {
                                victims.push(cell);
                            }
                        }
                    }
                }
                scanned += i;
                if i == 0 {
                    break;
                }
            }
            for cell in victims {
                match cell.evict(&file) {
                    Ok(n) => freed += n,
                    Err(e) => {
                        // Disk full or gone: keep everything in memory from now on.
                        self.budget.store(0, Ordering::Relaxed);
                        self.error(format!("{e}; spilling is off"));
                        return freed;
                    }
                }
            }
        }
        freed
    }
}

/// Called after resident memory grew.
fn maybe_evict() {
    let budget = M.budget.load(Ordering::Relaxed);
    if budget == 0 {
        return;
    }
    let resident = M.resident.load(Ordering::Relaxed);
    if resident > budget / 5 * 4 {
        M.wake.notify_all();
    }
    if resident <= budget {
        return;
    }
    let worker = {
        let mut c = M.config();
        c.pending = true;
        c.worker
    };
    M.wake.notify_one();
    // Far over budget (or no worker thread): this thread helps.
    if !worker || resident > budget.saturating_add(budget / 4) {
        M.sweep(low_water(budget), 1 << 16);
    }
}

fn low_water(budget: usize) -> usize {
    budget / 10 * 9
}

fn worker_loop() {
    loop {
        {
            let mut c = M.config();
            while !c.pending {
                c = M.wake.wait(c).unwrap_or_else(PoisonError::into_inner);
            }
            c.pending = false;
        }
        loop {
            let budget = M.budget.load(Ordering::Relaxed);
            if budget == 0 || M.resident.load(Ordering::Relaxed) <= budget {
                break;
            }
            if M.sweep(low_water(budget), 1 << 16) == 0 {
                // Everything is in use or was just touched: give the clock time to age it.
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }
}

/// Turn the scratch disk on (`dir` and a non-zero `budget_bytes`) or off. Tiles already on disk
/// stay readable either way. Changing the directory starts a new scratch file there; the old one
/// is deleted once its last tile is freed or changed.
pub fn configure(dir: Option<&Path>, budget_bytes: usize) -> Result<(), String> {
    let mut c = M.config();
    let dir = dir.filter(|_| budget_bytes > 0).map(Path::to_path_buf);
    if let Some(d) = &dir
        && let Err(e) = std::fs::create_dir_all(d)
    {
        let msg = format!("scratch disk {}: {e}; spilling is off", d.display());
        c.dir = None;
        c.file = None;
        c.last_error = Some(msg.clone());
        M.budget.store(0, Ordering::Relaxed);
        return Err(msg);
    }
    if dir != c.dir {
        c.file = None;
    }
    c.dir = dir.clone();
    c.last_error = None;
    #[cfg(not(target_arch = "wasm32"))]
    if dir.is_some() && !c.worker {
        c.worker = std::thread::Builder::new().name("photocraft-spill".into()).spawn(worker_loop).is_ok();
    }
    #[cfg(not(target_arch = "wasm32"))]
    if dir.is_some() && !c.writer && std::env::var("PHOTOCRAFT_WRITE_BEHIND").as_deref() != Ok("0") {
        c.writer = std::thread::Builder::new().name("photocraft-write-behind".into()).spawn(write_behind_loop).is_ok();
    }
    M.budget.store(if dir.is_some() { budget_bytes } else { 0 }, Ordering::Relaxed);
    drop(c);
    maybe_evict();
    Ok(())
}

/// Where the tile memory stands.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpillStats {
    /// Pixel bytes of live tiles in memory.
    pub resident_bytes: usize,
    pub live_tiles: usize,
    /// 0 = spilling off.
    pub budget_bytes: usize,
    pub dir: Option<PathBuf>,
    /// Bytes allocated in the current scratch file.
    pub scratch_bytes: u64,
    pub evictions: u64,
    pub reloads: u64,
    /// Compressed bytes written to scratch so far.
    pub written_bytes: u64,
    pub last_error: Option<String>,
}

pub fn stats() -> SpillStats {
    let c = M.config();
    SpillStats {
        resident_bytes: M.resident.load(Ordering::Relaxed),
        live_tiles: M.live.load(Ordering::Relaxed),
        budget_bytes: M.budget.load(Ordering::Relaxed),
        dir: c.dir.clone(),
        scratch_bytes: c.file.as_ref().map_or(0, |f| f.used()),
        evictions: M.evictions.load(Ordering::Relaxed),
        reloads: M.reloads.load(Ordering::Relaxed),
        written_bytes: M.written.load(Ordering::Relaxed),
        last_error: c.last_error.clone(),
    }
}

/// Whether spilling is on (tiles beyond the budget go to disk).
pub fn enabled() -> bool {
    M.budget.load(Ordering::Relaxed) > 0
}

/// Evict down to the budget now, on this thread (tests, and before a measurement).
pub fn settle() {
    let budget = M.budget.load(Ordering::Relaxed);
    // Stops when nothing more can be evicted (every remaining tile is in use).
    while budget > 0 && M.resident.load(Ordering::Relaxed) > budget {
        if M.sweep(budget, usize::MAX) == 0 {
            break;
        }
    }
}

fn log_error(msg: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("photocraft: {msg}");
    #[cfg(target_arch = "wasm32")]
    let _ = msg;
}

// Reads always take precedence over speculative scratch writes.
struct ActiveRead;
impl ActiveRead {
    fn new() -> Self {
        M.reading.fetch_add(1, Ordering::AcqRel);
        Self
    }
}
impl Drop for ActiveRead {
    fn drop(&mut self) {
        M.reading.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn write_behind_loop() {
    let mut hand = 0usize;
    loop {
        let config = M.config();
        let (config, _) = M.wake.wait_timeout(config, std::time::Duration::from_millis(100)).unwrap_or_else(PoisonError::into_inner);
        drop(config);
        let budget = M.budget.load(Ordering::Relaxed);
        if budget == 0
            || M.resident.load(Ordering::Relaxed) < budget / 5 * 4
            || M.reading.load(Ordering::Acquire) != 0
            || std::env::var("PHOTOCRAFT_WRITE_BEHIND").as_deref() == Ok("0")
        {
            continue;
        }
        let Some(file) = M.file() else { continue };
        let cells = {
            let registry = M.registry.lock().unwrap_or_else(PoisonError::into_inner);
            let mut cells = Vec::new();
            for _ in 0..64.min(registry.cells.len()) {
                if hand >= registry.cells.len() {
                    hand = 0;
                }
                if let Some(cell) = registry.cells.get(hand).and_then(Weak::upgrade) {
                    cells.push(cell);
                }
                hand = hand.saturating_add(1);
            }
            cells
        };
        // Cold tiles first; warm dirty tiles are prepared on a later pass too.
        let mut cells = cells;
        cells.sort_by_key(|cell| cell.referenced.load(Ordering::Relaxed));
        for cell in cells {
            if M.reading.load(Ordering::Acquire) != 0 || M.budget.load(Ordering::Relaxed) == 0 {
                break;
            }
            if let Err(error) = cell.prewrite(&file) {
                M.error(error);
                M.budget.store(0, Ordering::Relaxed);
                break;
            }
            std::thread::yield_now();
        }
    }
}
