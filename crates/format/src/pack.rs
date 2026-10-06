//! Append-only native document packs. A commit is one CRC-checked manifest and footer.
//! Tile records reuse scratch LZ4 blocks and are attached only after the footer is durable.
use crate::{
    FormatError, LoadOptions, Result, SaveOptions, SaveStats,
    convert::{self, Fetch, Loader, Sink},
    manifest::{FORMAT_VERSION, Manifest},
};
use photocraft_color::PixelFormat;
use photocraft_doc::Document;
use photocraft_raster::{
    Tile,
    spill::{PackedExtent, TileBacking},
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex, PoisonError, Weak};

pub const MAGIC: &[u8; 8] = b"PCRAFT2\0";
const FOOTER_MAGIC: &[u8; 8] = b"PCOMMIT2";
const HEADER: u64 = 32;
const FOOTER: usize = 40;
/// Unreferenced bytes a pack may carry before a save rewrites it compactly (also at least as
/// many as it references).
const COMPACT_GARBAGE: u64 = 64 << 20;

/// Whether the pack a save appended to is mostly superseded data: more unreferenced bytes than
/// referenced ones, and at least [`COMPACT_GARBAGE`].
pub fn wants_compaction(stats: &SaveStats) -> bool {
    let garbage = stats.pack_bytes.saturating_sub(stats.pack_live_bytes);
    garbage > stats.pack_live_bytes && garbage >= COMPACT_GARBAGE
}

trait Source: Send + Sync {
    fn len(&self) -> std::io::Result<u64>;
    fn read(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>>;
}
/// A factory retaining the caller's filesystem authority for independent reader handles.
pub type ReaderOpener = Arc<dyn Fn() -> std::io::Result<File> + Send + Sync>;
struct FileSource {
    main: Arc<File>,
    readers: Mutex<HashMap<std::thread::ThreadId, Arc<File>>>,
    opener: Option<ReaderOpener>,
    id: Option<[u8; 16]>,
}
impl FileSource {
    fn new(file: File, opener: Option<ReaderOpener>) -> Self {
        let mut header = [0; 32];
        let id = positioned_read(&file, &mut header, 0).ok().and_then(|()| header.get(8..24).and_then(|b| b.try_into().ok()));
        Self { main: Arc::new(file), readers: Mutex::new(HashMap::new()), opener, id }
    }
    fn reader(&self) -> Arc<File> {
        let Some(opener) = self.opener.as_ref() else { return self.main.clone() };
        let thread = std::thread::current().id();
        let mut readers = self.readers.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(file) = readers.get(&thread) {
            return file.clone();
        }
        if readers.len() >= 32 {
            return self.main.clone();
        }
        // A replaced pathname must never redirect an old document snapshot to another pack.
        let file = opener()
            .ok()
            .filter(|file| {
                let mut header = [0; 32];
                positioned_read(file, &mut header, 0).is_ok() && self.id.as_ref().is_some_and(|id| header.get(8..24) == Some(id.as_slice()))
            })
            .map(Arc::new)
            .unwrap_or_else(|| self.main.clone());
        readers.insert(thread, file.clone());
        file
    }
}
fn positioned_read(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            let n = file.seek_read(buf, offset)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            buf = buf.get_mut(n..).ok_or(std::io::ErrorKind::UnexpectedEof)?;
            offset = offset.checked_add(n as u64).ok_or(std::io::ErrorKind::UnexpectedEof)?;
        }
        Ok(())
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (file, buf, offset);
        Err(std::io::ErrorKind::Unsupported.into())
    }
}
impl Source for FileSource {
    fn len(&self) -> std::io::Result<u64> {
        self.main.metadata().map(|m| m.len())
    }
    fn read(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        let mut out = vec![0; len];
        positioned_read(&self.reader(), &mut out, offset)?;
        Ok(out)
    }
}
struct SliceSource<'a>(&'a [u8]);
impl Source for SliceSource<'_> {
    fn len(&self) -> std::io::Result<u64> {
        Ok(self.0.len() as u64)
    }
    fn read(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        let start = usize::try_from(offset).map_err(|_| std::io::ErrorKind::UnexpectedEof)?;
        let end = start.checked_add(len).ok_or(std::io::ErrorKind::UnexpectedEof)?;
        self.0.get(start..end).map(|bytes| bytes.to_vec()).ok_or_else(|| std::io::ErrorKind::UnexpectedEof.into())
    }
}
struct MemorySource(Arc<Vec<u8>>);
impl Source for MemorySource {
    fn len(&self) -> std::io::Result<u64> {
        Ok(self.0.len() as u64)
    }
    fn read(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        let start = usize::try_from(offset).map_err(|_| std::io::ErrorKind::UnexpectedEof)?;
        let end = start.checked_add(len).ok_or(std::io::ErrorKind::UnexpectedEof)?;
        self.0.get(start..end).map(|b| b.to_vec()).ok_or_else(|| std::io::ErrorKind::UnexpectedEof.into())
    }
}
struct Backing {
    source: Arc<dyn Source>,
    id: [u8; 16],
    limit: u64,
    anchor: (u64, [u8; 40]),
}
impl std::fmt::Debug for Backing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackBacking").field("id", &self.id).finish()
    }
}
impl TileBacking for Backing {
    fn identity(&self) -> [u8; 16] {
        self.id
    }
    fn anchor(&self) -> Option<(u64, [u8; 40])> {
        Some(self.anchor)
    }
    fn read_packed(&self, extent: PackedExtent) -> std::result::Result<Vec<u8>, String> {
        validate_extent(extent, self.limit).map_err(|e| e.to_string())?;
        let bytes = self.source.read(extent.offset, extent.len as usize).map_err(|e| e.to_string())?;
        if crc32fast::hash(&bytes) != extent.crc32 {
            return Err("document tile CRC mismatch".into());
        }
        Ok(bytes)
    }
}
fn validate_extent(e: PackedExtent, limit: u64) -> Result<()> {
    if e.offset < HEADER || e.offset.checked_add(u64::from(e.len)).is_none_or(|end| end > limit) {
        return Err(FormatError::corrupt("pack extent outside committed data"));
    }
    Ok(())
}
fn number(bytes: &[u8], start: usize) -> Option<u64> {
    Some(u64::from_le_bytes(bytes.get(start..start.checked_add(8)?)?.try_into().ok()?))
}
fn word(bytes: &[u8], start: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(start..start.checked_add(4)?)?.try_into().ok()?))
}
fn footer_crc(bytes: &[u8]) -> Option<u32> {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes.get(..28)?);
    h.update(bytes.get(32..40)?);
    Some(h.finalize())
}
fn header_id(source: &dyn Source) -> Result<[u8; 16]> {
    let bytes = source.read(0, HEADER as usize)?;
    if bytes.get(..8) != Some(MAGIC.as_slice()) || number(&bytes, 24) != Some(1) {
        return Err(FormatError::corrupt("invalid pack header"));
    }
    bytes.get(8..24).and_then(|b| b.try_into().ok()).ok_or_else(|| FormatError::corrupt("invalid pack identity"))
}
/// Find the newest complete commit, including after arbitrary trailing partial writes.
fn commit(source: &dyn Source, opts: &LoadOptions) -> Result<(Manifest, u64, (u64, [u8; 40]))> {
    let size = source.len()?;
    let mut end = size;
    while end >= HEADER + FOOTER as u64 {
        let start = end.saturating_sub(65536).max(HEADER);
        let bytes = source.read(start, (end - start) as usize)?;
        for i in (0..bytes.len().saturating_sub(FOOTER - 1)).rev() {
            let Some(f) = bytes.get(i..i + FOOTER) else { continue };
            if f.get(..8) != Some(FOOTER_MAGIC.as_slice()) || word(f, 28) != footer_crc(f) {
                continue;
            }
            let Some(offset) = number(f, 8) else { continue };
            let Some(len) = number(f, 16) else { continue };
            let position = start + i as u64;
            if number(f, 32) != Some(position + FOOTER as u64)
                || len > opts.max_manifest_bytes as u64
                || offset < HEADER
                || offset.checked_add(len) != Some(position)
            {
                continue;
            }
            let raw = source.read(offset, len as usize)?;
            if Some(crc32fast::hash(&raw)) != word(f, 24) {
                continue;
            }
            let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(&raw) else { continue };
            crate::migrate::migrate(&mut value)?;
            let manifest = serde_json::from_value(value)?;
            let anchor = f.try_into().map_err(|_| FormatError::corrupt("invalid footer size"))?;
            return Ok((manifest, offset, (position, anchor)));
        }
        if start == HEADER {
            break;
        }
        end = start + FOOTER as u64 - 1;
    }
    Err(FormatError::corrupt("pack contains no complete commit"))
}
fn reference(e: PackedExtent, raw: usize) -> String {
    format!("{}:{}:{}:{}", e.offset, e.len, e.crc32, raw)
}
fn parse_reference(s: &str, limit: u64) -> Result<(PackedExtent, usize)> {
    let mut parts = s.split(':');
    let mut next = || parts.next().and_then(|s| s.parse::<u64>().ok()).ok_or_else(|| FormatError::corrupt("invalid pack reference"));
    let offset = next()?;
    let len = u32::try_from(next()?).map_err(|_| FormatError::corrupt("oversized packed block"))?;
    let crc32 = u32::try_from(next()?).map_err(|_| FormatError::corrupt("invalid block checksum"))?;
    let raw = usize::try_from(next()?).map_err(|_| FormatError::corrupt("oversized raw block"))?;
    if parts.next().is_some() {
        return Err(FormatError::corrupt("invalid pack reference"));
    }
    let extent = PackedExtent { offset, len, crc32 };
    validate_extent(extent, limit)?;
    Ok((extent, raw))
}
/// Where a blob already lies in a pack. The pack is held weakly: this process-wide cache must
/// not keep a closed document's file open (its tiles hold the pack while the document lives).
#[derive(Clone)]
struct BlobSlot {
    bytes: Weak<Vec<u8>>,
    backing: Weak<Backing>,
    id: [u8; 16],
    extent: PackedExtent,
}
static BLOBS: LazyLock<Mutex<HashMap<usize, Vec<BlobSlot>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
fn prune_blobs() {
    BLOBS.lock().unwrap_or_else(PoisonError::into_inner).retain(|_, slots| {
        slots.retain(|slot| slot.bytes.strong_count() > 0 && slot.backing.strong_count() > 0);
        !slots.is_empty()
    });
}
fn remember_blob(blob: &Arc<Vec<u8>>, backing: &Arc<Backing>, extent: PackedExtent) {
    let mut cache = BLOBS.lock().unwrap_or_else(PoisonError::into_inner);

    let slots = cache.entry(Arc::as_ptr(blob) as usize).or_default();
    slots.retain(|slot| slot.bytes.upgrade().is_some_and(|b| Arc::ptr_eq(&b, blob)) && slot.backing.strong_count() > 0 && slot.id != backing.id);
    if slots.len() >= 4 {
        slots.remove(0);
    }
    slots.push(BlobSlot { bytes: Arc::downgrade(blob), backing: Arc::downgrade(backing), id: backing.id, extent });
}
struct PackFetch {
    backing: Arc<Backing>,
    opts: LoadOptions,
    total: u64,
    tiles: HashMap<String, Arc<Tile>>,
    blobs: HashMap<String, Arc<Vec<u8>>>,
}
impl PackFetch {
    fn account(&mut self, n: usize) -> Result<()> {
        self.total = self.total.checked_add(n as u64).ok_or_else(|| FormatError::LimitExceeded("pack byte count overflow".into()))?;
        if self.total > self.opts.max_total_bytes {
            return Err(FormatError::LimitExceeded("pack exceeds total byte limit".into()));
        }
        Ok(())
    }
    fn decoded(&self, e: PackedExtent, raw: usize) -> Result<Vec<u8>> {
        let packed = self.backing.read_packed(e).map_err(FormatError::corrupt)?;
        let bytes = lz4_flex::block::decompress(&packed, raw).map_err(|e| FormatError::corrupt(e.to_string()))?;
        if bytes.len() != raw {
            return Err(FormatError::corrupt("pack decoded length mismatch"));
        }
        Ok(bytes)
    }
}
impl Fetch for PackFetch {
    fn tile(&mut self, s: &str, len: usize) -> Result<Vec<u8>> {
        let (e, raw) = parse_reference(s, self.backing.limit)?;
        if raw != len {
            return Err(FormatError::corrupt("pack tile length mismatch"));
        }
        self.decoded(e, raw)
    }
    fn tile_object(&mut self, s: &str, format: PixelFormat) -> Result<Arc<Tile>> {
        let (e, raw) = parse_reference(s, self.backing.limit)?;
        if raw != convert::tile_len(&format) || e.len as usize > raw.saturating_add(raw / 255).saturating_add(32) {
            return Err(FormatError::corrupt("pack tile size mismatch"));
        }
        if let Some(tile) = self.tiles.get(s) {
            return Ok(tile.clone());
        }
        self.account(raw)?;
        let tile = if cfg!(target_endian = "big") {
            let mut bytes = self.decoded(e, raw)?;
            convert::swap_to_le(&mut bytes, format.sample);
            Arc::new(Tile::from_vec(bytes))
        } else {
            Arc::new(Tile::lazy(raw, self.backing.clone(), e))
        };
        self.tiles.insert(s.into(), tile.clone());
        Ok(tile)
    }
    fn blob(&mut self, s: &str) -> Result<Arc<Vec<u8>>> {
        if let Some(blob) = self.blobs.get(s) {
            return Ok(blob.clone());
        }
        let (e, raw) = parse_reference(s, self.backing.limit)?;
        if raw > self.opts.max_blob_bytes || e.len as usize > raw.saturating_add(raw / 255).saturating_add(32) {
            return Err(FormatError::LimitExceeded("pack blob exceeds byte limit".into()));
        }
        self.account(raw)?;
        let bytes = Arc::new(self.decoded(e, raw)?);
        self.blobs.insert(s.into(), bytes.clone());
        remember_blob(&bytes, &self.backing, e);
        Ok(bytes)
    }
}
fn load_source(source: Arc<dyn Source>, opts: &LoadOptions) -> Result<Document> {
    prune_blobs();
    let id = header_id(source.as_ref())?;
    let (manifest, limit, anchor) = commit(source.as_ref(), opts)?;
    let mut fetch = PackFetch { backing: Arc::new(Backing { source, id, limit, anchor }), opts: *opts, total: 0, tiles: HashMap::new(), blobs: HashMap::new() };
    let mut loader = Loader { fetch: &mut fetch, preserve_ids: opts.preserve_ids, max_id: 0, id_map: HashMap::new() };
    let doc = loader.document(&manifest.document)?;
    if opts.preserve_ids && !convert::reserve_ids_through(loader.max_id) {
        let source = fetch.backing.source.clone();
        return load_source(source, &LoadOptions { preserve_ids: false, ..*opts });
    }
    Ok(doc)
}
pub fn load_file(file: File, opts: &LoadOptions) -> Result<Document> {
    load_source(Arc::new(FileSource::new(file, None)), opts)
}
pub fn load_file_with_opener(file: File, opts: &LoadOptions, opener: ReaderOpener) -> Result<Document> {
    load_source(Arc::new(FileSource::new(file, Some(opener))), opts)
}
pub fn load_memory(bytes: Vec<u8>, opts: &LoadOptions) -> Result<Document> {
    load_source(Arc::new(MemorySource(Arc::new(bytes))), opts)
}
pub fn load_path(path: &Path, opts: &LoadOptions) -> Result<Document> {
    let owned = path.to_path_buf();
    load_file_with_opener(File::open(path)?, opts, Arc::new(move || File::open(&owned)))
}

#[derive(Default)]
struct Collect {
    tiles: HashMap<usize, (Arc<Tile>, PixelFormat)>,
    blobs: HashMap<usize, Arc<Vec<u8>>>,
    refs: HashMap<usize, String>,
}
impl Sink for Collect {
    fn tile(&mut self, format: PixelFormat, tile: &Arc<Tile>) -> String {
        let key = Arc::as_ptr(tile) as usize;
        self.tiles.entry(key).or_insert_with(|| (tile.clone(), format));
        self.refs.get(&key).cloned().unwrap_or_default()
    }
    fn blob(&mut self, blob: &Arc<Vec<u8>>) -> String {
        let key = Arc::as_ptr(blob) as usize;
        self.blobs.entry(key).or_insert_with(|| blob.clone());
        self.refs.get(&key).cloned().unwrap_or_default()
    }
}
fn append(file: &mut File, bytes: &[u8]) -> Result<PackedExtent> {
    let len = u32::try_from(bytes.len()).map_err(|_| FormatError::LimitExceeded("pack block exceeds 4 GiB".into()))?;
    let offset = file.seek(SeekFrom::End(0))?;
    file.write_all(bytes)?;
    Ok(PackedExtent { offset, len, crc32: crc32fast::hash(bytes) })
}
/// Write a durable append-only commit. No decoded document-sized allocation is held.
/// The caller supplies a read/write, non-truncating file handle (including capability handles).
pub fn save_file(doc: &Document, file: &mut File, opts: &SaveOptions) -> Result<SaveStats> {
    save_file_inner(doc, file, opts, None)
}
pub fn save_file_with_opener(doc: &Document, file: &mut File, opts: &SaveOptions, opener: ReaderOpener) -> Result<SaveStats> {
    save_file_inner(doc, file, opts, Some(opener))
}
fn save_file_inner(doc: &Document, file: &mut File, opts: &SaveOptions, opener: Option<ReaderOpener>) -> Result<SaveStats> {
    prune_blobs();
    let size = file.metadata()?.len();
    let id = if size == 0 {
        let mut seed = Vec::new();
        seed.extend_from_slice(&std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos().to_le_bytes());
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        seed.extend_from_slice(&NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed).to_le_bytes());
        seed.extend_from_slice(&std::process::id().to_le_bytes());
        let hash = blake3::hash(&seed);
        let mut id = [0; 16];
        id.copy_from_slice(hash.as_bytes().get(..16).unwrap_or(&[0; 16]));
        file.seek(SeekFrom::Start(0))?;
        file.write_all(MAGIC)?;
        file.write_all(&id)?;
        file.write_all(&1u64.to_le_bytes())?;
        id
    } else {
        header_id(&FileSource::new(file.try_clone()?, None))?
    };
    file.seek(SeekFrom::End(0))?;
    let mut collect = Collect::default();
    let _ = convert::doc_m(doc, &mut collect);
    let mut stats = SaveStats { tiles_total: collect.tiles.len(), blobs_total: collect.blobs.len(), ..Default::default() };
    // Bytes the new commit references (header, tiles, blobs, previews, manifest and footer).
    let mut live = HEADER + FOOTER as u64;
    let mut attach = Vec::new();
    let marker_file = file.try_clone()?;
    let mut matches = HashMap::<(u64, [u8; 40]), bool>::new();
    let mut belongs = |backing: &dyn TileBacking| -> bool {
        let Some(anchor) = backing.anchor() else { return false };
        *matches.entry(anchor).or_insert_with(|| {
            let mut bytes = [0; 40];
            positioned_read(&marker_file, &mut bytes, anchor.0).is_ok() && bytes == anchor.1
        })
    };
    for (key, (tile, format)) in &collect.tiles {
        let extent = if let Some((_, extent)) = tile.pack_backing(id).filter(|(backing, e)| validate_extent(*e, size).is_ok() && belongs(backing.as_ref())) {
            stats.tiles_reused += 1;
            extent
        } else {
            let packed = if cfg!(target_endian = "big") {
                let mut bytes = tile.try_bytes().map_err(FormatError::corrupt)?.to_vec();
                convert::swap_to_le(&mut bytes, format.sample);
                lz4_flex::block::compress(&bytes)
            } else {
                tile.packed_bytes().map_err(FormatError::corrupt)?
            };
            stats.tiles_written += 1;
            append(file, &packed)?
        };
        live += u64::from(extent.len);
        collect.refs.insert(*key, reference(extent, tile.len()));
        attach.push((tile.clone(), extent));
    }
    for (key, blob) in &collect.blobs {
        let old = BLOBS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .and_then(|slots| {
                let live = |slot: &&BlobSlot| slot.bytes.upgrade().is_some_and(|b| Arc::ptr_eq(&b, blob)) && slot.backing.strong_count() > 0;
                slots.iter().filter(live).find(|slot| slot.id == id).or_else(|| slots.iter().find(live))
            })
            .and_then(|slot| Some((slot.backing.upgrade()?, slot.extent)));
        let extent = match old {
            Some((backing, extent)) if backing.id == id && validate_extent(extent, size).is_ok() && belongs(backing.as_ref()) => extent,
            Some((backing, extent)) => {
                let packed = backing.read_packed(extent).map_err(FormatError::corrupt)?;
                stats.blobs_written += 1;
                append(file, &packed)?
            }
            None => {
                stats.blobs_written += 1;
                append(file, &lz4_flex::block::compress(blob))?
            }
        };
        live += u64::from(extent.len);
        collect.refs.insert(*key, reference(extent, blob.len()));
    }
    let mut preview = |image: &Option<photocraft_raster::Rgba8Image>| -> Result<Option<String>> {
        image
            .as_ref()
            .map(|image| {
                let png = crate::store::png(image)?;
                let e = append(file, &lz4_flex::block::compress(&png))?;
                live += u64::from(e.len);
                Ok(reference(e, png.len()))
            })
            .transpose()
    };
    let thumbnail = preview(&opts.thumbnail)?;
    let composite = preview(&opts.composite)?;
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        generator: format!("PhotoCraft pack {}", env!("CARGO_PKG_VERSION")),
        document: convert::doc_m(doc, &mut collect),
        thumbnail,
        composite,
    };
    let bytes = serde_json::to_vec(&manifest)?;
    stats.manifest_bytes = bytes.len();
    stats.pack_live_bytes = live + bytes.len() as u64;
    let offset = file.seek(SeekFrom::End(0))?;
    file.write_all(&bytes)?;
    let mut footer = Vec::with_capacity(FOOTER);
    footer.extend_from_slice(FOOTER_MAGIC);
    footer.extend_from_slice(&offset.to_le_bytes());
    footer.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    footer.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
    footer.extend_from_slice(&0u32.to_le_bytes());
    footer.extend_from_slice(&(offset + bytes.len() as u64 + FOOTER as u64).to_le_bytes());
    let crc = footer_crc(&footer).ok_or_else(|| FormatError::corrupt("invalid generated footer"))?;
    if let Some(out) = footer.get_mut(28..32) {
        out.copy_from_slice(&crc.to_le_bytes());
    }
    // Make payloads durable before publishing the footer; a power loss cannot expose
    // a durable commit pointing at writes which have not reached the disk yet.
    file.sync_all()?;
    file.write_all(&footer)?;
    file.sync_all()?;
    stats.pack_bytes = offset + bytes.len() as u64 + FOOTER as u64;
    let backing = Arc::new(Backing {
        source: Arc::new(FileSource::new(file.try_clone()?, opener)),
        id,
        limit: offset,
        anchor: (offset + bytes.len() as u64, footer.as_slice().try_into().map_err(|_| FormatError::corrupt("invalid generated footer size"))?),
    });
    for (tile, extent) in attach {
        tile.attach_backing(backing.clone(), extent);
    }

    for (key, blob) in &collect.blobs {
        if let Some(r) = collect.refs.get(key) {
            let (extent, _) = parse_reference(r, offset)?;
            remember_blob(blob, &backing, extent);
        }
    }
    Ok(stats)
}
pub fn save_path(doc: &Document, path: &Path, opts: &SaveOptions) -> Result<SaveStats> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
    let owned = path.to_path_buf();
    save_file_with_opener(doc, &mut file, opts, Arc::new(move || File::open(&owned)))
}

/// Save As writes a fresh compact pack, then atomically replaces the destination.
pub fn save_compact_path(doc: &Document, path: &Path, opts: &SaveOptions) -> Result<SaveStats> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = path.with_extension(format!("pcraft-{}-{stamp}-{}.tmp", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    let mut file = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&tmp)?;
    let owned = path.to_path_buf();
    let stats = save_file_with_opener(doc, &mut file, opts, Arc::new(move || File::open(&owned)))?;
    drop(file);
    std::fs::rename(tmp, path)?;
    Ok(stats)
}

pub fn read_manifest(bytes: &[u8]) -> Result<Manifest> {
    let source = SliceSource(bytes);
    header_id(&source)?;
    commit(&source, &LoadOptions::default()).map(|(manifest, _, _)| manifest)
}
pub fn read_thumbnail(bytes: &[u8]) -> Result<Option<Vec<u8>>> {
    let source = SliceSource(bytes);
    header_id(&source)?;
    let (manifest, limit, _) = commit(&source, &LoadOptions::default())?;
    let Some(reference) = manifest.thumbnail else { return Ok(None) };
    let (extent, raw) = parse_reference(&reference, limit)?;
    if raw > 64 << 20 || extent.len as usize > raw.saturating_add(raw / 255).saturating_add(32) {
        return Err(FormatError::LimitExceeded("pack preview exceeds byte limit".into()));
    }
    let bytes = source.read(extent.offset, extent.len as usize)?;
    if crc32fast::hash(&bytes) != extent.crc32 {
        return Err(FormatError::corrupt("pack preview CRC mismatch"));
    }
    let decoded = lz4_flex::block::decompress(&bytes, raw).map_err(|e| FormatError::corrupt(e.to_string()))?;
    if decoded.len() != raw {
        return Err(FormatError::corrupt("pack preview length mismatch"));
    }
    Ok(Some(decoded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_waits_for_garbage_to_outweigh_live_data() {
        let stats = |file: u64, live: u64| SaveStats { pack_bytes: file, pack_live_bytes: live, ..Default::default() };
        // Small packs never compact, however much of them is garbage.
        assert!(!wants_compaction(&stats(10 << 20, 1 << 20)));
        // Large ones once garbage exceeds what the newest commit references.
        assert!(!wants_compaction(&stats(1000 << 20, 600 << 20)));
        assert!(wants_compaction(&stats(1000 << 20, 400 << 20)));
        assert!(!wants_compaction(&stats(100 << 20, 60 << 20)));
    }
}
