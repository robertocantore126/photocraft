//! Brush plug-ins: Fotox's brush plugin ABI (D-096) on the same sandbox as the filters.
//!
//! A brush plug-in decides what a stroke does to the pixels under it; the stroke engine keeps the
//! tip, spacing, pressure, smoothing, selection, the coverage buffer (opacity ceiling, flow
//! build-up) and undo. It is called once per dirty rectangle of a stroke with the layer **as it
//! was when the stroke started** (premultiplied RGBA `0..=1`, or a mask's grey) and `k`, the
//! stroke's opacity × coverage per pixel. The `.wasm` files Fotox builds load here unchanged.
//!
//! Exports (the plugin SDK's `brush_plugin!` macro writes them):
//! * `fx_manifest() -> i64`: `ptr << 32 | len` of the manifest JSON;
//! * `fx_alloc(bytes) -> ptr`;
//! * `fx_rect(ptr, w, h, x0, y0) -> i32`: `ptr` holds [`HEADER_WORDS`] f32 (params `0..16`,
//!   colour `16..20`, lock-alpha flag `20`), then `w·h` RGBA pixels, then `w·h` `k`;
//! * `fx_gray(ptr, w, h, x0, y0) -> i32` (optional): the same for a mask, one channel.
//!
//! Protection: no imports, a memory cap and an instruction budget per call (about a second of work),
//! a fresh instance per call. A plug-in that traps, loops or runs too long is **stopped**: its
//! calls fail until it is loaded again, and the stroke keeps those pixels as they were.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use serde::Deserialize;
use wasmi::{Config, EnforcedLimits, Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc};

use crate::{Error, Limits, Result};

/// f32 words before the pixels: params `0..16`, colour `16..20`, flags `20`.
pub const HEADER_WORDS: usize = 32;
/// Option-bar params a plug-in can read.
pub const MAX_PARAMS: usize = 16;

/// How the stroke engine paints for the tool (manifest `brush`).
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct BrushHints {
    /// `classic` | `gaussian` | `feather`.
    pub profile: Option<String>,
    /// `build_up` (every native brush) | `max` (each pixel keeps the strongest coverage).
    pub accumulate: Option<String>,
    /// Dab spacing as a fraction of the diameter (unless the bar has `Spacing`).
    pub spacing: Option<f32>,
    /// The px field whose value is the fade on each side of `Size` (the Feather tip).
    pub feather_from: Option<String>,
}

/// A brush plug-in's manifest (Fotox's format).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BrushManifest {
    pub id: String,
    pub name: String,
    /// The toolbar slot whose flyout shows the tool (`eraser`, `brush`, `dodge`…).
    #[serde(default)]
    pub slot: String,
    #[serde(default)]
    pub icon: String,
    /// `foreground` | `background`: the colour handed to the plug-in.
    #[serde(default)]
    pub color: String,
    /// The option bar: `{type: num|range|select|toggle|brushpreset, text, value, unit, options, on}`.
    #[serde(default)]
    pub options: Vec<serde_json::Value>,
    /// The option-bar fields the plug-in reads, by label without the colon, in order.
    #[serde(default)]
    pub params: Vec<String>,
    #[serde(default)]
    pub brush: Option<BrushHints>,
}

impl BrushManifest {
    fn validate(&self) -> Result<()> {
        let ok = |s: &str| !s.is_empty() && s.len() <= 64 && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_');
        if !ok(&self.id) {
            return Err(Error::Manifest(format!("id {:?} must be a-z 0-9 - _ (at most 64)", self.id)));
        }
        if self.name.trim().is_empty() {
            return Err(Error::Manifest("name is empty".into()));
        }
        if self.params.len() > MAX_PARAMS {
            return Err(Error::Manifest(format!("{} params (at most {MAX_PARAMS})", self.params.len())));
        }
        Ok(())
    }
}

/// A loaded brush plug-in. Every call gets its own store and instance.
pub struct BrushPlugin {
    manifest: BrushManifest,
    engine: Engine,
    module: Module,
    limits: Limits,
    has_gray: bool,
    stopped: AtomicBool,
    why: RwLock<Option<String>>,
    source: Option<String>,
}

impl std::fmt::Debug for BrushPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrushPlugin").field("manifest", &self.manifest).field("source", &self.source).finish()
    }
}

/// Is `bytes` a brush plug-in (it exports `fx_rect`) rather than a filter?
pub fn is_brush_module(bytes: &[u8]) -> bool {
    let engine = Engine::default();
    Module::new(&engine, bytes).is_ok_and(|m| m.exports().any(|e| e.name() == "fx_rect"))
}

struct Live {
    store: Store<StoreLimits>,
    instance: wasmi::Instance,
    memory: wasmi::Memory,
}

fn wasm_err(e: wasmi::Error) -> Error {
    if e.as_trap_code() == Some(wasmi::TrapCode::OutOfFuel) { Error::Limit("instruction budget".into()) } else { Error::Trap(e.to_string()) }
}

impl Live {
    fn func<P: wasmi::WasmParams, R: wasmi::WasmResults>(&self, name: &str) -> Result<TypedFunc<P, R>> {
        self.instance.get_typed_func::<P, R>(&self.store, name).map_err(|e| Error::Abi(format!("export `{name}`: {e}")))
    }
}

impl BrushPlugin {
    /// Brush limits: 256 MiB of memory, 2 s per call (Fotox stopped a plug-in after 1–2 s).
    pub fn default_limits() -> Limits {
        Limits { max_memory_bytes: 256 << 20, wall_time_ms: 2_000, fuel_base: 50_000_000, fuel_per_sample: 20_000, ..Limits::default() }
    }

    pub fn load(bytes: &[u8], limits: Limits) -> Result<BrushPlugin> {
        if bytes.len() > limits.max_module_bytes {
            return Err(Error::Module(format!("module is {} bytes (limit {})", bytes.len(), limits.max_module_bytes)));
        }
        if !bytes.starts_with(b"\0asm") {
            return Err(Error::Module("not a WebAssembly binary".into()));
        }
        let mut config = Config::default();
        config.consume_fuel(true).enforced_limits(EnforcedLimits::strict()).set_max_recursion_depth(limits.max_recursion_depth.max(16));
        let engine = Engine::new(&config);
        let module = Module::new(&engine, bytes).map_err(|e| Error::Module(e.to_string()))?;
        if let Some(imp) = module.imports().next() {
            return Err(Error::Abi(format!("the module imports `{}.{}`: brush plug-ins get no host imports", imp.module(), imp.name())));
        }
        let has_gray = module.exports().any(|e| e.name() == "fx_gray");
        let mut p = BrushPlugin {
            manifest: BrushManifest { id: String::new(), name: String::new(), slot: String::new(), icon: String::new(), color: String::new(), options: Vec::new(), params: Vec::new(), brush: None },
            engine,
            module,
            limits,
            has_gray,
            stopped: AtomicBool::new(false),
            why: RwLock::new(None),
            source: None,
        };
        let mut live = p.instantiate()?;
        let f: TypedFunc<(), i64> = live.func("fx_manifest")?;
        live.store.set_fuel(p.limits.fuel_base).map_err(wasm_err)?;
        let packed = f.call(&mut live.store, ()).map_err(wasm_err)? as u64;
        let (ptr, len) = ((packed >> 32) as usize, (packed & 0xffff_ffff) as usize);
        if len > crate::manifest::MAX_MANIFEST_BYTES {
            return Err(Error::Manifest(format!("manifest is {len} bytes")));
        }
        let text = live.memory.data(&live.store).get(ptr..ptr.saturating_add(len)).ok_or_else(|| Error::Abi("fx_manifest points outside memory".into()))?;
        let manifest: BrushManifest = serde_json::from_slice(text).map_err(|e| Error::Manifest(e.to_string()))?;
        manifest.validate()?;
        p.manifest = manifest;
        live.func::<i32, i32>("fx_alloc")?;
        live.func::<RectArgs, i32>("fx_rect")?;
        Ok(p)
    }

    pub fn manifest(&self) -> &BrushManifest {
        &self.manifest
    }
    pub fn id(&self) -> &str {
        &self.manifest.id
    }
    pub fn has_gray(&self) -> bool {
        self.has_gray
    }
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
    /// Why the plug-in was stopped, if it was.
    pub fn stopped(&self) -> Option<String> {
        if !self.stopped.load(Ordering::Relaxed) {
            return None;
        }
        Some(self.why.read().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default())
    }

    fn instantiate(&self) -> Result<Live> {
        let limits = StoreLimitsBuilder::new().memory_size(self.limits.max_memory_bytes).memories(1).tables(4).table_elements(100_000).instances(1).build();
        let mut store = Store::new(&self.engine, limits);
        store.limiter(|l| l);
        store.set_fuel(self.limits.fuel_base).map_err(wasm_err)?;
        let linker = Linker::<StoreLimits>::new(&self.engine);
        let instance = linker.instantiate_and_start(&mut store, &self.module).map_err(wasm_err)?;
        let memory = instance.get_memory(&store, "memory").ok_or_else(|| Error::Abi("the module must export its memory as `memory`".into()))?;
        Ok(Live { store, instance, memory })
    }

    /// Runs `fx_rect` (`channels` = 4, premultiplied RGBA) or `fx_gray` (1) on one rectangle:
    /// `values` (`w·h·channels`) are overwritten with the result. A failure stops the plug-in.
    #[allow(clippy::too_many_arguments)]
    pub fn paint(&self, header: &[f32; HEADER_WORDS], values: &mut [f32], channels: usize, k: &[f32], w: u32, h: u32, x0: i32, y0: i32) -> Result<()> {
        if let Some(why) = self.stopped() {
            return Err(Error::Failed(format!("{} is stopped: {why}", self.manifest.name)));
        }
        let r = self.paint_inner(header, values, channels, k, w, h, x0, y0);
        if let Err(e) = &r {
            self.stopped.store(true, Ordering::Relaxed);
            *self.why.write().unwrap_or_else(|e| e.into_inner()) = Some(e.to_string());
        }
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn paint_inner(&self, header: &[f32; HEADER_WORDS], values: &mut [f32], channels: usize, k: &[f32], w: u32, h: u32, x0: i32, y0: i32) -> Result<()> {
        let n = w as usize * h as usize;
        if values.len() != n * channels || k.len() != n {
            return Err(Error::Params("buffer sizes do not match the rectangle".into()));
        }
        let name = match channels {
            4 => "fx_rect",
            1 if self.has_gray => "fx_gray",
            _ => return Err(Error::Abi("this brush cannot paint masks".into())),
        };
        let words = HEADER_WORDS + n * channels + n;
        let bytes = words.checked_mul(4).filter(|b| *b <= i32::MAX as usize && *b <= self.limits.max_memory_bytes).ok_or_else(|| Error::Limit("memory budget".into()))?;
        let mut live = self.instantiate()?;
        let alloc: TypedFunc<i32, i32> = live.func("fx_alloc")?;
        let rect: TypedFunc<RectArgs, i32> = live.func(name)?;
        live.store.set_fuel(self.limits.fuel_base).map_err(wasm_err)?;
        let ptr = alloc.call(&mut live.store, bytes as i32).map_err(wasm_err)? as u32 as usize;
        let end = ptr.checked_add(bytes).ok_or_else(|| Error::Abi("fx_alloc returned a block outside memory".into()))?;
        {
            let mem = live.memory.data_mut(&mut live.store);
            let dst = mem.get_mut(ptr..end).ok_or_else(|| Error::Abi("fx_alloc returned a block outside memory".into()))?;
            let words = header.iter().chain(values.iter()).chain(k.iter());
            for (d, v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                *d = v.to_le_bytes();
            }
        }
        let fuel = self.limits.fuel_base.saturating_add((n as u64).saturating_mul(self.limits.fuel_per_sample));
        live.store.set_fuel(fuel).map_err(wasm_err)?;
        let rc = rect.call(&mut live.store, (ptr as i32, w as i32, h as i32, x0, y0)).map_err(wasm_err)?;
        if rc != 0 {
            return Err(Error::Failed(format!("{name} returned {rc}")));
        }
        let mem = live.memory.data(&live.store);
        let start = ptr + HEADER_WORDS * 4;
        let src = mem.get(start..start + n * channels * 4).ok_or_else(|| Error::Abi("pixels no longer in memory".into()))?;
        for (v, s) in values.iter_mut().zip(src.as_chunks::<4>().0) {
            *v = f32::from_le_bytes(*s);
        }
        Ok(())
    }
}

/// `fx_rect(ptr, w, h, x0, y0)`.
type RectArgs = (i32, i32, i32, i32, i32);

static BRUSHES: RwLock<Vec<Arc<BrushPlugin>>> = RwLock::new(Vec::new());
static REVISION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Installs a brush plug-in, replacing one with the same id.
pub fn install(p: BrushPlugin) -> Arc<BrushPlugin> {
    let p = Arc::new(p);
    let mut all = BRUSHES.write().unwrap_or_else(|e| e.into_inner());
    all.retain(|q| q.id() != p.id());
    all.push(p.clone());
    all.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
    REVISION.fetch_add(1, Ordering::Relaxed);
    p
}

pub fn get(id: &str) -> Option<Arc<BrushPlugin>> {
    BRUSHES.read().unwrap_or_else(|e| e.into_inner()).iter().find(|p| p.id() == id).cloned()
}

pub fn list() -> Vec<Arc<BrushPlugin>> {
    BRUSHES.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Removes the brush plug-in loaded from `source` (a file that was deleted).
pub fn remove_source(source: &str) -> bool {
    let mut all = BRUSHES.write().unwrap_or_else(|e| e.into_inner());
    let before = all.len();
    all.retain(|p| p.source() != Some(source));
    let gone = all.len() != before;
    if gone {
        REVISION.fetch_add(1, Ordering::Relaxed);
    }
    gone
}

/// Changes whenever a brush plug-in is installed or removed.
pub fn revision() -> u64 {
    REVISION.load(Ordering::Relaxed)
}
