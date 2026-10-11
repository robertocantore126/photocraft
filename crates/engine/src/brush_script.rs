//! Single-file brush plug-ins (ported from Fotox's `fx_plugin::script`): a `.rs` dropped in the
//! brush plug-in folder is built into WebAssembly by PhotoCraft itself, so a plug-in can be one
//! file an AI writes (`plugins/AI-PROMPT.md`, copied into the folder, tells it how).
//!
//! The build is a hidden Cargo project per file under `%LOCALAPPDATA%\PhotoCraft\plugin-build`
//! (`PHOTOCRAFT_PLUGIN_BUILD` overrides it): the file as `src/lib.rs`, the SDK (embedded here) as
//! its only dependency, no build script, so building runs nothing but the compiler. All scripts
//! share one target folder, so the SDK compiles once. A failed build writes `<name>.errors.txt`
//! beside the source, ready to paste back to the AI; a good build deletes it. Needs Rust
//! (`cargo`) and the `wasm32-unknown-unknown` target.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The plug-in side of the ABI, built into every script (the crate keeps Fotox's name, so
/// plug-ins written for Fotox build unchanged).
const SDK_LIB: &str = include_str!("../../../plugins/sdk/src/lib.rs");
const SDK_TOML: &str = "[package]\nname = \"fotox-plugin\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\n";
/// The instructions for an AI writing a plug-in, kept in the plug-in folder.
pub const AI_PROMPT: &str = include_str!("../../../plugins/AI-PROMPT.md");
/// A build still running after this is killed.
const BUILD_TIMEOUT: Duration = Duration::from_secs(180);

/// The brush plug-in folder: `PHOTOCRAFT_PLUGINS`, else `%APPDATA%\Photocraft\plugins`.
pub fn folder() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("PHOTOCRAFT_PLUGINS") {
        return Some(PathBuf::from(dir));
    }
    std::env::var_os("APPDATA").map(|dir| PathBuf::from(dir).join("Photocraft").join("plugins"))
}

/// Fotox's plug-in folder (imported once, when PhotoCraft's is first made).
fn fotox_folder() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(|dir| PathBuf::from(dir).join("Fotox").join("plugins"))
}

/// Make the folder (copying Fotox's `.wasm` and `.rs` plug-ins into it the first time) and keep
/// `AI-PROMPT.md` current. Returns the folder.
pub fn prepare() -> Option<PathBuf> {
    let dir = folder()?;
    if !dir.exists() {
        std::fs::create_dir_all(&dir).ok()?;
        if let Some(fx) = fotox_folder().filter(|d| d.is_dir())
            && let Ok(rd) = std::fs::read_dir(fx)
        {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wasm") || x.eq_ignore_ascii_case("rs"))
                    && let Some(name) = p.file_name()
                {
                    let _ = std::fs::copy(&p, dir.join(name));
                }
            }
        }
    }
    let prompt = dir.join("AI-PROMPT.md");
    if std::fs::read_to_string(&prompt).ok().as_deref() != Some(AI_PROMPT) {
        let _ = std::fs::write(&prompt, AI_PROMPT);
    }
    Some(dir)
}

/// Where scripts are built.
pub fn build_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("PHOTOCRAFT_PLUGIN_BUILD") {
        return PathBuf::from(dir);
    }
    std::env::var_os("LOCALAPPDATA").map(|dir| PathBuf::from(dir).join("PhotoCraft")).unwrap_or_else(|| std::env::temp_dir().join("PhotoCraft")).join("plugin-build")
}

/// The crate name for a script: `pc_script_` and its file name, cleaned.
fn crate_name(path: &Path) -> String {
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
    let clean: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    format!("pc_script_{clean}")
}

/// Where the build's messages go when it fails: `<name>.errors.txt`.
pub fn errors_file(path: &Path) -> PathBuf {
    path.with_extension("errors.txt")
}

fn write_if_changed(path: &Path, contents: &str) -> std::io::Result<()> {
    if std::fs::read_to_string(path).is_ok_and(|old| old == contents) {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)
}

/// The script's crate folder and the wasm its build produces.
fn layout(path: &Path) -> (PathBuf, PathBuf) {
    let root = build_root();
    let name = crate_name(path);
    let wasm = root.join("target").join("wasm32-unknown-unknown").join("release").join(format!("{name}.wasm"));
    (root.join("scripts").join(&name), wasm)
}

/// The built wasm of `path`, if it is up to date with the source.
fn cached(path: &Path) -> Option<PathBuf> {
    let source = std::fs::read_to_string(path).ok()?;
    let (dir, wasm) = layout(path);
    let lib = dir.join("src").join("lib.rs");
    let fresh = std::fs::read_to_string(&lib).ok()? == source
        && std::fs::read_to_string(build_root().join("sdk").join("src").join("lib.rs")).is_ok_and(|sdk| sdk == SDK_LIB)
        && wasm.metadata().and_then(|m| m.modified()).ok()? >= lib.metadata().and_then(|m| m.modified()).ok()?;
    fresh.then_some(wasm)
}

/// `cargo`: `PHOTOCRAFT_CARGO`, else rustup's default place, else the PATH.
fn cargo() -> PathBuf {
    if let Some(cargo) = std::env::var_os("PHOTOCRAFT_CARGO") {
        return PathBuf::from(cargo);
    }
    if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        let candidate = PathBuf::from(home).join(".cargo").join("bin").join(if cfg!(windows) { "cargo.exe" } else { "cargo" });
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("cargo")
}

/// Build `path` (or take the up-to-date build) and return the wasm bytes.
pub fn build(path: &Path) -> Result<Vec<u8>, String> {
    let file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if let Some(wasm) = cached(path) {
        return std::fs::read(&wasm).map_err(|e| format!("plug-in {file}: {e}"));
    }
    let source = std::fs::read_to_string(path).map_err(|e| format!("plug-in {file}: {e}"))?;
    let root = build_root();
    let (dir, wasm) = layout(path);
    let name = crate_name(path);
    let setup = || -> std::io::Result<()> {
        write_if_changed(&root.join("sdk").join("Cargo.toml"), SDK_TOML)?;
        write_if_changed(&root.join("sdk").join("src").join("lib.rs"), SDK_LIB)?;
        let toml = format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lib]\ncrate-type = [\"cdylib\"]\n\n\
             [dependencies]\nfotox-plugin = {{ path = \"../../sdk\" }}\n\n\
             [profile.release]\nopt-level = 3\npanic = \"abort\"\ncodegen-units = 4\ndebug = false\nstrip = true\n\n[workspace]\n"
        );
        write_if_changed(&dir.join("Cargo.toml"), &toml)?;
        write_if_changed(&dir.join("src").join("lib.rs"), &source)
    };
    setup().map_err(|e| format!("plug-in {file}: cannot prepare its build in {}: {e}", root.display()))?;
    let mut command = Command::new(cargo());
    command
        .current_dir(&dir)
        .args(["build", "--release", "--target", "wasm32-unknown-unknown", "--color", "never", "--target-dir"])
        .arg(root.join("target"))
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET_DIR")
        .env_remove("RUSTFLAGS")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: no console flashing over the app.
        command.creation_flags(0x0800_0000);
    }
    let mut child = command.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            format!("plug-in {file}: Rust is not installed (cargo was not found). Install it from rustup.rs, or use a built .wasm")
        } else {
            format!("plug-in {file}: cannot start cargo: {e}")
        }
    })?;
    let stderr = child.stderr.take();
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut s) = stderr {
            let _ = s.read_to_string(&mut text);
        }
        text
    });
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() > BUILD_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(_) => break None,
        }
    };
    let output = reader.join().unwrap_or_default();
    let errors = errors_file(path);
    match status {
        Some(status) if status.success() => {
            let _ = std::fs::remove_file(&errors);
            std::fs::read(&wasm).map_err(|e| format!("plug-in {file}: the build made no {}: {e}", wasm.display()))
        }
        _ => {
            let messages: String = output
                .lines()
                .filter(|l| !l.trim_start().starts_with("Compiling") && !l.trim_start().starts_with("Updating") && !l.trim_start().starts_with("Locking"))
                .map(|l| format!("{l}\n"))
                .collect();
            let why = if status.is_none() {
                format!("the build took longer than {} s and was stopped", BUILD_TIMEOUT.as_secs())
            } else if output.contains("target may not be installed") || output.contains("can't find crate for `core`") {
                "the wasm32 target is missing: run `rustup target add wasm32-unknown-unknown` once".to_string()
            } else {
                "the code does not compile".to_string()
            };
            let report = format!(
                "PhotoCraft could not build the plug-in {file}: {why}.\n\n\
                 Paste this whole file to the AI that wrote the plug-in and ask it to fix the code.\n\n{messages}"
            );
            let _ = std::fs::write(&errors, report);
            Err(format!("plug-in {file}: {why} — the details are in {}", errors.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()))
        }
    }
}
