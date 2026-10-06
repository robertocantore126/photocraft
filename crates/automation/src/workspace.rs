//! Capability-based filesystem policy for untrusted automation paths.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cap_std::ambient_authority;
use cap_std::fs::{Dir, OpenOptions};
use serde_json::Value;

use crate::AutomationError;

const DENIED: &str = "automation filesystem access is not granted";

#[derive(Clone)]
struct RootCapability {
    dir: Arc<Dir>,
}

/// Separate directory capabilities for automation reads and writes.
///
/// Root paths are trusted launch-time configuration. Request paths are always
/// untrusted, forward-slash relative paths resolved by `cap-std` beneath the
/// held directory handle. Absolute paths, parent traversal, alternate
/// separators, Windows prefixes and empty components are rejected before I/O.
#[derive(Clone, Default)]
pub struct AuthorizedWorkspace {
    read: Option<RootCapability>,
    write: Option<RootCapability>,
}

impl AuthorizedWorkspace {
    /// Open the configured roots as capabilities. Either authority may be
    /// omitted; omitted authority fails closed.
    pub fn new(read_root: Option<&Path>, write_root: Option<&Path>) -> Result<Self, AutomationError> {
        Ok(Self { read: open_root(read_root, "read")?, write: open_root(write_root, "write")? })
    }

    /// Open a capability-scoped native pack reader without materializing the file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_read_file(&self, path: &str) -> Result<std::fs::File, AutomationError> {
        let relative = relative_path(path)?;
        let root = self.read.as_ref().ok_or_else(|| AutomationError::BadRequest(format!("{DENIED}: read authority is absent")))?;
        let file = root.dir.open(&relative).map_err(|e| file_error("read", path, e))?;
        if !file.metadata().map_err(|e| file_error("read", path, e))?.is_file() {
            return Err(AutomationError::BadRequest("automation path is not a regular file".into()));
        }
        Ok(file.into_std())
    }
    /// Open a capability-scoped non-truncating read/write handle for append-only commits.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_pack_writer(&self, path: &str) -> Result<std::fs::File, AutomationError> {
        self.pack_writer(path, false)
    }
    /// Create a new staging pack without overwriting any existing entry.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create_pack_writer(&self, path: &str) -> Result<std::fs::File, AutomationError> {
        self.pack_writer(path, true)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn pack_writer(&self, path: &str, new: bool) -> Result<std::fs::File, AutomationError> {
        let relative = relative_path(path)?;
        let root = self.write.as_ref().ok_or_else(|| AutomationError::BadRequest(format!("{DENIED}: write authority is absent")))?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).create_new(new);
        let file = root.dir.open_with(&relative, &options).map_err(|e| file_error("write", path, e))?;
        if !file.metadata().map_err(|e| file_error("write", path, e))?.is_file() {
            return Err(AutomationError::BadRequest("automation path is not a regular file".into()));
        }
        Ok(file.into_std())
    }

    /// Atomically publish a completed pack within the granted write directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn replace_file(&self, from: &str, to: &str) -> Result<(), AutomationError> {
        let from = relative_path(from)?;
        let to = relative_path(to)?;
        let root = self.write.as_ref().ok_or_else(|| AutomationError::BadRequest(format!("{DENIED}: write authority is absent")))?;
        root.dir.rename(&from, &root.dir, &to).map_err(|e| file_error("publish", to.to_string_lossy().as_ref(), e))
    }

    /// Import a native document through capability handles, preserving lazy pack tiles.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_document(&self, path: &str) -> Result<photocraft_io::ImportResult, AutomationError> {
        let file = self.open_read_file(path)?;
        let workspace = self.clone();
        let relative = path.to_string();
        let name = Path::new(path).file_name().and_then(|name| name.to_str()).unwrap_or(path);
        Ok(photocraft_io::import_file_with_opener(name, file, Arc::new(move || workspace.open_read_file(&relative).map_err(std::io::Error::other)))?)
    }
    /// Commit a pack without materializing a byte bundle or escaping the write root.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_document_pack(&self, doc: &photocraft_doc::Document, path: &str, mut compact: bool) -> Result<photocraft_format::SaveStats, AutomationError> {
        let mut existing = if compact { None } else { Some(self.open_pack_writer(path)?) };
        if let Some(file) = existing.as_mut() {
            let mut magic = [0; 8];
            compact = file.metadata().map_err(|e| file_error("write", path, e))?.len() > 0
                && (file.read_exact(&mut magic).is_err() || &magic != photocraft_format::pack::MAGIC);
        }
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let target = if compact { format!("{path}.{stamp}-{}.pack-tmp", NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) } else { path.to_string() };
        let mut file = if compact {
            existing.take();
            self.create_pack_writer(&target)?
        } else {
            existing.ok_or_else(|| AutomationError::BadRequest("missing pack writer".into()))?
        };
        let workspace = self.clone();
        let relative = path.to_string();
        let opener = Arc::new(move || {
            let relative = relative_path(&relative).map_err(std::io::Error::other)?;
            let root = workspace.write.as_ref().ok_or_else(|| std::io::Error::other("write authority is absent"))?;
            root.dir.open(&relative).map(|file| file.into_std())
        });
        let stats = photocraft_format::pack::save_file_with_opener(doc, &mut file, &Default::default(), opener)?;
        drop(file);
        if compact {
            self.replace_file(&target, path)?;
        }
        Ok(stats)
    }

    /// Read one regular file below the configured read root.
    pub fn read(&self, path: &str) -> Result<Vec<u8>, AutomationError> {
        let relative = relative_path(path)?;
        let root = self.read.as_ref().ok_or_else(|| AutomationError::BadRequest(format!("{DENIED}: read authority is absent")))?;
        let mut file = root.dir.open(&relative).map_err(|e| file_error("read", path, e))?;
        let metadata = file.metadata().map_err(|e| file_error("read", path, e))?;
        if !metadata.is_file() {
            return Err(AutomationError::BadRequest(format!("automation read path is not a regular file: `{path}`")));
        }
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|e| file_error("read", path, e))?;
        Ok(bytes)
    }

    /// Create or replace one file below the configured write root.
    ///
    /// The parent directory must already exist. `cap-std` performs path
    /// resolution and file creation relative to the held directory handle, so
    /// a non-existent final target is supported without ambient path access.
    pub fn write(&self, path: &str, bytes: &[u8]) -> Result<(), AutomationError> {
        let relative = relative_path(path)?;
        let root = self.write.as_ref().ok_or_else(|| AutomationError::BadRequest(format!("{DENIED}: write authority is absent")))?;
        let mut options = OpenOptions::new();
        options.write(true).create(true).truncate(true);
        let mut file = root.dir.open_with(&relative, &options).map_err(|e| file_error("write", path, e))?;
        file.write_all(bytes).map_err(|e| file_error("write", path, e))?;
        file.flush().map_err(|e| file_error("write", path, e))
    }
}

fn open_root(path: Option<&Path>, authority: &str) -> Result<Option<RootCapability>, AutomationError> {
    let Some(path) = path else { return Ok(None) };
    if path.as_os_str().is_empty() {
        return Err(AutomationError::BadRequest(format!("automation {authority} root is empty")));
    }
    let dir = Dir::open_ambient_dir(path, ambient_authority())
        .map_err(|e| AutomationError::Io(format!("cannot open automation {authority} root `{}`: {e}", path.display())))?;
    Ok(Some(RootCapability { dir: Arc::new(dir) }))
}

fn relative_path(raw: &str) -> Result<PathBuf, AutomationError> {
    if raw.is_empty() {
        return Err(path_error(raw, "path is empty"));
    }
    if raw.contains('\\') {
        return Err(path_error(raw, "alternate separators are not allowed; use `/`"));
    }
    if raw.starts_with('/') {
        return Err(path_error(raw, "absolute paths are not allowed"));
    }
    if raw.contains(':') {
        return Err(path_error(raw, "drive, device and stream prefixes are not allowed"));
    }
    for component in raw.split('/') {
        if component.is_empty() {
            return Err(path_error(raw, "empty path components are not allowed"));
        }
        if component == "." {
            return Err(path_error(raw, "`.` path components are not allowed"));
        }
        if component == ".." {
            return Err(path_error(raw, "parent traversal is not allowed"));
        }
        if component.ends_with(['.', ' ']) {
            return Err(path_error(raw, "path components ending in a dot or space are not allowed"));
        }
        if is_windows_device_name(component) {
            return Err(path_error(raw, "reserved device names are not allowed"));
        }
    }
    Ok(PathBuf::from(raw))
}

fn is_windows_device_name(component: &str) -> bool {
    let stem = component.split('.').next().unwrap_or(component).to_ascii_uppercase();
    matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CLOCK$" | "CONIN$" | "CONOUT$")
        || stem.strip_prefix("COM").is_some_and(|n| matches!(n, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"))
        || stem.strip_prefix("LPT").is_some_and(|n| matches!(n, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"))
}

fn path_error(path: &str, reason: &str) -> AutomationError {
    AutomationError::BadRequest(format!("automation path rejected: {reason}: `{path}`"))
}

fn file_error(operation: &str, path: &str, error: std::io::Error) -> AutomationError {
    let kind = error.kind();
    AutomationError::Io(format!("automation {operation} `{path}` failed ({kind:?}): {error}"))
}

/// Filesystem-bearing engine commands have not yet been converted to consume
/// directory capabilities. Automation must use `doc.open`, `doc.save` and
/// `doc.render` for file effects until those commands are migrated.
pub fn authorize_engine_command(id: &str, params: &Value) -> Result<(), AutomationError> {
    let safe_file_command = matches!(
        id,
        "file.new"
            | "file.close"
            | "file.closeAll"
            | "file.closeOthers"
            | "file.fileInfo"
            | "file.automate.fitImage"
            | "file.automate.conditionalModeChange"
            | "file.scripts.flattenAllLayerEffects"
            | "file.scripts.flattenAllMasks"
            | "file.scripts.deleteAllEmptyLayers"
            | "file.export.exportPreferences"
    );
    if id.starts_with("file.") && !safe_file_command {
        return Err(command_error(id));
    }
    if (id.starts_with("layer.smartObjects.") && id != "layer.smartObjects.convertToSmartObject")
        || matches!(
            id,
            "pattern.import"
                | "pattern.export"
                | "edit.presets.migratePresets"
                | "measurementLog.export"
                | "layer.videoLayers.newVideoLayerFromFile"
                | "layer.videoLayers.replaceFootage"
                | "layer.videoLayers.reloadFrame"
                | "image.applyDataSet"
        )
        || profile_command_may_read_ambient(id, params)
        || preferences_may_grant_ambient_paths(id, params)
        || params_contain_ambient_path(id, params)
    {
        return Err(command_error(id));
    }
    Ok(())
}

/// Desktop sessions may already contain user-configured ambient colour-profile
/// paths. Commands which implicitly resolve those settings are denied even
/// when their request parameters contain no path. Fresh headless sessions
/// cannot acquire such settings through the automation interface.
pub fn authorize_desktop_engine_command(id: &str, params: &Value) -> Result<(), AutomationError> {
    authorize_engine_command(id, params)?;
    if id.starts_with("image.mode.") {
        return Err(command_error(id));
    }
    Ok(())
}

fn params_contain_ambient_path(id: &str, params: &Value) -> bool {
    let keys: &[&str] = match id {
        "image.adjustments.colorLookup" | "layer.newAdjustmentLayer.colorLookup" | "layer.setAdjustment" => &["file"],
        "filter.distort.displace" => &["mapPath"],
        "layer.quickExportAsPng" | "layer.exportAs" => &["path"],
        "edit.assignProfile" | "edit.convertToProfile" | "edit.profileInfo" | "view.proofSetup" | "view.gamutWarning" => &["profile"],
        "edit.colorSettings" => &["workingRgb", "workingCmyk", "workingGray"],
        _ => &[],
    };
    keys.iter().any(|key| {
        params
            .get(*key)
            .and_then(Value::as_str)
            .is_some_and(|value| if matches!(*key, "file" | "mapPath" | "path") { !value.is_empty() } else { looks_like_path(value) })
    })
}

fn profile_command_may_read_ambient(id: &str, params: &Value) -> bool {
    if matches!(id, "color.profileMismatch" | "edit.colorSettings" | "view.proofSetup") {
        return true;
    }
    matches!(id, "edit.assignProfile" | "edit.convertToProfile" | "edit.profileInfo")
        && params.get("profile").and_then(Value::as_str).is_some_and(|profile| {
            matches!(profile, "working" | "default" | "working-cmyk" | "workingCmyk" | "working-rgb" | "workingRgb" | "working-gray" | "workingGray")
        })
}

fn preferences_may_grant_ambient_paths(id: &str, params: &Value) -> bool {
    if id != "prefs.set" {
        return false;
    }
    let direct = params
        .get("path")
        .and_then(Value::as_str)
        .is_some_and(|path| path == "colorSettings" || path.starts_with("colorSettings.") || path == "scriptEvents" || path.starts_with("scriptEvents."));
    let batch = params.get("values").and_then(Value::as_object).is_some_and(|values| {
        values.keys().any(|path| path == "colorSettings" || path.starts_with("colorSettings.") || path == "scriptEvents" || path.starts_with("scriptEvents."))
    });
    direct || batch
}

fn looks_like_path(value: &str) -> bool {
    value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.to_ascii_lowercase().ends_with(".icc")
        || value.to_ascii_lowercase().ends_with(".icm")
}

fn command_error(id: &str) -> AutomationError {
    AutomationError::BadRequest(format!("automation command `{id}` uses ambient filesystem paths and is disabled; use capability-scoped document methods"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots(name: &str) -> (PathBuf, PathBuf, AuthorizedWorkspace) {
        let base = std::env::temp_dir().join(format!("photocraft-workspace-{}-{name}", std::process::id()));
        let inside = base.join("inside");
        let outside = base.join("outside");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let workspace = AuthorizedWorkspace::new(Some(&inside), Some(&inside)).unwrap();
        (inside, outside, workspace)
    }

    #[test]
    fn valid_read_and_nonexistent_output_write_stay_in_root() {
        let (inside, _, workspace) = roots("valid");
        std::fs::write(inside.join("read.txt"), b"canary").unwrap();
        assert_eq!(workspace.read("read.txt").unwrap(), b"canary");
        workspace.write("new-output.txt", b"created").unwrap();
        assert_eq!(std::fs::read(inside.join("new-output.txt")).unwrap(), b"created");
    }

    #[test]
    fn rejects_absolute_traversal_mixed_prefix_and_malformed_paths_without_panicking() {
        let (_, _, workspace) = roots("reject");
        for path in [
            "",
            "/absolute",
            "../escape",
            "a/../escape",
            "a\\..\\escape",
            "a\\b",
            "C:/escape",
            "a//b",
            "./a",
            "name:stream",
            "NUL",
            "con.txt",
            "folder/COM1.log",
        ] {
            assert!(workspace.read(path).is_err(), "read accepted {path:?}");
            assert!(workspace.write(path, b"x").is_err(), "write accepted {path:?}");
            #[cfg(not(target_arch = "wasm32"))]
            {
                assert!(workspace.open_read_file(path).is_err());
                assert!(workspace.open_pack_writer(path).is_err());
                assert!(workspace.create_pack_writer(path).is_err());
                assert!(workspace.replace_file(path, "valid.pcraft").is_err());
                assert!(workspace.replace_file("valid.pcraft", path).is_err());
            }
        }
    }

    #[test]
    fn read_and_write_authority_are_separate() {
        let (inside, _, _) = roots("separate");
        std::fs::write(inside.join("read.txt"), b"canary").unwrap();
        let read_only = AuthorizedWorkspace::new(Some(&inside), None).unwrap();
        assert_eq!(read_only.read("read.txt").unwrap(), b"canary");
        assert!(read_only.write("blocked.txt", b"x").is_err());
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(read_only.open_pack_writer("blocked.pcraft").is_err());
            assert!(read_only.create_pack_writer("blocked.pcraft").is_err());
        }
        let write_only = AuthorizedWorkspace::new(None, Some(&inside)).unwrap();
        assert!(write_only.read("read.txt").is_err());
        #[cfg(not(target_arch = "wasm32"))]
        assert!(write_only.open_read_file("read.txt").is_err());
        write_only.write("written.txt", b"ok").unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn pack_handles_append_and_publish_without_truncation() {
        use std::io::{Seek, SeekFrom};
        let (inside, _, workspace) = roots("pack-handles");
        workspace.write("saved.pcraft", b"first").unwrap();
        let mut file = workspace.open_pack_writer("saved.pcraft").unwrap();
        file.seek(SeekFrom::End(0)).unwrap();
        file.write_all(b" second").unwrap();
        drop(file);
        assert_eq!(workspace.read("saved.pcraft").unwrap(), b"first second");
        assert!(workspace.create_pack_writer("saved.pcraft").is_err());
        let mut stage = workspace.create_pack_writer("stage.pcraft").unwrap();
        stage.write_all(b"replacement").unwrap();
        stage.sync_all().unwrap();
        drop(stage);
        workspace.replace_file("stage.pcraft", "saved.pcraft").unwrap();
        assert_eq!(workspace.read("saved.pcraft").unwrap(), b"replacement");
        assert!(!inside.join("stage.pcraft").exists());
    }

    #[test]
    fn missing_parent_is_a_stable_error_before_any_file_is_created() {
        let (inside, _, workspace) = roots("missing-parent");
        let error = workspace.write("missing/output.txt", b"x").unwrap_err().to_string();
        assert!(error.contains("automation write"), "{error}");
        assert!(!inside.join("missing").exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_rejected() {
        use std::os::unix::fs::symlink;

        let (inside, outside, workspace) = roots("symlink");
        std::fs::write(outside.join("secret.txt"), b"outside").unwrap();
        symlink(&outside, inside.join("link")).unwrap();
        assert!(workspace.read("link/secret.txt").is_err());
        assert!(workspace.write("link/new.txt", b"blocked").is_err());
        assert!(!outside.join("new.txt").exists());
    }

    #[cfg(windows)]
    #[test]
    fn windows_symlink_escape_is_rejected_when_supported() {
        use std::os::windows::fs::symlink_dir;

        let (inside, outside, workspace) = roots("windows-link");
        std::fs::write(outside.join("secret.txt"), b"outside").unwrap();
        if symlink_dir(&outside, inside.join("link")).is_err() {
            return;
        }
        assert!(workspace.read("link/secret.txt").is_err());
        assert!(workspace.write("link/new.txt", b"blocked").is_err());
        assert!(!outside.join("new.txt").exists());
    }

    #[cfg(windows)]
    #[test]
    fn windows_junction_escape_is_rejected_when_supported() {
        use std::process::Command;

        let (inside, outside, workspace) = roots("windows-junction");
        std::fs::write(outside.join("secret.txt"), b"outside").unwrap();
        let link = inside.join("junction");
        let status = Command::new("cmd.exe").args(["/D", "/C", "mklink", "/J"]).arg(&link).arg(&outside).status();
        if !status.is_ok_and(|status| status.success()) {
            return;
        }
        assert!(workspace.read("junction/secret.txt").is_err());
        assert!(workspace.write("junction/new.txt", b"blocked").is_err());
        assert!(!outside.join("new.txt").exists());
        let _ = std::fs::remove_dir(link);
    }

    #[test]
    fn filesystem_commands_fail_closed() {
        for id in [
            "file.openAs",
            "file.export.saveForWebLegacy",
            "pattern.import",
            "layer.smartObjects.exportContents",
            "measurementLog.export",
            "layer.videoLayers.reloadFrame",
            "image.applyDataSet",
            "edit.colorSettings",
        ] {
            assert!(authorize_engine_command(id, &serde_json::json!({})).is_err());
        }
        assert!(authorize_engine_command("file.new", &serde_json::json!({})).is_ok());
        assert!(authorize_engine_command("image.mode.cmyk", &serde_json::json!({})).is_ok());
        assert!(authorize_desktop_engine_command("image.mode.cmyk", &serde_json::json!({})).is_err());
        assert!(authorize_engine_command("filter.distort.displace", &serde_json::json!({"mapPath": "outside.png"})).is_err());
        assert!(authorize_engine_command("layer.setAdjustment", &serde_json::json!({"file": "outside.cube"})).is_err());
        assert!(authorize_engine_command("prefs.set", &serde_json::json!({"path": "colorSettings.workingRgb", "value": "outside.icc"})).is_err());
    }
}
