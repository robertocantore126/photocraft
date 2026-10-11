//! ONNX Runtime, loaded once per process (`load-dynamic`), and the models run on it.
//!
//! The library is looked for, in order: `PHOTOCRAFT_ORT_DYLIB`; next to the executable; the
//! models folders (Fotox keeps a DirectML build there); then the system's library search path.
//! Without it every AI feature says so.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::{AiError, Input, Tensor32};

fn lib_name() -> &'static str {
    if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// Where the runtime library would be, if anywhere.
pub fn find_library() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("PHOTOCRAFT_ORT_DYLIB").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    let beside = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join(lib_name())));
    let in_models = crate::models::model_dirs().into_iter().map(|d| d.join(lib_name()));
    beside.into_iter().chain(in_models).find(|p| p.is_file()).or_else(|| {
        let dirs = std::env::var_os("PATH").map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>()).unwrap_or_default();
        dirs.into_iter().map(|d| d.join(lib_name())).find(|p| p.is_file())
    })
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use ort::session::Session;
    use ort::value::Tensor;
    use std::sync::{Mutex, OnceLock, PoisonError};

    static STATE: OnceLock<Result<PathBuf, String>> = OnceLock::new();

    /// Load and initialise ONNX Runtime (once); the library's path, or why not.
    pub fn ensure() -> Result<PathBuf, AiError> {
        STATE
            .get_or_init(|| {
                let path = find_library().ok_or_else(|| format!("{} was not found (set PHOTOCRAFT_ORT_DYLIB or put it next to PhotoCraft)", lib_name()))?;
                let builder = ort::init_from(&path).map_err(|e| e.to_string())?;
                builder.with_name("photocraft").commit();
                Ok(path)
            })
            .clone()
            .map_err(AiError::Runtime)
    }

    /// A loaded model; `run` takes the inputs in the model's input order.
    pub struct Model {
        session: Mutex<Session>,
        pub inputs: Vec<String>,
        pub outputs: Vec<String>,
    }

    impl Model {
        /// Load `path`, DirectML first (Windows), the CPU otherwise (`PHOTOCRAFT_AI_CPU=1` forces
        /// the CPU).
        pub fn load(path: &Path) -> Result<Self, AiError> {
            ensure()?;
            let err = |e: ort::Error<ort::session::builder::SessionBuilder>| AiError::Inference(e.to_string());
            let mut builder = Session::builder()?;
            #[cfg(windows)]
            let providers = {
                use ort::ep::ExecutionProvider;
                let dml = ort::ep::DirectML::default();
                // DirectML only when the runtime lists it: registering it on a CPU-only build
                // crashes EfficientSAM's decoder instead of returning an error.
                if std::env::var_os("PHOTOCRAFT_AI_CPU").is_none() && dml.is_available().unwrap_or(false) {
                    // DirectML wants sequential execution without memory patterns; its graph
                    // fusion runs out of memory on BiRefNet at 1024².
                    builder = builder
                        .with_memory_pattern(false)
                        .map_err(err)?
                        .with_parallel_execution(false)
                        .map_err(err)?
                        .with_config_entry("ep.dml.disable_graph_fusion", "1")
                        .map_err(err)?;
                    vec![dml.build(), ort::ep::CPU::default().build()]
                } else {
                    vec![ort::ep::CPU::default().build()]
                }
            };
            #[cfg(not(windows))]
            let providers = [ort::ep::CPU::default().build()];
            builder = builder.with_execution_providers(providers).map_err(err)?;
            let session = builder.commit_from_file(path)?;
            let inputs = session.inputs().iter().map(|o| o.name().to_owned()).collect();
            let outputs = session.outputs().iter().map(|o| o.name().to_owned()).collect();
            Ok(Self { session: Mutex::new(session), inputs, outputs })
        }

        /// Run with `inputs` (the model's inputs, in order); every output as f32.
        pub fn run(&self, inputs: Vec<Input>) -> Result<Vec<Tensor32>, AiError> {
            let mut values = Vec::with_capacity(inputs.len());
            for (name, input) in self.inputs.iter().zip(inputs) {
                let value: ort::session::SessionInputValue<'_> = match input {
                    Input::F32(t) => Tensor::from_array((t.shape, t.data.into_boxed_slice()))?.into(),
                    Input::I64 { shape, data } => Tensor::from_array((shape, data.into_boxed_slice()))?.into(),
                };
                values.push((std::borrow::Cow::<str>::from(name.clone()), value));
            }
            let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
            let outputs = session.run(values)?;
            let mut out = Vec::with_capacity(self.outputs.len());
            for name in &self.outputs {
                let value = outputs.get(name.as_str()).ok_or_else(|| AiError::Inference(format!("no output {name}")))?;
                let (shape, data) = value.try_extract_tensor::<f32>()?;
                out.push(Tensor32::new(shape.iter().map(|d| (*d).max(0) as usize).collect(), data.to_vec()));
            }
            Ok(out)
        }
    }

    /// A model loaded once per process and kept.
    pub fn model(path: &Path) -> Result<Arc<Model>, AiError> {
        static MODELS: Mutex<Vec<(PathBuf, Arc<Model>)>> = Mutex::new(Vec::new());
        let mut models = MODELS.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, m)) = models.iter().find(|(p, _)| p == path) {
            return Ok(m.clone());
        }
        let m = Arc::new(Model::load(path)?);
        models.push((path.to_owned(), m.clone()));
        Ok(m)
    }
}

#[cfg(target_arch = "wasm32")]
mod native {
    use super::*;

    pub fn ensure() -> Result<PathBuf, AiError> {
        Err(AiError::Runtime("the web build has no local AI".into()))
    }

    pub struct Model {
        pub inputs: Vec<String>,
        pub outputs: Vec<String>,
    }

    impl Model {
        pub fn run(&self, _inputs: Vec<Input>) -> Result<Vec<Tensor32>, AiError> {
            Err(AiError::Runtime("the web build has no local AI".into()))
        }
    }

    pub fn model(_path: &Path) -> Result<Arc<Model>, AiError> {
        Err(AiError::Runtime("the web build has no local AI".into()))
    }
}

pub use native::{Model, ensure, model};

/// Can the AI run at all (runtime found, without loading it)?
pub fn available() -> bool {
    !cfg!(target_arch = "wasm32") && find_library().is_some()
}

/// The inputs in the model's order, matched by name.
pub fn ordered(model: &Model, mut named: Vec<(&str, Input)>) -> Result<Vec<Input>, AiError> {
    let mut out = Vec::with_capacity(model.inputs.len());
    for name in &model.inputs {
        let i = named.iter().position(|(n, _)| n == name).ok_or_else(|| AiError::Inference(format!("the model wants an input named {name}")))?;
        out.push(named.swap_remove(i).1);
    }
    Ok(out)
}

/// Load the selection models on a background thread (and run BiRefNet once on a blank image), so
/// the first Object Selection does not wait for them.
pub fn warm() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::models::{BIREFNET, EFFICIENT_SAM};
        let _ = std::thread::Builder::new().name("photocraft-ai-warm".into()).spawn(|| {
            if ensure().is_err() {
                return;
            }
            if let Some(dir) = BIREFNET.dir()
                && let Ok(m) = model(&dir.join(BIREFNET.files[0].file))
            {
                const SIDE: usize = 1024;
                let _ = m.run(vec![Tensor32::new(vec![1, 3, SIDE, SIDE], vec![0.0; 3 * SIDE * SIDE]).into()]);
            }
            if let Some(dir) = EFFICIENT_SAM.dir() {
                for f in EFFICIENT_SAM.files {
                    let _ = model(&dir.join(f.file));
                }
            }
        });
    }
}
