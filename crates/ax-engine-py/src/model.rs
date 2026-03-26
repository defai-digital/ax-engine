use std::path::Path;
use std::sync::{Arc, Mutex, Once};

use anyhow::{Context, anyhow, ensure};
use ax_core::backend::{BackendConfig, create_backend};
use ax_core::gguf::MappedModel;
use ax_core::model::{LlamaModel, ModelConfig};
use ax_core::tokenizer::Tokenizer;
use pyo3::prelude::*;

use crate::errors::py_runtime_error;
use crate::session::Session;

pub(crate) struct LoadedModel {
    pub(crate) mapped: MappedModel,
    pub(crate) config: ModelConfig,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model: LlamaModel,
}

// SAFETY: LoadedModel is immutable after construction. Mutable inference state
// such as KV caches, token history, sampler state, and logits buffers lives in
// Session state, not on the loaded model itself. This matches the engine's
// ownership model: `MappedModel` is read-only mmap data, `Tokenizer` is
// immutable, and `LlamaModel` methods operate on caller-owned mutable buffers.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

#[pyclass(unsendable, module = "ax_engine")]
pub struct Model {
    pub(crate) inner: Mutex<Option<Arc<LoadedModel>>>,
}

#[pymethods]
impl Model {
    #[staticmethod]
    #[pyo3(signature = (path, backend = None))]
    pub fn load(py: Python<'_>, path: &str, backend: Option<String>) -> PyResult<Self> {
        py.allow_threads(|| Self::load_inner(path, backend.as_deref()))
            .map_err(py_runtime_error)
    }

    #[getter]
    pub fn architecture(&self) -> PyResult<String> {
        Ok(self.loaded_model()?.model.arch_name().to_string())
    }

    #[getter]
    pub fn context_length(&self) -> PyResult<usize> {
        Ok(self.loaded_model()?.config.context_length as usize)
    }

    #[getter]
    pub fn vocab_size(&self) -> PyResult<usize> {
        Ok(self.loaded_model()?.tokenizer.vocab_size())
    }

    #[getter]
    pub fn bos_token_id(&self) -> PyResult<u32> {
        Ok(self.loaded_model()?.tokenizer.bos_id())
    }

    #[getter]
    pub fn eos_token_id(&self) -> PyResult<u32> {
        Ok(self.loaded_model()?.tokenizer.eos_id())
    }

    #[getter]
    pub fn model_name(&self) -> PyResult<Option<String>> {
        Ok(self
            .loaded_model()?
            .mapped
            .header
            .get_str("general.name")
            .map(str::to_string))
    }

    #[getter]
    pub fn support_note(&self) -> PyResult<Option<String>> {
        Ok(self
            .loaded_model()?
            .mapped
            .support_note()
            .map(str::to_string))
    }

    #[getter]
    pub fn closed(&self) -> bool {
        self.inner.lock().expect("model lock poisoned").is_none()
    }

    #[pyo3(signature = (text, add_special = false))]
    pub fn tokenize(&self, text: &str, add_special: bool) -> PyResult<Vec<u32>> {
        Ok(self.loaded_model()?.tokenizer.encode(text, add_special))
    }

    pub fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
        Ok(self.loaded_model()?.tokenizer.decode(&token_ids))
    }

    #[pyo3(signature = (ctx_size = None, seed = None))]
    pub fn session(&self, ctx_size: Option<usize>, seed: Option<u64>) -> PyResult<Session> {
        Session::new(self.loaded_model()?, ctx_size, seed).map_err(py_runtime_error)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) {
        self.close();
    }

    fn __repr__(&self) -> String {
        match self.loaded_model() {
            Ok(model) => format!(
                "Model(architecture={:?}, vocab_size={}, context_length={}, closed=False)",
                model.model.arch_name(),
                model.tokenizer.vocab_size(),
                model.config.context_length
            ),
            Err(_) => "Model(closed=True)".to_string(),
        }
    }

    pub fn close(&self) {
        let mut inner = self.inner.lock().expect("model lock poisoned");
        inner.take();
    }
}

impl Model {
    fn load_inner(path: &str, backend: Option<&str>) -> anyhow::Result<Self> {
        init_tracing();

        let model_path = Path::new(path);
        ensure!(
            model_path.exists(),
            "model path does not exist: {}",
            model_path.display()
        );

        let mapped = MappedModel::open(model_path)
            .with_context(|| format!("failed to open GGUF model: {}", model_path.display()))?;
        let config = ModelConfig::from_gguf(&mapped.header)
            .context("failed to read model configuration from GGUF metadata")?;
        let tokenizer = Tokenizer::from_gguf(&mapped.header)
            .context("failed to construct tokenizer from GGUF metadata")?;

        let profile_model_name = model_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(str::to_owned)
            .or_else(|| mapped.header.get_str("general.name").map(str::to_owned))
            .or_else(|| mapped.header.architecture().map(str::to_owned))
            .unwrap_or_else(|| "default".to_string());
        let profile_quant = mapped
            .predominant_quant()
            .map(|dtype| dtype.to_string())
            .unwrap_or_else(|| "default".to_string());
        let profile_architecture = mapped.header.architecture().unwrap_or("default");

        // Validate architecture before with_backend (which panics on unsupported arch)
        ax_core::model::arch_registry::forward_for_arch(&config.architecture)?;

        let backend_config = parse_backend(backend)?;
        let backend = create_backend(backend_config)?;
        backend.configure_for_model(&profile_model_name, &profile_quant, profile_architecture)?;

        let model = LlamaModel::with_backend(config.clone(), backend);

        Ok(Self {
            inner: Mutex::new(Some(Arc::new(LoadedModel {
                mapped,
                config,
                tokenizer,
                model,
            }))),
        })
    }

    pub(crate) fn loaded_model(&self) -> PyResult<Arc<LoadedModel>> {
        self.inner
            .lock()
            .expect("model lock poisoned")
            .clone()
            .ok_or_else(|| py_runtime_error(anyhow!("model is closed")))
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.take();
        }
    }
}

fn parse_backend(value: Option<&str>) -> anyhow::Result<BackendConfig> {
    match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(BackendConfig::default()),
        "cpu" => Ok(BackendConfig::Cpu),
        "metal" => Ok(BackendConfig::Metal),
        "hybrid" => Ok(BackendConfig::Hybrid),
        "hybrid_cpu_decode" | "hybrid-cpu-decode" | "hybrid_cpu" => {
            Ok(BackendConfig::HybridCpuDecode)
        }
        other => Err(anyhow!(
            "unsupported backend '{other}'; expected auto, cpu, metal, hybrid, or hybrid_cpu_decode"
        )),
    }
}

fn init_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("ax_core=warn,ax_engine_py=info")
            .try_init();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_backend_accepts_known_values() {
        assert_eq!(parse_backend(None).unwrap(), BackendConfig::Hybrid);
        assert_eq!(parse_backend(Some("cpu")).unwrap(), BackendConfig::Cpu);
        assert_eq!(parse_backend(Some("metal")).unwrap(), BackendConfig::Metal);
        assert_eq!(
            parse_backend(Some("hybrid_cpu_decode")).unwrap(),
            BackendConfig::HybridCpuDecode
        );
    }

    #[test]
    fn test_parse_backend_rejects_unknown_values() {
        let err = parse_backend(Some("cuda")).unwrap_err().to_string();
        assert!(err.contains("unsupported backend"));
    }
}
