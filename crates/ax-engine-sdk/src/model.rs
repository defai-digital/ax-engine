use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Once};

use anyhow::{Context, anyhow, ensure};
use ax_engine_core::backend::{BackendConfig, create_backend};
use ax_engine_core::gguf::MappedModel;
use ax_engine_core::model::{LlamaModel, ModelConfig, ModelFingerprint};
use ax_engine_core::tokenizer::Tokenizer;

use crate::session::{Session, SessionOptions};

pub(crate) struct NativeLoadedModel {
    pub(crate) model_id: String,
    pub(crate) mapped: MappedModel,
    pub(crate) config: ModelConfig,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model: LlamaModel,
    pub(crate) model_name: Option<String>,
    pub(crate) support_note: Option<String>,
}

// SAFETY: All fields are immutable after construction. Session/KV state lives
// behind synchronization primitives.
unsafe impl Send for NativeLoadedModel {}
unsafe impl Sync for NativeLoadedModel {}

#[derive(Clone)]
pub struct Model {
    inner: Arc<NativeLoadedModel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    #[default]
    Auto,
    Cpu,
    Metal,
    Hybrid,
    HybridCpuDecode,
}

#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    pub backend: BackendKind,
    pub context_length: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub architecture: String,
    pub context_length: usize,
    pub vocab_size: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub model_name: Option<String>,
    pub support_note: Option<String>,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Hybrid => "hybrid",
            Self::HybridCpuDecode => "hybrid_cpu_decode",
        }
    }

    #[cfg(test)]
    pub(crate) fn from_backend_config(config: BackendConfig) -> Self {
        match config {
            BackendConfig::Cpu => Self::Cpu,
            BackendConfig::Metal => Self::Metal,
            BackendConfig::Hybrid => Self::Hybrid,
            BackendConfig::HybridCpuDecode => Self::HybridCpuDecode,
        }
    }

    fn into_backend_config(self) -> BackendConfig {
        match self {
            Self::Auto => BackendConfig::default(),
            Self::Cpu => BackendConfig::Cpu,
            Self::Metal => BackendConfig::Metal,
            Self::Hybrid => BackendConfig::Hybrid,
            Self::HybridCpuDecode => BackendConfig::HybridCpuDecode,
        }
    }
}

impl FromStr for BackendKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "hybrid" => Ok(Self::Hybrid),
            "hybrid_cpu_decode" | "hybrid-cpu-decode" | "hybrid_cpu" => Ok(Self::HybridCpuDecode),
            other => Err(anyhow!(
                "unsupported backend '{other}'; expected auto, cpu, metal, hybrid, or hybrid_cpu_decode"
            )),
        }
    }
}

impl Model {
    pub fn load(path: impl AsRef<Path>, options: LoadOptions) -> anyhow::Result<Self> {
        let model_path = path.as_ref();
        ensure!(
            model_path.exists(),
            "model path does not exist: {}",
            model_path.display()
        );

        init_runtime();

        let mapped = MappedModel::open(model_path)
            .with_context(|| format!("failed to open GGUF model: {}", model_path.display()))?;
        let mut config = ModelConfig::from_gguf(&mapped.header)
            .context("failed to read model configuration from GGUF metadata")?;
        if let Some(context_length) = options.context_length {
            ensure!(
                context_length > 0,
                "context_length must be greater than zero"
            );
            if context_length != config.context_length {
                config.context_length = context_length;
            }
        }
        let tokenizer = Tokenizer::from_gguf(&mapped.header)
            .context("failed to construct tokenizer from GGUF metadata")?;

        let model_id = infer_model_id(model_path, &mapped);
        let model_name = mapped.header.get_str("general.name").map(str::to_owned);
        let support_note = mapped.support_note().map(str::to_owned);

        let backend = create_backend(options.backend.into_backend_config())?;
        let fingerprint = ModelFingerprint::from_mapped_model(Some(model_path), &mapped, &config);
        backend.configure_for_fingerprint(&fingerprint)?;

        let model = LlamaModel::with_backend(config.clone(), backend)?;

        tracing::info!(
            architecture = %model.arch_name(),
            model_id = %model_id,
            "model loaded"
        );

        Ok(Self {
            inner: Arc::new(NativeLoadedModel {
                model_id,
                mapped,
                config,
                tokenizer,
                model,
                model_name,
                support_note,
            }),
        })
    }

    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            id: self.inner.model_id.clone(),
            architecture: self.inner.model.arch_name().to_string(),
            context_length: self.inner.config.context_length as usize,
            vocab_size: self.inner.tokenizer.vocab_size(),
            bos_token_id: self.inner.tokenizer.bos_id(),
            eos_token_id: self.inner.tokenizer.eos_id(),
            model_name: self.inner.model_name.clone(),
            support_note: self.inner.support_note.clone(),
        }
    }

    pub fn id(&self) -> &str {
        &self.inner.model_id
    }

    pub fn architecture(&self) -> &str {
        self.inner.model.arch_name()
    }

    pub fn context_length(&self) -> usize {
        self.inner.config.context_length as usize
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.tokenizer.vocab_size()
    }

    pub fn bos_token_id(&self) -> u32 {
        self.inner.tokenizer.bos_id()
    }

    pub fn eos_token_id(&self) -> u32 {
        self.inner.tokenizer.eos_id()
    }

    pub fn model_name(&self) -> Option<&str> {
        self.inner.model_name.as_deref()
    }

    pub fn support_note(&self) -> Option<&str> {
        self.inner.support_note.as_deref()
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Vec<u32> {
        self.tokenizer().encode(text, add_special)
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        self.tokenizer().decode(token_ids)
    }

    pub fn session(&self, options: SessionOptions) -> anyhow::Result<Session> {
        Session::new(self.clone(), options)
    }

    pub(crate) fn config(&self) -> &ModelConfig {
        &self.inner.config
    }

    pub(crate) fn tokenizer(&self) -> &Tokenizer {
        &self.inner.tokenizer
    }

    pub(crate) fn native(&self) -> &NativeLoadedModel {
        &self.inner
    }
}

fn infer_model_id(model_path: &Path, mapped: &MappedModel) -> String {
    model_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_owned)
        .or_else(|| mapped.header.get_str("general.name").map(str::to_owned))
        .or_else(|| mapped.header.architecture().map(str::to_owned))
        .unwrap_or_else(|| "default".to_string())
}

fn init_runtime() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = ax_engine_core::scheduler::init_global_threadpool();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_kind_from_str_accepts_known_values() {
        assert_eq!("auto".parse::<BackendKind>().unwrap(), BackendKind::Auto);
        assert_eq!("cpu".parse::<BackendKind>().unwrap(), BackendKind::Cpu);
        assert_eq!("metal".parse::<BackendKind>().unwrap(), BackendKind::Metal);
        assert_eq!(
            "hybrid_cpu_decode".parse::<BackendKind>().unwrap(),
            BackendKind::HybridCpuDecode
        );
    }

    #[test]
    fn test_backend_kind_from_backend_config_maps_variants() {
        assert_eq!(
            BackendKind::from_backend_config(BackendConfig::HybridCpuDecode),
            BackendKind::HybridCpuDecode
        );
    }

    #[test]
    fn test_backend_kind_from_str_rejects_unknown_values() {
        let err = "cuda".parse::<BackendKind>().unwrap_err().to_string();
        assert!(err.contains("unsupported backend"));
    }
}
