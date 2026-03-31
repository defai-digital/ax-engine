use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Once};

use anyhow::{Context, anyhow, ensure};
use ax_engine_core::backend::{BackendConfig, create_backend};
use ax_engine_core::gguf::MappedModel;
use ax_engine_core::model::{LlamaModel, ModelConfig};
use ax_engine_core::tokenizer::Tokenizer;

use crate::llama_cpp_process::LlamaCppProcess;
use crate::routing::{InferenceBackendKind, RoutingPolicy};
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

pub(crate) struct LlamaCppLoadedModel {
    pub(crate) model_id: String,
    pub(crate) config: ModelConfig,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) process: Arc<LlamaCppProcess>,
    pub(crate) architecture: String,
    pub(crate) model_name: Option<String>,
    pub(crate) support_note: Option<String>,
    pub(crate) routing_message: Option<String>,
}

pub(crate) enum LoadedModel {
    Native(Box<NativeLoadedModel>),
    LlamaCpp(Box<LlamaCppLoadedModel>),
}

// SAFETY: Both variants are immutable after construction. Session/KV state and
// subprocess lifecycle coordination live behind synchronization primitives.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

#[derive(Clone)]
pub struct Model {
    inner: Arc<LoadedModel>,
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
    pub backend: InferenceBackendKind,
    pub routing: Option<String>,
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

        let routing = RoutingPolicy::from_env()?.resolve(model_path)?;
        match routing.backend {
            InferenceBackendKind::Native => {
                init_runtime();
                tracing::info!(
                    architecture = %routing.support.architecture,
                    predominant_quant = %routing
                        .support
                        .predominant_quant
                        .map(|dtype| dtype.to_string())
                        .unwrap_or_else(|| "unknown".to_string()),
                    route_source = routing.source,
                    "model loaded natively"
                );
                Self::load_native(model_path, options)
            }
            InferenceBackendKind::LlamaCpp => {
                let reason = routing.reason.clone();
                if let Some(reason) = reason.as_deref() {
                    tracing::warn!(%reason, "native support unavailable; routing to llama.cpp");
                }
                Self::load_llama_cpp(model_path, options, reason)
            }
        }
    }

    fn load_native(model_path: &Path, options: LoadOptions) -> anyhow::Result<Self> {
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
        let profile_quant = mapped
            .predominant_quant()
            .map(|dtype| dtype.to_string())
            .unwrap_or_else(|| "default".to_string());
        let profile_architecture = mapped.header.architecture().unwrap_or("default");

        let backend = create_backend(options.backend.into_backend_config())?;
        backend.configure_for_model(&model_id, &profile_quant, profile_architecture)?;

        let model = LlamaModel::with_backend(config.clone(), backend)?;

        Ok(Self {
            inner: Arc::new(LoadedModel::Native(Box::new(NativeLoadedModel {
                model_id,
                mapped,
                config,
                tokenizer,
                model,
                model_name,
                support_note,
            }))),
        })
    }

    fn load_llama_cpp(
        model_path: &Path,
        options: LoadOptions,
        routing_message: Option<String>,
    ) -> anyhow::Result<Self> {
        let mapped = MappedModel::open(model_path)
            .with_context(|| format!("failed to open GGUF model: {}", model_path.display()))?;
        let mut config = ModelConfig::from_gguf(&mapped.header)
            .context("failed to read model configuration from GGUF metadata")?;
        if let Some(context_length) = options.context_length {
            ensure!(
                context_length > 0,
                "context_length must be greater than zero"
            );
            config.context_length = context_length;
        }

        let tokenizer = Tokenizer::from_gguf(&mapped.header)
            .context("failed to construct tokenizer from GGUF metadata")?;
        let model_id = infer_model_id(model_path, &mapped);
        let architecture = mapped.header.architecture().unwrap_or("llama").to_string();
        let model_name = mapped.header.get_str("general.name").map(str::to_owned);
        let support_note = mapped.support_note().map(str::to_owned);
        let process = Arc::new(LlamaCppProcess::spawn(model_path, config.context_length)?);

        tracing::info!(
            model = %model_path.display(),
            port = process.port(),
            "llama.cpp subprocess ready"
        );

        Ok(Self {
            inner: Arc::new(LoadedModel::LlamaCpp(Box::new(LlamaCppLoadedModel {
                model_id,
                config,
                tokenizer,
                process,
                architecture,
                model_name,
                support_note,
                routing_message,
            }))),
        })
    }

    pub fn info(&self) -> ModelInfo {
        match self.inner.as_ref() {
            LoadedModel::Native(model) => ModelInfo {
                id: model.model_id.clone(),
                architecture: model.model.arch_name().to_string(),
                context_length: model.config.context_length as usize,
                vocab_size: model.tokenizer.vocab_size(),
                bos_token_id: model.tokenizer.bos_id(),
                eos_token_id: model.tokenizer.eos_id(),
                model_name: model.model_name.clone(),
                support_note: model.support_note.clone(),
                backend: InferenceBackendKind::Native,
                routing: None,
            },
            LoadedModel::LlamaCpp(model) => ModelInfo {
                id: model.model_id.clone(),
                architecture: model.architecture.clone(),
                context_length: model.config.context_length as usize,
                vocab_size: model.tokenizer.vocab_size(),
                bos_token_id: model.tokenizer.bos_id(),
                eos_token_id: model.tokenizer.eos_id(),
                model_name: model.model_name.clone(),
                support_note: model.support_note.clone(),
                backend: InferenceBackendKind::LlamaCpp,
                routing: model.routing_message.clone(),
            },
        }
    }

    pub fn id(&self) -> &str {
        match self.inner.as_ref() {
            LoadedModel::Native(model) => &model.model_id,
            LoadedModel::LlamaCpp(model) => &model.model_id,
        }
    }

    pub fn architecture(&self) -> &str {
        match self.inner.as_ref() {
            LoadedModel::Native(model) => model.model.arch_name(),
            LoadedModel::LlamaCpp(model) => &model.architecture,
        }
    }

    pub fn context_length(&self) -> usize {
        self.config().context_length as usize
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer().vocab_size()
    }

    pub fn bos_token_id(&self) -> u32 {
        self.tokenizer().bos_id()
    }

    pub fn eos_token_id(&self) -> u32 {
        self.tokenizer().eos_id()
    }

    pub fn model_name(&self) -> Option<&str> {
        match self.inner.as_ref() {
            LoadedModel::Native(model) => model.model_name.as_deref(),
            LoadedModel::LlamaCpp(model) => model.model_name.as_deref(),
        }
    }

    pub fn support_note(&self) -> Option<&str> {
        match self.inner.as_ref() {
            LoadedModel::Native(model) => model.support_note.as_deref(),
            LoadedModel::LlamaCpp(model) => model.support_note.as_deref(),
        }
    }

    pub fn inference_backend_kind(&self) -> InferenceBackendKind {
        match self.inner.as_ref() {
            LoadedModel::Native(_) => InferenceBackendKind::Native,
            LoadedModel::LlamaCpp(_) => InferenceBackendKind::LlamaCpp,
        }
    }

    pub fn routing_message(&self) -> Option<&str> {
        match self.inner.as_ref() {
            LoadedModel::Native(_) => None,
            LoadedModel::LlamaCpp(model) => model.routing_message.as_deref(),
        }
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
        match self.inner.as_ref() {
            LoadedModel::Native(model) => &model.config,
            LoadedModel::LlamaCpp(model) => &model.config,
        }
    }

    pub(crate) fn tokenizer(&self) -> &Tokenizer {
        match self.inner.as_ref() {
            LoadedModel::Native(model) => &model.tokenizer,
            LoadedModel::LlamaCpp(model) => &model.tokenizer,
        }
    }

    pub(crate) fn native_loaded(&self) -> Option<&NativeLoadedModel> {
        match self.inner.as_ref() {
            LoadedModel::Native(model) => Some(model),
            LoadedModel::LlamaCpp(_) => None,
        }
    }

    pub(crate) fn llama_cpp_loaded(&self) -> Option<&LlamaCppLoadedModel> {
        match self.inner.as_ref() {
            LoadedModel::Native(_) => None,
            LoadedModel::LlamaCpp(model) => Some(model),
        }
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
