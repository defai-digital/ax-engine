//! C-compatible type definitions matching llama.h

use std::os::raw::c_int;

use ax_core::gguf::MappedModel as CoreMappedModel;
use ax_core::kv::ModelKv;
use ax_core::model::{LlamaModel as CoreLlamaModel, ModelConfig};
use ax_core::tokenizer::Tokenizer;

/// Token type alias (matches llama.h: typedef int32_t llama_token).
pub type LlamaToken = i32;

/// Opaque model handle.
/// Owns the GGUF mmap, config, tokenizer, and forward-pass model.
pub struct LlamaModel {
    pub(crate) mapped: CoreMappedModel,
    pub(crate) config: ModelConfig,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model: CoreLlamaModel,
}

/// Opaque context handle.
/// Owns inference state: KV cache, logits buffer, position, and RNG.
pub struct LlamaContext {
    pub(crate) model: *const LlamaModel,
    pub(crate) kv: ModelKv,
    pub(crate) logits: Vec<f32>,
    pub(crate) position: usize,
    pub(crate) n_ctx: u32,
    pub(crate) rng_state: u64,
}

/// Model loading parameters.
#[repr(C)]
pub struct LlamaModelParams {
    pub n_gpu_layers: c_int,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

/// Context creation parameters.
#[repr(C)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub seed: u32,
}

impl Default for LlamaModelParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            use_mmap: true,
            use_mlock: false,
        }
    }
}

impl Default for LlamaContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 2048,
            n_threads: 0,
            n_threads_batch: 0,
            seed: u32::MAX,
        }
    }
}
