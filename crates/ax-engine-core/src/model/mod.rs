//! Model runtime facade, architecture registry, and shared inference helpers.

pub mod arch;
pub mod config;
pub mod decode;
pub(crate) mod execution_plan;
pub mod fingerprint;
pub mod forward;
mod inference_model;
pub(crate) mod layer_ops;
pub(crate) mod moe_utils;
#[cfg(target_os = "macos")]
mod prefill_schedule;
pub mod registry;
pub(crate) mod shared;
pub mod weights;
pub use arch::qwen3_5 as qwen35;
pub use arch::{gemma3, gemma4, llama, qwen3_5};
pub use registry as arch_registry;

pub use config::ModelConfig;
pub use decode::{
    DecodeControl, DecodeIntent, DecodeMetalPerfSummary, DecodeMode, DecodeRunConfig,
    DecodeRunResult, DecodeSelection, run_decode, select_decode_mode,
};
pub use fingerprint::ModelFingerprint;
pub use forward::ForwardPass;
pub use inference_model::InferenceModel;
#[allow(deprecated)]
pub use inference_model::LlamaModel;
pub use weights::WeightStore;
