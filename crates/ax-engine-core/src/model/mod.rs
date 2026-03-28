pub mod arch_registry;
pub mod config;
pub mod decode;
pub(crate) mod execution_plan;
pub mod forward;
pub mod gemma3;
pub(crate) mod layer_ops;
pub mod llama;
#[cfg(target_os = "macos")]
mod prefill_schedule;
pub mod qwen3;
pub mod qwen35;
pub mod qwen3_moe;
pub(crate) mod shared;
pub mod weights;

pub use config::ModelConfig;
pub use decode::{
    DecodeControl, DecodeIntent, DecodeMetalPerfSummary, DecodeMode, DecodeRunConfig,
    DecodeRunResult, DecodeSelection, run_decode, select_decode_mode,
};
pub use forward::ForwardPass;
pub use llama::LlamaModel;
pub use weights::WeightStore;
