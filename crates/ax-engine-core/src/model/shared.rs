//! Shared helpers used by all three model implementations (LLaMA, Gemma3, Qwen3).

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::backend::Backend;
use crate::backend::metal::MetalOps;
use crate::compute::rms_norm;
use crate::gguf::tensor::GgmlType;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::weights::WeightStore;

const LAYER_SUFFIXES: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

include!("shared/runtime.rs");
include!("shared/gpu_decode.rs");
include!("shared/gpu_batch.rs");
#[cfg(test)]
mod tests;
