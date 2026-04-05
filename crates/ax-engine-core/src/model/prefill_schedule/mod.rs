//! Pre-computed prefill dispatch schedule ("graph IR").
//!
//! Instead of evaluating execution plans inline during Metal encoding,
//! the schedule is built once upfront with all decisions resolved —
//! kernel variant, buffer pointers, dimensions, barrier positions.
//! The encoding loop becomes a tight match over the flat op list.
//!
//! Ports llama.cpp's graph-compute pattern: pre-build the graph, then
//! encode it with minimal per-dispatch overhead.

use std::sync::OnceLock;

use crate::backend::metal::MetalOps;
use crate::gguf::tensor::GgmlType;
use crate::model::execution_plan::{
    DecodeExecutionPlan, GpuBatchPrefillExecutionPlan, LlamaPrefillQkvPostPlan,
    PrefillAttentionPlan, PrefillFfnActivationPlan, PrefillLogitsPlan, PrefillProjectionInputPlan,
    PrefillResidualHandoffPlan, PrefillWoInputPlan,
};
use crate::model::shared::{
    encode_batch_logits, encode_dequant_batch, encode_dequant_batch_f16in,
    encode_dequant_batch_pair_f16in, encode_dequant_matvec,
};

include!("common.rs");
include!("qwen3_5.rs");
#[cfg(test)]
mod tests;
