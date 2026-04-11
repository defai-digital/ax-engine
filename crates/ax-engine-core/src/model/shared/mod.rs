//! Shared helpers reused across the architecture forward-pass implementations.

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

mod gpu_batch;
mod gpu_decode;
mod moe;
mod runtime;
#[cfg(test)]
mod tests;

pub(crate) use gpu_batch::*;
pub(crate) use gpu_decode::*;
pub(crate) use moe::*;
pub(crate) use runtime::*;
