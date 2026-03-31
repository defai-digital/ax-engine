//! LLaMA-family transformer forward pass.
//!
//! Implements the full inference pipeline:
//!   1. Token embedding lookup
//!   2. For each layer:
//!      a. RMSNorm (attention norm)
//!      b. Q/K/V projections (matmul)
//!      c. RoPE on Q and K
//!      d. KV cache update
//!      e. Multi-head attention
//!      f. Output projection (matmul)
//!      g. Residual add
//!      h. RMSNorm (FFN norm)
//!      i. Gate/Up projections (matmul)
//!      j. SiLU activation + element-wise multiply
//!      k. Down projection (matmul)
//!      l. Residual add
//!   3. Final RMSNorm
//!   4. LM head projection -> logits
//!
//! # v2 API changes
//! - `kv_cache: &mut KvCache` → `kv: &mut ModelKv`
//! - GPU decode path uses `kv.as_gpu_mut()` directly, no Mutex locking
//! - `forward_batch` GPU gate uses `ctx.backend.use_gpu_decode()` instead of AX_CPU_ONLY
//! - `gpu_kv.advance()` → `gpu_kv.finalize_token()`
//! - `gpu_kv.advance_by(n)` → `gpu_kv.finalize_batch(n)` (no CPU mirror sync)
//! - Paged KV deferred to v2.1 (stubs kept but disabled)

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::backend::metal::MetalOps;
use crate::compute::attention::{self, AttentionParams};
use crate::compute::rope;
use crate::compute::silu;
use crate::gguf::tensor::GgmlType;
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::decode::DecodeIntent;
use crate::model::execution_plan::{
    DecodeEncoderPlan, DecodeExecutionPlan, DecodeScratchPlan, GpuBatchPrefillExecutionPlan,
    GpuDecodeExecutionPlan, PrefillAttentionPlan, PrefillExecutionPlan, PrefillFfnActivationPlan,
    PrefillLogitsPlan, PrefillMode, PrefillProjectionInputPlan, PrefillResidualHandoffPlan,
    PrefillWoInputPlan,
};
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    apply_attention_norm_single, apply_output_norm_single, cache_output_head_keys,
    encode_batch_logits, encode_dequant_batch, encode_dequant_batch_f16in,
    encode_dequant_batch_pair_f16in, encode_dequant_matvec, encode_dequant_matvec_with_config,
    ensure_precomputed_linear_f16, ensure_precomputed_linear_f16_many,
    ensure_precomputed_lm_head_f16, gpu_decode_quant_supported,
    gpu_prefill_q5k_small_n_auto_eligible, gpu_prefill_uses_q5k,
    write_normalized_single_logits_with_breakdown,
};
use crate::model::weights::WeightStore;
use std::sync::OnceLock;

/// Timing helper: records elapsed time into `ops.$field` if ops is Some.
macro_rules! timed {
    ($ops:expr, $field:ident, $body:expr) => {{
        if let Some(ref mut ops) = $ops {
            let _t = OpTimer::start();
            let _r = $body;
            ops.$field += _t.elapsed();
            _r
        } else {
            $body
        }
    }};
}

/// Whether explicit Metal barriers are enabled for single-token decode path.
///
/// Controlled by `AX_METAL_DECODE_BARRIERS`:
/// - `1` / `true` / `on`            -> enabled
/// - unset / `0` / `false` / `off`  -> disabled (default)
///
/// Default OFF: llama.cpp uses zero barriers in decode. The sequential
/// Metal encoder on Apple Silicon UMA provides ordering guarantees —
/// each dispatch completes before the next starts, and writes are
/// immediately visible via cache coherency. Barriers are redundant.
pub(crate) fn metal_decode_barriers_enabled() -> bool {
    match std::env::var("AX_METAL_DECODE_BARRIERS") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false, // llama.cpp uses 0 barriers; safe on Apple Silicon UMA
    }
}

/// Whether mistral-style HD128 prefill should write attention output in f16.
///
/// Controlled by `AX_METAL_PREFILL_MISTRAL_F16OUT`:
/// - `1` / `true` / `on` -> enabled
/// - unset / other       -> disabled (default)
fn metal_prefill_attn_f16out_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_MISTRAL_F16OUT") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn metal_prefill_use_cached0_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("AX_METAL_PREFILL_USE_CACHED0") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}

fn metal_prefill_split_rope_append_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("AX_METAL_PREFILL_SPLIT_ROPE_APPEND") {
            Ok(v) => {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "on"
            }
            Err(_) => true,
        },
    )
}

include!("llama/gpu.rs");
include!("llama/core.rs");
include!("llama/forward.rs");
include!("llama/model.rs");
#[cfg(test)]
mod tests;
