//! Qwen3-family transformer forward pass.
//!
//! Differences from LLaMA:
//!   1. QKV bias terms: after each Q/K/V projection, a bias vector is added (when present)
//!   2. Per-head QK normalization: RMSNorm on Q and K vectors per-head before RoPE (when present)
//!
//! # v2 API changes
//! - `kv_cache: &mut KvCache` → `kv: &mut ModelKv`
//! - GPU decode path uses `kv.as_gpu_mut()` directly, no Mutex locking
//! - `forward_batch` GPU gate uses `ctx.backend.use_gpu_decode()` instead of AX_CPU_ONLY
//! - `gpu_kv.advance()` → `gpu_kv.finalize_token()`
//! - `gpu_kv.advance_by(n)` → `gpu_kv.finalize_batch(n)` (no CPU mirror sync)
//!
//! # v2 Qwen3 Bias Strategy
//!
//! v1 applied QKV biases per-token inside the batched prefill loop:
//!   for i in 0..n_tokens {
//!       copy q[i] → scratch; add_bias(scratch); per_head_norm(scratch); rope(scratch);
//!       copy_back(scratch → batch[i]); copy K/V to GPU KV cache;
//!   }
//!
//! This caused O(n_tokens × 7) Metal dispatches + barriers, defeating the purpose of batching.
//!
//! v2 solution: `encode_elementwise_add_batch` applies the bias to all token rows in one
//! batched dispatch (O(1) dispatches), then the existing batched `encode_qk_norm_rope_batch`
//! and `encode_kv_append_batch` handle the rest. This is the same O(1) strategy used for
//! the no-bias path, extended to the bias-present path.

use crate::backend::metal::MetalOps;

use crate::compute::attention;
use crate::compute::rope;
use crate::compute::silu;
use crate::gguf::tensor::GgmlType;
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::execution_plan::{
    DecodeEncoderPlan, DecodeExecutionPlan, GpuBatchPrefillExecutionPlan, GpuDecodeExecutionPlan,
    PrefillAttentionPlan, PrefillExecutionPlan, PrefillFfnActivationPlan, PrefillLogitsPlan,
    PrefillMode, PrefillProjectionInputPlan, PrefillResidualHandoffPlan, Qwen3DecodeLayerPlan,
    Qwen3PrefillQkvPost, qwen3_layer_plan_for_gpu,
};
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    apply_attention_norm_single, apply_optional_attention_qk_norm_single, apply_output_norm_single,
    cache_attention_qk_norm_keys, cache_optional_prefixed_f32_key, cache_output_head_keys,
    encode_batch_logits, encode_dequant_batch, encode_dequant_batch_f16in,
    encode_dequant_batch_pair_f16in, encode_dequant_matvec, encode_dequant_matvec_with_config,
    ensure_precomputed_linear_f16, ensure_precomputed_linear_f16_many,
    ensure_precomputed_lm_head_f16, gpu_decode_quant_supported,
    gpu_prefill_q5k_small_n_auto_eligible, gpu_prefill_uses_q5k,
    write_normalized_single_logits_with_breakdown,
};
use crate::model::weights::WeightStore;

/// Timing helper.
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

/// Qwen3-family forward pass implementation.
///
/// Key difference from LLaMA: adds QKV bias after projections.
#[derive(Debug)]
pub struct Qwen3Forward;

fn unsupported_qwen3_layout_reason(
    architecture: &str,
    has_fused_qkv: bool,
    has_ssm: bool,
) -> Option<&'static str> {
    if architecture == "qwen35" || has_fused_qkv || has_ssm {
        Some(
            "unsupported qwen35 hybrid layout: found fused attention/SSM tensors \
             (attn_qkv/ssm_*). This model cannot run through Qwen3Forward and needs \
             a dedicated qwen35 implementation.",
        )
    } else {
        None
    }
}

pub(crate) fn ensure_supported_qwen3_layout(
    weights: &WeightStore,
    cfg: &ModelConfig,
) -> anyhow::Result<()> {
    let has_fused_qkv = weights.has("blk.0.attn_qkv.weight");
    let has_ssm = weights.has("blk.0.ssm_a")
        || weights.has("blk.0.ssm_alpha.weight")
        || weights.has("blk.0.ssm_conv1d.weight");

    if let Some(reason) = unsupported_qwen3_layout_reason(&cfg.architecture, has_fused_qkv, has_ssm)
    {
        anyhow::bail!(reason);
    }

    Ok(())
}

include!("qwen3/gpu.rs");
include!("qwen3/core.rs");
include!("qwen3/forward.rs");
#[cfg(test)]
mod tests;
