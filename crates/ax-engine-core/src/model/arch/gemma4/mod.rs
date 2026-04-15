//! Gemma4-family transformer forward pass.
//!
//! Key differences from Gemma3:
//!   1. Per-layer variable head_dim: SWA layers use head_dim_swa (256),
//!      global layers use head_dim_global (512)
//!   2. Per-layer variable n_kv_heads: SWA layers have more KV heads (16),
//!      global layers have fewer (4)
//!   3. V=K on global layers: no attn_v.weight, V is copied from K then RMS-normed
//!   4. V normalization: raw RMSNorm on V (no learned weight)
//!   5. Attention scale = 1.0 (QK norms handle scaling)
//!   6. Per-layer output scale: layer_output_scale.weight scalar multiplier
//!   7. Final logit softcapping: tanh(logits/cap)*cap
//!   8. Proportional RoPE on global layers: only ~25% of dims rotated
//!   9. Norm weight shift = 0.0 (Gemma3 adds +1.0 in GGUF)

use crate::backend::Backend;
use crate::backend::metal::MetalOps;
use crate::compute::{attention, rms_norm, rope, silu};
use crate::gguf::tensor::GgmlType;
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::execution_plan::{
    DecodeExecutionPlan, PrefillFfnActivationPlan, PrefillLogitsPlan, PrefillProjectionInputPlan,
};
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    apply_attention_norm_single, apply_optional_attention_qk_norm_single, apply_output_norm_single,
    encode_batch_logits, encode_dequant_batch, encode_dequant_batch_f16in,
    encode_dequant_batch_pair_f16in, encode_dequant_matvec, gpu_decode_quant_supported,
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

/// Gemma4-family forward pass implementation.
#[derive(Debug)]
pub struct Gemma4Forward;

impl Gemma4Forward {
    pub(crate) fn has_any_moe_layer(config: &ModelConfig, weights: &WeightStore) -> bool {
        if config.n_expert.unwrap_or(0) == 0 {
            return false;
        }

        (0..config.n_layers as usize)
            .any(|layer| weights.has(&format!("blk.{layer}.ffn_gate_inp.weight")))
    }

    /// Returns true when EVERY MoE layer has Q8_0 for BOTH `ffn_gate_up_exps`
    /// and `ffn_down_exps`. Empirically, that exact combination GPU-hangs in
    /// `moe_mul_mat_id_q8_0` on this Gemma-4 dispatch shape (pre-existing in
    /// HEAD 43541e0). Q6_K schemes promote `ffn_down_exps` to Q8_0 but keep
    /// `ffn_gate_up_exps` at a lower bit width and run fine on the GPU. The
    /// Q8_0 scheme has both at Q8_0 and triggers the hang.
    pub(crate) fn all_layers_q8_0_expert_down(
        config: &ModelConfig,
        weights: &WeightStore,
    ) -> bool {
        use crate::gguf::GgmlType;
        if config.n_expert.unwrap_or(0) == 0 {
            return false;
        }
        let mut saw_any = false;
        for layer in 0..config.n_layers as usize {
            let down = weights
                .raw_with_dtype(&format!("blk.{layer}.ffn_down_exps.weight"))
                .ok();
            let gate_up = weights
                .raw_with_dtype(&format!("blk.{layer}.ffn_gate_up_exps.weight"))
                .ok();
            match (down, gate_up) {
                (Some((_, d)), Some((_, g))) => {
                    if d != GgmlType::Q8_0 || g != GgmlType::Q8_0 {
                        return false;
                    }
                    saw_any = true;
                }
                _ => {}
            }
        }
        saw_any
    }

    /// Determine whether a layer uses sliding window attention.
    /// Same pattern as Gemma3: every Nth layer (default 6) is global.
    pub(crate) fn use_sliding_window(layer: usize, config: &ModelConfig) -> bool {
        match (config.sliding_window_size, config.sliding_window_pattern) {
            (Some(_size), Some(pattern)) if pattern > 0 => {
                layer % (pattern as usize) != (pattern as usize - 1)
            }
            _ => false,
        }
    }

    /// Get the per-layer attention dimensions for a Gemma4 model.
    /// Returns (head_dim, n_kv_heads, q_dim, kv_dim) for the given layer.
    pub(crate) fn layer_dims(layer: usize, config: &ModelConfig) -> (usize, usize, usize, usize) {
        let n_heads = config.n_heads as usize;
        let is_swa = Self::use_sliding_window(layer, config);

        if is_swa {
            let hd = config.gemma4_head_dim_swa.unwrap_or(config.head_dim) as usize;
            let nkv = config.gemma4_n_kv_heads_swa.unwrap_or(config.n_kv_heads) as usize;
            (hd, nkv, n_heads * hd, nkv * hd)
        } else {
            let hd = config.gemma4_head_dim_global.unwrap_or(config.head_dim) as usize;
            let nkv = config.gemma4_n_kv_heads_global.unwrap_or(config.n_kv_heads) as usize;
            (hd, nkv, n_heads * hd, nkv * hd)
        }
    }

    /// Whether the given layer has a V=K configuration (no separate V projection).
    pub(crate) fn layer_v_equals_k(
        layer: usize,
        config: &ModelConfig,
        weights: &WeightStore,
    ) -> bool {
        !Self::use_sliding_window(layer, config)
            && !weights.has(&format!("blk.{layer}.attn_v.weight"))
    }

    pub(crate) fn gpu_prefill_chunk_len(config: &ModelConfig, n_tokens: usize) -> Option<usize> {
        match config.sliding_window_size {
            Some(window) if n_tokens > window as usize => Some(window as usize),
            _ => None,
        }
    }
}

include!("core.rs");
include!("forward.rs");

#[cfg(test)]
mod tests;
