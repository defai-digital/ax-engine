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

use crate::compute::{attention, rms_norm, rope, silu};
use crate::kv::ModelKv;
use crate::metrics::OpBreakdown;
use crate::metrics::counters::OpTimer;
use crate::model::config::ModelConfig;
use crate::model::forward::{ForwardContext, ForwardPass};
use crate::model::shared::{
    apply_attention_norm_single, apply_optional_attention_qk_norm_single, apply_output_norm_single,
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
}

include!("forward.rs");

#[cfg(test)]
mod tests;
