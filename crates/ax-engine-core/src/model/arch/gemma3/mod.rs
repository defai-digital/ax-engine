//! Gemma3-family transformer forward pass.
//!
//! Key differences from LLaMA:
//!   1. Q/K projection sizes use n_heads * head_dim (head_dim may differ from embedding_dim / n_heads)
//!   2. Sliding window attention for most layers (every Nth layer is global)
//!   3. GELU activation in FFN instead of SiLU
//!   4. Weight tying: LM head reuses token_embd.weight when output.weight is absent
//!   5. Per-head RMSNorm on Q and K vectors after projection, before RoPE (qk_norm)
//!   6. Post-attention and post-FFN RMSNorm (before residual add)
//!   7. Different RoPE freq base for local (sliding window) vs global layers
//!   8. Embedding scaling by sqrt(embedding_dim)
//!
//! # v2 API changes
//! - `kv_cache: &mut KvCache` → `kv: &mut ModelKv`
//! - GPU decode path uses `kv.as_gpu_mut()` directly, no Mutex locking
//! - `forward_batch` GPU gate uses `ctx.backend.use_gpu_decode()` instead of AX_CPU_ONLY
//! - `gpu_kv.advance()` → `gpu_kv.finalize_token()`
//! - `gpu_kv.advance_by(n)` → `gpu_kv.finalize_batch(n)` (no CPU mirror sync)

use crate::backend::metal::MetalOps;
use crate::compute::attention;
use crate::compute::rms_norm;
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
    PrefillMode, PrefillProjectionInputPlan, PrefillResidualHandoffPlan,
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

/// Gemma3-family forward pass implementation.
#[derive(Debug)]
pub struct Gemma3Forward;

include!("gpu.rs");
include!("core.rs");
include!("forward.rs");
#[cfg(test)]
mod tests;
