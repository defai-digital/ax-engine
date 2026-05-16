use mlx_sys::MlxArray;

use super::super::ModelConfig;
use super::super::turboquant_context::TurboQuantModelDecodeContext;
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

/// Full layer forward for Mixtral.
///
/// Architecture: Mistral backbone (uniform SWA via `cfg.global_sliding_window`)
/// with sparse MoE FFN (top-K routing, no shared experts). The Qwen3-style
/// softmax router in `shared/mlp.rs` is reused — Mixtral uses the same
/// `FfnGateInp + FfnGateExps` weight layout with `moe_norm_topk_prob = false`.
///
/// Delegates to `standard::layer_forward`; the MoE path is activated
/// automatically when `w.router_proj.is_some()` and `!cfg.gemma4_moe_router`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    shared_mask: Option<&Option<MlxArray>>,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    super::standard::layer_forward(
        cfg,
        w,
        hidden,
        cache,
        layer_idx,
        token_offset,
        None, // no per-layer inputs
        shared_mask,
        turboquant_context,
    )
}
