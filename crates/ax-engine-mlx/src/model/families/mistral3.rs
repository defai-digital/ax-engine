use mlx_sys::MlxArray;

use super::super::ModelConfig;
use super::super::turboquant_context::TurboQuantModelDecodeContext;
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

/// Full layer forward for Mistral 3.
///
/// Architecture: standard GQA with uniform sliding-window attention on every
/// layer, SwiGLU dense FFN, no QK norm, no MoE. The sliding window is read
/// from `cfg.global_sliding_window` (populated from `sliding_window_size` in
/// the manifest) and applied uniformly via `layer_params`.
///
/// Delegates directly to `standard::layer_forward`; the uniform SWA is handled
/// transparently by `layer_params` falling back to `cfg.global_sliding_window`
/// when `cfg.layer_configs` is empty.
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
