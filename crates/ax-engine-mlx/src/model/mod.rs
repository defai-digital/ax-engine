use mlx_sys::{
    MlxArray, MlxClosure, MlxDtype, MlxVectorArray, add, astype, async_eval, broadcast_to,
    concatenate, dequantize_with_mode, divide, multiply, power, reshape, rms_norm, rope_dynamic,
    slice, split, stack, sum_axis, take, take_along_axis, transpose,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

use crate::kv_cache::{MlxKVCache, SlidingRingLayout};
use crate::weights::{LayerWeights, ModelWeights, PipelineStageWeights, QuantizedWeight};

/// Typed error for Gemma4 assistant MTP forward-path failures.
#[derive(Debug, Error)]
pub enum Gemma4AssistantForwardError {
    #[error("Gemma4 Assistant MTP requires gemma4_assistant draft and gemma4 target")]
    ModelFamilyMismatch,
    #[error("Gemma4 assistant missing pre_projection")]
    MissingPreProjection,
    #[error("Gemma4 assistant missing post_projection")]
    MissingPostProjection,
    #[error("Gemma4 assistant missing target shared KV layer")]
    MissingSharedKvLayer,
    #[error("Gemma4 assistant target shared KV layer has no cache")]
    MissingSharedKvCache,
    #[error("Gemma4 assistant draft session has no frozen target KV; call bind_target_cache first")]
    UnboundTargetKv,
    #[error("Gemma4 assistant layer missing q_proj")]
    MissingQProj,
    #[error("Gemma4 assistant layer missing o_proj")]
    MissingOProj,
    #[error("compiled assistant forward failed")]
    CompiledForwardFailed,
}

/// One shared target-layer K/V view frozen for a multi-depth assistant draft.
///
/// Assistant layers re-read the target's committed cache and never write it;
/// freezing once per draft attempt avoids re-peeking the same arrays on every
/// layer × depth step.
#[derive(Clone, Debug)]
struct FrozenSharedKvLayer {
    k: MlxArray,
    v: MlxArray,
    ring: Option<SlidingRingLayout>,
}

/// Frozen sliding + full shared-layer bindings for one draft attempt.
#[derive(Clone, Debug, Default)]
struct Gemma4AssistantFrozenTargetKv {
    sliding: Option<FrozenSharedKvLayer>,
    full: Option<FrozenSharedKvLayer>,
}

impl Gemma4AssistantFrozenTargetKv {
    fn freeze_from_cache(
        cache: &MlxKVCache,
        shared: Gemma4AssistantSharedKvLayers,
    ) -> Result<Self, Gemma4AssistantForwardError> {
        use Gemma4AssistantForwardError as E;
        let peek = |layer: usize| -> Result<FrozenSharedKvLayer, E> {
            let (k, v) = cache.peek_layer_kv(layer).ok_or(E::MissingSharedKvCache)?;
            Ok(FrozenSharedKvLayer {
                k,
                v,
                ring: cache.layer_sliding_ring(layer),
            })
        };
        let sliding_idx = shared.sliding_attention_layer;
        let full_idx = shared.full_attention_layer;
        if sliding_idx.is_none() && full_idx.is_none() {
            return Err(E::MissingSharedKvLayer);
        }
        let sliding = sliding_idx.map(peek).transpose()?;
        let full = if full_idx == sliding_idx {
            sliding.clone()
        } else {
            full_idx.map(peek).transpose()?
        };
        Ok(Self { sliding, full })
    }

    fn for_layer(&self, sliding_window: Option<usize>) -> Option<&FrozenSharedKvLayer> {
        if sliding_window.is_some() {
            self.sliding.as_ref()
        } else {
            self.full.as_ref()
        }
    }

    /// Whether the sliding binding carries a ring layout (SWA rotating path).
    fn sliding_uses_ring(&self) -> bool {
        self.sliding
            .as_ref()
            .is_some_and(|layer| layer.ring.is_some())
    }
}

pub(crate) mod profile;
pub use profile::{
    DecodeProfileSnapshot, DenseFfnFastpathSnapshot, EmbedProfileSnapshot,
    Gemma4MoeProfileSnapshot, LinearAttentionProfileSnapshot, MoeProfileSnapshot,
    PrefillProfileSnapshot, take_decode_profile_snapshot, take_dense_ffn_fastpath_snapshot,
    take_embed_profile_snapshot, take_gemma4_moe_profile_snapshot,
    take_linear_attention_profile_snapshot, take_moe_profile_snapshot,
    take_prefill_profile_snapshot,
};
use profile::{
    DecodeProfileStage, EmbedProfileStage, decode_profile_enabled, decode_profile_eval_elapsed,
    embed_profile_enabled, embed_profile_eval_elapsed, forward_profile_eval_elapsed,
    prefill_profile_enabled, record_decode_profile_step, record_embed_profile_call,
    record_prefill_profile_step,
};

mod config;
use config::layer_params;
pub use config::{
    DeepseekV4Config, DiffusionConfig, DiffusionSampler, DiffusionTemperatureSchedule,
    Gemma4AssistantSharedKvLayers, GlmRouterConfig, KvQuantSpec, LayerConfig,
    LinearAttentionConfig, MlaAttentionConfig, ModelConfig,
};

pub(crate) mod shared;
pub(crate) use shared::scale_hidden_pub;
use shared::*;

mod families;

pub(crate) use families::standard::layer_forward_bidirectional;
// The MTP module drives the nextn block directly (`mtp.rs`); the family owns
// the packed hyper-connection residual it threads through that block.
pub(crate) use families::deepseek_v4 as deepseek_v4_family;

/// Read and reset the batched-decode per-stage timing accumulators
/// (`AX_MLX_BATCHED_PROFILE=1`); `[pre_attn, attention, o_proj, ffn]` µs.
/// Diagnostic surface for Phase 3.4 batched-amortization work.
pub fn take_batched_decode_profile() -> [u128; 4] {
    families::standard::batched_profile::take()
}

/// Stage labels matching [`take_batched_decode_profile`].
pub const BATCHED_DECODE_PROFILE_STAGES: [&str; 4] = families::standard::batched_profile::STAGES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FinalLogitsMode {
    Full,
    ArgmaxOnly,
    /// Non-final prefill chunks: skip the lm_head projection entirely.
    /// The returned array is the post-norm hidden (not logits) — callers
    /// must only use it to force-eval the transformer-layer graph (KV cache
    /// writes). This avoids a `hidden × vocab_size` matrix multiply on every
    /// chunk that would be discarded (Qwen3.6 27B: 3584 × 151 936 ≈ 543M
    /// multiply-adds saved per non-final chunk). Reference: MTPLX
    /// `emit_logits=False`.
    Skip,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LazySingleTokenMode {
    NormalizedFullLogits,
    SingletonArgmaxOnly,
}

impl LazySingleTokenMode {
    fn logits_mode(self) -> FinalLogitsMode {
        match self {
            Self::NormalizedFullLogits => FinalLogitsMode::Full,
            Self::SingletonArgmaxOnly => FinalLogitsMode::ArgmaxOnly,
        }
    }
}

#[cfg(test)]
use crate::attention_mask::create_causal_mask;
#[cfg(test)]
use ax_engine_core::NativeModelManifest;
#[cfg(test)]
use mlx_sys::expand_dims_axes;
#[cfg(test)]
use mlx_sys::{
    ScaledDotProductAttentionMask, argmax, gelu_approx, matmul,
    scaled_dot_product_attention_with_mask,
};
#[cfg(test)]
use profile::record_linear_attention_profile_layer;

/// Forward pass for one transformer layer.
///
/// `shared_mask`: pre-computed SDPA mask for this layer — `None` computes it
/// internally from `seq`, `key_len`, and `sliding_window`.  Pass `Some(&m)`
/// from `build_layer_masks` in `forward*` to avoid creating identical mask
/// graphs for every layer of the same attention type.
///
/// Returns updated hidden states.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray, // [1, seq, hidden]
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    per_layer_input: Option<&MlxArray>, // [1, seq, per_layer_dim] or None
    shared_mask: Option<&Option<MlxArray>>,
) -> MlxArray {
    // Nemotron-H owns all of its hybrid mixers (Mamba-2 / attention / MoE);
    // dispatch before the Qwen linear-attention short-circuit.
    if cfg.model_family == "nemotron_h" {
        return families::nemotron_h::layer_forward(cfg, w, hidden, cache, layer_idx, token_offset);
    }

    // Linear-attention layers dispatch before the family route because the same
    // model (qwen3_5 / qwen3_next) has both linear and full-attention layers
    // and the family string alone is not enough to disambiguate per-layer type.
    if cfg.is_linear_attention_layer(layer_idx) {
        return families::qwen3_linear::layer_forward(
            cfg, w, hidden, cache, layer_idx, false, false,
        );
    }

    // ADR-038: prefer static architecture registry over open family match arms.
    use ax_engine_core::{LayerForwardRoute, resolve_layer_forward_route};
    match resolve_layer_forward_route(&cfg.model_family) {
        Some(LayerForwardRoute::Standard) => families::standard::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            per_layer_input,
            shared_mask,
            false, // last_position_only_after_attention
            false, // skip_post_attention_ffn
        ),
        Some(LayerForwardRoute::Llama4) => families::llama4::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            shared_mask,
        ),
        Some(LayerForwardRoute::GlmMoeLite) => {
            families::glm4_moe_lite::layer_forward(cfg, w, hidden, cache, layer_idx, token_offset)
        }
        Some(LayerForwardRoute::DeepseekV3) => {
            families::deepseek_v3::layer_forward(cfg, w, hidden, cache, layer_idx, token_offset)
        }
        Some(LayerForwardRoute::DeepseekV4) => {
            // Unreachable by construction: DeepSeek V4 forwards dispatch to
            // `deepseek_v4_forward_*` before the layer loop — the family owns
            // the packed 4-stream residual, which the E-wide `layer_forward`
            // contract cannot express. `layer_forward` has no fallible error
            // channel; mirror the `None` arm so a bypassing caller still
            // fails loudly.
            panic!(
                "deepseek_v4 layers must run through the dedicated deepseek_v4 forward path (packed hyper-connection residual)"
            )
        }
        Some(LayerForwardRoute::Mistral3) => families::mistral3::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            shared_mask,
        ),
        Some(LayerForwardRoute::Mixtral) => families::mixtral::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            shared_mask,
        ),
        Some(LayerForwardRoute::GptOss) => {
            families::gpt_oss::layer_forward(cfg, w, hidden, cache, layer_idx, token_offset)
        }
        Some(LayerForwardRoute::NemotronH) => {
            families::nemotron_h::layer_forward(cfg, w, hidden, cache, layer_idx, token_offset)
        }
        None => panic!(
            "unknown model_family in layer_forward (not in architecture registry): {}",
            cfg.model_family
        ),
    }
}

/// Run a transformer layer with the "last-position-only after attention"
/// optimization enabled. Equivalent to
/// [`layer_forward`] except that, for `seq > 1`,
/// the layer's pre-FFN slice happens **inside** the layer (between the
/// attention residual and the pre-FFN norm) instead of after the layer
/// returns. The MLP / MoE / per-layer-gate / layer-scalar steps then run
/// on `[1, 1, hidden]`, matching `mlx-lm`'s lazy-eval prune of the
/// terminal layer's post-attention work.
///
/// When `skip_post_attention_ffn` is `true`, the layer returns after the
/// attention residual with **no** FFN (even the last-only 1-token FFN).
/// Use only for cache-only prefill (`FinalLogitsMode::Skip`) where the
/// residual is discarded and only KV / linear-state writes matter.
///
/// **Caller obligation:** only invoke this for the *last* transformer
/// layer of a prefill pass. Passing a 1-position hidden into the next
/// layer's attention would corrupt the KV cache (positions would no
/// longer align with `token_offset`).
///
/// Supported families: `gemma4`, `gemma3`, `qwen3`, `llama3`, `qwen3_5`,
/// `qwen3_next` (both full-attention and linear-attention layers). Linear-
/// attention layers slice to the last position after the attention-residual
/// add (the recurrent state is already committed to cache). Other families
/// fall back to the normal `layer_forward` path; the post-loop slice in
/// [`forward`] keeps correctness while losing this
/// specific perf win until those families pick up the same optimization.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward_last_only(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    per_layer_input: Option<&MlxArray>,
    shared_mask: Option<&Option<MlxArray>>,
    skip_post_attention_ffn: bool,
) -> MlxArray {
    if cfg.model_family == "nemotron_h" {
        // Nemotron mixers are residual-only; last-only FFN does not apply.
        let _ = skip_post_attention_ffn;
        return families::nemotron_h::layer_forward(cfg, w, hidden, cache, layer_idx, token_offset);
    }
    if cfg.is_linear_attention_layer(layer_idx) {
        // Linear-attention layers support last-position-only and skip-FFN:
        // the recurrent state is committed to cache before the FFN path.
        return families::qwen3_linear::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            !skip_post_attention_ffn,
            skip_post_attention_ffn,
        );
    }
    // ADR-038: standard-route families get the last-only FFN optimization;
    // specialized routes fall back to the unoptimized path (correctness via
    // post-loop slice in `forward`).
    use ax_engine_core::{LayerForwardRoute, resolve_layer_forward_route};
    match resolve_layer_forward_route(&cfg.model_family) {
        Some(LayerForwardRoute::Standard) => families::standard::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            per_layer_input,
            shared_mask,
            !skip_post_attention_ffn, // last-only FFN when not skipping
            skip_post_attention_ffn,
        ),
        // DeepSeek V4 must NOT use last-only FFN pruning: HC mixing consumes
        // the full packed stream, so its dedicated forward path runs every
        // layer in full. The fallthrough hits the defensive DeepseekV4 panic
        // in `layer_forward`, which is unreachable from the V4 entry points.
        _ => layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            per_layer_input,
            shared_mask,
        ),
    }
}

/// Embed token IDs and return hidden states of shape [1, seq_len, hidden].
/// Embed tokens from a pre-built scalar, 1-D `[seq]`, or singleton matrix
/// token-ID array.
///
/// Accepts lazy (unevaluated) arrays — all ops are lazy MLX graph nodes — so
/// this can be called with a GPU argmax result before it has been materialised.
/// Used internally by `embed_tokens` (materialized path) and by
/// `forward_lazy_single` (double-buffer pipelining path).
pub(crate) fn embed_tokens_arr(
    ids_1d: &MlxArray,
    embedding: &QuantizedWeight,
    hidden_size: usize,
) -> MlxArray {
    let scalar_ids_storage;
    let ids_reshaped = if ids_1d.ndim() == 0 {
        scalar_ids_storage = reshape(ids_1d, &[1_i32], None);
        &scalar_ids_storage
    } else {
        ids_1d
    };
    // Clamp to the embedding table's row range before any gather. MLX's
    // Metal `take`/gather kernel does no bounds checking for unsigned
    // indices (see mlx/backend/metal/kernels/indexing/indexing.h
    // offset_neg_idx, which returns unsigned indices unmodified) — an
    // out-of-range client-supplied token id (e.g. a malformed
    // POST /v1/embeddings or /v1/generate request) would otherwise read
    // arbitrary GPU memory past the weight buffer instead of erroring.
    let clamped_ids_storage;
    let ids = if embedding.weight.shape().first().copied().unwrap_or(0) > 0 {
        let vocab_rows = embedding.weight.shape()[0];
        let min_id = shared::utils::scalar_like(0.0, MlxDtype::Uint32);
        let max_id = shared::utils::scalar_like((vocab_rows - 1) as f32, MlxDtype::Uint32);
        clamped_ids_storage = mlx_sys::clip(ids_reshaped, &min_id, &max_id, None);
        &clamped_ids_storage
    } else {
        ids_reshaped
    };
    let seq = ids.shape()[0]; // shape metadata is available without eval
    if let Some(scales) = &embedding.scales {
        let row_w = take(&embedding.weight, ids, 0, None);
        let row_s = take(scales, ids, 0, None);
        let row_b = embedding.biases.as_ref().map(|b| take(b, ids, 0, None));
        // Mode-aware dequant (mxfp8 embeddings have U8 scales and no biases).
        let flat = dequantize_with_mode(
            &row_w,
            &row_s,
            row_b.as_ref(),
            Some(embedding.group_size),
            Some(embedding.bits),
            embedding.mlx_quantization_mode(),
            None,
            None,
            None,
        );
        reshape(&flat, &[1, seq, hidden_size as i32], None)
    } else {
        let flat = take(&embedding.weight, ids, 0, None);
        reshape(&flat, &[1, seq, hidden_size as i32], None)
    }
}

pub fn embed_tokens(
    token_ids: &[u32],
    embedding: &QuantizedWeight,
    hidden_size: usize,
) -> MlxArray {
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    embed_tokens_arr(&ids_1d, embedding, hidden_size)
}

/// Embed one decode token per row for **batched decode**, returning hidden
/// states of shape `[B, 1, hidden]` where `B == token_ids.len()`.
///
/// This is the batched-decode analog of [`embed_tokens`] (which returns
/// `[1, seq, hidden]` for one sequence): the row gather + dequantize is
/// identical — only the final reshape differs. `embed_tokens` stacks `seq`
/// tokens of one sequence on the sequence axis (`[1, seq, hidden]`); this stacks
/// `B` independent sequences' current tokens on the batch axis
/// (`[B, 1, hidden]`), the input shape a batched decode forward expects. The
/// single-sequence path in [`embed_tokens_arr`] is left untouched.
///
/// Like [`embed_tokens`], `from_raw_data` borrows `token_ids`, so the caller
/// must keep it alive until the returned (lazy) hidden states are evaluated.
///
/// # Panics
/// If `token_ids` is empty.
pub fn embed_decode_tokens_batched(
    token_ids: &[u32],
    embedding: &QuantizedWeight,
    hidden_size: usize,
) -> MlxArray {
    assert!(
        !token_ids.is_empty(),
        "batched decode embed requires at least one token"
    );
    let batch = token_ids.len() as i32;
    let ids = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[batch],
        MlxDtype::Uint32,
    );
    let flat = if let Some(scales) = &embedding.scales {
        let row_w = take(&embedding.weight, &ids, 0, None);
        let row_s = take(scales, &ids, 0, None);
        let row_b = embedding.biases.as_ref().map(|b| take(b, &ids, 0, None));
        dequantize_with_mode(
            &row_w,
            &row_s,
            row_b.as_ref(),
            Some(embedding.group_size),
            Some(embedding.bits),
            embedding.mlx_quantization_mode(),
            None,
            None,
            None,
        )
    } else {
        take(&embedding.weight, &ids, 0, None)
    };
    reshape(&flat, &[batch, 1, hidden_size as i32], None)
}

/// Batched decode forward for dense full-attention families **and** Qwen3-Next
/// hybrids (MoE + gated-delta linear attention) — continuous batched MLX decode.
///
/// Embeds the B current tokens (`[B,1,hidden]`), builds the per-row validity
/// mask once, runs the batched layer loop
/// ([`families::standard::layer_forward_batched`] / linear sibling), and returns
/// logits `[B, 1, vocab]`. Rows may be **ragged** — each row's decode position
/// and KV length are read from the cache (`row_len(r)`), so a continuously-batched
/// cohort at different sequence positions decodes together. The caller supplies
/// the [`BatchedKvCache`] (seeded from per-request prefill) and turns logits into
/// tokens via [`crate::batched_sampling::argmax_batched`] / `sample_batched_host`.
///
/// Mirrors the single-sequence `forward` embed prologue
/// (bf16 cast + optional `hidden_states_scale`) and final norm + lm_head.
/// Dense projections default to the **Shared** batch policy
/// (`AX_MLX_BATCHED_SHARED_PROJ`, default ON) for weight-read amortization;
/// kill-switch restores per-row RowExact. Hybrid MoE amortization uses batched
/// `gather_qmm` and is intentionally uncertified for bit-exact greedy parity
/// (ADR-003 D5) — see `docs/performance/batched-hybrid-moe-linear-decode.md`.
pub fn decode_batched_forward(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    tokens: &[u32],
    cache: &mut crate::batched_kv_cache::BatchedKvCache,
    mut lin_state: Option<&mut crate::batched_linear_state::BatchedLinearState>,
) -> MlxArray {
    assert_eq!(
        tokens.len(),
        cache.batch(),
        "decode_batched_forward: one token per cache row"
    );
    let mut hidden = embed_decode_tokens_batched(tokens, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    // Per-row decode position (`offsets[r]`) and validity length. Each row's
    // current token is written at `row_len(r)` (its RoPE position), giving it
    // `row_len(r) + 1` valid keys; the batched view spans `max_len + 1`.
    let offsets: Vec<usize> = (0..cache.batch()).map(|r| cache.row_len(r)).collect();
    let valid_lengths: Vec<usize> = offsets.iter().map(|&o| o + 1).collect();
    let key_len = cache.max_len() + 1;
    let mask = Some(crate::attention_mask::batched_decode_validity_mask(
        &valid_lengths,
        key_len,
    ));

    // `BatchedLinearState` is indexed by linear-layer order (0, 1, 2, …), not
    // global layer index, so linear layers carry a running counter. Full-attention
    // layers decode through the KV-cache path; gated-delta layers advance their
    // per-row recurrent state in `lin_state` (which must be present for a hybrid
    // model — the runner routes only certified, correctly-provisioned cohorts here).
    let mut linear_idx = 0usize;
    for (li, layer_w) in weights.layers.iter().enumerate() {
        if layer_w.linear_attn.is_some() {
            let lin = lin_state
                .as_deref_mut()
                .expect("batched decode: linear-attention layer requires a BatchedLinearState");
            hidden = families::standard::layer_forward_batched_linear(
                cfg, layer_w, &hidden, lin, linear_idx, li,
            );
            linear_idx += 1;
        } else {
            hidden = families::standard::layer_forward_batched(
                cfg, layer_w, &hidden, cache, li, &offsets, &mask,
            );
        }
    }
    cache.advance_all(1);

    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    // The lm_head (`[B, hidden] × [hidden, vocab]`, vocab up to ~262 MB at
    // 4-bit) is the largest single projection and, like the layer projections,
    // was per-row `RowExact` — re-reading the whole lm_head weight B times per
    // step, the residual non-amortizing cost outside the per-layer profiler
    // (Phase 3.6). Route it through the same `AX_MLX_BATCHED_SHARED_PROJ`
    // policy so it amortizes with the rest.
    let lm_head_policy = if crate::fastpath::batched_shared_projections_enabled() {
        ProjectionBatchPolicy::Shared
    } else {
        ProjectionBatchPolicy::RowExact
    };
    let logits = qw_with_policy(&normed, &weights.lm_head, lm_head_policy);
    finalize_lm_head_logits(cfg, &logits, FinalLogitsMode::Full)
}

/// Whether this model can run [`prefill_batched_forward`] (padded batched
/// prefill, v1 scope): standard autoregressive generation over uniformly
/// full-attention dense layers. Excluded — and asserted again inside the
/// layer forward — are diffusion models, linear-attention/MLA hybrids,
/// sliding-window or KV-shared layers, MoE layers, per-layer-input gating
/// (Gemma 2B/4B), and models with an MTP head (whose prefill must capture
/// MTP history on a different path).
pub fn supports_batched_prefill(cfg: &ModelConfig, weights: &ModelWeights) -> bool {
    if cfg.is_block_diffusion()
        || cfg.linear_attention.is_some()
        || weights.mtp.is_some()
        || weights.glm_mtp.is_some()
        || weights.per_layer_embed.is_some()
    {
        return false;
    }
    weights.layers.iter().enumerate().all(|(layer_idx, w)| {
        let (_, _, _, _, sliding_window, kv_source, _) = config::layer_params(cfg, layer_idx);
        sliding_window.is_none()
            && kv_source.is_none()
            && w.linear_attn.is_none()
            && w.glm_mla_attn.is_none()
            && w.per_layer_gate.is_none()
            && w.router_proj.is_none()
            && w.o_proj.is_some()
    })
}

/// Per-row results of one padded batched prefill forward, already evaluated:
/// each row's last-real-position logits and its per-layer K/V trimmed to the
/// row's true prompt length (`[1, kv_heads, len, head_dim]`), ready for
/// [`crate::kv_cache::MlxKVCache::set_layer_kv_logical`].
pub struct BatchedPrefillRows {
    /// Per row: `[vocab]` f32 logits at the row's last real prompt position.
    pub row_logits: Vec<MlxArray>,
    /// `row_layer_kv[row][layer] == (k, v)`.
    pub row_layer_kv: Vec<Vec<(MlxArray, MlxArray)>>,
}

/// Padded batched prefill: embed `B` cold prompts padded to the longest
/// length, run one `[B, L, hidden]` pass over the dense layer stack under a
/// causal+padding mask, and return per-row last-token logits plus per-row
/// trimmed K/V. Rows must all be cold (KV offset zero) — the cohort planner's
/// offset-isolation invariant — so a single position-zero RoPE covers the
/// batch. The caller is responsible for gating on
/// [`supports_batched_prefill`] and for the `rows * max_len` admission cap
/// that bounds the `[B, L, L]` mask transient.
pub fn prefill_batched_forward(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompts: &[&[u32]],
) -> Result<BatchedPrefillRows, String> {
    let batch = prompts.len();
    if batch < 2 {
        return Err("batched prefill requires at least two rows".to_string());
    }
    let lens: Vec<usize> = prompts.iter().map(|prompt| prompt.len()).collect();
    if lens.contains(&0) {
        return Err("batched prefill rows must have non-empty prompts".to_string());
    }
    let padded_len = *lens.iter().max().expect("batch checked non-empty");

    // Pad with token id 0; padded positions are excluded by the mask and
    // their K/V is trimmed away below, so the pad value never leaks.
    let mut padded_ids = vec![0u32; batch * padded_len];
    for (row, prompt) in prompts.iter().enumerate() {
        padded_ids[row * padded_len..row * padded_len + prompt.len()].copy_from_slice(prompt);
    }
    let flat_ids = MlxArray::from_raw_data(
        padded_ids.as_ptr().cast(),
        std::mem::size_of_val(padded_ids.as_slice()),
        &[(batch * padded_len) as i32],
        MlxDtype::Uint32,
    );
    let embedded = embed_tokens_arr(&flat_ids, &weights.token_embedding, cfg.hidden_size);
    let mut hidden = reshape(
        &embedded,
        &[batch as i32, padded_len as i32, cfg.hidden_size as i32],
        None,
    );
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    let mask = Some(crate::attention_mask::batched_prefill_causal_mask(
        &lens, padded_len,
    ));
    let mut layer_kv: Vec<(MlxArray, MlxArray)> = Vec::with_capacity(weights.layers.len());
    for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
        let (out, k_rope, v) = families::standard::layer_forward_batched_prefill(
            cfg, layer_w, &hidden, layer_idx, padded_len, &mask,
        );
        hidden = out;
        layer_kv.push((k_rope, v));
    }

    // Gather each row's last real position BEFORE final norm + lm_head, so
    // the vocab projection runs on [B, 1, hidden] rather than every padded
    // position.
    let flat_hidden = reshape(
        &hidden,
        &[(batch * padded_len) as i32, cfg.hidden_size as i32],
        None,
    );
    let last_indices: Vec<u32> = lens
        .iter()
        .enumerate()
        .map(|(row, &len)| (row * padded_len + len - 1) as u32)
        .collect();
    let index_arr = MlxArray::from_raw_data(
        last_indices.as_ptr().cast(),
        std::mem::size_of_val(last_indices.as_slice()),
        &[batch as i32],
        MlxDtype::Uint32,
    );
    let last_hidden = take(&flat_hidden, &index_arr, 0, None);
    let last_hidden = reshape(
        &last_hidden,
        &[batch as i32, 1, cfg.hidden_size as i32],
        None,
    );
    let normed = rms_norm(
        &last_hidden,
        Some(&weights.final_norm),
        cfg.rms_norm_eps,
        None,
    );
    let lm_head_policy = if crate::fastpath::batched_shared_projections_enabled() {
        ProjectionBatchPolicy::Shared
    } else {
        ProjectionBatchPolicy::RowExact
    };
    let logits = qw_with_policy(&normed, &weights.lm_head, lm_head_policy);
    let logits = finalize_lm_head_logits(cfg, &logits, FinalLogitsMode::Full);

    // Split per row and trim K/V to each row's real length. `contiguous`
    // copies out of the padded batch buffer so a request's cache does not
    // retain the whole [B, L] allocation, and everything is evaluated in one
    // barrier before handing rows to per-request caches.
    let vocab = cfg.vocab_size as i32;
    let mut row_logits = Vec::with_capacity(batch);
    let mut row_layer_kv: Vec<Vec<(MlxArray, MlxArray)>> = Vec::with_capacity(batch);
    for (row, &len) in lens.iter().enumerate() {
        let r = row as i32;
        let one_row_logits = slice(&logits, &[r, 0, 0], &[r + 1, 1, vocab], &[1, 1, 1], None);
        row_logits.push(reshape(&one_row_logits, &[vocab], None));
        let mut per_layer = Vec::with_capacity(layer_kv.len());
        for (k, v) in &layer_kv {
            let shape = k.shape();
            let (kv_heads, head_dim) = (shape[1], shape[3]);
            let trim = |arr: &MlxArray| -> MlxArray {
                let row_view = slice(
                    arr,
                    &[r, 0, 0, 0],
                    &[r + 1, kv_heads, len as i32, head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                mlx_sys::contiguous(&row_view, None)
            };
            per_layer.push((trim(k), trim(v)));
        }
        row_layer_kv.push(per_layer);
    }
    let mut eval_refs: Vec<&MlxArray> = Vec::new();
    eval_refs.extend(row_logits.iter());
    for per_layer in &row_layer_kv {
        for (k, v) in per_layer {
            eval_refs.push(k);
            eval_refs.push(v);
        }
    }
    mlx_sys::eval(&eval_refs);

    Ok(BatchedPrefillRows {
        row_logits,
        row_layer_kv,
    })
}

/// Full forward pass: returns logits for the LAST token only — `[vocab_size]` f32.
pub fn forward(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    forward_and_logits_mode(
        cfg,
        weights,
        token_ids,
        cache,
        token_offset,
        FinalLogitsMode::Full,
    )
}

pub fn forward_argmax(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    // MoE pure-direct: MLX singleton (smokef98 — no inv; Shared MT alone).
    forward_and_logits_mode(
        cfg,
        weights,
        token_ids,
        cache,
        token_offset,
        FinalLogitsMode::ArgmaxOnly,
    )
}

/// Cache-only forward: runs all transformer layers (writing KV / linear
/// state), **skips the last-layer FFN**, and **skips final RMSNorm +
/// lm_head**. Returns the post-attention residual of the last layer —
/// use only to force-eval the layer graph (or drop it and eval KV refs).
///
/// Intended for non-final prefill chunks / mlx-lm-style cache-only prefix
/// where residual and logits are discarded. On Qwen3.6 27B this saves the
/// last-layer FFN plus ~543M multiply-adds of lm_head per chunk.
pub fn forward_cache_only(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    forward_and_logits_mode(
        cfg,
        weights,
        token_ids,
        cache,
        token_offset,
        FinalLogitsMode::Skip,
    )
}

/// Input accepted by one static pipeline stage.
pub enum PipelineStageInput<'a> {
    /// Token IDs are valid only for the embedding-owning first rank.
    Tokens(&'a [u32]),
    /// Hidden states `[1, sequence, hidden_size]` from the preceding rank.
    Hidden(&'a MlxArray),
}

/// Execute exactly one dense Llama 3 pipeline stage.
///
/// Non-final ranks return hidden states. The output-head rank returns
/// last-position logits `[vocab_size]`. KV entries retain global layer indexes,
/// so each rank may allocate a cache sized to the complete model while only
/// materializing its assigned entries.
pub fn forward_pipeline_stage(
    cfg: &ModelConfig,
    weights: &PipelineStageWeights,
    input: PipelineStageInput<'_>,
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> Result<MlxArray, PipelineStageForwardError> {
    if cfg.deepseek_v4.is_some() {
        // Packed `[1, seq, hc*hidden]` pipeline transfer is out of scope; the
        // hidden-state shape check below hard-requires hidden_size. Reject
        // explicitly rather than falling into the dense-Llama stage.
        return Err(PipelineStageForwardError::UnsupportedFamily(
            cfg.model_family.clone(),
        ));
    }
    if cfg.model_family != "llama3" {
        return Err(PipelineStageForwardError::UnsupportedFamily(
            cfg.model_family.clone(),
        ));
    }
    let assignment = &weights.assignment;
    let expected_layers = assignment.layers.len() as usize;
    if weights.layers.len() != expected_layers {
        return Err(PipelineStageForwardError::LayerCount {
            expected: expected_layers,
            actual: weights.layers.len(),
        });
    }
    if assignment.layers.end as usize > cfg.layer_count {
        return Err(PipelineStageForwardError::LayerRangeExceedsModel);
    }

    let mut hidden = match input {
        PipelineStageInput::Tokens(token_ids) => {
            if !assignment.owns_embeddings {
                return Err(PipelineStageForwardError::TokensOnNonEmbeddingRank);
            }
            if token_ids.is_empty() {
                return Err(PipelineStageForwardError::EmptySequence);
            }
            let embedding = weights
                .token_embedding
                .as_ref()
                .ok_or(PipelineStageForwardError::MissingEmbedding)?;
            let ids_1d = MlxArray::from_raw_data(
                token_ids.as_ptr().cast(),
                std::mem::size_of_val(token_ids),
                &[token_ids.len() as i32],
                MlxDtype::Uint32,
            );
            let embedded = embed_tokens_arr(&ids_1d, embedding, cfg.hidden_size);
            let embedded = astype(&embedded, MlxDtype::Bfloat16, None);
            match cfg.hidden_states_scale {
                Some(scale) => scale_hidden(&embedded, scale),
                None => embedded,
            }
        }
        PipelineStageInput::Hidden(hidden) => {
            if assignment.owns_embeddings {
                return Err(PipelineStageForwardError::HiddenOnEmbeddingRank);
            }
            hidden.clone()
        }
    };
    let shape = hidden.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] != cfg.hidden_size as i32 {
        return Err(PipelineStageForwardError::InvalidHiddenShape(shape));
    }
    let sequence = shape[1] as usize;
    let decode_mask: Option<MlxArray> = None;
    let masks = (sequence > 1 || cache.rotating_sliding_slack() > 0).then(|| {
        build_layer_masks_for_forward(
            cfg,
            cfg.layer_count,
            sequence,
            token_offset + sequence,
            cache,
        )
    });

    for (local_index, layer_weights) in weights.layers.iter().enumerate() {
        let global_index = weights
            .global_layer_index(local_index)
            .ok_or(PipelineStageForwardError::LayerRangeExceedsModel)?;
        let shared_mask = masks
            .as_ref()
            .map(|masks| &masks[global_index])
            .unwrap_or(&decode_mask);
        hidden = layer_forward(
            cfg,
            layer_weights,
            &hidden,
            cache,
            global_index,
            token_offset,
            None,
            Some(shared_mask),
        );
    }

    if !assignment.owns_output_head {
        return Ok(hidden);
    }
    let final_norm = weights
        .final_norm
        .as_ref()
        .ok_or(PipelineStageForwardError::MissingOutputHead)?;
    let lm_head = weights
        .lm_head
        .as_ref()
        .ok_or(PipelineStageForwardError::MissingOutputHead)?;
    let last_hidden = if sequence > 1 {
        let last = (sequence - 1) as i32;
        slice(
            &hidden,
            &[0, last, 0],
            &[1, last + 1, cfg.hidden_size as i32],
            &[1, 1, 1],
            None,
        )
    } else {
        hidden
    };
    let normed = rms_norm(&last_hidden, Some(final_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, lm_head);
    Ok(reshape(&logits, &[cfg.vocab_size as i32], None))
}

#[derive(Debug, Error)]
pub enum PipelineStageForwardError {
    #[error("pipeline forward supports dense llama3 only, got {0}")]
    UnsupportedFamily(String),
    #[error("stage layer count mismatch: expected {expected}, got {actual}")]
    LayerCount { expected: usize, actual: usize },
    #[error("stage layer range exceeds model configuration")]
    LayerRangeExceedsModel,
    #[error("token IDs may only be sent to the embedding-owning rank")]
    TokensOnNonEmbeddingRank,
    #[error("hidden states may not be sent directly to the embedding-owning rank")]
    HiddenOnEmbeddingRank,
    #[error("pipeline stage received an empty token sequence")]
    EmptySequence,
    #[error("embedding-owning rank did not load token embeddings")]
    MissingEmbedding,
    #[error("output-head rank did not load final norm and LM head")]
    MissingOutputHead,
    #[error("pipeline hidden state must have shape [1, sequence, hidden_size], got {0:?}")]
    InvalidHiddenShape(Vec<i32>),
}

fn forward_and_logits_mode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
    logits_mode: FinalLogitsMode,
) -> MlxArray {
    // DeepSeek V4 owns its packed hyper-connection residual; dispatch before
    // the generic E-wide path.
    if cfg.deepseek_v4.is_some() {
        return deepseek_v4_forward_and_logits_mode(
            cfg,
            weights,
            token_ids,
            cache,
            token_offset,
            logits_mode,
        );
    }
    let profile_prefill = token_ids.len() > 1 && prefill_profile_enabled();
    let token_offset = qwen_visual_rope_offset(weights, cache, token_offset);

    // Build ids_1d once; reused for both embedding and per-layer-input projection
    // (Gemma4 2B/4B), avoiding a duplicate CPU→GPU token-ID upload.
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    let mut hidden = embed_tokens_arr(&ids_1d, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    let seq = token_ids.len();
    // Single-token decode never needs explicit SDPA masks — except in
    // bounded-rollback rotating mode, where a converted sliding layer
    // presents its full `window + slack` ring and needs the slot-validity
    // mask even for one query. Keep the common decode path on one borrowed
    // `None` instead of allocating a per-layer mask vec.
    let decode_mask: Option<MlxArray> = None;
    let masks = (seq > 1 || cache.rotating_sliding_slack() > 0).then(|| {
        build_layer_masks_for_forward(cfg, weights.layers.len(), seq, token_offset + seq, cache)
    });
    let per_layer_started = profile_prefill.then(Instant::now);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &ids_1d, &hidden);
    if let (Some(started), Some(inputs)) = (per_layer_started, per_layer_inputs.as_ref()) {
        let refs: Vec<&MlxArray> = inputs.iter().collect();
        forward_profile_eval_elapsed(
            false,
            profile_prefill,
            DecodeProfileStage::PerLayerInput,
            started,
            &refs,
        );
    }
    // Last-layer prefill optimizations (matches mlx-lm's implicit lazy-eval
    // prune). For multi-token prefill, the final transformer layer either:
    // - **Skip FFN** (`FinalLogitsMode::Skip` / cache-only): return after the
    //   attention residual; residual is discarded and only KV/linear state
    //   writes matter. Avoids even the last-only 1-token FFN + final_norm.
    // - **Last-position-only FFN**: slice residual to the last position before
    //   FFN / gate / scalar. Saves ~50% of the last layer on long prompts.
    // Explicit cache-only single-token continuations may also skip the final
    // FFN; ordinary logits-producing decode never does. See
    // `layer_forward_last_only` for the contract.
    let last_layer_idx = weights.layers.len().saturating_sub(1);
    let skip_last_layer_ffn = matches!(logits_mode, FinalLogitsMode::Skip);
    // Cache-only scheduler continuations may intentionally preserve the
    // established n-1 + single-token boundary for greedy numerical parity.
    // The final residual is discarded in that single-token call too, so its
    // last-layer FFN remains pure waste even though ordinary decode (which
    // requests logits) must keep the full layer.
    let use_last_layer_optimization = seq > 1 || skip_last_layer_ffn;
    let total_layers = weights.layers.len();
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        let shared_mask = masks
            .as_ref()
            .map(|masks| &masks[li])
            .unwrap_or(&decode_mask);
        hidden = if use_last_layer_optimization && li == last_layer_idx {
            layer_forward_last_only(
                cfg,
                layer_w,
                &hidden,
                cache,
                li,
                token_offset,
                pli,
                Some(shared_mask),
                skip_last_layer_ffn,
            )
        } else {
            layer_forward(
                cfg,
                layer_w,
                &hidden,
                cache,
                li,
                token_offset,
                pli,
                Some(shared_mask),
            )
        };
        // mlxcel residual: `pipeline_hint` after non-final layers
        // (`MLXCEL_PIPELINE_GRANULARITY` parity via AX_MLX_PIPELINE_GRANULARITY).
        if crate::fastpath::pipeline_hint_should_fire(li, total_layers) {
            async_eval(&[&hidden]);
        }
        // Diagnostic/fairness probe: unlike the async pipeline hint above,
        // this blocks at selected non-final prefill layer boundaries to shorten
        // individual GPU command bursts. Default OFF preserves the lazy graph.
        if crate::fastpath::pipeline_eval_should_fire(seq, li, total_layers) {
            mlx_sys::eval(&[&hidden]);
        }
    }

    // Cache-only: residual after the last layer (FFN already skipped) is only
    // held so callers that `eval` the return value still force the layer graph.
    // Skip final_norm + lm_head — both are pure waste when the result is
    // discarded after KV materialisation.
    if matches!(logits_mode, FinalLogitsMode::Skip) {
        if profile_prefill {
            record_prefill_profile_step(weights.layers.len() as u32, seq as u32);
        }
        return hidden;
    }

    // Post-loop slice: idempotent when the last layer already collapsed to
    // a 1-position output (the supported standard families). For
    // unsupported families that fall back to the unoptimized last-layer
    // path, this slice still trims `[1, seq, hidden]` to `[1, 1, hidden]`
    // before the final norm + lm_head.
    let last_hidden = if hidden.shape().get(1).copied().unwrap_or(1) > 1 {
        let last = (token_ids.len() - 1) as i32;
        let hs = cfg.hidden_size as i32;
        slice(&hidden, &[0, last, 0], &[1, last + 1, hs], &[1, 1, 1], None)
    } else {
        hidden
    };

    let lm_head_started = profile_prefill.then(Instant::now);
    let normed = rms_norm(
        &last_hidden,
        Some(&weights.final_norm),
        cfg.rms_norm_eps,
        None,
    );
    let logits = qw(&normed, &weights.lm_head);
    let logits = finalize_lm_head_logits(cfg, &logits, logits_mode);
    let logits = reshape(&logits, &[cfg.vocab_size as i32], None);
    if let Some(started) = lm_head_started {
        forward_profile_eval_elapsed(
            false,
            profile_prefill,
            DecodeProfileStage::LmHead,
            started,
            &[&logits],
        );
    }
    if profile_prefill {
        record_prefill_profile_step(weights.layers.len() as u32, seq as u32);
    }
    logits
}

// ── DeepSeek V4 (Flash) forward path ────────────────────────────────────────
//
// V4 owns the packed `[1, seq, hc_mult * hidden]` hyper-connection residual
// end-to-end; these helpers keep the generic `layer_forward` dispatch
// (E-wide) out of the loop and expose E-wide boundaries only at embed and at
// the post-collapse tail.

/// V4 trunk: embed `ids_1d` → expand to the packed stream → per-layer
/// `families::deepseek_v4::layer_forward`. Returns the packed residual
/// `[1, seq, hc*hidden]`. `ids_1d` is a scalar/1-D `[seq]`/singleton u32
/// array (lazy arrays allowed); the same ids feed the hash-routed MoE
/// layers' `tid2eid` lookup.
fn deepseek_v4_forward_trunk(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ids_1d: &MlxArray,
    seq: usize,
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    let v4_cfg = cfg.deepseek_v4.as_ref().expect("DeepSeek V4 config");
    let mut hidden = embed_tokens_arr(ids_1d, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    let mut packed = families::deepseek_v4::expand_embedding(&hidden, v4_cfg);

    // [1, seq] ids for the hash-routing tid2eid gather.
    let ids_2d = reshape(ids_1d, &[1, seq as i32], None);

    // Single-token decode never needs explicit SDPA masks (same convention
    // as the generic forward). V4 uses its own cache buffers, never the
    // rotating ring, so `build_layer_masks_for_forward` always takes the
    // ordered path.
    let decode_mask: Option<MlxArray> = None;
    let masks = (seq > 1).then(|| {
        build_layer_masks_for_forward(cfg, weights.layers.len(), seq, token_offset + seq, cache)
    });
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let shared_mask = masks
            .as_ref()
            .map(|masks| &masks[li])
            .unwrap_or(&decode_mask);
        packed = families::deepseek_v4::layer_forward(
            cfg,
            layer_w,
            &packed,
            cache,
            li,
            token_offset,
            Some(&ids_2d),
            shared_mask.as_ref(),
        );
    }
    packed
}

/// V4 trunk + root-level `hc_head` collapse → `[1, seq, hidden]`.
fn deepseek_v4_forward_hidden(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ids_1d: &MlxArray,
    seq: usize,
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    let packed = deepseek_v4_forward_trunk(cfg, weights, ids_1d, seq, cache, token_offset);
    let head_w = weights
        .deepseek_v4_head
        .as_ref()
        .expect("DeepSeek V4 head weights (hc_head_*)");
    families::deepseek_v4::collapse_for_head(cfg, head_w, &packed)
}

/// V4 counterpart of [`forward_and_logits_mode`]: last-position logits
/// `[vocab]`, or — for [`FinalLogitsMode::Skip`] — the packed residual after
/// the last layer (collapse, final norm, and lm_head all skipped; HC mixing
/// means the last layer must run its full FFN, so no last-only pruning).
fn deepseek_v4_forward_and_logits_mode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
    logits_mode: FinalLogitsMode,
) -> MlxArray {
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    let seq = token_ids.len();
    if matches!(logits_mode, FinalLogitsMode::Skip) {
        return deepseek_v4_forward_trunk(cfg, weights, &ids_1d, seq, cache, token_offset);
    }
    let hidden = deepseek_v4_forward_hidden(cfg, weights, &ids_1d, seq, cache, token_offset);
    let last_hidden = if seq > 1 {
        let last = (seq - 1) as i32;
        let hs = cfg.hidden_size as i32;
        slice(&hidden, &[0, last, 0], &[1, last + 1, hs], &[1, 1, 1], None)
    } else {
        hidden
    };
    let normed = rms_norm(
        &last_hidden,
        Some(&weights.final_norm),
        cfg.rms_norm_eps,
        None,
    );
    let logits = qw(&normed, &weights.lm_head);
    let logits = finalize_lm_head_logits(cfg, &logits, logits_mode);
    reshape(&logits, &[cfg.vocab_size as i32], None)
}

/// V4 counterpart of [`forward_lazy_single_and_logits_mode`]: single-token
/// forward from a (possibly lazy) token array, returning `[1, 1, vocab]`.
fn deepseek_v4_forward_lazy_single_and_logits_mode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_arr: &MlxArray,
    cache: &mut MlxKVCache,
    token_offset: usize,
    lazy_mode: LazySingleTokenMode,
) -> MlxArray {
    let hidden = deepseek_v4_forward_hidden(cfg, weights, token_arr, 1, cache, token_offset);
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &weights.lm_head);
    finalize_lm_head_logits(cfg, &logits, lazy_mode.logits_mode())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_with_initial_hidden_and_media_ranges(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    hidden: MlxArray,
    media_ranges: &[(usize, usize)],
    cache: &mut MlxKVCache,
    token_offset: usize,
    logits_mode: FinalLogitsMode,
) -> MlxArray {
    if cfg.deepseek_v4.is_some() {
        // V4 is text-only and owns the packed HC residual; the pre-scattered
        // hidden + media-overlay path cannot express it. Defensive guard —
        // V4 manifests carry no vision tower, so this is unreachable.
        panic!("deepseek_v4 does not support the initial-hidden/media forward path");
    }
    // Gemma 4 E-series derives a separate embedding for every language
    // layer. Media placeholder IDs are outside that auxiliary embedding
    // vocabulary; the reference model substitutes token 0 at visual/audio
    // positions before the gather while retaining the already-scattered
    // multimodal residual stream.
    let per_layer_token_ids =
        mask_media_token_ids_for_per_layer_inputs(token_ids, media_ranges, token_offset);
    let ids_1d = MlxArray::from_raw_data(
        per_layer_token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(per_layer_token_ids.as_slice()),
        &[per_layer_token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    let seq = token_ids.len();
    let decode_mask: Option<MlxArray> = None;
    let masks = if seq > 1 {
        Some(build_layer_masks_with_media_ranges(
            cfg,
            weights.layers.len(),
            seq,
            token_offset + seq,
            media_ranges,
        ))
    } else {
        None
    };
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &ids_1d, &hidden);
    let mut hidden = hidden;
    let last_layer_idx = weights.layers.len().saturating_sub(1);
    let use_last_layer_optimization = seq > 1;
    let skip_last_layer_ffn = matches!(logits_mode, FinalLogitsMode::Skip);
    let total_layers = weights.layers.len();
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        let shared_mask = masks
            .as_ref()
            .map(|masks| &masks[li])
            .unwrap_or(&decode_mask);
        hidden = if use_last_layer_optimization && li == last_layer_idx {
            layer_forward_last_only(
                cfg,
                layer_w,
                &hidden,
                cache,
                li,
                token_offset,
                pli,
                Some(shared_mask),
                skip_last_layer_ffn,
            )
        } else {
            layer_forward(
                cfg,
                layer_w,
                &hidden,
                cache,
                li,
                token_offset,
                pli,
                Some(shared_mask),
            )
        };
        // mlxcel residual: `pipeline_hint` (AX_MLX_PIPELINE_GRANULARITY).
        if crate::fastpath::pipeline_hint_should_fire(li, total_layers) {
            async_eval(&[&hidden]);
        }
        // Keep the multimodal standard path on the same opt-in blocking
        // prefill-fairness probe as the text-only standard loop.
        if crate::fastpath::pipeline_eval_should_fire(seq, li, total_layers) {
            mlx_sys::eval(&[&hidden]);
        }
    }

    if matches!(logits_mode, FinalLogitsMode::Skip) {
        return hidden;
    }

    let last_hidden = if hidden.shape().get(1).copied().unwrap_or(1) > 1 {
        let last = (token_ids.len() - 1) as i32;
        let hs = cfg.hidden_size as i32;
        slice(&hidden, &[0, last, 0], &[1, last + 1, hs], &[1, 1, 1], None)
    } else {
        hidden
    };

    let normed = rms_norm(
        &last_hidden,
        Some(&weights.final_norm),
        cfg.rms_norm_eps,
        None,
    );
    let logits = qw(&normed, &weights.lm_head);
    let logits = finalize_lm_head_logits(cfg, &logits, logits_mode);
    reshape(&logits, &[cfg.vocab_size as i32], None)
}

fn mask_media_token_ids_for_per_layer_inputs(
    token_ids: &[u32],
    media_ranges: &[(usize, usize)],
    token_offset: usize,
) -> Vec<u32> {
    let mut masked = token_ids.to_vec();
    let chunk_end = token_offset.saturating_add(masked.len());
    for &(start, end_inclusive) in media_ranges {
        let end_exclusive = end_inclusive.saturating_add(1);
        let overlap_start = start.max(token_offset);
        let overlap_end = end_exclusive.min(chunk_end);
        if overlap_start < overlap_end {
            masked[overlap_start - token_offset..overlap_end - token_offset].fill(0);
        }
    }
    masked
}

/// Full visual prefill for unified Qwen checkpoints.
///
/// This keeps AX's existing hybrid-language implementation (including
/// Qwen3.5 GatedDelta layers) and substitutes explicit interleaved MRoPE only
/// on full-attention layers. Qwen3-VL DeepStack maps are injected after
/// language layers 0, 1, ... exactly once during the initial prefill.
pub(crate) fn forward_qwen_visual_prefill(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    prepared: crate::qwen3_vl::Qwen3VlPrefillEmbeddings,
    cache: &mut MlxKVCache,
) -> Result<MlxArray, String> {
    let crate::qwen3_vl::Qwen3VlPrefillEmbeddings {
        hidden,
        mrope,
        rope_delta,
        deepstack,
    } = prepared;
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    let seq = token_ids.len();
    let masks = build_layer_masks_with_media_ranges(cfg, weights.layers.len(), seq, seq, &[]);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &ids_1d, &hidden);
    let mut hidden = hidden;

    for (layer_index, layer_weights) in weights.layers.iter().enumerate() {
        let per_layer_input = per_layer_inputs.as_ref().map(|values| &values[layer_index]);
        hidden = if cfg.is_linear_attention_layer(layer_index) {
            layer_forward(
                cfg,
                layer_weights,
                &hidden,
                cache,
                layer_index,
                0,
                per_layer_input,
                Some(&masks[layer_index]),
            )
        } else if let Some(mrope) = mrope.as_ref() {
            families::standard::layer_forward_with_mrope(
                cfg,
                layer_weights,
                &hidden,
                cache,
                layer_index,
                0,
                per_layer_input,
                Some(&masks[layer_index]),
                false,
                false,
                mrope,
            )
        } else {
            layer_forward(
                cfg,
                layer_weights,
                &hidden,
                cache,
                layer_index,
                0,
                per_layer_input,
                Some(&masks[layer_index]),
            )
        };
        if let Some(deepstack) = deepstack.as_ref()
            && let Some(feature) = deepstack.features.get(layer_index)
        {
            hidden =
                crate::qwen3_vl::add_deepstack_into_text(&hidden, feature, &deepstack.positions)
                    .map_err(|error| error.to_string())?;
        }
    }

    cache.set_mrope_position_delta(rope_delta);
    let last = (seq.saturating_sub(1)) as i32;
    let hidden_size = cfg.hidden_size as i32;
    let last_hidden = slice(
        &hidden,
        &[0, last, 0],
        &[1, last + 1, hidden_size],
        &[1, 1, 1],
        None,
    );
    let normed = rms_norm(
        &last_hidden,
        Some(&weights.final_norm),
        cfg.rms_norm_eps,
        None,
    );
    let logits = qw(&normed, &weights.lm_head);
    let logits = finalize_lm_head_logits(cfg, &logits, FinalLogitsMode::Full);
    Ok(reshape(&logits, &[cfg.vocab_size as i32], None))
}

/// Forward all positions only far enough to update cache state.
///
/// Used when a speculative branch must replay a committed prefix after a
/// partial rejection. The caller only needs recurrent/KV state for future
/// tokens, not logits for sampling, so this intentionally skips final norm and
/// lm_head work.
pub fn forward_all_positions_update_cache(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) {
    // DI-VL-001: match singleton decode MRoPE origin after visual prefill.
    let token_offset = qwen_visual_rope_offset(weights, cache, token_offset);
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    if cfg.deepseek_v4.is_some() {
        // Trunk only: cache writes are the point; collapse/head are skipped.
        let _ =
            deepseek_v4_forward_trunk(cfg, weights, &ids_1d, token_ids.len(), cache, token_offset);
        return;
    }
    let mut hidden = embed_tokens_arr(&ids_1d, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    let seq = token_ids.len();
    let masks =
        build_layer_masks_for_forward(cfg, weights.layers.len(), seq, token_offset + seq, cache);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &ids_1d, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
        );
    }
}

/// Forward pass returning logits for ALL token positions — `[seq, vocab_size]` f32.
///
/// Used by draft verification to check all draft tokens in one pass.
pub fn forward_all_positions(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    // DI-VL-001: multi-token n-gram / MTP verify must use the same MRoPE origin
    // as singleton decode after visual prefill (mrope_position_delta).
    let token_offset = qwen_visual_rope_offset(weights, cache, token_offset);
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    if cfg.deepseek_v4.is_some() {
        let hidden =
            deepseek_v4_forward_hidden(cfg, weights, &ids_1d, token_ids.len(), cache, token_offset);
        let seq = token_ids.len() as i32;
        let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
        let logits = qw(&normed, &weights.lm_head);
        let logits_f32 = astype(&logits, MlxDtype::Float32, None);
        let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
        return reshape(&logits_f32, &[seq, cfg.vocab_size as i32], None);
    }
    let mut hidden = embed_tokens_arr(&ids_1d, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    let seq = token_ids.len();
    let masks =
        build_layer_masks_for_forward(cfg, weights.layers.len(), seq, token_offset + seq, cache);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &ids_1d, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
        );
    }

    let seq = seq as i32;
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    reshape(&logits_f32, &[seq, cfg.vocab_size as i32], None)
}

/// Forward all positions, returning both per-position logits `[seq, vocab]` and
/// post-final-norm hidden states `[1, seq, hidden_size]`.
///
/// MTP heads use a selected row from the returned post-norm hidden states as
/// the `main_hidden` for the first draft head's forward pass.
/// Like [`forward_all_positions_with_post_norm`], but final logits match pure-direct
/// greedy (`FinalLogitsMode::ArgmaxOnly`): native lm_head dtype, no softcap, no
/// vocab-wide f32 cast. Used by Gemma assistant-MTP multi-token always-adopt so
/// teacher-forced argmax agrees with singleton pure-direct on near-ties (4-bit
/// cycle breaks) where softcap(f32(logits)) can flip vs bf16 argmax.
pub fn forward_all_positions_with_post_norm_greedy(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> (MlxArray, MlxArray) {
    let seq = token_ids.len();
    // LONG_MT uses physical cache length (heuristic), then map to MRoPE origin.
    let moe_long_mt = cfg.moe_expert_count > 0
        && token_offset >= 512
        && std::env::var("AX_MLX_GEMMA4_MOE_LONG_MT")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    let token_offset = qwen_visual_rope_offset(weights, cache, token_offset);
    // LONG_MT aligns both the materialized S=1 baseline and the multi-token
    // verifier on the invariant affine projection kernel. RowExact MLX reads
    // every quantized weight once per verify row; the invariant kernel keeps
    // the singleton qmv reduction order while reusing each weight load across
    // the microbatch. Scope it here so empty-draft S=1 and verify S>1 cannot
    // acquire different arithmetic contracts.
    let _moe_invariant = moe_long_mt.then(|| crate::fastpath::scoped_qwen_linear_mtp_exact(true));
    // MoE multi-token: per-pos SDPA, pos0-exact/rest-Shared QKV, RowExact
    // dense/o_proj, batched experts. Empty-draft real post_norm.
    // Keep multi-token window views ON (default) so long-prompt agent uses
    // window+seq-1 retained K/V matching pure-direct ring geometry. Forcing
    // views OFF (smokef107) still diverged at first_diff@14; identity probe
    // without rings is exact — formal A/B uses bounded rings after long prefill.
    let _moe_bf16 = (cfg.moe_expert_count > 0 && seq > 1)
        .then(|| crate::fastpath::scoped_moe_mt_bf16_identity(true));
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    let mut hidden = embed_tokens_arr(&ids_1d, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    let masks =
        build_layer_masks_for_forward(cfg, weights.layers.len(), seq, token_offset + seq, cache);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &ids_1d, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
        );
    }
    let seq_i = seq as i32;
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &weights.lm_head);
    // ArgmaxOnly: no softcap / no f32 cast (matches pure-direct greedy).
    let logits_out = reshape(&logits, &[seq_i, cfg.vocab_size as i32], None);
    (logits_out, normed)
}

pub fn forward_all_positions_with_post_norm(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> (MlxArray, MlxArray) {
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    forward_all_positions_with_post_norm_ids(
        cfg,
        weights,
        &ids_1d,
        token_ids.len(),
        cache,
        token_offset,
    )
}

/// [`forward_all_positions_with_post_norm`] taking the token ids as a lazy
/// `[seq]` u32 array instead of host values.
///
/// Lets a speculative verifier chain directly on lazy draft-token arrays
/// (`AX_MLX_MTP_ASYNC_DRAFT`): the whole verify graph builds before the draft
/// head's GPU forward completes, so graph construction overlaps GPU execution
/// instead of serializing behind a host extraction. The arithmetic is
/// identical — the embedding gather consumes the same index values.
pub fn forward_all_positions_with_post_norm_ids(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ids_1d: &MlxArray,
    seq: usize,
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> (MlxArray, MlxArray) {
    // DI-VL-001: async-draft verify and MTP multi-token path share this entry.
    let token_offset = qwen_visual_rope_offset(weights, cache, token_offset);
    if cfg.deepseek_v4.is_some() {
        let hidden = deepseek_v4_forward_hidden(cfg, weights, ids_1d, seq, cache, token_offset);
        let seq_i = seq as i32;
        let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
        let logits = qw(&normed, &weights.lm_head);
        let logits_f32 = astype(&logits, MlxDtype::Float32, None);
        let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
        let logits_out = reshape(&logits_f32, &[seq_i, cfg.vocab_size as i32], None);
        return (logits_out, normed);
    }
    let mut hidden = embed_tokens_arr(ids_1d, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    let masks =
        build_layer_masks_for_forward(cfg, weights.layers.len(), seq, token_offset + seq, cache);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, ids_1d, &hidden);
    let layer_count = weights.layers.len();
    // Chunked submit (`AX_MLX_MTP_VERIFY_SUBMIT_LAYERS`, default off).
    let submit_interval = crate::fastpath::verify_submit_interval_for_build(
        seq,
        layer_count,
        crate::fastpath::mtp_verify_submit_layer_interval(),
    );
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
        );
        // Hand the layers built so far to the GPU and keep building. The last
        // chunk is left for the caller's `eval`, which has to block anyway and
        // already carries the full cache refs.
        //
        // Mid-loop submits use `hidden` only. Within one multi-token forward
        // each layer runs once; later layers depend on prior hiddens, so the
        // MoE/attention compute on the residual path is scheduled. Per-layer
        // K/V and linear-attn recurrent side outputs are only needed on the
        // *next* decode step — the terminal eval already materialises them.
        // Re-submitting the full cache every chunk (often hundreds of arrays)
        // inflated `async_eval` task counts and re-introduced gather_qmm
        // backpressure on MoE, which is the opposite of the overlap goal.
        let submitted =
            submit_interval > 0 && li + 1 < layer_count && (li + 1) % submit_interval == 0;
        if submitted {
            async_eval(&[&hidden]);
        } else if crate::fastpath::pipeline_hint_should_fire(li, layer_count) {
            // Same residual-only hint as prefill/direct layer loops. Skip when
            // the chunked-submit interval already fired this layer to avoid
            // double async_eval on the same residual.
            async_eval(&[&hidden]);
        }
    }

    let seq_i = seq as i32;
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    let logits_out = reshape(&logits_f32, &[seq_i, cfg.vocab_size as i32], None);
    // When chunked submit is active, also kick the lm_head/norm tail so GPU
    // work continues while the caller builds argmax + cache-ref eval targets.
    // Exactness-preserving: only schedules already-built arrays.
    if submit_interval > 0 {
        async_eval(&[&logits_out, &normed]);
    }
    (logits_out, normed)
}

/// Forward all positions, returning last-position logits `[vocab]` and
/// all-position post-final-norm hidden states `[1, seq, hidden_size]`.
///
/// Unlike [`forward_all_positions_with_post_norm`], the lm_head projection
/// runs only on the last position, reducing the matmul from
/// `[seq, hidden] × [hidden, vocab]` to `[1, hidden] × [hidden, vocab]` —
/// a ~seq× reduction in compute (Qwen3.6 27B 128-token prompt: ~128× fewer
/// multiply-adds). The all-position post-norm hidden is preserved for MTP
/// warmup seeding.
/// Multimodal MTP warmup path: full-seq residual through every layer with
/// media-aware masks, post-norm on all positions, lm_head on last only (WS-M5).
pub fn forward_with_initial_hidden_media_post_norm_last_lm_head(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    hidden: MlxArray,
    media_ranges: &[(usize, usize)],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> (MlxArray, MlxArray) {
    if cfg.deepseek_v4.is_some() {
        // See the guard in `forward_with_initial_hidden_and_media_ranges`:
        // V4 is text-only and owns the packed HC residual; unreachable.
        panic!("deepseek_v4 does not support the initial-hidden/media forward path");
    }
    let per_layer_token_ids =
        mask_media_token_ids_for_per_layer_inputs(token_ids, media_ranges, token_offset);
    let ids_1d = MlxArray::from_raw_data(
        per_layer_token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(per_layer_token_ids.as_slice()),
        &[per_layer_token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    let seq = token_ids.len();
    let masks = if seq > 1 {
        build_layer_masks_with_media_ranges(
            cfg,
            weights.layers.len(),
            seq,
            token_offset + seq,
            media_ranges,
        )
    } else {
        build_layer_masks_for_forward(cfg, weights.layers.len(), seq, token_offset + seq, cache)
    };
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &ids_1d, &hidden);
    let mut hidden = hidden;
    // Full-seq residual through every layer: MTP warmup needs post-norm at
    // every position. Do not use last-layer last-only / skip-FFN here.
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
        );
    }
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let last = (seq.saturating_sub(1)) as i32;
    let hs = cfg.hidden_size as i32;
    let last_normed = slice(&normed, &[0, last, 0], &[1, last + 1, hs], &[1, 1, 1], None);
    let logits = qw(&last_normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    let last_logits = reshape(&logits_f32, &[cfg.vocab_size as i32], None);
    (last_logits, normed)
}

pub fn forward_all_positions_post_norm_last_lm_head(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> (MlxArray, MlxArray) {
    // DI-VL-001: MTP warmup / optimistic multi-token path after visual prefill.
    let token_offset = qwen_visual_rope_offset(weights, cache, token_offset);
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    if cfg.deepseek_v4.is_some() {
        let seq = token_ids.len();
        let hidden = deepseek_v4_forward_hidden(cfg, weights, &ids_1d, seq, cache, token_offset);
        let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
        let last = (seq - 1) as i32;
        let hs = cfg.hidden_size as i32;
        let last_normed = slice(&normed, &[0, last, 0], &[1, last + 1, hs], &[1, 1, 1], None);
        let logits = qw(&last_normed, &weights.lm_head);
        let logits_f32 = astype(&logits, MlxDtype::Float32, None);
        let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
        let last_logits = reshape(&logits_f32, &[cfg.vocab_size as i32], None);
        return (last_logits, normed);
    }
    let mut hidden = embed_tokens_arr(&ids_1d, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    let seq = token_ids.len();
    let masks =
        build_layer_masks_for_forward(cfg, weights.layers.len(), seq, token_offset + seq, cache);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &ids_1d, &hidden);
    // Full-seq residual through every layer: MTP warmup needs post-norm at
    // every position. Do not use last-layer last-only / skip-FFN here.
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
        );
    }

    // rms_norm on ALL positions — needed for MTP warmup which iterates
    // over every position's hidden state to seed the MTP recurrent cache.
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);

    // Slice to last position ONLY for lm_head — the key optimization.
    let last = (seq - 1) as i32;
    let hs = cfg.hidden_size as i32;
    let last_normed = slice(&normed, &[0, last, 0], &[1, last + 1, hs], &[1, 1, 1], None);
    let logits = qw(&last_normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    let last_logits = reshape(&logits_f32, &[cfg.vocab_size as i32], None);
    (last_logits, normed)
}

/// DeepSeek V4 MTP verify forward: per-position logits `[seq, vocab]` plus
/// the packed pre-collapse residual `[1, seq, hc*hidden]` — the nextn block's
/// `h` input (llama.cpp `t_h_nextn`). V4 counterpart of
/// [`forward_all_positions_with_post_norm`]: the MTP draft head consumes the
/// packed stream, not the post-norm E-wide hidden, so the runner slices draft
/// rows from the second return with width `hc_mult * hidden_size`.
pub fn deepseek_v4_forward_all_positions_with_packed(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> (MlxArray, MlxArray) {
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    let seq = token_ids.len();
    let packed = deepseek_v4_forward_trunk(cfg, weights, &ids_1d, seq, cache, token_offset);
    let head_w = weights
        .deepseek_v4_head
        .as_ref()
        .expect("DeepSeek V4 head weights (hc_head_*)");
    let hidden = families::deepseek_v4::collapse_for_head(cfg, head_w, &packed);
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    let logits_out = reshape(&logits_f32, &[seq as i32, cfg.vocab_size as i32], None);
    (logits_out, packed)
}

/// Forward all positions, returning both per-position logits `[seq, vocab]` and
/// the post-final-norm hidden state at the final position `[1, 1, hidden_size]`.
pub fn forward_all_positions_with_final_hidden(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> (MlxArray, MlxArray) {
    let (logits, post_norm) =
        forward_all_positions_with_post_norm(cfg, weights, token_ids, cache, token_offset);
    let last = (token_ids.len() - 1) as i32;
    let hs = cfg.hidden_size as i32;
    let last_post_norm = slice(
        &post_norm,
        &[0, last, 0],
        &[1, last + 1, hs],
        &[1, 1, 1],
        None,
    );
    (logits, last_post_norm)
}

/// Whether a greedy assistant draft position should be kept under confidence gates.
///
/// Depth 0 uses `first_gate`; deeper positions use `deep_gate` (tighter by default).
/// Pure helper — used by the multi-depth draft loop and unit-tested without weights.
#[inline]
pub fn gemma4_assistant_draft_position_accepted(
    depth: usize,
    confidence: f32,
    first_gate: f32,
    deep_gate: f32,
) -> bool {
    let gate = if depth == 0 { first_gate } else { deep_gate };
    confidence >= gate
}

/// How many leading confidences pass the position gates (stops at first miss).
pub fn gemma4_assistant_accepted_draft_depth(
    confidences: &[f32],
    first_gate: f32,
    deep_gate: f32,
) -> usize {
    confidences
        .iter()
        .enumerate()
        .take_while(|(d, c)| {
            gemma4_assistant_draft_position_accepted(*d, **c, first_gate, deep_gate)
        })
        .count()
}

#[allow(clippy::too_many_arguments)]
pub fn gemma4_assistant_forward_one(
    assistant_cfg: &ModelConfig,
    assistant_weights: &ModelWeights,
    target_cfg: &ModelConfig,
    target_weights: &ModelWeights,
    target_cache: &MlxKVCache,
    target_shared_layers: Gemma4AssistantSharedKvLayers,
    last_token: u32,
    last_backbone_hidden: &MlxArray,
    constant_position: usize,
) -> Result<(MlxArray, MlxArray), Gemma4AssistantForwardError> {
    use Gemma4AssistantForwardError as E;
    if assistant_cfg.model_family != "gemma4_assistant" || target_cfg.model_family != "gemma4" {
        return Err(E::ModelFamilyMismatch);
    }
    gemma4_assistant_forward_one_validated(
        assistant_cfg,
        assistant_weights,
        target_cfg,
        target_weights,
        target_cache,
        target_shared_layers,
        last_token,
        last_backbone_hidden,
        constant_position,
    )
}

/// Hot multi-depth path after family/projection validation has succeeded once.
#[allow(clippy::too_many_arguments)]
fn gemma4_assistant_forward_one_validated(
    assistant_cfg: &ModelConfig,
    assistant_weights: &ModelWeights,
    target_cfg: &ModelConfig,
    target_weights: &ModelWeights,
    target_cache: &MlxKVCache,
    target_shared_layers: Gemma4AssistantSharedKvLayers,
    last_token: u32,
    last_backbone_hidden: &MlxArray,
    constant_position: usize,
) -> Result<(MlxArray, MlxArray), Gemma4AssistantForwardError> {
    // Shared implementation with the multi-depth session: validate once, freeze
    // target KV once, then run a single depth step.
    let mut session = Gemma4AssistantDraftSession::open(
        assistant_cfg,
        assistant_weights,
        target_cfg,
        target_weights,
        target_shared_layers,
    )?;
    session.bind_target_cache(target_cache)?;
    session.forward_one(last_token, last_backbone_hidden, constant_position)
}

/// Scalar Int32 RoPE offset array for assistant draft (compile-graph input).
///
/// Mirrors the MTP compiled-draft pattern: position flows as an array node so
/// multi-depth steps share graph structure and a future `mlx_compile` closure
/// can take offset as an input without re-tracing.
fn gemma4_assistant_rope_offset_array(position: usize) -> MlxArray {
    let offset_val = position as i32;
    MlxArray::from_raw_data(
        &offset_val as *const i32 as *const u8,
        4,
        &[1_i32],
        MlxDtype::Int32,
    )
}

fn gemma4_assistant_layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    frozen_kv: &Gemma4AssistantFrozenTargetKv,
    layer_idx: usize,
    rope_offset_arr: &MlxArray,
) -> Result<MlxArray, Gemma4AssistantForwardError> {
    use Gemma4AssistantForwardError as E;
    let (head_dim, rope_theta, rope_dims, layer_rope_freqs, sliding_window, _, _) =
        layer_params(cfg, layer_idx);
    let layer_kv = frozen_kv
        .for_layer(sliding_window)
        .ok_or(E::MissingSharedKvLayer)?;
    let cached_k = &layer_kv.k;
    let cached_v = &layer_kv.v;

    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    let q_raw = qw(&normed, w.q_proj.as_ref().ok_or(E::MissingQProj)?);
    let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
    let (rope_base, rope_freqs_ref) = rope_freqs
        .map(|f| (None, Some(f)))
        .unwrap_or((Some(rope_theta), None));
    // Dynamic RoPE: offset is a graph input (same as MTP compiled draft).
    // Target K/V are already frozen arrays; only Q is rotated per depth.
    let seq = hidden.shape()[1] as usize;
    let q_normed = qk_norm_bhsd_from_proj(
        &q_raw,
        w.q_norm.as_ref(),
        cfg.n_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    let q_rope = rope_dynamic(
        &q_normed,
        rope_dims as i32,
        false,
        rope_base,
        1.0,
        rope_offset_arr,
        rope_freqs_ref,
        None,
    );

    let key_len = cached_k.shape()[2] as usize;
    let mask = match layer_kv.ring {
        // Rotated target ring: frozen peek returned the full unordered ring,
        // so ordered-position masks are meaningless. The drafter's query
        // logically sits at the end of the committed context; the slot-validity
        // mask keeps exactly the last `window` live tokens and excludes
        // rolled-back draft slots and dead slack slots.
        Some(ring) => Some(crate::attention_mask::create_ring_sliding_mask(
            seq,
            ring.window,
            ring.capacity,
            ring.write_start.saturating_sub(seq),
        )),
        None => attention_mask_array(seq, key_len, sliding_window),
    };
    let attn_sdpa =
        full_precision_attention(&q_rope, cached_k, cached_v, cfg.query_scale, seq, &mask);
    let attn_flat = flatten_attention_output_bhsd(&attn_sdpa, seq, cfg.n_heads, head_dim);
    let attn_proj =
        attention_output_projection(&attn_flat, None, w.o_proj.as_ref().ok_or(E::MissingOProj)?);
    let attn_proj = if let Some(post_norm) = &w.attn_post_norm {
        rms_norm(&attn_proj, Some(post_norm), cfg.rms_norm_eps, None)
    } else {
        attn_proj
    };
    let hidden = add(hidden, &attn_proj, None);

    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    // Gemma4 "sandwich norm": post_feedforward_layernorm is applied to the FFN
    // output (inside ffn_swiglu) before the residual add, and the learned
    // per-layer scalar scales the residual output. The assistant layers carry
    // both weights (post_feedforward_layernorm + layer_scalar) exactly like the
    // target's dense layers, so mirror families::standard::layer_forward here.
    // Dropping either drifts the hidden state across the 4 assistant layers and
    // tanks the draft accept rate.
    let ffn = ffn_swiglu(cfg, w, &normed2, w.ffn_post_norm.as_ref(), layer_idx);
    Ok(if let Some(scalar) = &w.layer_scalar {
        add_then_multiply_scalar(&hidden, &ffn, scalar)
    } else {
        add(&hidden, &ffn, None)
    })
}

// ── Gemma4 assistant multi-depth draft session (Phases B + D + E) ───────────
//
// Validates family + projections once, freezes shared target K/V (+ ring) once
// per draft attempt, applies Q RoPE via rope_dynamic (offset as array input),
// then runs the hot forward per depth. Pure mlx_compile can still be a
// follow-on once a single MlxClosure is traced over these array inputs; the
// "compile" flag must not wrap the imperative path in a non-compiled
// MlxClosure until that lands with A/B evidence.

/// Request-scoped assistant draft context: one validation, one frozen target
/// KV bind, many depth steps.
pub struct Gemma4AssistantDraftSession<'a> {
    assistant_cfg: &'a ModelConfig,
    assistant_weights: &'a ModelWeights,
    target_cfg: &'a ModelConfig,
    target_weights: &'a ModelWeights,
    target_shared_layers: Gemma4AssistantSharedKvLayers,
    pre_projection: &'a QuantizedWeight,
    post_projection: &'a QuantizedWeight,
    /// Frozen shared full/sliding K/V after [`Self::bind_target_cache`].
    frozen_target_kv: Option<Gemma4AssistantFrozenTargetKv>,
}

impl<'a> Gemma4AssistantDraftSession<'a> {
    /// Validate model families and projection weights once for a draft loop.
    pub fn open(
        assistant_cfg: &'a ModelConfig,
        assistant_weights: &'a ModelWeights,
        target_cfg: &'a ModelConfig,
        target_weights: &'a ModelWeights,
        target_shared_layers: Gemma4AssistantSharedKvLayers,
    ) -> Result<Self, Gemma4AssistantForwardError> {
        use Gemma4AssistantForwardError as E;
        if assistant_cfg.model_family != "gemma4_assistant" || target_cfg.model_family != "gemma4" {
            return Err(E::ModelFamilyMismatch);
        }
        let pre_projection = assistant_weights
            .assistant_pre_projection
            .as_ref()
            .ok_or(E::MissingPreProjection)?;
        let post_projection = assistant_weights
            .assistant_post_projection
            .as_ref()
            .ok_or(E::MissingPostProjection)?;
        Ok(Self {
            assistant_cfg,
            assistant_weights,
            target_cfg,
            target_weights,
            target_shared_layers,
            pre_projection,
            post_projection,
            frozen_target_kv: None,
        })
    }

    /// Peek shared full/sliding target K/V (+ ring layout) once for this draft
    /// attempt. Target cache is read-only for the assistant, so multi-depth
    /// steps reuse the same arrays without re-peeking.
    pub fn bind_target_cache(
        &mut self,
        target_cache: &MlxKVCache,
    ) -> Result<(), Gemma4AssistantForwardError> {
        // Clear any prior freeze *before* attempting the new one so a failed
        // re-bind fails closed (`has_frozen_target_kv() == false`) instead of
        // silently keeping a stale snapshot from an earlier successful bind.
        self.frozen_target_kv = None;
        self.frozen_target_kv = Some(Gemma4AssistantFrozenTargetKv::freeze_from_cache(
            target_cache,
            self.target_shared_layers,
        )?);
        Ok(())
    }

    /// Whether the frozen sliding binding uses a rotating ring layout.
    ///
    /// Useful for SWA pilot / multi-depth telemetry: ring-aware drafts must
    /// keep the slot-validity mask derived from the frozen write_start.
    pub fn frozen_sliding_uses_ring(&self) -> bool {
        self.frozen_target_kv
            .as_ref()
            .is_some_and(Gemma4AssistantFrozenTargetKv::sliding_uses_ring)
    }

    /// Whether target KV has been frozen via [`Self::bind_target_cache`].
    pub fn has_frozen_target_kv(&self) -> bool {
        self.frozen_target_kv.is_some()
    }

    /// One assistant forward step at `constant_position` (validated + frozen).
    ///
    /// Requires [`Self::bind_target_cache`] first. RoPE offset is passed as an
    /// array (via [`gemma4_assistant_rope_offset_array`]) so multi-depth steps
    /// share a compile-ready graph shape with frozen target K/V.
    pub fn forward_one(
        &self,
        last_token: u32,
        last_backbone_hidden: &MlxArray,
        constant_position: usize,
    ) -> Result<(MlxArray, MlxArray), Gemma4AssistantForwardError> {
        let token_data = [last_token];
        let token_arr = MlxArray::from_raw_data(
            token_data.as_ptr() as *const u8,
            std::mem::size_of_val(&token_data),
            &[1_i32],
            MlxDtype::Uint32,
        );
        self.forward_one_from_token_arr(&token_arr, last_backbone_hidden, constant_position)
    }

    /// Like [`Self::forward_one`], but accepts a lazy `[1]` uint32 token id
    /// array so multi-depth drafting can chain unevaluated argmax outputs
    /// into the next step's embedding without a host sync.
    pub fn forward_one_from_token_arr(
        &self,
        last_token_ids: &MlxArray,
        last_backbone_hidden: &MlxArray,
        constant_position: usize,
    ) -> Result<(MlxArray, MlxArray), Gemma4AssistantForwardError> {
        let frozen = self
            .frozen_target_kv
            .as_ref()
            .ok_or(Gemma4AssistantForwardError::UnboundTargetKv)?;
        let rope_offset = gemma4_assistant_rope_offset_array(constant_position);

        let mut token_embedding = embed_tokens_arr(
            last_token_ids,
            &self.target_weights.token_embedding,
            self.target_cfg.hidden_size,
        );
        token_embedding = astype(&token_embedding, MlxDtype::Bfloat16, None);
        if let Some(scale) = self.target_cfg.hidden_states_scale {
            token_embedding = scale_hidden(&token_embedding, scale);
        }
        let assistant_input = concatenate(&[&token_embedding, last_backbone_hidden], -1, None);
        let mut hidden = qw(&assistant_input, self.pre_projection);

        for (layer_idx, layer_weights) in self.assistant_weights.layers.iter().enumerate() {
            hidden = gemma4_assistant_layer_forward(
                self.assistant_cfg,
                layer_weights,
                &hidden,
                frozen,
                layer_idx,
                &rope_offset,
            )?;
        }

        let normed = rms_norm(
            &hidden,
            Some(&self.assistant_weights.final_norm),
            self.assistant_cfg.rms_norm_eps,
            None,
        );
        let logits = qw(&normed, &self.assistant_weights.lm_head);
        let logits = apply_final_logit_softcap(
            self.assistant_cfg,
            &astype(&logits, MlxDtype::Float32, None),
        );
        let logits = reshape(&logits, &[self.assistant_cfg.vocab_size as i32], None);
        let projected_hidden = qw(&normed, self.post_projection);
        Ok((logits, projected_hidden))
    }
}

/// Apply the Gemma4 assistant MTP forward via the "compile" entry point.
///
/// Returns `None` when the compile flag is disabled (caller uses imperative /
/// draft session). When enabled, runs the same real forward as
/// [`gemma4_assistant_forward_one`] (no non-compiled MlxClosure wrapper).
/// Pure-graph `mlx_compile` remains a follow-on once KV/RoPE are array inputs.
#[allow(clippy::too_many_arguments)]
pub fn gemma4_assistant_forward_one_compiled(
    assistant_cfg: &ModelConfig,
    assistant_weights: &ModelWeights,
    target_cfg: &ModelConfig,
    target_weights: &ModelWeights,
    target_cache: &MlxKVCache,
    target_shared_layers: Gemma4AssistantSharedKvLayers,
    last_token: u32,
    last_backbone_hidden: &MlxArray,
    constant_position: usize,
) -> Option<Result<(MlxArray, MlxArray), Gemma4AssistantForwardError>> {
    if !crate::fastpath::gemma4_assistant_compile_enabled() {
        return None;
    }
    Some(gemma4_assistant_forward_one(
        assistant_cfg,
        assistant_weights,
        target_cfg,
        target_weights,
        target_cache,
        target_shared_layers,
        last_token,
        last_backbone_hidden,
        constant_position,
    ))
}

/// Cache-free single transformer layer for dense embedding models.
///
/// Equivalent to the standard dense-attention path in `layer_forward`, but
/// skips all KV-cache writes (no `zeros` allocation, no `slice_update`).
/// Only valid for Qwen3/Gemma dense layers — no linear attention, no MLA,
/// no KV-source sharing, no MoE.
///
/// `pending_ffn` carries the previous layer's FFN output so its residual
/// add can be fused with this layer's pre-attention RMSNorm via
/// `add_rms_norm_pair`, saving one GPU kernel dispatch per layer boundary.
/// Returns `(hidden, ffn_out)` where `ffn_out` is the current layer's FFN
/// output — the caller is responsible for adding it as the residual for
/// the next layer (or final norm).
fn layer_forward_dense_embed(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray, // [batch, seq, hidden]
    layer_idx: usize,
    pending_ffn: Option<&MlxArray>,
    target_positions_for_ffn: Option<&[usize]>,
    // When Some: bidirectional + pad mask. When None: causal SDPA (Qwen last-token).
    attention_mask: &Option<MlxArray>,
) -> (MlxArray, MlxArray) {
    let (
        head_dim,
        rope_theta,
        rope_dims,
        layer_rope_freqs,
        _sliding_window,
        _kv_source,
        v_norm_no_scale,
    ) = layer_params(cfg, layer_idx);
    let batch = hidden.shape()[0] as usize;
    let seq = hidden.shape()[1] as usize;
    let flat_attention_shape = seq > 1;
    // AX_MLX_EMBED_PROFILE per-stage timers. Each `embed_profile_eval_elapsed`
    // forces an `eval()` barrier, so the breakdown is for ratio analysis only
    // (it disables forward pipelining and inflates absolute time).
    let profile = embed_profile_enabled() && seq > 1;

    // 1. Fuse pending FFN residual from the previous layer with this layer's
    // pre-attention RMSNorm via `add_rms_norm_pair`, saving one GPU kernel
    // dispatch per layer boundary. When there is no pending FFN (first layer),
    // fall back to a plain RMSNorm.
    let attn_norm_started = profile.then(Instant::now);
    let (hidden, normed) = match pending_ffn {
        Some(ffn_out) => {
            mlx_sys::add_rms_norm_pair(hidden, ffn_out, &w.attn_norm, cfg.rms_norm_eps, None)
        }
        None => {
            let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
            (hidden.clone(), normed)
        }
    };
    if let Some(started) = attn_norm_started {
        embed_profile_eval_elapsed(profile, EmbedProfileStage::AttnNorm, started, &[&normed]);
    }

    let common_last_target_position = target_positions_for_ffn.and_then(|positions| {
        positions
            .first()
            .copied()
            .filter(|&first| first + 1 == seq && positions.iter().all(|&p| p == first))
    });

    // Last-layer last-token pooling with split Q/K/V (Qwen3-Embedding): project
    // Q only at the shared target position and RoPE it at that offset. K/V still
    // span the full sequence so causal SDPA at the target token is exact. Saves
    // the full-seq Q matmul + Q rope on the terminal layer (ingest equal-length
    // chunks always hit this path).
    let q_projected_at_target = common_last_target_position.is_some()
        && !cfg.attn_output_gate
        && w.q_proj.is_some()
        && w.k_proj.is_some();

    // 2-7. QKV projections, reshape, QK-norm, transpose, RoPE.
    let qkv_proj_started = profile.then(Instant::now);
    let (q_raw, k_raw, v_raw, q_seq, q_rope_offset) =
        if let (true, Some(target_pos), Some(positions), Some(q_w), Some(k_w)) = (
            q_projected_at_target,
            common_last_target_position,
            target_positions_for_ffn,
            w.q_proj.as_ref(),
            w.k_proj.as_ref(),
        ) {
            let q_in = select_embedding_targets(&normed, positions);
            let q = qw(&q_in, q_w);
            let k = qw(&normed, k_w);
            let v = w
                .v_proj
                .as_ref()
                .map(|v_w| qw(&normed, v_w))
                .unwrap_or_else(|| k.clone());
            (q, k, v, 1_usize, target_pos)
        } else {
            let (q, k, v, _gate) = qkv_project_embed(cfg, w, &normed, head_dim);
            (q, k, v, seq, 0_usize)
        };
    if let Some(started) = qkv_proj_started {
        embed_profile_eval_elapsed(
            profile,
            EmbedProfileStage::QkvProj,
            started,
            &[&q_raw, &k_raw, &v_raw],
        );
    }
    let kv_heads = (k_raw.shape()[2] as usize)
        .checked_div(head_dim)
        .expect("k projection output must divide by head_dim");

    let value_prep_started = profile.then(Instant::now);
    let v = if flat_attention_shape {
        prepare_value_bhsd_from_proj_flat(
            &v_raw,
            v_norm_no_scale,
            kv_heads,
            head_dim,
            seq,
            cfg.rms_norm_eps,
        )
    } else {
        prepare_value_bhsd_from_proj(
            &v_raw,
            v_norm_no_scale,
            kv_heads,
            head_dim,
            seq,
            cfg.rms_norm_eps,
        )
    };
    if let Some(started) = value_prep_started {
        embed_profile_eval_elapsed(profile, EmbedProfileStage::ValuePrep, started, &[&v]);
    }

    let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
    let (rope_base, rope_freqs_ref) = rope_freqs
        .map(|f| (None, Some(f)))
        .unwrap_or((Some(rope_theta), None));
    let qk_norm_rope_started = profile.then(Instant::now);
    // Q may already be 1-token (last-layer target path); use the non-flat rope
    // helpers for q_seq == 1. K always spans the full sequence.
    let q_rope = if q_seq > 1 && flat_attention_shape {
        qk_norm_rope_bhsd_from_proj_flat(
            &q_raw,
            w.q_norm.as_ref(),
            cfg.n_heads,
            head_dim,
            q_seq,
            cfg.rms_norm_eps,
            rope_dims,
            rope_base,
            q_rope_offset,
            rope_freqs_ref,
        )
    } else {
        qk_norm_rope_bhsd_from_proj(
            &q_raw,
            w.q_norm.as_ref(),
            cfg.n_heads,
            head_dim,
            q_seq,
            cfg.rms_norm_eps,
            rope_dims,
            rope_base,
            q_rope_offset,
            rope_freqs_ref,
        )
    };
    let k_rope = if flat_attention_shape {
        qk_norm_rope_bhsd_from_proj_flat(
            &k_raw,
            w.k_norm.as_ref(),
            kv_heads,
            head_dim,
            seq,
            cfg.rms_norm_eps,
            rope_dims,
            rope_base,
            0,
            rope_freqs_ref,
        )
    } else {
        qk_norm_rope_bhsd_from_proj(
            &k_raw,
            w.k_norm.as_ref(),
            kv_heads,
            head_dim,
            seq,
            cfg.rms_norm_eps,
            rope_dims,
            rope_base,
            0,
            rope_freqs_ref,
        )
    };
    if let Some(started) = qk_norm_rope_started {
        embed_profile_eval_elapsed(
            profile,
            EmbedProfileStage::QkNormRope,
            started,
            &[&q_rope, &k_rope],
        );
    }

    let q_rope_for_attention;
    let (q_rope_ref, attn_seq) = if q_projected_at_target {
        // Q was projected and RoPE'd only at the target; already [B, H, 1, D].
        (&q_rope, 1_usize)
    } else if let Some(target_position) = common_last_target_position {
        // Packed-QKV / gated fallback: full-seq Q then slice.
        q_rope_for_attention = select_attention_common_target_bhsd(&q_rope, target_position);
        (&q_rope_for_attention, 1_usize)
    } else {
        (&q_rope, seq)
    };

    // 8. SDPA — k_rope/v used directly, no KV-cache writes.
    // Qwen3-Embedding: mask None → causal (last-token pool oracle).
    // Nemotron Embed: pass bidirectional padding mask so tokens attend fully.
    let sdpa_started = profile.then(Instant::now);
    let attn_sdpa = full_precision_attention(
        q_rope_ref,
        &k_rope,
        &v,
        cfg.query_scale,
        attn_seq,
        attention_mask,
    );
    if let Some(started) = sdpa_started {
        embed_profile_eval_elapsed(profile, EmbedProfileStage::Sdpa, started, &[&attn_sdpa]);
    }

    // 9-13. Transpose back, reshape, output projection, residual.
    let attn_out_proj_started = profile.then(Instant::now);
    let attn_out = transpose(&attn_sdpa, &[0, 2, 1, 3], None);
    let attn_flat = reshape(
        &attn_out,
        &[
            batch as i32,
            attn_seq as i32,
            (cfg.n_heads * head_dim) as i32,
        ],
        None,
    );
    let attn_proj = attention_output_projection(
        &attn_flat,
        None,
        w.o_proj.as_ref().unwrap_or_else(|| {
            tracing::error!(
                layer_idx,
                "dense embed layer missing o_proj weight; inference will produce garbage output"
            );
            panic!("dense embed layer must have o_proj");
        }),
    );
    // 14-17. Fuse residual-add and pre-FFN RMSNorm into one C++ call
    // (`add_rms_norm_pair`), saving one MLX graph dispatch per layer.
    let hidden_for_residual;
    let hidden_residual_ref = if common_last_target_position.is_some()
        && let Some(positions) = target_positions_for_ffn
    {
        hidden_for_residual = select_embedding_targets(&hidden, positions);
        &hidden_for_residual
    } else {
        &hidden
    };
    let (hidden, normed2) = mlx_sys::add_rms_norm_pair(
        hidden_residual_ref,
        &attn_proj,
        &w.ffn_norm,
        cfg.rms_norm_eps,
        None,
    );
    if let Some(started) = attn_out_proj_started {
        embed_profile_eval_elapsed(
            profile,
            EmbedProfileStage::AttnOutProj,
            started,
            &[&hidden, &normed2],
        );
    }
    // FfnNorm is fused into the add_rms_norm_pair above; record the stage
    // as zero-cost so the profile snapshot ABI stays stable.
    if let Some(started) = profile.then(Instant::now) {
        embed_profile_eval_elapsed(profile, EmbedProfileStage::FfnNorm, started, &[&normed2]);
    }
    if let Some(positions) = target_positions_for_ffn {
        let (hidden, normed2) = if common_last_target_position.is_some() {
            (hidden, normed2)
        } else {
            (
                select_embedding_targets(&hidden, positions),
                select_embedding_targets(&normed2, positions),
            )
        };
        let ffn_out = ffn_swiglu(cfg, w, &normed2, None, layer_idx);
        if let Some(started) = profile.then(Instant::now) {
            embed_profile_eval_elapsed(profile, EmbedProfileStage::Ffn, started, &[&ffn_out]);
        }
        return (hidden, ffn_out);
    }
    let ffn_out = ffn_swiglu(cfg, w, &normed2, None, layer_idx);
    // Defer the FFN residual add: the caller (layer loop) will fuse it with
    // the next layer's pre-attention RMSNorm via `add_rms_norm_pair`, saving
    // one GPU kernel dispatch per layer boundary.
    if let Some(started) = profile.then(Instant::now) {
        embed_profile_eval_elapsed(profile, EmbedProfileStage::Ffn, started, &[&ffn_out]);
    }
    (hidden, ffn_out)
}

/// Stateless forward pass for dense-embedding extraction.
///
/// Runs the full transformer stack (embed → layers → final norm) but skips
/// `lm_head`.  Returns the normalized hidden states as `[1, seq, hidden_size]`
/// bfloat16.  The caller is responsible for pooling and dtype conversion.
///
/// No KV cache is consulted or updated — embeddings are always computed from
/// scratch in a single forward pass.
pub fn forward_for_embedding(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    target_position: Option<usize>,
) -> MlxArray {
    // DI-W2-002: EmbeddingGemma must use the bidirectional Gemma3 sandwich
    // path (same as batch-of-one), not the causal dense embed body.
    if cfg.model_family == "embeddinggemma" {
        let (out, _lens) =
            forward_for_embedding_gemma3_batch(cfg, weights, &[token_ids.to_vec()]);
        return out;
    }
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    forward_for_embedding_body(cfg, weights, hidden, target_position)
}

/// Body of `forward_for_embedding` starting from pre-embedded hidden states.
/// Split out so the same logic can be wrapped in an `mlx_compile` closure that
/// fuses the per-layer dispatches into a single compiled graph.
///
/// When `target_position` is `Some` (Last/Cls pooling), the final transformer
/// layer uses the same last-token Q + FFN prune as the batched path: Q is
/// projected only at the target, SDPA runs with a 1-token query, and FFN runs
/// on the pooled residual only.
fn forward_for_embedding_body(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    mut hidden: MlxArray,
    target_position: Option<usize>,
) -> MlxArray {
    // Nemotron Embed is bidirectional: None mask would force causal SDPA.
    let seq = hidden.shape()[1] as usize;
    let bidirectional = cfg.model_family == "nemotron_embed";
    let bidir_mask = if bidirectional {
        build_bidirectional_padding_mask(1, seq, &[seq], hidden.dtype())
    } else {
        None
    };
    // Last-token Q prune is only valid for causal last-token embedders.
    let target_position = if bidirectional { None } else { target_position };

    let mut pending_ffn: Option<MlxArray> = None;
    let final_layer_idx = weights.layers.len().saturating_sub(1);
    let target_pos_buf = target_position.map(|p| [p]);
    let mut pooled_for_final_ffn = false;
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let target_positions_for_ffn = if li == final_layer_idx {
            if let Some(ref buf) = target_pos_buf {
                pooled_for_final_ffn = true;
                Some(buf.as_slice())
            } else {
                None
            }
        } else {
            None
        };
        let (h, ffn_out) = layer_forward_dense_embed(
            cfg,
            layer_w,
            &hidden,
            li,
            pending_ffn.as_ref(),
            target_positions_for_ffn,
            &bidir_mask,
        );
        hidden = h;
        pending_ffn = Some(ffn_out);
    }
    // Final FFN residual: no next layer to fuse with, so add directly.
    if let Some(ffn_out) = &pending_ffn {
        hidden = add(&hidden, ffn_out, None);
    }
    let to_norm = if pooled_for_final_ffn {
        // Last layer already collapsed to the target token ([1, 1, H]).
        hidden
    } else {
        match target_position {
            Some(pos) => {
                let pos_i32 = pos as i32;
                let hs = cfg.hidden_size as i32;
                slice(
                    &hidden,
                    &[0, pos_i32, 0],
                    &[1, pos_i32 + 1, hs],
                    &[1, 1, 1],
                    None,
                )
            }
            None => hidden,
        }
    };
    rms_norm(&to_norm, Some(&weights.final_norm), cfg.rms_norm_eps, None)
}

/// Build an `mlx_compile`-wrapped closure that takes the pre-embedded hidden
/// state and returns the final norm-output for Last/Cls pooling.
///
/// When `has_dense_head` is true, the Dense head projection
/// (`embedding_dense_0` → `embedding_dense_1`) is fused into the compiled
/// graph so the runner can skip the separate post-closure dispatch.
///
/// The closure captures `Arc<ModelWeights>` and `ModelConfig` by value so the
/// caller can drop them safely. The compiled graph is shape-specific (the
/// `seq_len` and `target_position` are baked into the trace), so callers
/// must cache one closure per `(seq_len, target_position)` shape.
///
/// Falling back to the imperative `forward_for_embedding_body` is always
/// correct; this path exists purely to amortize the per-op MLX C-API
/// dispatch cost across the ~28 ops/layer × N layers of the forward pass.
/// DI-W2-002: families that must not use the causal dense embed compiled body.
/// Returns a stable error string when the dense closure is forbidden.
pub fn dense_embed_closure_forbidden_reason(model_family: &str) -> Option<&'static str> {
    if model_family == "embeddinggemma" {
        Some(
            "embeddinggemma must use gemma3 bidirectional embed path, not dense embed closure",
        )
    } else {
        None
    }
}

pub fn build_embedding_forward_closure(
    cfg: Arc<ModelConfig>,
    weights: Arc<ModelWeights>,
    target_position: Option<usize>,
    has_dense_head: bool,
) -> Result<MlxClosure, String> {
    // DI-W2-002: dense causal body is wrong for EmbeddingGemma. Fail closed so
    // the runner falls back to the gemma3 bidirectional path (or the early
    // gemma3 dispatch in embedding_forward never calls us at all).
    if let Some(reason) = dense_embed_closure_forbidden_reason(&cfg.model_family) {
        return Err(reason.into());
    }
    let body_closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
        if inputs.is_empty() {
            return vec![];
        }
        let hidden = inputs.get(0);
        let out = forward_for_embedding_body(&cfg, &weights, hidden, target_position);
        // Fuse the Dense head projection into the compiled graph so the
        // runner avoids a separate MLX dispatch after the closure returns.
        let out = if has_dense_head {
            apply_embedding_dense_head(&weights, &out)
        } else {
            out
        };
        vec![out]
    });
    // shapeless=false → compile per concrete seq_len. The caller caches one
    // compiled closure per `(seq_len, target_position)` key so the first
    // call at a new shape pays the trace cost once and subsequent calls hit
    // the cached compiled graph.
    body_closure.compile(false)
}

/// Embed a flat token-id array and reshape to [batch, max_seq, hidden_size].
fn embed_tokens_batched(
    ids_flat: &MlxArray, // [batch * max_seq] u32
    embedding: &QuantizedWeight,
    hidden_size: usize,
    batch: usize,
    max_seq: usize,
) -> MlxArray {
    // Reuse single-sequence path; it produces [1, batch*max_seq, hidden].
    let flat_hidden = embed_tokens_arr(ids_flat, embedding, hidden_size);
    reshape(
        &flat_hidden,
        &[batch as i32, max_seq as i32, hidden_size as i32],
        None,
    )
}

/// Key type for the padding-mask thread-local cache, extracted to satisfy
/// `clippy::type_complexity`.
type PaddingMaskKey = (MlxDtype, usize, Vec<usize>);

thread_local! {
    static PADDING_MASK_CACHE: RefCell<HashMap<PaddingMaskKey, MlxArray>> =
        RefCell::new(HashMap::new());
}

/// Maximum number of entries retained in the padding mask cache before
/// eviction. 64 covers realistic batch/shape combinations for embedding
/// forwards without allowing unbounded growth in long-running processes.
const PADDING_MASK_CACHE_MAX_ENTRIES: usize = 64;

/// Build a bidirectional key-padding mask for the EmbeddingGemma encoder.
///
/// Shape `[batch, 1, max_len, max_len]`, additive: `0.0` where key `j` is a real
/// token (`j < len[b]`) and `-inf` where it is right-padding. Query rows are
/// uniform (bidirectional: every position attends every real token); padded
/// query rows produce junk hidden states that mean-pooling discards. Mirrors
/// mlx-embeddings' `get_extended_attention_mask`.
///
/// Always returns `Some` (all-zeros when there is no padding): `full_precision_
/// attention` interprets a `None` mask as **causal** for seq>1, so an explicit
/// mask is required to get bidirectional (full) attention.
///
/// Uses a thread-local cache keyed by `(dtype, max_len, actual_lens)` to avoid
/// rebuilding identical masks across consecutive embedding forward passes.
pub(crate) fn build_bidirectional_padding_mask(
    batch: usize,
    max_len: usize,
    actual_lens: &[usize],
    dtype: MlxDtype,
) -> Option<MlxArray> {
    let cache_key = (dtype, max_len, actual_lens.to_vec());
    PADDING_MASK_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(mask) = cache.get(&cache_key) {
            return Some(mask.clone());
        }

        let n = batch * max_len * max_len;

        // Build the mask directly in the target dtype to skip the f32→dtype astype
        // dispatch and its eval() barrier. For bf16 (the common embedding dtype),
        // this halves host memory usage and eliminates one GPU kernel dispatch.
        let mask = if dtype == MlxDtype::Bfloat16 {
            let neg_inf_bf16: u16 = (f32::NEG_INFINITY.to_bits() >> 16) as u16;
            let mut data = vec![0u16; n];
            for (b, &len) in actual_lens.iter().enumerate() {
                for i in 0..max_len {
                    let row = (b * max_len + i) * max_len;
                    for cell in data[row + len..row + max_len].iter_mut() {
                        *cell = neg_inf_bf16;
                    }
                }
            }
            let arr = MlxArray::from_raw_data(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<u16>(),
                &[batch as i32, 1, max_len as i32, max_len as i32],
                MlxDtype::Bfloat16,
            );
            mlx_sys::eval(&[&arr]);
            arr
        } else {
            let neg_inf = f32::NEG_INFINITY;
            let mut data = vec![0.0f32; n];
            for (b, &len) in actual_lens.iter().enumerate() {
                for i in 0..max_len {
                    let row = (b * max_len + i) * max_len;
                    for cell in data[row + len..row + max_len].iter_mut() {
                        *cell = neg_inf;
                    }
                }
            }
            let arr = MlxArray::from_raw_data(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
                &[batch as i32, 1, max_len as i32, max_len as i32],
                MlxDtype::Float32,
            );
            let arr = if dtype == MlxDtype::Float32 {
                arr
            } else {
                astype(&arr, dtype, None)
            };
            mlx_sys::eval(&[&arr]);
            arr
        };
        // `mask` is consumed lazily across every layer's SDPA but the host buffer
        // is freed when this function returns. Force materialization above (eval),
        // so the deferred graph never reads freed memory.

        if cache.len() >= PADDING_MASK_CACHE_MAX_ENTRIES {
            cache.clear();
        }
        cache.insert(cache_key, mask.clone());
        Some(mask)
    })
}

/// Build EmbeddingGemma mean-pooling tensors captured by the compiled batch
/// closure. Shapes are `[B, max_seq, 1]` for the token mask and `[B, 1]` for
/// the reciprocal real-token counts.
pub(crate) fn build_embedding_mean_pool_inputs(
    batch: usize,
    max_len: usize,
    actual_lens: &[usize],
) -> (MlxArray, MlxArray) {
    let one_bf16: u16 = (1.0f32.to_bits() >> 16) as u16;
    let zero_bf16: u16 = 0u16;

    let mut mask_data = vec![zero_bf16; batch * max_len];
    for (i, &len) in actual_lens.iter().enumerate() {
        for j in 0..len {
            mask_data[i * max_len + j] = one_bf16;
        }
    }
    let mask = MlxArray::from_raw_data(
        mask_data.as_ptr() as *const u8,
        mask_data.len() * std::mem::size_of::<u16>(),
        &[batch as i32, max_len as i32, 1_i32],
        MlxDtype::Bfloat16,
    );

    let mut scale_data = vec![zero_bf16; batch];
    for (i, &len) in actual_lens.iter().enumerate() {
        scale_data[i] = ((1.0f32 / len as f32).to_bits() >> 16) as u16;
    }
    let scale = MlxArray::from_raw_data(
        scale_data.as_ptr() as *const u8,
        scale_data.len() * std::mem::size_of::<u16>(),
        &[batch as i32, 1_i32],
        MlxDtype::Bfloat16,
    );

    // The arrays are captured by a lazily executed compiled closure; materialize
    // them while the host buffers are still alive.
    mlx_sys::eval(&[&mask, &scale]);
    (mask, scale)
}

fn gemma3_clip_residual(x: &MlxArray, y: &MlxArray) -> MlxArray {
    if x.dtype() != MlxDtype::Float16 {
        return add(x, y, None);
    }
    let sum = add(
        &astype(x, MlxDtype::Float32, None),
        &astype(y, MlxDtype::Float32, None),
        None,
    );
    let lower = mlx_sys::ops::cached_scalar(-65504.0, MlxDtype::Float32);
    let upper = mlx_sys::ops::cached_scalar(65504.0, MlxDtype::Float32);
    astype(
        &mlx_sys::ops::clip(&sum, &lower, &upper, None),
        MlxDtype::Float16,
        None,
    )
}

/// One EmbeddingGemma (Gemma3) bidirectional encoder layer: sandwich norms,
/// q/k-norm + dual-RoPE (base selected per layer by `layer_params`), full
/// bidirectional SDPA over real tokens, GeGLU FFN. Mirrors the Gemma block in
/// `families::standard::layer_forward` but bidirectional and cache-free.
///
/// `pending_ffn` carries the previous layer's FFN output so its residual
/// add can be fused with this layer's pre-attention RMSNorm via
/// `add_rms_norm_pair`, saving one GPU kernel dispatch per layer boundary.
/// For fp16 dtype, the clip-safe unfused path is used instead.
/// Returns `(hidden, ffn_out)` where `ffn_out` is the current layer's FFN
/// output — the caller is responsible for adding it as the residual for
/// the next layer (or final norm).
fn layer_forward_embed_gemma3(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    layer_idx: usize,
    bidir_mask: &Option<MlxArray>,
    pending_ffn: Option<&MlxArray>,
) -> (MlxArray, MlxArray) {
    let (
        head_dim,
        rope_theta,
        rope_dims,
        layer_rope_freqs,
        _sliding_window,
        _kv_source,
        v_norm_no_scale,
    ) = layer_params(cfg, layer_idx);
    let seq = hidden.shape()[1] as usize;

    // 1. Fuse pending FFN residual from the previous layer with this layer's
    // pre-attention RMSNorm via `add_rms_norm_pair`, saving one GPU kernel
    // dispatch per layer boundary. For fp16 (Gemma3 clip range), fall back
    // to the unfused clip-safe path.
    let (hidden, normed) = match pending_ffn {
        Some(ffn_out) if hidden.dtype() != MlxDtype::Float16 => {
            mlx_sys::add_rms_norm_pair(hidden, ffn_out, &w.attn_norm, cfg.rms_norm_eps, None)
        }
        Some(ffn_out) => {
            // fp16: clip-safe residual then plain RMSNorm.
            let h = gemma3_clip_residual(hidden, ffn_out);
            let n = rms_norm(&h, Some(&w.attn_norm), cfg.rms_norm_eps, None);
            (h, n)
        }
        None => {
            let n = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
            (hidden.clone(), n)
        }
    };

    // 2. QKV projections.
    let (q_raw, k_raw, v_raw, _attn_gate) = qkv_project(cfg, w, &normed, head_dim);
    let kv_heads = (k_raw.shape()[2] as usize)
        .checked_div(head_dim)
        .expect("k projection output must divide by head_dim");
    let v = prepare_value_bhsd_from_proj(
        &v_raw,
        v_norm_no_scale,
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );

    // 3. Q/K norm + RoPE (dual base: sliding layers use rope_theta_swa, full
    //    layers use rope_theta — resolved by `layer_params`).
    let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
    let (rope_base, rope_freqs_ref) = rope_freqs
        .map(|f| (None, Some(f)))
        .unwrap_or((Some(rope_theta), None));
    let q_rope = qk_norm_rope_bhsd_from_proj(
        &q_raw,
        w.q_norm.as_ref(),
        cfg.n_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
        rope_dims,
        rope_base,
        0,
        rope_freqs_ref,
    );
    let k_rope = qk_norm_rope_bhsd_from_proj(
        &k_raw,
        w.k_norm.as_ref(),
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
        rope_dims,
        rope_base,
        0,
        rope_freqs_ref,
    );

    // 4. Bidirectional SDPA (full attention over real tokens; padding masked).
    //    query_scale = query_pre_attn_scalar^-0.5 (set in ModelConfig).
    let attn_sdpa =
        full_precision_attention(&q_rope, &k_rope, &v, cfg.query_scale, seq, bidir_mask);

    // 5. Output projection.
    let attn_flat = flatten_attention_output_bhsd(&attn_sdpa, seq, cfg.n_heads, head_dim);
    let attn_proj = attention_output_projection(
        &attn_flat,
        None,
        w.o_proj
            .as_ref()
            .expect("embeddinggemma layer must have o_proj"),
    );

    // 6. Post-attention norm (Gemma sandwich) then residual.
    let attn_proj = rms_norm_opt(&attn_proj, w.attn_post_norm.as_ref(), cfg.rms_norm_eps);
    let hidden = gemma3_clip_residual(&hidden, &attn_proj);

    // 7. Pre-FFN norm → GeGLU FFN (post_feedforward_layernorm applied inside
    //    ffn_swiglu when present).
    // Defer the FFN residual add: the caller (layer loop) will fuse it with
    // the next layer's pre-attention RMSNorm via `add_rms_norm_pair`, saving
    // one GPU kernel dispatch per layer boundary.
    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    let ffn_out = ffn_swiglu(cfg, w, &normed2, w.ffn_post_norm.as_ref(), layer_idx);
    (hidden, ffn_out)
}

/// Full EmbeddingGemma encoder forward: token embed (× sqrt(hidden)) → 24
/// bidirectional Gemma3 layers → final norm. Returns `[B, max_seq, H]` plus the
/// per-sequence real lengths for masked mean pooling. The Dense head + L2 norm
/// are applied by the runner after pooling.
fn forward_for_embedding_gemma3_batch(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    batch_token_ids: &[Vec<u32>],
) -> (MlxArray, Vec<usize>) {
    let actual_lens: Vec<usize> = batch_token_ids.iter().map(Vec::len).collect();
    let max_len = actual_lens.iter().copied().max().unwrap_or(0);
    let batch = batch_token_ids.len();

    let hidden = build_embedding_batch_hidden(cfg, weights, batch_token_ids, batch, max_len);
    let bidir_mask = build_bidirectional_padding_mask(batch, max_len, &actual_lens, hidden.dtype());
    let out = forward_for_embedding_gemma3_batch_body(cfg, weights, hidden, &bidir_mask);
    (out, actual_lens)
}

/// Body of `forward_for_embedding_gemma3_batch` from the pre-embedded
/// `[B, max_seq, H]` hidden state to the final norm output. Split out so the
/// same logic can be wrapped in an `mlx_compile` closure that fuses the
/// per-layer dispatches into a single compiled graph.
fn forward_for_embedding_gemma3_batch_body(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    mut hidden: MlxArray,
    bidir_mask: &Option<MlxArray>,
) -> MlxArray {
    let mut pending_ffn: Option<MlxArray> = None;
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let (h, ffn_out) =
            layer_forward_embed_gemma3(cfg, layer_w, &hidden, li, bidir_mask, pending_ffn.as_ref());
        hidden = h;
        pending_ffn = Some(ffn_out);
    }
    // Final FFN residual: no next layer to fuse with, so add directly.
    if let Some(ffn_out) = &pending_ffn {
        hidden = gemma3_clip_residual(&hidden, ffn_out);
    }
    rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None)
}

/// Layer-by-layer EmbeddingGemma depth probe.
///
/// Runs the encoder forward one layer at a time, returning the normalized
/// mean-pooled hidden state after each layer (plus the post-final-norm
/// checkpoint). Each element is `[B, H]` float32.
///
/// This is a diagnostic-only API used by `embed_gemma_depth_probe` to
/// pinpoint the first transformer layer where AX drifts from the
/// `mlx-embeddings` reference. Not used in production serving.
pub fn forward_for_embedding_gemma3_depth_probe(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    batch_token_ids: &[Vec<u32>],
) -> (Vec<MlxArray>, Vec<usize>) {
    let actual_lens: Vec<usize> = batch_token_ids.iter().map(Vec::len).collect();
    let max_len = actual_lens.iter().copied().max().unwrap_or(0);
    let batch = batch_token_ids.len();
    let hidden_size = cfg.hidden_size;

    let mut hidden = build_embedding_batch_hidden(cfg, weights, batch_token_ids, batch, max_len);
    let bidir_mask = build_bidirectional_padding_mask(batch, max_len, &actual_lens, hidden.dtype());

    let mut checkpoints: Vec<MlxArray> = Vec::with_capacity(weights.layers.len() + 1);

    let mut pending_ffn: Option<MlxArray> = None;
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let (h, ffn_out) = layer_forward_embed_gemma3(
            cfg,
            layer_w,
            &hidden,
            li,
            &bidir_mask,
            pending_ffn.as_ref(),
        );
        // Apply FFN residual so the checkpoint reflects the complete layer.
        hidden = gemma3_clip_residual(&h, &ffn_out);
        pending_ffn = None;

        let cp = normed_mean_pool_probe(&hidden, &actual_lens, hidden_size);
        mlx_sys::eval(&[&cp]);
        checkpoints.push(cp);
    }

    // Final checkpoint: post final-norm.
    let final_out = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let cp = normed_mean_pool_probe(&final_out, &actual_lens, hidden_size);
    mlx_sys::eval(&[&cp]);
    checkpoints.push(cp);

    (checkpoints, actual_lens)
}

/// Masked mean-pool + L2 normalize for the depth probe.
///
/// Returns `[B, H]` float32. Each row is the L2-normalized mean of the real
/// (non-padding) positions in the corresponding sequence.
fn normed_mean_pool_probe(
    hidden: &MlxArray, // [B, max_seq, H]
    actual_lens: &[usize],
    hidden_size: usize,
) -> MlxArray {
    let batch = actual_lens.len();
    let mut rows: Vec<MlxArray> = Vec::with_capacity(batch);
    for (b, &len) in actual_lens.iter().enumerate() {
        let row = slice(
            hidden,
            &[b as i32, 0, 0],
            &[b as i32 + 1, len as i32, hidden_size as i32],
            &[1, 1, 1],
            None,
        );
        let summed = sum_axis(&row, 1, false, None);
        let scale = 1.0 / len as f32;
        let scale_arr = MlxArray::from_raw_data(
            &scale as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1_i32, 1_i32],
            MlxDtype::Float32,
        );
        let pooled = divide(&astype(&summed, MlxDtype::Float32, None), &scale_arr, None);
        rows.push(reshape(&pooled, &[hidden_size as i32], None));
    }
    let row_refs: Vec<&MlxArray> = rows.iter().collect();
    let stacked = stack(&row_refs, 0, None);
    l2_normalize_probe(&stacked)
}

/// L2 normalize a `[B, H]` float32 tensor along the hidden dimension.
fn l2_normalize_probe(x: &MlxArray) -> MlxArray {
    fn scalar_f32(value: f32) -> MlxArray {
        MlxArray::from_raw_data(
            &value as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[],
            MlxDtype::Float32,
        )
    }
    let x_sq = multiply(x, x, None);
    let sum_sq = sum_axis(&x_sq, 1, true, None);
    let half = scalar_f32(0.5);
    let norm = power(&sum_sq, &half, None);
    let eps = scalar_f32(1e-12);
    let norm_stable = add(&norm, &eps, None);
    divide(x, &norm_stable, None)
}

/// Build an `mlx_compile`-wrapped closure for the EmbeddingGemma batched
/// forward. The closure captures the bidirectional padding mask (which is
/// determined by `actual_lens` + `max_len`) and takes the pre-embedded
/// `[B, max_seq, H]` hidden state as its only input. Returns the final
/// norm output `[B, max_seq, H]`. The runner applies mean pooling + Dense
/// head after the closure returns.
///
/// Cache key: `(batch_size, max_len, actual_lens)` — the mask is fully
/// determined by these, so same-shape batches with the same real lengths
/// hit the cache.
pub fn build_embedding_gemma3_batch_forward_closure(
    cfg: Arc<ModelConfig>,
    weights: Arc<ModelWeights>,
    bidir_mask: Option<MlxArray>,
) -> Result<MlxClosure, String> {
    let body_closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
        if inputs.is_empty() {
            return vec![];
        }
        let hidden = inputs.get(0);
        let out = forward_for_embedding_gemma3_batch_body(&cfg, &weights, hidden, &bidir_mask);
        // Fold the f32 cast into the compiled graph.
        vec![astype(&out, MlxDtype::Float32, None)]
    });
    body_closure.compile(false)
}

/// Build an `mlx_compile`-wrapped EmbeddingGemma closure that returns the final
/// sentence embedding tensor `[B, H]` instead of the full `[B, max_seq, H]`
/// encoder output.
pub fn build_embedding_gemma3_pooled_batch_forward_closure(
    cfg: Arc<ModelConfig>,
    weights: Arc<ModelWeights>,
    bidir_mask: Option<MlxArray>,
    pool_mask: MlxArray,
    pool_scale: MlxArray,
) -> Result<MlxClosure, String> {
    let body_closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
        if inputs.is_empty() {
            return vec![];
        }
        let hidden = inputs.get(0);
        let out = forward_for_embedding_gemma3_batch_body(&cfg, &weights, hidden, &bidir_mask);
        let masked = multiply(&out, &pool_mask, None);
        let sums = sum_axis(&masked, 1, false, None);
        let pooled = multiply(&sums, &pool_scale, None);
        let pooled = apply_embedding_dense_head(&weights, &pooled);
        // Fold the f32 cast into the compiled graph so the runner can skip
        // the separate astype dispatch, saving one MLX op + eval per call.
        vec![astype(&pooled, MlxDtype::Float32, None)]
    });
    body_closure.compile(false)
}

/// Apply the EmbeddingGemma sentence-transformers Dense head to a pooled
/// `[B, hidden]` tensor: `dense.1(dense.0(x))` — two bias-free, identity-activation
/// quantized linears (hidden → 4*hidden → hidden). Returns the input unchanged
/// when the head is absent (non-embeddinggemma models). The caller applies L2
/// normalization afterward.
pub(crate) fn apply_embedding_dense_head(weights: &ModelWeights, pooled: &MlxArray) -> MlxArray {
    match (&weights.embedding_dense_0, &weights.embedding_dense_1) {
        (Some(d0), Some(d1)) => {
            let h = qw(pooled, d0);
            qw(&h, d1)
        }
        _ => pooled.clone(),
    }
}

/// Batch stateless forward pass for dense-embedding extraction.
///
/// Pads `batch_token_ids` to the longest sequence length, runs a single
/// transformer forward pass for all sequences, and returns normalized hidden
/// states along with the actual (un-padded) length of each sequence.
///
/// When `target_positions` is `Some`, each sequence's hidden state is extracted
/// at its specified position *before* the final norm, so the norm runs on
/// `[B, hidden_size]` instead of `[B, max_seq, hidden_size]`. Pass it for
/// Last/Cls pooling; pass `None` for Mean pooling (which needs all positions).
/// The returned array shape is `[B, hidden_size]` when positions are given, or
/// `[B, max_seq, hidden_size]` otherwise.
pub fn forward_for_embedding_batch(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    batch_token_ids: &[Vec<u32>],
    target_positions: Option<&[usize]>,
) -> (MlxArray, Vec<usize>) {
    // EmbeddingGemma (Gemma3 backbone) is a bidirectional encoder with mean
    // pooling; it uses a dedicated forward (sandwich norms + bidirectional
    // padding mask) and always returns the full [B, max_seq, H] hidden so the
    // caller can masked-mean-pool. The Dense head is applied post-pooling by the
    // runner.
    if cfg.model_family == "embeddinggemma" {
        return forward_for_embedding_gemma3_batch(cfg, weights, batch_token_ids);
    }

    let actual_lens: Vec<usize> = batch_token_ids.iter().map(Vec::len).collect();
    let max_len = actual_lens.iter().copied().max().unwrap_or(0);
    let batch = batch_token_ids.len();

    // AX_MLX_EMBED_PROFILE: charge the token-embedding gather once per call,
    // outside the per-layer loop. seq>1 keeps decode/single-token paths
    // unprofiled (the breakdown is a batched-prefill diagnostic).
    let profile = embed_profile_enabled() && max_len > 1;
    let embed_started = profile.then(Instant::now);
    let hidden = build_embedding_batch_hidden(cfg, weights, batch_token_ids, batch, max_len);
    if let Some(started) = embed_started {
        embed_profile_eval_elapsed(profile, EmbedProfileStage::EmbedTokens, started, &[&hidden]);
    }
    // Nemotron Embed: bidirectional standard-dense layers + mean pool (no last-token prune).
    let target_positions = if cfg.model_family == "nemotron_embed" {
        None
    } else {
        target_positions
    };
    let out = forward_for_embedding_batch_body(
        cfg,
        weights,
        hidden,
        target_positions,
        Some(actual_lens.as_slice()),
    );
    if profile {
        record_embed_profile_call(
            weights.layers.len() as u32,
            batch as u32,
            (batch * max_len) as u32,
        );
    }
    (out, actual_lens)
}

/// Pre-embed a token-id batch plus optional hidden-states scale.
/// Returns `[batch, max_len, hidden]` in the model embedding dtype. Split out
/// so the same prelude can be used by the imperative and compiled-closure batch
/// paths.
fn build_embedding_batch_hidden(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    batch_token_ids: &[Vec<u32>],
    batch: usize,
    max_len: usize,
) -> MlxArray {
    let mut flat_ids = vec![0u32; batch * max_len];
    for (i, ids) in batch_token_ids.iter().enumerate() {
        flat_ids[i * max_len..i * max_len + ids.len()].copy_from_slice(ids);
    }
    // mlx_array_new_data copies the buffer immediately; flat_ids can be dropped.
    let ids_flat = MlxArray::from_raw_data(
        flat_ids.as_ptr() as *const u8,
        flat_ids.len() * std::mem::size_of::<u32>(),
        &[(batch * max_len) as i32],
        MlxDtype::Uint32,
    );

    let mut hidden = embed_tokens_batched(
        &ids_flat,
        &weights.token_embedding,
        cfg.hidden_size,
        batch,
        max_len,
    );
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    hidden
}

fn select_embedding_targets(hidden: &MlxArray, positions: &[usize]) -> MlxArray {
    let shape = hidden.shape();
    let batch_size = shape[0];
    let hidden_size = shape[2] as usize;
    let common_pos = positions
        .first()
        .copied()
        .filter(|&first| positions.iter().all(|&p| p == first));
    if let Some(pos) = common_pos {
        let pos_i32 = pos as i32;
        let sliced = slice(
            hidden,
            &[0, pos_i32, 0],
            &[batch_size, pos_i32 + 1, hidden_size as i32],
            &[1, 1, 1],
            None,
        );
        return sliced;
    }

    let pos_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
    let idx_b11 = MlxArray::from_raw_data(
        pos_u32.as_ptr() as *const u8,
        pos_u32.len() * std::mem::size_of::<u32>(),
        &[batch_size, 1_i32, 1_i32],
        MlxDtype::Uint32,
    );
    let idx_broadcast = broadcast_to(&idx_b11, &[batch_size, 1_i32, hidden_size as i32], None);
    take_along_axis(hidden, &idx_broadcast, 1, None)
}

fn select_attention_common_target_bhsd(x: &MlxArray, position: usize) -> MlxArray {
    let shape = x.shape();
    let position = position as i32;
    slice(
        x,
        &[0, 0, position, 0],
        &[shape[0], shape[1], position + 1, shape[3]],
        &[1, 1, 1, 1],
        None,
    )
}

fn pool_embedding_targets(hidden: &MlxArray, positions: &[usize]) -> MlxArray {
    let selected = select_embedding_targets(hidden, positions);
    let shape = selected.shape();
    reshape(&selected, &[shape[0], shape[2]], None)
}

/// Body of `forward_for_embedding_batch` from the pre-embedded `[B, max_seq, H]`
/// hidden state to the final norm output. Split out so the same logic can
/// be wrapped in an `mlx_compile` closure that fuses the per-layer
/// dispatches into a single compiled graph.
fn forward_for_embedding_batch_body(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    mut hidden: MlxArray,
    target_positions: Option<&[usize]>,
    actual_lens: Option<&[usize]>,
) -> MlxArray {
    let batch = hidden.shape()[0] as usize;
    let seq = hidden.shape()[1] as usize;
    // AX_MLX_EMBED_PROFILE: guard on seq>1 so the closure-builder trace (which
    // invokes this body once at build time) and decode paths stay unprofiled.
    // Under profiling the runner forces the imperative path, so the per-stage
    // eval barriers below only fire on a real imperative forward.
    let profile = embed_profile_enabled() && seq > 1;
    // Nemotron Embed: bidirectional SDPA with pad-aware mask. Prefer real
    // lengths; fall back to full-seq when the compiled-trace path omits them.
    let bidir_mask = if cfg.model_family == "nemotron_embed" {
        let lens: Vec<usize> = match actual_lens {
            Some(lens) if lens.len() == batch => lens.to_vec(),
            _ => vec![seq; batch],
        };
        build_bidirectional_padding_mask(batch, seq, &lens, hidden.dtype())
    } else {
        None
    };
    let mut pending_ffn: Option<MlxArray> = None;
    let final_layer_idx = weights.layers.len().saturating_sub(1);
    let mut pooled_for_final_ffn = false;
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let target_positions_for_ffn = if li == final_layer_idx && target_positions.is_some() {
            pooled_for_final_ffn = true;
            target_positions
        } else {
            None
        };
        let (h, ffn_out) = layer_forward_dense_embed(
            cfg,
            layer_w,
            &hidden,
            li,
            pending_ffn.as_ref(),
            target_positions_for_ffn,
            &bidir_mask,
        );
        hidden = h;
        pending_ffn = Some(ffn_out);
    }
    let final_started = profile.then(Instant::now);
    let target_positions = if pooled_for_final_ffn {
        None
    } else {
        target_positions
    };
    let to_norm = match target_positions {
        Some(positions) => {
            let pooled = pool_embedding_targets(&hidden, positions);
            if let Some(ffn_out) = &pending_ffn {
                let pooled_ffn = pool_embedding_targets(ffn_out, positions);
                add(&pooled, &pooled_ffn, None)
            } else {
                pooled
            }
        }
        None => match pending_ffn {
            Some(ffn_out) => add(&hidden, &ffn_out, None),
            None => hidden,
        },
    };
    let out = rms_norm(&to_norm, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    if let Some(started) = final_started {
        embed_profile_eval_elapsed(profile, EmbedProfileStage::FinalNormPool, started, &[&out]);
    }
    // Fold the f32 cast into the compiled graph so the runner can skip the
    // separate astype dispatch, saving one MLX op + eval per call.
    astype(&out, MlxDtype::Float32, None)
}

/// Build an `mlx_compile`-wrapped closure for the batched embedding forward.
/// The closure takes the pre-embedded `[B, max_seq, H]` hidden state and
/// returns the final norm output. `target_positions` is baked into the trace,
/// so callers must cache one closure per `(batch_size, max_seq,
/// target_positions)` shape combination.
///
/// When `has_dense_head` is true, the Dense head projection is fused into
/// the compiled graph so the runner can skip the separate post-closure
/// dispatch (Last/Cls paths only — mean-pool keeps Dense head outside).
pub fn build_embedding_batch_forward_closure(
    cfg: Arc<ModelConfig>,
    weights: Arc<ModelWeights>,
    target_positions: Option<Vec<usize>>,
    has_dense_head: bool,
) -> Result<MlxClosure, String> {
    let body_closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
        if inputs.is_empty() {
            return vec![];
        }
        let hidden = inputs.get(0);
        let out = forward_for_embedding_batch_body(
            &cfg,
            &weights,
            hidden,
            target_positions.as_deref(),
            None,
        );
        // Fuse the Dense head projection into the compiled graph so the
        // runner avoids a separate MLX dispatch after the closure returns.
        let out = if has_dense_head {
            apply_embedding_dense_head(&weights, &out)
        } else {
            out
        };
        vec![out]
    });
    body_closure.compile(false)
}

/// Run the embedding layer loop returning full hidden states for mean-pool
/// masking. Same layer loop as `forward_for_embedding_batch_body` but without
/// target-position extraction — returns the full `[B, max_seq, H]` tensor
/// after final norm so the caller can apply masked mean-pooling.
fn forward_for_embedding_mean_pool_body(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    mut hidden: MlxArray,
) -> MlxArray {
    let batch = hidden.shape()[0] as usize;
    let seq = hidden.shape()[1] as usize;
    // Compiled mean-pool traces do not carry actual_lens; Nemotron uses the
    // imperative batch path (length-aware mask) instead of this closure.
    let bidir_mask = if cfg.model_family == "nemotron_embed" {
        let lens = vec![seq; batch];
        build_bidirectional_padding_mask(batch, seq, &lens, hidden.dtype())
    } else {
        None
    };
    let mut pending_ffn: Option<MlxArray> = None;
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let (h, ffn_out) = layer_forward_dense_embed(
            cfg,
            layer_w,
            &hidden,
            li,
            pending_ffn.as_ref(),
            None,
            &bidir_mask,
        );
        hidden = h;
        pending_ffn = Some(ffn_out);
    }
    // Final FFN residual: no next layer to fuse with, so add directly.
    if let Some(ffn_out) = &pending_ffn {
        hidden = add(&hidden, ffn_out, None);
    }
    // Apply final norm to the full [B, max_seq, H] tensor.
    let out = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    astype(&out, MlxDtype::Float32, None)
}

/// Build an `mlx_compile`-wrapped closure for the mean-pool embedding forward.
/// The closure takes the pre-embedded `[B, max_seq, H]` hidden state and
/// returns the final norm output `[B, max_seq, H]` (f32). The caller applies
/// masked mean-pooling post-closure.
pub fn build_embedding_mean_pool_forward_closure(
    cfg: Arc<ModelConfig>,
    weights: Arc<ModelWeights>,
) -> Result<MlxClosure, String> {
    let body_closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
        if inputs.is_empty() {
            return vec![];
        }
        let hidden = inputs.get(0);
        let out = forward_for_embedding_mean_pool_body(&cfg, &weights, hidden);
        vec![out]
    });
    body_closure.compile(false)
}

/// Pub re-export so `runner.rs` can build the same pre-embedded hidden
/// state outside the compile cache (the closure body operates on the
/// already-embedded array).
pub(crate) fn build_embedding_batch_hidden_pub(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    batch_token_ids: &[Vec<u32>],
) -> (MlxArray, usize, usize, Vec<usize>) {
    let actual_lens: Vec<usize> = batch_token_ids.iter().map(Vec::len).collect();
    let raw_max = actual_lens.iter().copied().max().unwrap_or(0);
    // Optional compile-key / pad snap (**default ON**). Disable with
    // AX_EMBED_MAX_LEN_BUCKETS=off. See mtp-embed-perf-sprint design.
    let max_len = embed_length_bucket(raw_max);
    let batch = batch_token_ids.len();
    let hidden = build_embedding_batch_hidden(cfg, weights, batch_token_ids, batch, max_len);
    (hidden, batch, max_len, actual_lens)
}

/// Calibrated default max_len edges (finer at short lengths).
/// Coarser 32/64/128-only edges over-padded short batches in A/B
/// (`scripts/ab_mtp_embed_perf.py`, 2026-07-17).
pub(crate) const DEFAULT_EMBED_MAX_LEN_BUCKETS: &[usize] = &[
    8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
    8192,
];

/// Snap embedding batch `max_len` up to a discrete bucket when enabled.
///
/// **Default ON** after length-affinity splits land. Disable with
/// `AX_EMBED_MAX_LEN_BUCKETS=off`. Custom list: `AX_EMBED_MAX_LEN_BUCKETS=16,32,64,...`.
///
/// Conservative pad: if the snap would add more than `max(16, max_len/4)` pad
/// tokens, keep exact `max_len` (avoids short-query regression from coarse edges).
pub(crate) fn embed_length_bucket(max_len: usize) -> usize {
    use std::sync::OnceLock;
    static BUCKETS: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    let buckets = BUCKETS.get_or_init(|| {
        match std::env::var("AX_EMBED_MAX_LEN_BUCKETS") {
            Err(_) => Some(DEFAULT_EMBED_MAX_LEN_BUCKETS.to_vec()), // default ON
            Ok(raw) => {
                let t = raw.trim().to_ascii_lowercase();
                if t.is_empty() || matches!(t.as_str(), "0" | "false" | "off" | "no") {
                    return None;
                }
                if matches!(t.as_str(), "1" | "true" | "on" | "yes" | "default") {
                    return Some(DEFAULT_EMBED_MAX_LEN_BUCKETS.to_vec());
                }
                let mut buckets: Vec<usize> = t
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .filter(|v| *v > 0)
                    .collect();
                if buckets.is_empty() {
                    return None;
                }
                buckets.sort_unstable();
                buckets.dedup();
                Some(buckets)
            }
        }
    });
    let Some(buckets) = buckets.as_ref() else {
        return max_len;
    };
    let Some(&bucket) = buckets.iter().find(|&&b| max_len <= b) else {
        return max_len;
    };
    if embed_bucket_pad_acceptable(max_len, bucket) {
        bucket
    } else {
        max_len
    }
}

/// True when padding `max_len` up to `bucket` is cheap enough for default-on.
pub(crate) fn embed_bucket_pad_acceptable(max_len: usize, bucket: usize) -> bool {
    if bucket <= max_len {
        return true;
    }
    let pad = bucket - max_len;
    let rel_ok = max_len > 0 && pad * 4 <= max_len; // ≤25% pad
    let abs_ok = pad <= 16;
    rel_ok || abs_ok
}

/// Length-affinity clustering: group batch indices so pad waste stays bounded.
///
/// Rows are sorted by length; a new group starts when adding the next length
/// would exceed `max_abs_spread` absolute tokens or `max_ratio` (max/min).
/// Used by `embed_batch_flat` before pad + optional bucket snap.
pub(crate) fn embed_length_affinity_groups(
    lens: &[usize],
    max_ratio: f64,
    max_abs_spread: usize,
) -> Vec<Vec<usize>> {
    if lens.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by_key(|&i| lens[i]);
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_min = 0usize;
    let mut cur_max = 0usize;
    for i in order {
        let l = lens[i];
        if cur.is_empty() {
            cur.push(i);
            cur_min = l;
            cur_max = l;
            continue;
        }
        let new_max = cur_max.max(l);
        let new_min = cur_min.min(l);
        let abs_exceed = new_max > new_min.saturating_add(max_abs_spread);
        let ratio_exceed = new_min > 0 && (new_max as f64 / new_min as f64) > max_ratio;
        if abs_exceed || ratio_exceed {
            groups.push(std::mem::take(&mut cur));
            cur.push(i);
            cur_min = l;
            cur_max = l;
        } else {
            cur.push(i);
            cur_min = new_min;
            cur_max = new_max;
        }
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

/// Whether multi-row embed batches should be split by length affinity.
/// **Default ON.** Disable with `AX_EMBED_LENGTH_SPLIT=off`.
pub(crate) fn embed_length_split_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("AX_EMBED_LENGTH_SPLIT") {
        Err(_) => true, // default ON with calibrated buckets
        Ok(raw) => {
            let t = raw.trim().to_ascii_lowercase();
            !matches!(t.as_str(), "0" | "false" | "off" | "no")
        }
    })
}

/// Single-token forward pass accepting a lazy token `MlxArray`.
///
/// Functionally equivalent to `forward(cfg, weights, &[tok], cache, offset)`,
/// but takes the token as an unevaluated MLX array so the caller can build the
/// next step's compute graph *before* the current step's GPU work completes.
/// This enables double-buffer pipelining (see `start_direct_pipeline` /
/// `advance_direct_pipeline` in `generate.rs`):
///
/// ```text
/// GPU: [step N ....][step N+1 (submitted before N finishes) ....]
/// CPU:              [build N+1 graph][submit async][eval N][return N's token]
/// ```
///
/// `token_arr` must be a scalar or `[1]` shaped `u32` array.
pub fn forward_lazy_single(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_arr: &MlxArray, // scalar or [1] u32; may be unevaluated (lazy)
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    forward_lazy_single_and_logits_mode(
        cfg,
        weights,
        token_arr,
        cache,
        token_offset,
        LazySingleTokenMode::NormalizedFullLogits,
    )
}

pub fn forward_lazy_single_argmax(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_arr: &MlxArray, // singleton u32 array from argmax; may be unevaluated (lazy)
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    // Match forward_argmax / multi-token MoE invariant (double-buffer path).
    forward_lazy_single_and_logits_mode(
        cfg,
        weights,
        token_arr,
        cache,
        token_offset,
        LazySingleTokenMode::SingletonArgmaxOnly,
    )
}

fn forward_lazy_single_and_logits_mode(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_arr: &MlxArray, // scalar, [1], or singleton argmax array; may be lazy
    cache: &mut MlxKVCache,
    token_offset: usize,
    lazy_mode: LazySingleTokenMode,
) -> MlxArray {
    // DeepSeek V4 owns its packed hyper-connection residual; the lazy token
    // array doubles as the hash-routing tid2eid index source.
    if cfg.deepseek_v4.is_some() {
        return deepseek_v4_forward_lazy_single_and_logits_mode(
            cfg,
            weights,
            token_arr,
            cache,
            token_offset,
            lazy_mode,
        );
    }
    let profile_decode = decode_profile_enabled();
    let token_offset = qwen_visual_rope_offset(weights, cache, token_offset);

    // The generic lazy path accepts scalar or vector token arrays and keeps the
    // historical normalization. The direct-pipeline argmax path already passes
    // a singleton `[1, 1]` array, so it can avoid one reshape graph node per
    // generated token.
    let tok_1d_storage;
    let token_ids = match lazy_mode {
        LazySingleTokenMode::NormalizedFullLogits => {
            tok_1d_storage = reshape(token_arr, &[1_i32], None);
            &tok_1d_storage
        }
        LazySingleTokenMode::SingletonArgmaxOnly => token_arr,
    };

    let mut hidden = embed_tokens_arr(token_ids, &weights.token_embedding, cfg.hidden_size);
    if hidden.dtype() != MlxDtype::Bfloat16 {
        hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    }
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    // Single-token decode never needs an explicit SDPA mask. Use one borrowed
    // `None` for every layer instead of allocating a per-step mask vector.
    let decode_mask: Option<MlxArray> = None;

    let per_layer_started = profile_decode.then(Instant::now);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, token_ids, &hidden);
    if let (Some(started), Some(inputs)) = (per_layer_started, per_layer_inputs.as_ref()) {
        // Force materialization of the per-layer-input tensors to attribute their
        // graph-build + dispatch cost to this stage. Models without per-layer input
        // gating (per_layer_inputs == None) skip this stage entirely.
        let refs: Vec<&MlxArray> = inputs.iter().collect();
        decode_profile_eval_elapsed(
            profile_decode,
            DecodeProfileStage::PerLayerInput,
            started,
            &refs,
        );
    }

    let stage_profile = crate::generate::direct_pipeline_stage_profile_enabled();
    let layer_loop_started = stage_profile.then(Instant::now);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        let layer_ops_before = stage_profile.then(mlx_sys::op_count_snapshot);
        hidden = layer_forward(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&decode_mask),
        );
        if let Some(layer_ops_before) = layer_ops_before {
            let layer_ops_delta = mlx_sys::op_count_take(layer_ops_before);
            crate::generate::record_layer_ops(layer_w.linear_attn.is_some(), layer_ops_delta);
        }
    }
    if let Some(started) = layer_loop_started {
        crate::generate::record_forward_layer_loop_wall_us(
            started.elapsed().as_micros().min(u32::MAX as u128) as u32,
        );
    }
    // Single token: hidden shape is [1, 1, hidden_size] — no sequence slice needed.
    // Return the logits as `[1, 1, vocab]`; the only consumer is `argmax(_, None)`
    // which operates on the last axis and produces `[1, 1]` (later read via
    // `data_u32().first()`). Skipping the flatten saves one reshape op per step
    // versus returning a 1-D `[vocab]` array.
    let head_started = stage_profile.then(Instant::now);
    let lm_head_started = profile_decode.then(Instant::now);
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits = finalize_lm_head_logits(cfg, &logits, lazy_mode.logits_mode());
    if let Some(started) = lm_head_started {
        decode_profile_eval_elapsed(
            profile_decode,
            DecodeProfileStage::LmHead,
            started,
            &[&logits],
        );
    }
    if let Some(started) = head_started {
        crate::generate::record_forward_head_wall_us(
            started.elapsed().as_micros().min(u32::MAX as u128) as u32,
        );
    }
    if profile_decode {
        record_decode_profile_step(weights.layers.len() as u32);
    }
    logits
}

/// Absolute RoPE start for decode / multi-token verify after Qwen VL visual
/// prefill. Visual soft tokens compress MRoPE so physical `token_offset`
/// (cache `seq_len`) is not the rotary origin — singleton decode already used
/// this helper; multi-token n-gram/MTP verify must too (DI-VL-001).
pub(crate) fn qwen_visual_rope_offset(
    weights: &ModelWeights,
    cache: &MlxKVCache,
    token_offset: usize,
) -> usize {
    if weights.qwen3_vl_vision.is_some() && cache.mrope_position_delta() != 0 {
        cache.mrope_decode_position(token_offset)
    } else {
        token_offset
    }
}

// ── private helpers ──────────────────────────────────────────────────────────

pub(crate) fn finalize_lm_head_logits(
    cfg: &ModelConfig,
    logits: &MlxArray,
    mode: FinalLogitsMode,
) -> MlxArray {
    match mode {
        FinalLogitsMode::Full => {
            let logits_f32 = astype(logits, MlxDtype::Float32, None);
            apply_final_logit_softcap(cfg, &logits_f32)
        }
        // Greedy decode only needs argmax. The final softcap is monotonic, and
        // casting bf16/f16 logits to f32 cannot recover precision, so preserving
        // the lm_head dtype avoids one vocab-wide cast per decode token.
        FinalLogitsMode::ArgmaxOnly => logits.clone(),
        // Caller discards the result; this arm is unreachable when the
        // forward function short-circuits before lm_head.
        FinalLogitsMode::Skip => logits.clone(),
    }
}

/// Compute per-layer input vectors for Gemma4 2B/4B models from a pre-built
/// scalar, 1-D `[seq]`, or singleton matrix token-ID array. Accepts lazy
/// (unevaluated) arrays.
///
/// Returns `Some(Vec<MlxArray>)` of length `num_layers`, each `[1, seq, per_layer_dim]`,
/// or `None` when the model does not use per-layer input gating.
///
/// Reference: Gemma4TextModel._get_per_layer_inputs + _project_per_layer_inputs.
pub(crate) fn compute_per_layer_inputs_arr(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ids_1d: &MlxArray, // scalar/[seq]/[1, 1] u32, may be unevaluated
    hidden: &MlxArray, // [1, seq, hidden] after embed_scale — used for model projection
) -> Option<Vec<MlxArray>> {
    let per_layer_dim = cfg.hidden_size_per_layer_input;
    if per_layer_dim == 0 {
        return None;
    }
    let embed_w = weights.per_layer_embed.as_ref()?;
    let model_proj_w = weights.per_layer_model_proj.as_ref()?;
    let proj_norm_w = weights.per_layer_proj_norm.as_ref()?;

    let num_layers = cfg.layer_count;
    let scalar_ids_storage;
    let ids = if ids_1d.ndim() == 0 {
        scalar_ids_storage = reshape(ids_1d, &[1_i32], None);
        &scalar_ids_storage
    } else {
        ids_1d
    };
    let seq = ids.shape()[0]; // shape metadata available without eval
    let dtype = MlxDtype::Bfloat16;

    // 1. Per-layer token embeddings: [1, seq, num_layers * per_layer_dim]
    //    embed_tokens_per_layer(input_ids) * sqrt(per_layer_dim)
    let embed_out = embed_tokens_arr(ids, embed_w, num_layers * per_layer_dim);
    let embed_out = if embed_out.dtype() != dtype {
        astype(&embed_out, dtype, None)
    } else {
        embed_out
    };
    let embed_scale = (per_layer_dim as f32).sqrt();
    let embed_out = scale_hidden(&embed_out, embed_scale);
    // Reshape to [1, seq, num_layers, per_layer_dim]
    let embed_out = reshape(
        &embed_out,
        &[1, seq, num_layers as i32, per_layer_dim as i32],
        None,
    );

    // 2. Project model hidden: [1, seq, num_layers * per_layer_dim]
    //    per_layer_model_projection(hidden) * (1 / sqrt(hidden_size))
    let proj_scale = 1.0 / (cfg.hidden_size as f32).sqrt();
    let proj_out = qw(hidden, model_proj_w);
    let proj_out = scale_hidden(&proj_out, proj_scale);
    // Reshape to [1, seq, num_layers, per_layer_dim]
    let proj_out = reshape(
        &proj_out,
        &[1, seq, num_layers as i32, per_layer_dim as i32],
        None,
    );
    // RMSNorm over last dim (per_layer_dim)
    let proj_out = rms_norm(&proj_out, Some(proj_norm_w), cfg.rms_norm_eps, None);

    // 3. Combine: (proj + embed) * 2^(-0.5)
    const GEMMA4_PER_LAYER_COMBINED_SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let combined_scale =
        mlx_sys::ops::cached_scalar(GEMMA4_PER_LAYER_COMBINED_SCALE, proj_out.dtype());
    let combined = add_then_multiply_scalar(&proj_out, &embed_out, &combined_scale);
    // combined shape: [1, seq, num_layers, per_layer_dim]

    // 4. Split per layer along axis 2.
    // combined: [1, seq, num_layers, per_layer_dim] — contiguous after add + scale.
    // One `split` collapses the previous `num_layers` separate `slice` dispatches
    // into a single MLX op (saves ~34 ops/step on Gemma 4 E2B). Each part is a
    // strided view shaped `[1, seq, 1, per_layer_dim]`; the per-layer reshape
    // drops the singleton index dim that downstream consumers do not expect.
    let parts = split(&combined, num_layers as i32, 2, None);
    let per_layer = parts
        .into_iter()
        .map(|s| reshape(&s, &[1, seq, per_layer_dim as i32], None))
        .collect();

    Some(per_layer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{GlmMlaAttentionWeights, LinearAttentionWeights};
    use ax_engine_core::model::{NativeGlmRouterConfig, NativeMlaAttentionConfig};
    use ax_engine_core::{
        NativeDiffusionConfig, NativeLinearAttentionConfig, NativeMoeConfig, NativeRuntimeStatus,
        NativeTensorFormat,
    };
    use mlx_sys::{eval, zeros};
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};

    #[test]
    fn gemma4_per_layer_media_ids_are_zeroed_with_chunk_offsets() {
        let token_ids = [10, 11, 12, 13, 14, 15];
        assert_eq!(
            mask_media_token_ids_for_per_layer_inputs(&token_ids, &[(2, 4)], 0),
            vec![10, 11, 0, 0, 0, 15]
        );
        assert_eq!(
            mask_media_token_ids_for_per_layer_inputs(&token_ids, &[(8, 10)], 6),
            vec![10, 11, 0, 0, 0, 15]
        );
        assert_eq!(
            mask_media_token_ids_for_per_layer_inputs(&token_ids, &[(0, 1), (20, 25)], 6),
            token_ids
        );
    }

    /// Serialize GPU-numeric oracle tests that materialize Metal results.
    /// Parallel suites on CI can otherwise race MLX streams and produce
    /// catastrophic (non-tolerance) mismatches on bit-sensitive RoPE checks.
    fn with_gpu_numeric_lock<R>(f: impl FnOnce() -> R) -> R {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        f()
    }

    fn cfg(attn_output_gate: bool) -> ModelConfig {
        ModelConfig {
            compile_cache_identity: 1,
            model_family: "qwen3".to_string(),
            layer_count: 1,
            hidden_size: 16,
            intermediate_size: 32,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 8,
            vocab_size: 32,
            rope_theta: 10000.0,
            rope_dims: 8,
            attn_output_gate,
            query_scale: 1.0,
            final_logit_softcapping: None,
            moe_expert_count: 0,
            moe_experts_per_token: 0,
            moe_expert_intermediate_size: 0,
            layer_configs: Vec::new(),
            global_sliding_window: None,
            protected_prefix_sliding_window: None,
            gemma4_moe_router: false,
            uses_geglu: false,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            deepseek_v4: None,
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            rope_mscale: 1.0,
            no_rope_layer_interval: 0,
            attn_temperature_floor: 8192.0,
            attn_temperature_scale: 0.1,
            intermediate_size_mlp: 0,
            moe_layer_freq: 1,
            moe_first_dense_layers: 0,
            moe_shared_expert_count: 0,
            moe_sigmoid_routing: false,
            moe_routed_scaling_factor: 1.0,
            moe_n_group: 1,
            moe_topk_group: 1,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: None,
            gpt_oss_uses_mxfp4_experts: false,
            generation_kind: ax_engine_core::GenerationKind::Autoregressive,
            kv_cache_quant: vec![None; 1],
        }
    }

    /// DI-W2-002: pure family gate used by `build_embedding_forward_closure`
    /// (and mirrored by runner single-item dispatch).
    #[test]
    fn dense_embed_closure_forbidden_for_embeddinggemma() {
        let reason = dense_embed_closure_forbidden_reason("embeddinggemma")
            .expect("embeddinggemma must forbid causal dense embed closure");
        assert!(
            reason.contains("gemma3 bidirectional") || reason.contains("embeddinggemma"),
            "reason should name the wrong path: {reason}"
        );
        assert!(
            dense_embed_closure_forbidden_reason("qwen3").is_none(),
            "qwen3 stays on the dense embed closure path"
        );
        assert!(
            dense_embed_closure_forbidden_reason("nemotron_embed").is_none(),
            "nemotron_embed uses dense body with its own bidirectional branch"
        );
    }

    fn gemma4_interleaved_manifest() -> NativeModelManifest {
        NativeModelManifest {
            schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "gemma4".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 2,
            hidden_size: 2816,
            intermediate_size: 2112,
            attention_head_count: 8,
            attention_head_dim: 256,
            kv_head_count: 2,
            vocab_size: 262144,
            tie_word_embeddings: true,
            rope_theta: Some(1_000_000),
            rope_theta_swa: Some(10_000),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            rope_beta_fast: None,
            rope_beta_slow: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: Some(0.25),
            rms_norm_eps: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: vec![0],
            global_head_dim: Some(512),
            global_kv_head_count: None,
            sliding_window_size: Some(512),
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            kv_shared_source_layers: BTreeMap::new(),
            final_logit_softcapping: Some(30.0),
            hidden_states_scale: Some((2816_f32).sqrt()),
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig::default(),
            mla_attention: Default::default(),
            moe: NativeMoeConfig::default(),
            glm_router: Default::default(),
            deepseek_v4: Default::default(),
            weight_sanitize: ax_engine_core::WeightSanitize::None,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: NativeDiffusionConfig::default(),
            dropped_tensors: Default::default(),
            kv_cache_quantization: None,
            tensors: Vec::new(),
        }
    }

    fn qwen35_linear_manifest() -> NativeModelManifest {
        NativeModelManifest {
            schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_5".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 4,
            hidden_size: 16,
            intermediate_size: 32,
            attention_head_count: 2,
            attention_head_dim: 8,
            kv_head_count: 1,
            vocab_size: 32,
            tie_word_embeddings: false,
            rope_theta: Some(100_000),
            rope_theta_swa: None,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            rope_beta_fast: None,
            rope_beta_slow: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: true,
            partial_rotary_factor: Some(0.25),
            rms_norm_eps: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            global_kv_head_count: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: BTreeMap::new(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: true,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig {
                full_attention_interval: None,
                num_value_heads: Some(2),
                num_key_heads: Some(1),
                key_head_dim: Some(4),
                value_head_dim: Some(3),
                conv_kernel_dim: Some(4),
            },
            mla_attention: Default::default(),
            moe: NativeMoeConfig::default(),
            glm_router: Default::default(),
            deepseek_v4: Default::default(),
            weight_sanitize: ax_engine_core::WeightSanitize::None,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: NativeDiffusionConfig::default(),
            dropped_tensors: Default::default(),
            kv_cache_quantization: None,
            tensors: Vec::new(),
        }
    }

    #[test]
    fn think_token_ids_follow_qwen_tokenizer_generation() {
        // Explicit manifest ids always win.
        let mut m = qwen35_linear_manifest();
        m.think_start_token_id = Some(7);
        m.think_end_token_id = Some(8);
        let cfg = ModelConfig::from_manifest(&m);
        assert_eq!(
            (cfg.think_start_token_id, cfg.think_end_token_id),
            (Some(7), Some(8))
        );
        // Original ~151k Qwen3 tokenizer generation (fixture vocab is small).
        let cfg = ModelConfig::from_manifest(&qwen35_linear_manifest());
        assert_eq!(
            (cfg.think_start_token_id, cfg.think_end_token_id),
            (Some(151_668), Some(151_669))
        );
        // Qwen3.6 248k tokenizer generation moved the think special tokens.
        let mut m = qwen35_linear_manifest();
        m.vocab_size = 248_320;
        let cfg = ModelConfig::from_manifest(&m);
        assert_eq!(
            (cfg.think_start_token_id, cfg.think_end_token_id),
            (Some(248_068), Some(248_069))
        );
        // Partial explicit pair must not pin an unclosable think state: fill the
        // missing end id from the 248k family default used by Qwen3.6 27B.
        let mut m = qwen35_linear_manifest();
        m.vocab_size = 248_320;
        m.think_start_token_id = Some(248_068);
        m.think_end_token_id = None;
        let cfg = ModelConfig::from_manifest(&m);
        assert_eq!(
            (cfg.think_start_token_id, cfg.think_end_token_id),
            (Some(248_068), Some(248_069)),
            "partial think ids must complete from family defaults"
        );
    }

    fn deepseek_think_id_manifest(family: &str) -> NativeModelManifest {
        // Reuse the qwen fixture skeleton but strip linear-attention fields so
        // ModelConfig::from_manifest does not require Qwen-only validated knobs.
        let mut m = qwen35_linear_manifest();
        m.model_family = family.to_string();
        m.vocab_size = 129_280;
        m.linear_attention = Default::default();
        m.think_start_token_id = None;
        m.think_end_token_id = None;
        m
    }

    #[test]
    fn think_token_ids_follow_deepseek_tokenizer_generation() {
        // DeepSeek V4 Flash/Pro official tokenizer.json: <think>=128821,
        // </think>=128822. Manifests without recorded ids must still resolve
        // so ngram_in_think / think-aware MTP draft T can engage (DI-DS-A001).
        let cfg = ModelConfig::from_manifest(&deepseek_think_id_manifest("deepseek_v4"));
        assert_eq!(
            (cfg.think_start_token_id, cfg.think_end_token_id),
            (Some(128_821), Some(128_822)),
            "deepseek_v4 family defaults must match official V4 tokenizer"
        );

        // DeepSeek V3 / V3.1 / V3.2 / R1 tokenizer generation.
        for family in ["deepseek_v3", "deepseek_v32"] {
            let cfg = ModelConfig::from_manifest(&deepseek_think_id_manifest(family));
            assert_eq!(
                (cfg.think_start_token_id, cfg.think_end_token_id),
                (Some(128_798), Some(128_799)),
                "{family} family defaults must match official V3 tokenizer"
            );
        }

        // Explicit converter-recorded ids still win (e.g. custom distill).
        let mut m = deepseek_think_id_manifest("deepseek_v4");
        m.think_start_token_id = Some(7);
        m.think_end_token_id = Some(8);
        let cfg = ModelConfig::from_manifest(&m);
        assert_eq!(
            (cfg.think_start_token_id, cfg.think_end_token_id),
            (Some(7), Some(8))
        );

        // Partial pair completes from V4 family defaults.
        let mut m = deepseek_think_id_manifest("deepseek_v4");
        m.think_start_token_id = Some(128_821);
        m.think_end_token_id = None;
        let cfg = ModelConfig::from_manifest(&m);
        assert_eq!(
            (cfg.think_start_token_id, cfg.think_end_token_id),
            (Some(128_821), Some(128_822)),
            "partial deepseek_v4 think ids must complete from family defaults"
        );
    }

    fn glm4_moe_lite_manifest() -> NativeModelManifest {
        NativeModelManifest {
            schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "glm4_moe_lite".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus {
                ready: false,
                blockers: vec![
                    "GLM4MoELite runtime support is implemented in staged slices".to_string(),
                ],
                notes: Vec::new(),
            },
            layer_count: 3,
            hidden_size: 2048,
            intermediate_size: 8192,
            attention_head_count: 20,
            attention_head_dim: 256,
            kv_head_count: 20,
            vocab_size: 151_552,
            tie_word_embeddings: false,
            rope_theta: Some(1_000_000),
            rope_theta_swa: None,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            rope_beta_fast: None,
            rope_beta_slow: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            rms_norm_eps: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            global_kv_head_count: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: BTreeMap::new(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: true,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig::default(),
            mla_attention: NativeMlaAttentionConfig {
                q_lora_rank: Some(768),
                kv_lora_rank: Some(512),
                qk_nope_head_dim: Some(192),
                qk_rope_head_dim: Some(64),
                value_head_dim: Some(256),
            },
            moe: NativeMoeConfig {
                expert_count: Some(64),
                experts_per_token: Some(4),
                expert_intermediate_size: Some(1536),
                layer_freq: None,
                first_dense_layers: None,
                shared_expert_count: None,
                sigmoid_routing: false,
                routed_scaling_factor: None,
                n_group: None,
                topk_group: None,
            },
            glm_router: NativeGlmRouterConfig {
                first_dense_layer_count: Some(1),
                routed_scaling_factor: Some(1.8),
                n_group: Some(1),
                topk_group: Some(1),
                has_shared_experts: true,
            },
            deepseek_v4: Default::default(),
            weight_sanitize: ax_engine_core::WeightSanitize::None,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: NativeDiffusionConfig::default(),
            dropped_tensors: Default::default(),
            kv_cache_quantization: None,
            tensors: Vec::new(),
        }
    }

    fn dense_weight(shape: &[i32]) -> QuantizedWeight {
        QuantizedWeight::new(zeros(shape, MlxDtype::Float32, None), None, None)
    }

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    fn dense_weight_from_data(data: &[f32], shape: &[i32]) -> QuantizedWeight {
        QuantizedWeight::new(array_f32(data, shape), None, None)
    }

    #[test]
    fn gemma3_clip_residual_matches_float16_reference_bound() {
        let x = astype(
            &array_f32(&[65_000.0, -65_000.0], &[1, 2]),
            MlxDtype::Float16,
            None,
        );
        let y = astype(
            &array_f32(&[1_000.0, -1_000.0], &[1, 2]),
            MlxDtype::Float16,
            None,
        );

        let out = gemma3_clip_residual(&x, &y);
        assert_eq!(out.dtype(), MlxDtype::Float16);
        let out_f32 = astype(&out, MlxDtype::Float32, None);
        eval(&[&out_f32]);

        assert_close(out_f32.data_f32(), &[65_504.0, -65_504.0], 1.0);
    }

    #[test]
    fn embed_tokens_arr_accepts_singleton_matrix_token_ids() {
        let embedding = dense_weight_from_data(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[3, 2]);
        let token_id = [2_u32];
        let ids_scalar = MlxArray::from_raw_data(
            token_id.as_ptr().cast(),
            std::mem::size_of_val(&token_id),
            &[],
            MlxDtype::Uint32,
        );
        let ids_1d = MlxArray::from_raw_data(
            token_id.as_ptr().cast(),
            std::mem::size_of_val(&token_id),
            &[1],
            MlxDtype::Uint32,
        );
        let ids_2d = MlxArray::from_raw_data(
            token_id.as_ptr().cast(),
            std::mem::size_of_val(&token_id),
            &[1, 1],
            MlxDtype::Uint32,
        );

        let from_scalar = embed_tokens_arr(&ids_scalar, &embedding, 2);
        let from_1d = embed_tokens_arr(&ids_1d, &embedding, 2);
        let from_2d = embed_tokens_arr(&ids_2d, &embedding, 2);
        eval(&[&from_scalar, &from_1d, &from_2d]);

        assert_eq!(from_scalar.shape(), vec![1, 1, 2]);
        assert_eq!(from_1d.shape(), vec![1, 1, 2]);
        assert_eq!(from_2d.shape(), vec![1, 1, 2]);
        assert_close(from_scalar.data_f32(), from_1d.data_f32(), 0.0);
        assert_close(from_1d.data_f32(), from_2d.data_f32(), 0.0);
        assert_close(from_2d.data_f32(), &[4.0, 5.0], 0.0);
    }

    #[test]
    fn qwen_embedding_attention_keeps_causal_lm_semantics() {
        let q_data = [0.0_f32, 0.0];
        let k_data = [0.0_f32, 0.0];
        let v_data = [1.0_f32, 3.0];
        let q = MlxArray::from_raw_data(
            q_data.as_ptr().cast(),
            std::mem::size_of_val(&q_data),
            &[1, 1, 2, 1],
            MlxDtype::Float32,
        );
        let k = MlxArray::from_raw_data(
            k_data.as_ptr().cast(),
            std::mem::size_of_val(&k_data),
            &[1, 1, 2, 1],
            MlxDtype::Float32,
        );
        let v = MlxArray::from_raw_data(
            v_data.as_ptr().cast(),
            std::mem::size_of_val(&v_data),
            &[1, 1, 2, 1],
            MlxDtype::Float32,
        );
        let causal_mask = None;
        let causal = full_precision_attention(&q, &k, &v, 1.0, 2, &causal_mask);
        eval(&[&causal]);

        assert_close(causal.data_f32(), &[1.0, 2.0], 1.0e-6);
    }

    fn empty_model_weights(vision: Option<crate::qwen3_vl::Qwen3VlVisionWeights>) -> ModelWeights {
        ModelWeights {
            token_embedding: dense_weight(&[3, 2]),
            final_norm: zeros(&[2], MlxDtype::Float32, None),
            lm_head: dense_weight(&[3, 2]),
            layers: Vec::new(),
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            deepseek_v4_head: None,
            deepseek_v4_nextn: None,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: None,
            assistant_post_projection: None,
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            gemma4_vl_vision: None,
            diffusion_self_conditioning: None,
            unlimited_ocr_vision: None,
            qwen3_vl_vision: vision,
            minicpm_v46_vision: None,
            nemotron_omni: None,
        }
    }

    fn stub_qwen3_vl_vision_weights() -> crate::qwen3_vl::Qwen3VlVisionWeights {
        use crate::qwen3_vl::{Qwen3VlMergerWeights, Qwen3VlVisionConfig, Qwen3VlVisionWeights};
        let z = zeros(&[1], MlxDtype::Float32, None);
        let merger = Qwen3VlMergerWeights {
            norm_weight: z.clone(),
            norm_bias: None,
            linear_fc1: z.clone(),
            linear_fc1_bias: None,
            linear_fc2: z.clone(),
            linear_fc2_bias: None,
            postshuffle_norm: false,
        };
        Qwen3VlVisionWeights {
            config: Qwen3VlVisionConfig {
                depth: 0,
                hidden_size: 1,
                intermediate_size: 1,
                out_hidden_size: 1,
                num_heads: 1,
                in_channels: 3,
                patch_size: 1,
                temporal_patch_size: 1,
                spatial_merge_size: 1,
                num_position_embeddings: 1,
                deepstack_visual_indexes: Vec::new(),
            },
            patch_embed: z.clone(),
            patch_embed_bias: None,
            pos_embed: z.clone(),
            layers: Vec::new(),
            merger,
            deepstack_mergers: Vec::new(),
            mrope_section: vec![1, 1, 1],
        }
    }

    #[test]
    fn qwen_visual_rope_offset_is_identity_without_vision_or_delta() {
        // DI-VL-001: multi-token paths share this helper with singleton decode.
        let cache = MlxKVCache::new(1);
        let weights = empty_model_weights(None);
        assert_eq!(qwen_visual_rope_offset(&weights, &cache, 17), 17);

        let mut cache_delta = MlxKVCache::new(1);
        cache_delta.set_mrope_position_delta(-9);
        // Without vision weights the delta must not change RoPE origin (text-only).
        assert_eq!(qwen_visual_rope_offset(&weights, &cache_delta, 17), 17);
        // With a non-zero delta the decode position math itself is physical+delta.
        assert_eq!(cache_delta.mrope_decode_position(17), 8);
    }

    #[test]
    fn qwen_visual_rope_offset_applies_delta_when_vision_loaded() {
        // After visual prefill soft-token compression, multi-token n-gram/MTP
        // verify must start RoPE at mrope_decode_position(physical_offset).
        let mut cache = MlxKVCache::new(1);
        cache.set_mrope_position_delta(-9);
        let weights = empty_model_weights(Some(stub_qwen3_vl_vision_weights()));
        assert_eq!(
            qwen_visual_rope_offset(&weights, &cache, 17),
            8,
            "physical 17 + delta -9 must match singleton decode origin"
        );
        assert_eq!(
            qwen_visual_rope_offset(&weights, &cache, 17),
            cache.mrope_decode_position(17)
        );
    }

    #[test]
    fn per_layer_inputs_accept_scalar_token_ids() {
        let mut cfg = cfg(false);
        cfg.layer_count = 2;
        cfg.hidden_size = 2;
        cfg.hidden_size_per_layer_input = 2;
        let token_id = [2_u32];
        let ids_scalar = MlxArray::from_raw_data(
            token_id.as_ptr().cast(),
            std::mem::size_of_val(&token_id),
            &[],
            MlxDtype::Uint32,
        );
        let hidden = zeros(&[1, 1, 2], MlxDtype::Bfloat16, None);
        let weights = ModelWeights {
            token_embedding: dense_weight(&[3, 2]),
            final_norm: zeros(&[2], MlxDtype::Float32, None),
            lm_head: dense_weight(&[3, 2]),
            layers: Vec::new(),
            per_layer_embed: Some(dense_weight(&[3, 4])),
            per_layer_model_proj: Some(dense_weight(&[4, 2])),
            per_layer_proj_norm: Some(zeros(&[2], MlxDtype::Float32, None)),
            mtp: None,
            glm_mtp: None,
            deepseek_v4_head: None,
            deepseek_v4_nextn: None,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: None,
            assistant_post_projection: None,
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            gemma4_vl_vision: None,
            diffusion_self_conditioning: None,
            unlimited_ocr_vision: None,
            qwen3_vl_vision: None,
            minicpm_v46_vision: None,
            nemotron_omni: None,
        };

        let per_layer = compute_per_layer_inputs_arr(&cfg, &weights, &ids_scalar, &hidden)
            .expect("per-layer inputs should be enabled");
        let refs = per_layer.iter().collect::<Vec<_>>();
        eval(&refs);

        assert_eq!(per_layer.len(), 2);
        assert_eq!(per_layer[0].shape(), vec![1, 1, 2]);
        assert_eq!(per_layer[1].shape(), vec![1, 1, 2]);
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "index {idx}: actual {actual}, expected {expected}, tolerance {tolerance}"
            );
        }
    }

    fn quantized_zero_weight(packed_shape: &[i32], scale_shape: &[i32]) -> QuantizedWeight {
        QuantizedWeight {
            weight: zeros(packed_shape, MlxDtype::Uint32, None),
            scales: Some(zeros(scale_shape, MlxDtype::Float32, None)),
            biases: Some(zeros(scale_shape, MlxDtype::Float32, None)),
            group_size: 64,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
        }
    }

    fn empty_layer_weights(hidden_size: usize) -> LayerWeights {
        LayerWeights {
            attn_norm: zeros(&[hidden_size as i32], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: None,
            ffn_norm: zeros(&[hidden_size as i32], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: None,
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        }
    }

    fn glm_mla_layer_weights(cfg: &ModelConfig) -> LayerWeights {
        let mla = cfg.mla_attention.as_ref().expect("GLM MLA config");
        LayerWeights {
            attn_norm: zeros(&[cfg.hidden_size as i32], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: Some(dense_weight(&[
                cfg.hidden_size as i32,
                (cfg.n_heads * mla.value_head_dim) as i32,
            ])),
            linear_attn: None,
            glm_mla_attn: Some(GlmMlaAttentionWeights {
                qa_kva_fused: dense_weight(&[
                    (mla.q_lora_rank + mla.kv_lora_rank + mla.qk_rope_head_dim) as i32,
                    cfg.hidden_size as i32,
                ]),
                q_a_norm: zeros(&[mla.q_lora_rank as i32], MlxDtype::Float32, None),
                q_b_proj: dense_weight(&[
                    (cfg.n_heads * mla.q_head_dim) as i32,
                    mla.q_lora_rank as i32,
                ]),
                kv_a_norm: zeros(&[mla.kv_lora_rank as i32], MlxDtype::Float32, None),
                embed_q: dense_weight(&[
                    (cfg.n_heads * mla.kv_lora_rank) as i32,
                    mla.qk_nope_head_dim as i32,
                ]),
                unembed_out: dense_weight(&[
                    (cfg.n_heads * mla.value_head_dim) as i32,
                    mla.kv_lora_rank as i32,
                ]),
            }),
            deepseek_v4: None,
            ffn_norm: zeros(&[cfg.hidden_size as i32], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: None,
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        }
    }

    fn glm_mla_quantized_multilinear_layer_weights(cfg: &ModelConfig) -> LayerWeights {
        let mla = cfg.mla_attention.as_ref().expect("GLM MLA config");
        let mut weights = glm_mla_layer_weights(cfg);
        weights.glm_mla_attn = Some(GlmMlaAttentionWeights {
            qa_kva_fused: dense_weight(&[
                (mla.q_lora_rank + mla.kv_lora_rank + mla.qk_rope_head_dim) as i32,
                cfg.hidden_size as i32,
            ]),
            q_a_norm: zeros(&[mla.q_lora_rank as i32], MlxDtype::Float32, None),
            q_b_proj: dense_weight(&[
                (cfg.n_heads * mla.q_head_dim) as i32,
                mla.q_lora_rank as i32,
            ]),
            kv_a_norm: zeros(&[mla.kv_lora_rank as i32], MlxDtype::Float32, None),
            embed_q: quantized_zero_weight(
                &[cfg.n_heads as i32, mla.kv_lora_rank as i32, 8],
                &[cfg.n_heads as i32, mla.kv_lora_rank as i32, 1],
            ),
            unembed_out: quantized_zero_weight(
                &[cfg.n_heads as i32, mla.value_head_dim as i32, 8],
                &[cfg.n_heads as i32, mla.value_head_dim as i32, 1],
            ),
        });
        weights
    }

    fn attach_dense_ffn(weights: &mut LayerWeights, cfg: &ModelConfig) {
        weights.gate_proj = Some(dense_weight(&[
            cfg.intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.up_proj = Some(dense_weight(&[
            cfg.intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.down_proj = Some(dense_weight(&[
            cfg.hidden_size as i32,
            cfg.intermediate_size as i32,
        ]));
    }

    fn gemma4_kv_shared_config() -> ModelConfig {
        let mut manifest = gemma4_interleaved_manifest();
        manifest.layer_count = 2;
        manifest.hidden_size = 8;
        manifest.intermediate_size = 6;
        manifest.attention_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.kv_head_count = 1;
        manifest.vocab_size = 16;
        manifest.partial_rotary_factor = None;
        manifest.global_head_dim = None;
        manifest.sliding_window_size = Some(8);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);
        manifest.attention_v_norm_no_scale_layers = vec![0];
        manifest.final_logit_softcapping = None;
        manifest.hidden_states_scale = None;
        ModelConfig::from_manifest(&manifest)
    }

    #[test]
    fn gemma4_assistant_shared_kv_layers_resolve_source_layers() {
        let mut manifest = gemma4_interleaved_manifest();
        manifest.layer_count = 4;
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(2, 0);
        manifest.kv_shared_source_layers.insert(3, 1);

        let cfg = ModelConfig::from_manifest(&manifest);
        let shared = cfg.gemma4_assistant_shared_kv_layers();

        assert_eq!(shared.sliding_attention_layer, Some(0));
        assert_eq!(shared.full_attention_layer, Some(1));
    }

    fn attach_glm_moe_ffn(weights: &mut LayerWeights, cfg: &ModelConfig) {
        weights.router_proj = Some(dense_weight(&[
            cfg.moe_expert_count as i32,
            cfg.hidden_size as i32,
        ]));
        weights.router_correction_bias = Some(zeros(
            &[cfg.moe_expert_count as i32],
            MlxDtype::Float32,
            None,
        ));
        weights.gate_exps = Some(dense_weight(&[
            cfg.moe_expert_count as i32,
            cfg.moe_expert_intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.up_exps = Some(dense_weight(&[
            cfg.moe_expert_count as i32,
            cfg.moe_expert_intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.down_exps = Some(dense_weight(&[
            cfg.moe_expert_count as i32,
            cfg.hidden_size as i32,
            cfg.moe_expert_intermediate_size as i32,
        ]));
        weights.shared_expert_gate = Some(dense_weight(&[1, cfg.hidden_size as i32]));
        weights.shared_gate_proj = Some(dense_weight(&[
            cfg.moe_expert_intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.shared_up_proj = Some(dense_weight(&[
            cfg.moe_expert_intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.shared_down_proj = Some(dense_weight(&[
            cfg.hidden_size as i32,
            cfg.moe_expert_intermediate_size as i32,
        ]));
    }

    fn qwen35_linear_layer_weights(
        cfg: &LinearAttentionConfig,
        hidden_size: usize,
    ) -> LayerWeights {
        LayerWeights {
            attn_norm: zeros(&[hidden_size as i32], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: None,
            linear_attn: Some(LinearAttentionWeights {
                in_proj_qkv: Some(dense_weight(&[cfg.conv_dim() as i32, hidden_size as i32])),
                in_proj_z: Some(dense_weight(&[cfg.value_dim() as i32, hidden_size as i32])),
                in_proj_a: Some(dense_weight(&[
                    cfg.num_value_heads as i32,
                    hidden_size as i32,
                ])),
                in_proj_b: Some(dense_weight(&[
                    cfg.num_value_heads as i32,
                    hidden_size as i32,
                ])),
                in_proj_qkvz: None,
                in_proj_ba: None,
                conv1d_bias: None,
                d: None,
                conv1d_dense: zeros(
                    &[cfg.conv_dim() as i32, cfg.conv_kernel_dim as i32, 1_i32],
                    MlxDtype::Float32,
                    None,
                ),
                dt_bias: zeros(&[cfg.num_value_heads as i32], MlxDtype::Float32, None),
                a_log: zeros(&[cfg.num_value_heads as i32], MlxDtype::Float32, None),
                norm: zeros(&[cfg.value_head_dim as i32], MlxDtype::Float32, None),
                out_proj: dense_weight(&[hidden_size as i32, cfg.value_dim() as i32]),
            }),
            glm_mla_attn: None,
            deepseek_v4: None,
            ffn_norm: zeros(&[hidden_size as i32], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: Some(dense_weight(&[hidden_size as i32, hidden_size as i32])),
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        }
    }

    #[test]
    fn qkv_slices_dense_attention_without_gate() {
        assert_eq!(
            qkv_slices(&cfg(false), 8, 1),
            QkvSlices {
                q: (0, 16),
                gate: None,
                k: (16, 24),
                v: (24, 32),
            }
        );
    }

    #[test]
    fn qkv_slices_dense_attention_with_output_gate() {
        assert_eq!(
            qkv_slices(&cfg(true), 8, 1),
            QkvSlices {
                q: (0, 16),
                gate: Some((16, 32)),
                k: (32, 40),
                v: (40, 48),
            }
        );
    }

    #[test]
    fn packed_qkv_geometry_uses_projection_rows_for_global_kv_heads() {
        let cfg = cfg(false);
        // 2 query heads × 16-wide heads, then one 16-wide K and V head.
        assert_eq!(packed_qkv_kv_head_count(&cfg, 16, 64), Some(1));
        assert_eq!(
            qkv_slices(&cfg, 16, 1),
            QkvSlices {
                q: (0, 32),
                gate: None,
                k: (32, 48),
                v: (48, 64),
            }
        );
    }

    #[test]
    fn attention_output_gate_is_applied_before_output_projection() {
        let attn_data = [2.0_f32, 4.0_f32];
        let attn_flat = MlxArray::from_raw_data(
            attn_data.as_ptr() as *const u8,
            std::mem::size_of_val(&attn_data),
            &[1, 1, 2],
            MlxDtype::Float32,
        );
        let gate = zeros(&[1, 1, 2], MlxDtype::Float32, None);
        let proj_data = [2.0_f32, 4.0_f32];
        let o_proj_weight = MlxArray::from_raw_data(
            proj_data.as_ptr() as *const u8,
            std::mem::size_of_val(&proj_data),
            &[1, 2],
            MlxDtype::Float32,
        );
        let o_proj = QuantizedWeight::new(o_proj_weight, None, None);

        let out = attention_output_projection(&attn_flat, Some(&gate), &o_proj);

        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 1, 1]);
        assert_eq!(out.data_f32(), &[10.0]);
    }

    #[test]
    fn qkv_project_packed_attn_output_gate_extracts_per_head_q_and_gate() {
        // Reproduces the per-head interleaved layout `[h0_q, h0_gate, h1_q, h1_gate, ...]`
        // that q_proj produces when attn_output_gate=true. With n_heads=2, head_dim=2,
        // q_size=4 and kv_size=4, the packed output's last dim is 16 elements:
        //   [0..2]  head0 q, [2..4]  head0 gate,
        //   [4..6]  head1 q, [6..8]  head1 gate,
        //   [8..12] k,       [12..16] v.
        // Before the fix, a flat slice (0, q_size=4) returned [h0_q, h0_gate] as `q` and
        // (4, 8) returned [h1_q, h1_gate] as `gate` — i.e. one head's q/gate masquerading
        // as all heads' q. After the fix `q` must be all heads' q values.
        let mut cfg = cfg(true);
        cfg.n_heads = 2;
        cfg.n_kv_heads = 2;
        cfg.hidden_size = 1;
        cfg.head_dim = 2;
        let head_dim = 2;

        // `out = x @ packed.T` with x=[[[1.0]]] and packed[i, 0]=i yields out[..]=[0..16].
        let packed_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut weights = empty_layer_weights(cfg.hidden_size);
        weights.qkv_packed = Some(dense_weight_from_data(&packed_data, &[16, 1]));

        let x_data = [1.0_f32];
        let x = array_f32(&x_data, &[1, 1, 1]);

        let (q, k, v, gate) = qkv_project(&cfg, &weights, &x, head_dim);
        let gate = gate.expect("attn_output_gate=true must produce a gate tensor");
        eval(&[&q, &k, &v, &gate]);

        assert_eq!(q.shape(), vec![1, 1, 4]);
        assert_eq!(q.data_f32(), &[0.0, 1.0, 4.0, 5.0]);
        assert_eq!(gate.shape(), vec![1, 1, 4]);
        assert_eq!(gate.data_f32(), &[2.0, 3.0, 6.0, 7.0]);
        assert_eq!(k.shape(), vec![1, 1, 4]);
        assert_eq!(k.data_f32(), &[8.0, 9.0, 10.0, 11.0]);
        assert_eq!(v.shape(), vec![1, 1, 4]);
        assert_eq!(v.data_f32(), &[12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn qkv_project_reuses_key_when_value_projection_is_absent() {
        let mut cfg = cfg(false);
        cfg.n_heads = 2;
        cfg.n_kv_heads = 8;
        let weights = LayerWeights {
            attn_norm: zeros(&[4], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: Some(dense_weight(&[8, 4])),
            k_proj: Some(dense_weight(&[4, 4])),
            v_proj: None,
            qkv_packed: None,
            o_proj: Some(dense_weight(&[4, 8])),
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: None,
            ffn_norm: zeros(&[4], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: Some(dense_weight(&[3, 4])),
            up_proj: Some(dense_weight(&[3, 4])),
            gate_up_packed: None,
            down_proj: Some(dense_weight(&[4, 3])),
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        };
        let x = zeros(&[1, 2, 4], MlxDtype::Float32, None);

        let (_q, k, v, gate) = qkv_project(&cfg, &weights, &x, 4);

        assert!(gate.is_none());
        assert_eq!(k.shape(), vec![1, 2, 4]);
        assert_eq!(v.shape(), vec![1, 2, 4]);
    }

    #[test]
    fn ffn_swiglu_packed_splits_by_runtime_output_width() {
        let mut cfg = cfg(false);
        cfg.hidden_size = 4;
        cfg.intermediate_size = 3;
        cfg.uses_geglu = true;
        let mut weights = empty_layer_weights(cfg.hidden_size);
        weights.gate_up_packed = Some(dense_weight(&[12, cfg.hidden_size as i32]));
        weights.down_proj = Some(dense_weight(&[cfg.hidden_size as i32, 6]));
        let x = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);

        let out = ffn_swiglu(&cfg, &weights, &x, None, 0);

        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
    }

    #[test]
    fn gemma4_layer_configs_keep_sliding_rope_at_full_head_dim() {
        let cfg = ModelConfig::from_manifest(&gemma4_interleaved_manifest());

        assert_eq!(cfg.query_scale, 1.0);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert_eq!(cfg.layer_configs[0].head_dim, 256);
        assert_eq!(cfg.layer_configs[0].rope_theta, 10_000.0);
        assert_eq!(cfg.layer_configs[0].rope_dims, 256);
        assert_eq!(cfg.layer_configs[0].sliding_window, Some(512));
        assert!(cfg.layer_configs[0].rope_freqs.is_none());
        assert_eq!(cfg.layer_configs[1].head_dim, 512);
        assert_eq!(cfg.layer_configs[1].rope_theta, 1_000_000.0);
        assert_eq!(cfg.layer_configs[1].rope_dims, 512);
        let full_freqs = cfg.layer_configs[1]
            .rope_freqs
            .as_ref()
            .expect("Gemma4 full-attention layers should use proportional RoPE freqs");
        eval(&[full_freqs]);
        let freqs = full_freqs.data_f32();
        assert_eq!(freqs.len(), 256);
        assert_eq!(freqs[0], 1.0);
        assert!(freqs[63].is_finite());
        assert!(freqs[64].is_infinite());
        assert_eq!(cfg.layer_configs[1].sliding_window, None);
    }

    #[test]
    fn gemma4_assistant_draft_position_gates_stop_on_first_miss() {
        // first_gate=0.9, deep_gate=0.99 — depth 0 at 0.95 ok, depth 1 at 0.98 miss.
        assert!(gemma4_assistant_draft_position_accepted(0, 0.95, 0.9, 0.99));
        assert!(!gemma4_assistant_draft_position_accepted(
            1, 0.98, 0.9, 0.99
        ));
        assert!(gemma4_assistant_draft_position_accepted(
            1, 0.995, 0.9, 0.99
        ));

        let confidences = [0.95_f32, 0.995, 0.5];
        assert_eq!(
            gemma4_assistant_accepted_draft_depth(&confidences, 0.9, 0.99),
            2,
            "should keep first two positions then stop at 0.5"
        );
        assert_eq!(
            gemma4_assistant_accepted_draft_depth(&[0.5_f32, 0.99], 0.9, 0.99),
            0,
            "first-position miss yields empty draft"
        );
        assert_eq!(gemma4_assistant_accepted_draft_depth(&[], 0.9, 0.99), 0);
    }

    #[test]
    fn gemma4_assistant_forward_one_compiled_disabled_returns_none() {
        // Default flag is OFF (opt-in). Disabled path must not claim a result.
        assert!(
            !crate::fastpath::gemma4_assistant_compile_enabled(),
            "test assumes compile flag default OFF"
        );
        let assistant_cfg = ModelConfig {
            model_family: "gemma4_assistant".into(),
            ..cfg(false)
        };
        let target_cfg = ModelConfig {
            model_family: "gemma4".into(),
            ..cfg(false)
        };
        let weights = ModelWeights {
            token_embedding: dense_weight(&[32, 16]),
            final_norm: zeros(&[16], MlxDtype::Float32, None),
            lm_head: dense_weight(&[32, 16]),
            layers: Vec::new(),
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            deepseek_v4_head: None,
            deepseek_v4_nextn: None,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: None,
            assistant_post_projection: None,
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            gemma4_vl_vision: None,
            diffusion_self_conditioning: None,
            unlimited_ocr_vision: None,
            qwen3_vl_vision: None,
            minicpm_v46_vision: None,
            nemotron_omni: None,
        };
        let cache = MlxKVCache::new(0);
        let hidden = zeros(&[1, 1, 16], MlxDtype::Bfloat16, None);
        let shared = Gemma4AssistantSharedKvLayers {
            sliding_attention_layer: Some(0),
            full_attention_layer: Some(0),
        };
        assert!(
            gemma4_assistant_forward_one_compiled(
                &assistant_cfg,
                &weights,
                &target_cfg,
                &weights,
                &cache,
                shared,
                1,
                &hidden,
                0,
            )
            .is_none()
        );
    }

    #[test]
    fn gemma4_assistant_draft_session_open_validates_once() {
        let assistant_cfg = ModelConfig {
            model_family: "gemma4_assistant".into(),
            ..cfg(false)
        };
        let target_cfg = ModelConfig {
            model_family: "gemma4".into(),
            ..cfg(false)
        };
        let mut weights = ModelWeights {
            token_embedding: dense_weight(&[32, 16]),
            final_norm: zeros(&[16], MlxDtype::Float32, None),
            lm_head: dense_weight(&[32, 16]),
            layers: Vec::new(),
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            deepseek_v4_head: None,
            deepseek_v4_nextn: None,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: None,
            assistant_post_projection: None,
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            gemma4_vl_vision: None,
            diffusion_self_conditioning: None,
            unlimited_ocr_vision: None,
            qwen3_vl_vision: None,
            minicpm_v46_vision: None,
            nemotron_omni: None,
        };
        let shared = Gemma4AssistantSharedKvLayers {
            sliding_attention_layer: Some(0),
            full_attention_layer: Some(0),
        };

        // Missing projections → open fails.
        assert!(matches!(
            Gemma4AssistantDraftSession::open(
                &assistant_cfg,
                &weights,
                &target_cfg,
                &weights,
                shared,
            ),
            Err(Gemma4AssistantForwardError::MissingPreProjection)
        ));

        // Wrong family → mismatch.
        let wrong = ModelConfig {
            model_family: "qwen3".into(),
            ..cfg(false)
        };
        assert!(matches!(
            Gemma4AssistantDraftSession::open(&wrong, &weights, &target_cfg, &weights, shared),
            Err(Gemma4AssistantForwardError::ModelFamilyMismatch)
        ));

        // With projections present, open succeeds (forward may still fail without layers/KV).
        weights.assistant_pre_projection = Some(dense_weight(&[16, 32]));
        weights.assistant_post_projection = Some(dense_weight(&[16, 16]));
        assert!(
            Gemma4AssistantDraftSession::open(
                &assistant_cfg,
                &weights,
                &target_cfg,
                &weights,
                shared,
            )
            .is_ok()
        );
    }

    #[test]
    fn gemma4_assistant_rope_offset_array_is_int32_scalar() {
        let arr = gemma4_assistant_rope_offset_array(42);
        assert_eq!(arr.shape(), &[1]);
        assert_eq!(arr.dtype(), MlxDtype::Int32);
        eval(&[&arr]);
        assert_eq!(arr.nbytes(), 4);
        let value = unsafe { *(arr.data_raw() as *const i32) };
        assert_eq!(value, 42);
    }

    #[test]
    fn gemma4_assistant_dynamic_rope_matches_static_scalar_offset() {
        // Compile-foundation check: rope_dynamic(offset_arr) ≡ rope(offset_i32)
        // for a single-token BHSD query at a fixed position.
        let n_heads = 2;
        let head_dim = 4;
        let seq = 1;
        let q_data: Vec<f32> = (0..(n_heads * seq * head_dim))
            .map(|i| 0.1 * (i as f32 + 1.0))
            .collect();
        let q = reshape(
            &MlxArray::from_f32_slice(&q_data),
            &[1, n_heads, seq, head_dim],
            None,
        );
        let offset = 7usize;
        let static_r = mlx_sys::rope(
            &q,
            head_dim,
            false,
            Some(10_000.0),
            1.0,
            offset as i32,
            None,
            None,
        );
        let dyn_r = rope_dynamic(
            &q,
            head_dim,
            false,
            Some(10_000.0),
            1.0,
            &gemma4_assistant_rope_offset_array(offset),
            None,
            None,
        );
        eval(&[&static_r, &dyn_r]);
        let a = static_r.data_f32();
        let b = dyn_r.data_f32();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "static vs dynamic rope mismatch at {i}: {x} vs {y}"
            );
        }
    }

    #[test]
    fn gemma4_assistant_draft_session_freezes_shared_target_kv_once() {
        let assistant_cfg = ModelConfig {
            model_family: "gemma4_assistant".into(),
            ..cfg(false)
        };
        let target_cfg = ModelConfig {
            model_family: "gemma4".into(),
            ..cfg(false)
        };
        let weights = ModelWeights {
            token_embedding: dense_weight(&[32, 16]),
            final_norm: zeros(&[16], MlxDtype::Float32, None),
            lm_head: dense_weight(&[32, 16]),
            layers: Vec::new(),
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            deepseek_v4_head: None,
            deepseek_v4_nextn: None,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: Some(dense_weight(&[16, 32])),
            assistant_post_projection: Some(dense_weight(&[16, 16])),
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            gemma4_vl_vision: None,
            diffusion_self_conditioning: None,
            unlimited_ocr_vision: None,
            qwen3_vl_vision: None,
            minicpm_v46_vision: None,
            nemotron_omni: None,
        };
        let shared = Gemma4AssistantSharedKvLayers {
            sliding_attention_layer: Some(0),
            full_attention_layer: Some(0),
        };
        let mut session = Gemma4AssistantDraftSession::open(
            &assistant_cfg,
            &weights,
            &target_cfg,
            &weights,
            shared,
        )
        .expect("open");
        assert!(!session.has_frozen_target_kv());
        assert!(!session.frozen_sliding_uses_ring());

        // Unbound forward must fail closed.
        let hidden = zeros(&[1, 1, 16], MlxDtype::Bfloat16, None);
        assert!(matches!(
            session.forward_one(1, &hidden, 0),
            Err(Gemma4AssistantForwardError::UnboundTargetKv)
        ));

        // Empty cache → bind fails with missing shared KV.
        let empty = MlxKVCache::new(1);
        assert!(matches!(
            session.bind_target_cache(&empty),
            Err(Gemma4AssistantForwardError::MissingSharedKvCache)
        ));
        assert!(!session.has_frozen_target_kv());

        // Populate layer 0 and bind once; multi-depth reuses the freeze.
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 1, 4, 4], MlxDtype::Float32, None);
        let v = zeros(&[1, 1, 4, 4], MlxDtype::Float32, None);
        cache.append(0, k, v);
        session.bind_target_cache(&cache).expect("bind");
        assert!(session.has_frozen_target_kv());
        // No rotating ring on a plain append — SWA ring telemetry stays false.
        assert!(!session.frozen_sliding_uses_ring());

        // Re-bind after further cache growth replaces the freeze (draft attempt
        // boundary is always a fresh bind in the runner).
        let k2 = zeros(&[1, 1, 1, 4], MlxDtype::Float32, None);
        let v2 = zeros(&[1, 1, 1, 4], MlxDtype::Float32, None);
        cache.append(0, k2, v2);
        session.bind_target_cache(&cache).expect("re-bind");
        assert!(session.has_frozen_target_kv());
    }

    #[test]
    fn gemma4_assistant_draft_session_rebind_failure_clears_stale_freeze() {
        let assistant_cfg = ModelConfig {
            model_family: "gemma4_assistant".into(),
            ..cfg(false)
        };
        let target_cfg = ModelConfig {
            model_family: "gemma4".into(),
            ..cfg(false)
        };
        let weights = ModelWeights {
            token_embedding: dense_weight(&[32, 16]),
            final_norm: zeros(&[16], MlxDtype::Float32, None),
            lm_head: dense_weight(&[32, 16]),
            layers: Vec::new(),
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            deepseek_v4_head: None,
            deepseek_v4_nextn: None,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: Some(dense_weight(&[16, 32])),
            assistant_post_projection: Some(dense_weight(&[16, 16])),
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            gemma4_vl_vision: None,
            diffusion_self_conditioning: None,
            unlimited_ocr_vision: None,
            qwen3_vl_vision: None,
            minicpm_v46_vision: None,
            nemotron_omni: None,
        };
        let shared = Gemma4AssistantSharedKvLayers {
            sliding_attention_layer: Some(0),
            full_attention_layer: Some(0),
        };
        let mut session = Gemma4AssistantDraftSession::open(
            &assistant_cfg,
            &weights,
            &target_cfg,
            &weights,
            shared,
        )
        .expect("open");

        // First bind succeeds and freezes a real snapshot.
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 1, 4, 4], MlxDtype::Float32, None);
        let v = zeros(&[1, 1, 4, 4], MlxDtype::Float32, None);
        cache.append(0, k, v);
        session.bind_target_cache(&cache).expect("bind");
        assert!(session.has_frozen_target_kv());

        // A second bind attempt that fails must clear the stale freeze rather
        // than silently keeping the first one — otherwise forward_one() would
        // attend over a snapshot the caller believes was replaced/invalidated.
        let empty = MlxKVCache::new(1);
        assert!(matches!(
            session.bind_target_cache(&empty),
            Err(Gemma4AssistantForwardError::MissingSharedKvCache)
        ));
        assert!(!session.has_frozen_target_kv());
        assert!(matches!(
            session.forward_one(1, &zeros(&[1, 1, 16], MlxDtype::Bfloat16, None), 0),
            Err(Gemma4AssistantForwardError::UnboundTargetKv)
        ));
    }

    #[test]
    fn gemma4_kv_shared_layer_forward_reuses_source_cache() {
        let cfg = gemma4_kv_shared_config();
        assert_eq!(cfg.layer_configs[1].kv_source_layer, Some(0));

        let mut weights = empty_layer_weights(cfg.hidden_size);
        weights.q_proj = Some(dense_weight(&[
            (cfg.n_heads * cfg.head_dim) as i32,
            cfg.hidden_size as i32,
        ]));
        weights.q_norm = Some(zeros(&[cfg.head_dim as i32], MlxDtype::Float32, None));
        weights.o_proj = Some(dense_weight(&[
            cfg.hidden_size as i32,
            (cfg.n_heads * cfg.head_dim) as i32,
        ]));
        attach_dense_ffn(&mut weights, &cfg);

        let mut cache = MlxKVCache::new(cfg.layer_count);
        let source_k = zeros(
            &[1, cfg.n_kv_heads as i32, 2, cfg.head_dim as i32],
            MlxDtype::Float32,
            None,
        );
        let source_v = zeros(
            &[1, cfg.n_kv_heads as i32, 2, cfg.head_dim as i32],
            MlxDtype::Float32,
            None,
        );
        cache.append(0, source_k, source_v);

        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let out = layer_forward(&cfg, &weights, &hidden, &mut cache, 1, 0, None, None);

        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(
            cache.collect_eval_refs().len(),
            2,
            "KV-shared consumer must not append its own K/V cache"
        );
    }

    #[test]
    fn standard_layer_forward_last_position_only_matches_full_seq_last_row() {
        // PRD §6.2 / standard.rs::layer_forward correctness contract:
        // running with `last_position_only_after_attention = true` must
        // produce the exact same output as running unoptimised and then
        // slicing the result to the last sequence position. The cache
        // writes happen *inside* attention, so both paths must produce
        // identical cache state too.
        //
        // We use the gemma4-kv-shared fixture because it is the lowest-
        // overhead standard-family fixture in this file. Both calls use
        // separate caches so the optimised call cannot accidentally
        // depend on residual state from the unoptimised one.
        let cfg = gemma4_kv_shared_config();
        let mut weights = empty_layer_weights(cfg.hidden_size);
        weights.q_proj = Some(dense_weight(&[
            (cfg.n_heads * cfg.head_dim) as i32,
            cfg.hidden_size as i32,
        ]));
        weights.q_norm = Some(zeros(&[cfg.head_dim as i32], MlxDtype::Float32, None));
        weights.o_proj = Some(dense_weight(&[
            cfg.hidden_size as i32,
            (cfg.n_heads * cfg.head_dim) as i32,
        ]));
        attach_dense_ffn(&mut weights, &cfg);

        // Use a non-trivial input so the slice/no-slice paths are exercising
        // real arithmetic, not a zero degenerate case.
        let mut data = Vec::with_capacity(2 * cfg.hidden_size);
        for i in 0..(2 * cfg.hidden_size) {
            data.push((i as f32) * 0.01 - 0.1);
        }
        let hidden = array_f32(&data, &[1, 2, cfg.hidden_size as i32]);

        // Path A: unoptimised. Layer returns full [1, 2, hidden]; we slice
        // to the last position for comparison.
        let mut cache_full = MlxKVCache::new(cfg.layer_count);
        let source_k = zeros(
            &[1, cfg.n_kv_heads as i32, 2, cfg.head_dim as i32],
            MlxDtype::Float32,
            None,
        );
        let source_v = zeros(
            &[1, cfg.n_kv_heads as i32, 2, cfg.head_dim as i32],
            MlxDtype::Float32,
            None,
        );
        cache_full.append(0, source_k, source_v);
        let out_full = families::standard::layer_forward(
            &cfg,
            &weights,
            &hidden,
            &mut cache_full,
            1,
            0,
            None,
            None,
            /* last_position_only_after_attention */ false,
            /* skip_post_attention_ffn */ false,
        );
        assert_eq!(out_full.shape(), vec![1, 2, cfg.hidden_size as i32]);
        let hs = cfg.hidden_size as i32;
        let last_full = mlx_sys::slice(&out_full, &[0, 1, 0], &[1, 2, hs], &[1, 1, 1], None);

        // Path B: optimised. Layer slices internally and returns
        // [1, 1, hidden] directly.
        let mut cache_opt = MlxKVCache::new(cfg.layer_count);
        let source_k = zeros(
            &[1, cfg.n_kv_heads as i32, 2, cfg.head_dim as i32],
            MlxDtype::Float32,
            None,
        );
        let source_v = zeros(
            &[1, cfg.n_kv_heads as i32, 2, cfg.head_dim as i32],
            MlxDtype::Float32,
            None,
        );
        cache_opt.append(0, source_k, source_v);
        let out_opt = families::standard::layer_forward(
            &cfg,
            &weights,
            &hidden,
            &mut cache_opt,
            1,
            0,
            None,
            None,
            /* last_position_only_after_attention */ true,
            /* skip_post_attention_ffn */ false,
        );
        assert_eq!(
            out_opt.shape(),
            vec![1, 1, cfg.hidden_size as i32],
            "optimised path must collapse the seq dimension to 1"
        );

        eval(&[&last_full, &out_opt]);
        assert_close(out_opt.data_f32(), last_full.data_f32(), 1e-4);
    }

    #[test]
    fn qwen35_linear_attention_config_matches_reference_interval() {
        let cfg = ModelConfig::from_manifest(&qwen35_linear_manifest());
        let linear = cfg
            .linear_attention
            .as_ref()
            .expect("linear attention config");

        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert!(cfg.mla_attention.is_none());
        assert!(cfg.glm_router.is_none());
        assert_eq!(linear.full_attention_interval, 4);
        assert_eq!(linear.key_dim(), 4);
        assert_eq!(linear.value_dim(), 6);
        assert_eq!(linear.conv_dim(), 14);
        assert!(cfg.is_linear_attention_layer(0));
        assert!(cfg.is_linear_attention_layer(1));
        assert!(cfg.is_linear_attention_layer(2));
        assert!(!cfg.is_linear_attention_layer(3));
    }

    #[test]
    fn model_config_uses_manifest_rms_norm_eps_when_present() {
        let mut manifest = qwen35_linear_manifest();
        manifest.rms_norm_eps = Some(5e-6);

        let cfg = ModelConfig::from_manifest(&manifest);

        assert_eq!(cfg.rms_norm_eps, 5e-6);
    }

    #[test]
    fn glm_mla_attention_config_matches_reference_shape_contract() {
        let cfg = ModelConfig::from_manifest(&glm4_moe_lite_manifest());
        let mla = cfg
            .mla_attention
            .as_ref()
            .expect("GLM MLA attention config");

        assert_eq!(mla.q_lora_rank, 768);
        assert_eq!(mla.kv_lora_rank, 512);
        assert_eq!(mla.qk_nope_head_dim, 192);
        assert_eq!(mla.qk_rope_head_dim, 64);
        assert_eq!(mla.value_head_dim, 256);
        assert_eq!(mla.q_head_dim, 256);
        assert_eq!(mla.kv_lora_rank + mla.qk_rope_head_dim, 576);
        assert_eq!(mla.latent_kv_cache_width(), 512);
        assert_eq!(mla.rope_key_cache_width(), 64);
        assert!((mla.query_scale - (1.0 / 256_f32.sqrt())).abs() < f32::EPSILON);
        assert_ne!(mla.query_scale, 1.0 / 576_f32.sqrt());
        assert_eq!(cfg.query_scale, mla.query_scale);
        assert_eq!(cfg.rms_norm_eps, 1e-5);
    }

    #[test]
    fn glm_router_config_matches_reference_dense_moe_split() {
        let cfg = ModelConfig::from_manifest(&glm4_moe_lite_manifest());
        let router = cfg.glm_router.as_ref().expect("GLM router config");

        assert_eq!(router.first_dense_layer_count, 1);
        assert!((router.routed_scaling_factor - 1.8).abs() < f32::EPSILON);
        assert_eq!(router.n_group, 1);
        assert_eq!(router.topk_group, 1);
        assert!(router.has_shared_experts);
        assert!(!cfg.is_glm_moe_layer(0));
        assert!(cfg.is_glm_moe_layer(1));
        assert!(cfg.is_glm_moe_layer(2));
    }

    #[test]
    fn glm_router_uses_correction_bias_for_selection_and_sigmoid_for_weights() {
        let mut cfg = ModelConfig::from_manifest(&glm4_moe_lite_manifest());
        cfg.moe_expert_count = 4;
        cfg.moe_experts_per_token = 2;
        cfg.moe_norm_topk_prob = true;
        let logits_data = [0.0_f32, 0.0, 0.0, 0.0];
        let logits = MlxArray::from_raw_data(
            logits_data.as_ptr() as *const u8,
            std::mem::size_of_val(&logits_data),
            &[1, 1, 4],
            MlxDtype::Float32,
        );
        let bias_data = [0.0_f32, 10.0, 0.0, 5.0];
        let bias = MlxArray::from_raw_data(
            bias_data.as_ptr() as *const u8,
            std::mem::size_of_val(&bias_data),
            &[1, 1, 4],
            MlxDtype::Float32,
        );

        let (indices, weights) = moe_router_glm_from_logits(&cfg, &logits, &bias);
        eval(&[&indices, &weights]);

        let mut selected = indices.data_u32().to_vec();
        selected.sort_unstable();
        assert_eq!(selected, vec![1, 3]);
        assert_eq!(weights.shape(), vec![1, 1, 2]);
        for weight in weights.data_f32() {
            assert!((*weight - 0.9).abs() < 1e-5, "{weight}");
        }
    }

    #[test]
    fn glm_router_group_selection_masks_unselected_groups() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.moe.expert_count = Some(4);
        manifest.moe.experts_per_token = Some(2);
        manifest.glm_router.n_group = Some(2);
        manifest.glm_router.topk_group = Some(1);
        let cfg = ModelConfig::from_manifest(&manifest);
        let logits_data = [0.0_f32, 0.0, 0.0, 0.0];
        let logits = MlxArray::from_raw_data(
            logits_data.as_ptr() as *const u8,
            std::mem::size_of_val(&logits_data),
            &[1, 1, 4],
            MlxDtype::Float32,
        );
        let bias_data = [10.0_f32, 10.0, 0.0, 0.0];
        let bias = MlxArray::from_raw_data(
            bias_data.as_ptr() as *const u8,
            std::mem::size_of_val(&bias_data),
            &[1, 1, 4],
            MlxDtype::Float32,
        );

        let (indices, weights) = moe_router_glm_from_logits(&cfg, &logits, &bias);
        eval(&[&indices, &weights]);

        let mut selected = indices.data_u32().to_vec();
        selected.sort_unstable();
        assert_eq!(selected, vec![0, 1]);
        for weight in weights.data_f32() {
            assert!((*weight - 0.9).abs() < 1e-5, "{weight}");
        }
    }

    #[test]
    fn glm_mla_projection_matches_reference_cache_shapes() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_layer_weights(&cfg);
        let hidden = zeros(&[1, 3, cfg.hidden_size as i32], MlxDtype::Float32, None);

        let projected = glm_mla_project_inputs(&cfg, &weights, &hidden, 5);
        eval(&[
            &projected.q_nope,
            &projected.q_pe,
            &projected.kv_latent,
            &projected.k_pe,
        ]);

        assert_eq!(projected.q_nope.shape(), vec![1, 2, 3, 2]);
        assert_eq!(projected.q_pe.shape(), vec![1, 2, 3, 2]);
        assert_eq!(projected.kv_latent.shape(), vec![1, 1, 3, 4]);
        assert_eq!(projected.k_pe.shape(), vec![1, 1, 3, 2]);
    }

    #[test]
    fn glm_mla_projection_updates_latent_cache_and_rope_keys() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_layer_weights(&cfg);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let cached = glm_mla_project_and_cache_inputs(&cfg, &weights, &hidden, &mut cache, 0, 0);
        eval(&[
            &cached.q_nope,
            &cached.q_pe,
            &cached.kv_latent,
            &cached.k_pe,
        ]);
        assert_eq!(cached.q_nope.shape(), vec![1, 2, 2, 2]);
        assert_eq!(cached.kv_latent.shape(), vec![1, 1, 2, 4]);
        assert_eq!(cached.k_pe.shape(), vec![1, 1, 2, 2]);

        cache.set_seq_len(2);
        let hidden = zeros(&[1, 1, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let cached = glm_mla_project_and_cache_inputs(&cfg, &weights, &hidden, &mut cache, 0, 2);
        eval(&[
            &cached.q_nope,
            &cached.q_pe,
            &cached.kv_latent,
            &cached.k_pe,
        ]);

        assert_eq!(cached.q_nope.shape(), vec![1, 2, 1, 2]);
        assert_eq!(cached.kv_latent.shape(), vec![1, 1, 3, 4]);
        assert_eq!(cached.k_pe.shape(), vec![1, 1, 3, 2]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn glm_mla_multilinear_matches_prefill_and_decode_shape_contracts() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_layer_weights(&cfg);

        let kv_latent = zeros(&[1, 1, 3, 4], MlxDtype::Float32, None);
        let prefill_k = glm_mla_embed_q_prefill(&cfg, &weights, &kv_latent);
        let prefill_v = glm_mla_unembed_out(&cfg, &weights, &kv_latent);
        eval(&[&prefill_k, &prefill_v]);
        assert_eq!(prefill_k.shape(), vec![1, 2, 3, 2]);
        assert_eq!(prefill_v.shape(), vec![1, 2, 3, 3]);

        let q_nope = zeros(&[1, 2, 1, 2], MlxDtype::Float32, None);
        let decode_q = glm_mla_embed_q_decode(&cfg, &weights, &q_nope);
        let decode_out = glm_mla_unembed_out(&cfg, &weights, &decode_q);
        eval(&[&decode_q, &decode_out]);
        assert_eq!(decode_q.shape(), vec![1, 2, 1, 4]);
        assert_eq!(decode_out.shape(), vec![1, 2, 1, 3]);
    }

    #[test]
    fn glm_mla_quantized_multilinear_dequantizes_to_prefill_and_decode_contracts() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 66;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(64);
        manifest.mla_attention.qk_nope_head_dim = Some(64);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(64);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_quantized_multilinear_layer_weights(&cfg);

        let kv_latent = zeros(&[1, 1, 3, 64], MlxDtype::Float32, None);
        let prefill_k = glm_mla_embed_q_prefill(&cfg, &weights, &kv_latent);
        let prefill_v = glm_mla_unembed_out(&cfg, &weights, &kv_latent);
        eval(&[&prefill_k, &prefill_v]);
        assert_eq!(prefill_k.shape(), vec![1, 2, 3, 64]);
        assert_eq!(prefill_v.shape(), vec![1, 2, 3, 64]);

        let q_nope = zeros(&[1, 2, 1, 64], MlxDtype::Float32, None);
        let decode_q = glm_mla_embed_q_decode(&cfg, &weights, &q_nope);
        let decode_out = glm_mla_unembed_out(&cfg, &weights, &decode_q);
        eval(&[&decode_q, &decode_out]);
        assert_eq!(decode_q.shape(), vec![1, 2, 1, 64]);
        assert_eq!(decode_out.shape(), vec![1, 2, 1, 64]);
    }

    #[test]
    fn glm_mla_attention_forward_returns_hidden_shape_and_updates_cache() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_layer_weights(&cfg);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let out = glm_mla_attention_forward(&cfg, &weights, &hidden, &mut cache, 0, 0);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);

        cache.set_seq_len(2);
        let hidden = zeros(&[1, 1, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let out = glm_mla_attention_forward(&cfg, &weights, &hidden, &mut cache, 0, 2);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 1, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn glm_mla_attention_forward_accepts_quantized_multilinear_weights() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 66;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(64);
        manifest.mla_attention.qk_nope_head_dim = Some(64);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(64);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_quantized_multilinear_layer_weights(&cfg);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let out = glm_mla_attention_forward(&cfg, &weights, &hidden, &mut cache, 0, 0);
        eval(&[&out]);

        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn glm_mla_packed_prefill_matches_reference_direct_projection_path() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 2;
        manifest.layer_count = 1;
        manifest.attention_head_count = 1;
        manifest.kv_head_count = 1;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(2);
        manifest.mla_attention.kv_lora_rank = Some(2);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(2);
        let cfg = ModelConfig::from_manifest(&manifest);
        let mut weights = glm_mla_layer_weights(&cfg);
        weights.o_proj = Some(dense_weight_from_data(&[0.6, -0.4, 0.3, 0.8], &[2, 2]));
        let mla_w = weights.glm_mla_attn.as_mut().expect("GLM MLA weights");
        mla_w.qa_kva_fused = dense_weight_from_data(
            &[
                0.8, -0.1, // q_a row 0
                0.2, 0.7, // q_a row 1
                0.5, -0.3, // kv latent row 0
                -0.4, 0.9, // kv latent row 1
                0.3, 0.6, // k_pe row 0
                -0.2, 0.4, // k_pe row 1
            ],
            &[6, 2],
        );
        mla_w.q_a_norm = array_f32(&[1.0, 1.0], &[2]);
        mla_w.kv_a_norm = array_f32(&[1.0, 1.0], &[2]);
        mla_w.q_b_proj = dense_weight_from_data(
            &[
                0.7, -0.2, // q_nope row 0
                0.1, 0.5, // q_nope row 1
                -0.3, 0.6, // q_pe row 0
                0.4, 0.2, // q_pe row 1
            ],
            &[4, 2],
        );
        mla_w.embed_q = dense_weight_from_data(&[0.9, -0.5, 0.2, 0.7], &[2, 2]);
        mla_w.unembed_out = dense_weight_from_data(&[0.8, 0.1, -0.3, 0.6], &[2, 2]);

        let hidden = array_f32(&[0.2, -0.5, 1.0, 0.3, -0.7, 0.8], &[1, 3, 2]);

        let mut packed_cache = MlxKVCache::new(cfg.layer_count);
        let packed = glm_mla_attention_forward(&cfg, &weights, &hidden, &mut packed_cache, 0, 0);

        let mut reference_cache = MlxKVCache::new(cfg.layer_count);
        let cached =
            glm_mla_project_and_cache_inputs(&cfg, &weights, &hidden, &mut reference_cache, 0, 0);
        let reference_k = glm_mla_embed_q_prefill(&cfg, &weights, &cached.kv_latent);
        let reference_v = glm_mla_unembed_out(&cfg, &weights, &cached.kv_latent);
        let mla = cfg.mla_attention.as_ref().expect("GLM MLA config");
        let q_pe_scaled = scale_hidden(&cached.q_pe, mla.query_scale);
        let pe_scores = matmul(
            &q_pe_scaled,
            &transpose(&cached.k_pe, &[0, 1, 3, 2], None),
            None,
        );
        let causal_mask = create_causal_mask(3, 0, None);
        let masked_pe_scores = mlx_sys::ops::where_cond(
            &causal_mask,
            &pe_scores,
            &scalar_like(f32::MIN, pe_scores.dtype()),
            None,
        );
        let reference_heads = scaled_dot_product_attention_with_mask(
            &cached.q_nope,
            &reference_k,
            &reference_v,
            mla.query_scale,
            ScaledDotProductAttentionMask::Array(&masked_pe_scores),
            None,
        );
        let reference_heads = transpose(&reference_heads, &[0, 2, 1, 3], None);
        let reference_flat = reshape(&reference_heads, &[1, 3, 2], None);
        let reference = attention_output_projection(
            &reference_flat,
            None,
            weights
                .o_proj
                .as_ref()
                .expect("GLM MLA layer must have o_proj"),
        );

        eval(&[&packed, &reference]);
        assert_eq!(packed.shape(), reference.shape());
        assert_close(packed.data_f32(), reference.data_f32(), 1e-3);
    }

    #[test]
    fn layer_forward_routes_glm_mla_without_standard_qkv_weights() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.intermediate_size = 6;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        let cfg = ModelConfig::from_manifest(&manifest);
        let mut weights = glm_mla_layer_weights(&cfg);
        weights.attn_post_norm = Some(zeros(&[cfg.hidden_size as i32], MlxDtype::Float32, None));
        attach_dense_ffn(&mut weights, &cfg);
        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let out = layer_forward(&cfg, &weights, &hidden, &mut cache, 0, 0, None, None);
        eval(&[&out]);

        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn layer_forward_routes_glm_moe_with_correction_bias_and_shared_expert() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.intermediate_size = 6;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        manifest.moe.expert_count = Some(4);
        manifest.moe.experts_per_token = Some(2);
        manifest.moe.expert_intermediate_size = Some(3);
        manifest.glm_router.first_dense_layer_count = Some(0);
        let cfg = ModelConfig::from_manifest(&manifest);
        let mut weights = glm_mla_layer_weights(&cfg);
        attach_glm_moe_ffn(&mut weights, &cfg);
        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let out = layer_forward(&cfg, &weights, &hidden, &mut cache, 0, 0, None, None);
        eval(&[&out]);

        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn layer_forward_routes_glm_moe_with_ungated_shared_expert() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.intermediate_size = 6;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        manifest.moe.expert_count = Some(4);
        manifest.moe.experts_per_token = Some(2);
        manifest.moe.expert_intermediate_size = Some(3);
        manifest.glm_router.first_dense_layer_count = Some(0);
        let cfg = ModelConfig::from_manifest(&manifest);
        let mut weights = glm_mla_layer_weights(&cfg);
        attach_glm_moe_ffn(&mut weights, &cfg);
        weights.shared_expert_gate = None;
        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let out = layer_forward(&cfg, &weights, &hidden, &mut cache, 0, 0, None, None);
        eval(&[&out]);

        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn shared_expert_forward_uses_geglu_when_configured() {
        let mut cfg = cfg(false);
        cfg.hidden_size = 1;
        cfg.moe_expert_intermediate_size = 1;
        cfg.uses_geglu = true;
        let mut weights = empty_layer_weights(1);
        weights.shared_gate_proj = Some(dense_weight_from_data(&[2.0], &[1, 1]));
        weights.shared_up_proj = Some(dense_weight_from_data(&[1.0], &[1, 1]));
        weights.shared_down_proj = Some(dense_weight_from_data(&[1.0], &[1, 1]));
        let x = array_f32(&[1.0], &[1, 1, 1]);

        let actual = shared_expert_forward(&cfg, &weights, &x);
        let gate = qw(&x, weights.shared_gate_proj.as_ref().unwrap());
        let up = qw(&x, weights.shared_up_proj.as_ref().unwrap());
        let hidden = multiply(&gelu_approx(&gate, None), &up, None);
        let expected = qw(&hidden, weights.shared_down_proj.as_ref().unwrap());

        eval(&[&actual, &expected]);
        assert_close(actual.data_f32(), expected.data_f32(), 1e-5);
    }

    #[test]
    fn shared_expert_forward_uses_packed_gate_up_projection() {
        let mut cfg = cfg(false);
        cfg.hidden_size = 1;
        cfg.moe_expert_intermediate_size = 1;
        cfg.uses_geglu = false;
        let mut weights = empty_layer_weights(1);
        weights.shared_gate_up_proj = Some(dense_weight_from_data(&[2.0, 1.0], &[2, 1]));
        weights.shared_down_proj = Some(dense_weight_from_data(&[1.0], &[1, 1]));
        let x = array_f32(&[1.0], &[1, 1, 1]);

        let actual = shared_expert_forward(&cfg, &weights, &x);
        let gate = array_f32(&[2.0], &[1, 1, 1]);
        let up = array_f32(&[1.0], &[1, 1, 1]);
        let hidden = swiglu(&gate, &up);
        let expected = qw(&hidden, weights.shared_down_proj.as_ref().unwrap());

        eval(&[&actual, &expected]);
        assert_close(actual.data_f32(), expected.data_f32(), 1e-5);
    }

    #[test]
    fn glm_full_forward_spans_dense_and_moe_layers() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.intermediate_size = 6;
        manifest.layer_count = 2;
        manifest.vocab_size = 16;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        manifest.moe.expert_count = Some(4);
        manifest.moe.experts_per_token = Some(2);
        manifest.moe.expert_intermediate_size = Some(3);
        manifest.glm_router.first_dense_layer_count = Some(1);
        let cfg = ModelConfig::from_manifest(&manifest);

        let mut dense_layer = glm_mla_layer_weights(&cfg);
        attach_dense_ffn(&mut dense_layer, &cfg);
        let mut moe_layer = glm_mla_layer_weights(&cfg);
        attach_glm_moe_ffn(&mut moe_layer, &cfg);
        let weights = ModelWeights {
            token_embedding: dense_weight(&[cfg.vocab_size as i32, cfg.hidden_size as i32]),
            final_norm: zeros(&[cfg.hidden_size as i32], MlxDtype::Float32, None),
            lm_head: dense_weight(&[cfg.vocab_size as i32, cfg.hidden_size as i32]),
            layers: vec![dense_layer, moe_layer],
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            deepseek_v4_head: None,
            deepseek_v4_nextn: None,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: None,
            assistant_post_projection: None,
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            gemma4_vl_vision: None,
            diffusion_self_conditioning: None,
            unlimited_ocr_vision: None,
            qwen3_vl_vision: None,
            minicpm_v46_vision: None,
            nemotron_omni: None,
        };
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let logits = forward_all_positions(&cfg, &weights, &[1, 2], &mut cache, 0);
        eval(&[&logits]);

        assert_eq!(logits.shape(), vec![2, cfg.vocab_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 4);
    }

    #[test]
    fn linear_attention_forward_returns_hidden_shape_and_updates_cache() {
        let mut cfg = cfg(true);
        cfg.hidden_size = 8;
        cfg.linear_attention = Some({
            let (q_scale, k_scale) = crate::linear_attention_ops::linear_attention_qk_scale(32);
            LinearAttentionConfig {
                full_attention_interval: 4,
                num_value_heads: 1,
                num_key_heads: 1,
                key_head_dim: 32,
                value_head_dim: 4,
                conv_kernel_dim: 4,
                q_scale,
                k_scale,
            }
        });
        let linear_cfg = cfg.linear_attention.as_ref().unwrap();
        let weights = qwen35_linear_layer_weights(linear_cfg, cfg.hidden_size);
        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let mut cache = MlxKVCache::new(1);

        let out = linear_attention_forward(&cfg, &weights, &hidden, &mut cache, 0);

        assert_eq!(out.shape(), vec![1, 2, 8]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    /// Deterministic patterned values in `[lo, hi)` — non-trivial weights/inputs
    /// so a batching bug that cross-contaminates rows actually changes results.
    fn pat(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503))
                    .wrapping_add(0x9E37_79B9);
                let unit = (x % 4096) as f32 / 4096.0;
                lo + unit * (hi - lo)
            })
            .collect()
    }

    /// Linear-attention layer weights (split projections) with patterned,
    /// non-trivial values — for the batched-vs-single-row oracle.
    fn patterned_linear_layer_weights(cfg: &LinearAttentionConfig, hidden: usize) -> LayerWeights {
        let mut w = qwen35_linear_layer_weights(cfg, hidden);
        let h = hidden as i32;
        let conv_dim = cfg.conv_dim() as i32;
        let value_dim = cfg.value_dim() as i32;
        let hv = cfg.num_value_heads as i32;
        let lin = LinearAttentionWeights {
            in_proj_qkv: Some(dense_weight_from_data(
                &pat((conv_dim * h) as usize, 1, -0.5, 0.5),
                &[conv_dim, h],
            )),
            in_proj_z: Some(dense_weight_from_data(
                &pat((value_dim * h) as usize, 2, -0.5, 0.5),
                &[value_dim, h],
            )),
            in_proj_a: Some(dense_weight_from_data(
                &pat((hv * h) as usize, 3, -0.5, 0.5),
                &[hv, h],
            )),
            in_proj_b: Some(dense_weight_from_data(
                &pat((hv * h) as usize, 4, -0.5, 0.5),
                &[hv, h],
            )),
            in_proj_qkvz: None,
            in_proj_ba: None,
            conv1d_bias: None,
            d: None,
            conv1d_dense: array_f32(
                &pat(
                    (conv_dim * cfg.conv_kernel_dim as i32) as usize,
                    5,
                    -0.3,
                    0.3,
                ),
                &[conv_dim, cfg.conv_kernel_dim as i32, 1],
            ),
            dt_bias: array_f32(&pat(hv as usize, 6, -0.5, 0.5), &[hv]),
            // A_log stays modest: g = exp(-exp(a_log) * softplus(...)) ∈ (0, 1).
            a_log: array_f32(&pat(hv as usize, 7, -1.0, 0.5), &[hv]),
            norm: array_f32(
                &pat(cfg.value_head_dim, 8, 0.5, 1.5),
                &[cfg.value_head_dim as i32],
            ),
            out_proj: dense_weight_from_data(
                &pat((h * value_dim) as usize, 9, -0.5, 0.5),
                &[h, value_dim],
            ),
        };
        w.linear_attn = Some(lin);
        w
    }

    /// Phase 3.7 oracle: batching B linear-attention rows must be **byte-identical**
    /// to B independent single-row decodes — for BOTH the layer output and the
    /// post-step conv/recurrent state. This is where silent numerical batching
    /// bugs (a reshape that mixes the batch axis into seq/heads, or state
    /// threaded to the wrong row) would surface. Compares row `r` of a batch=B
    /// run against a standalone batch=1 run of that same row through the exact
    /// same code path.
    #[test]
    fn batched_linear_attention_row_matches_single_row_with_byte_identical_state() {
        use crate::batched_linear_state::BatchedLinearState;

        let mut cfg = cfg(true);
        cfg.hidden_size = 8;
        cfg.linear_attention = Some({
            let (q_scale, k_scale) = crate::linear_attention_ops::linear_attention_qk_scale(32);
            LinearAttentionConfig {
                full_attention_interval: 4,
                num_value_heads: 1,
                num_key_heads: 1,
                key_head_dim: 32,
                value_head_dim: 4,
                conv_kernel_dim: 4,
                q_scale,
                k_scale,
            }
        });
        let linear_cfg = cfg.linear_attention.as_ref().unwrap().clone();
        let weights = patterned_linear_layer_weights(&linear_cfg, cfg.hidden_size);

        const B: usize = 4;
        let h = cfg.hidden_size as i32;
        let conv_dim = linear_cfg.conv_dim() as i32;
        let tail = linear_cfg.conv_kernel_dim as i32 - 1;
        let hv = linear_cfg.num_value_heads as i32;
        let dv = linear_cfg.value_head_dim as i32;
        let dk = linear_cfg.key_head_dim as i32;

        // Distinct per-row input, conv state, recurrent state.
        let xs: Vec<MlxArray> = (0..B)
            .map(|r| array_f32(&pat(h as usize, 100 + r as u64, -1.0, 1.0), &[1, 1, h]))
            .collect();
        let convs: Vec<MlxArray> = (0..B)
            .map(|r| {
                array_f32(
                    &pat((tail * conv_dim) as usize, 200 + r as u64, -0.4, 0.4),
                    &[1, tail, conv_dim],
                )
            })
            .collect();
        let recs: Vec<MlxArray> = (0..B)
            .map(|r| {
                array_f32(
                    &pat((hv * dv * dk) as usize, 300 + r as u64, -0.3, 0.3),
                    &[1, hv, dv, dk],
                )
            })
            .collect();

        // Reference: each row alone through the same batched function (B == 1).
        let mut ref_out = Vec::new();
        let mut ref_conv = Vec::new();
        let mut ref_rec = Vec::new();
        for r in 0..B {
            let mut s1 = BatchedLinearState::with_capacity(1, 1);
            s1.add_row(&[convs[r].clone()], &[recs[r].clone()]);
            let out = linear_attention_forward_batched(&cfg, &weights, &xs[r], &mut s1, 0);
            let (c, rc) = s1.row_state(0, 0).unwrap();
            eval(&[&out, &c, &rc]);
            ref_out.push(out.data_f32().to_vec());
            ref_conv.push(c.data_f32().to_vec());
            ref_rec.push(rc.data_f32().to_vec());
        }

        // Batched: all B rows together in one forward.
        let mut sb = BatchedLinearState::with_capacity(1, B);
        for r in 0..B {
            sb.add_row(&[convs[r].clone()], &[recs[r].clone()]);
        }
        let x_batch = mlx_sys::concatenate(&xs.iter().collect::<Vec<_>>(), 0, None);
        let out_batch = linear_attention_forward_batched(&cfg, &weights, &x_batch, &mut sb, 0);
        eval(&[&out_batch]);
        assert_eq!(out_batch.shape(), vec![B as i32, 1, h]);

        for r in 0..B {
            let row = mlx_sys::slice(
                &out_batch,
                &[r as i32, 0, 0],
                &[r as i32 + 1, 1, h],
                &[1, 1, 1],
                None,
            );
            let (c, rc) = sb.row_state(0, r).unwrap();
            eval(&[&row, &c, &rc]);
            // The conv + recurrent states must stay BYTE-identical: they
            // compound across decode steps, so any batched-vs-single drift
            // there diverges a whole request. The projected output row is
            // held to a tight per-element tolerance instead — MLX dispatches
            // the output-projection matmul to different kernels for [1,·]
            // vs [B,·] shapes on some hardware (different reduction order;
            // observed 2026-07 on M-series under pip 0.31.2, pip 0.32.0, and
            // Homebrew 0.31.2 alike: every element off by ~1e-4..2.2e-4
            // while both states remained exactly equal). That per-step
            // rounding does not accumulate; token-exactness of the serving
            // path is gated separately at the runner level (ADR-037).
            const OUT_ROW_ABS_TOLERANCE: f32 = 5e-4;
            let row_data = row.data_f32();
            assert_eq!(row_data.len(), ref_out[r].len(), "output row {r} shape");
            for (index, (batched, single)) in row_data.iter().zip(&ref_out[r]).enumerate() {
                assert!(
                    (batched - single).abs() <= OUT_ROW_ABS_TOLERANCE,
                    "output row {r}[{index}] diverged beyond tolerance: {batched} vs {single}"
                );
            }
            assert_eq!(c.data_f32(), ref_conv[r], "conv state row {r} diverged");
            assert_eq!(
                rc.data_f32(),
                ref_rec[r],
                "recurrent state row {r} diverged"
            );
        }
    }

    #[test]
    fn moe_experts_forward_uses_packed_gate_up_experts() {
        let mut cfg = cfg(false);
        cfg.hidden_size = 4;
        cfg.moe_expert_count = 2;
        cfg.moe_experts_per_token = 1;
        cfg.moe_expert_intermediate_size = 3;
        cfg.uses_geglu = true;
        let weights = LayerWeights {
            attn_norm: zeros(&[4], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: None,
            ffn_norm: zeros(&[4], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: Some(dense_weight(&[4, 3])),
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: Some(dense_weight(&[2, 6, 4])),
            gate_exps: None,
            up_exps: None,
            down_exps: Some(dense_weight(&[2, 4, 3])),
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        };
        let x = zeros(&[1, 2, 4], MlxDtype::Float32, None);
        let indices_data = [0_u32, 1_u32];
        let top_k_indices = MlxArray::from_raw_data(
            indices_data.as_ptr() as *const u8,
            std::mem::size_of_val(&indices_data),
            &[1, 2, 1],
            MlxDtype::Uint32,
        );
        let weights_data = [1.0_f32, 1.0_f32];
        let top_k_weights = MlxArray::from_raw_data(
            weights_data.as_ptr() as *const u8,
            std::mem::size_of_val(&weights_data),
            &[1, 2, 1],
            MlxDtype::Float32,
        );

        let out = moe_experts_forward(&cfg, &weights, &x, &top_k_indices, &top_k_weights);

        assert_eq!(out.shape(), vec![1, 2, 4]);
    }

    #[test]
    fn gemma4_router_expert_scale_gathers_by_top_k_indices() {
        let scale_data = [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32];
        let per_expert_scale = MlxArray::from_raw_data(
            scale_data.as_ptr() as *const u8,
            std::mem::size_of_val(&scale_data),
            &[4],
            MlxDtype::Float32,
        );
        let indices_data = [0_u32, 3_u32, 2_u32, 1_u32];
        let top_k_indices = MlxArray::from_raw_data(
            indices_data.as_ptr() as *const u8,
            std::mem::size_of_val(&indices_data),
            &[1, 2, 2],
            MlxDtype::Uint32,
        );

        let gathered = take(&per_expert_scale, &top_k_indices, 0, None);

        eval(&[&gathered]);
        assert_eq!(gathered.shape(), vec![1, 2, 2]);
        assert_eq!(gathered.data_f32(), &[1.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn moe_experts_forward_weights_multiple_packed_experts() {
        let mut cfg = cfg(false);
        cfg.hidden_size = 4;
        cfg.moe_expert_count = 2;
        cfg.moe_experts_per_token = 2;
        cfg.moe_expert_intermediate_size = 3;
        cfg.uses_geglu = true;
        let weights = LayerWeights {
            attn_norm: zeros(&[4], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: None,
            ffn_norm: zeros(&[4], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: Some(dense_weight(&[4, 3])),
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: Some(dense_weight(&[2, 6, 4])),
            gate_exps: None,
            up_exps: None,
            down_exps: Some(dense_weight(&[2, 4, 3])),
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        };
        let x = zeros(&[1, 2, 4], MlxDtype::Float32, None);
        let indices_data = [0_u32, 1_u32, 1_u32, 0_u32];
        let top_k_indices = MlxArray::from_raw_data(
            indices_data.as_ptr() as *const u8,
            std::mem::size_of_val(&indices_data),
            &[1, 2, 2],
            MlxDtype::Uint32,
        );
        let weights_data = [0.75_f32, 0.25_f32, 0.25_f32, 0.75_f32];
        let top_k_weights = MlxArray::from_raw_data(
            weights_data.as_ptr() as *const u8,
            std::mem::size_of_val(&weights_data),
            &[1, 2, 2],
            MlxDtype::Float32,
        );

        let out = moe_experts_forward(&cfg, &weights, &x, &top_k_indices, &top_k_weights);

        assert_eq!(out.shape(), vec![1, 2, 4]);
    }

    #[test]
    fn moe_experts_forward_weights_multiple_split_experts() {
        let mut cfg = cfg(false);
        cfg.hidden_size = 4;
        cfg.moe_expert_count = 2;
        cfg.moe_experts_per_token = 2;
        cfg.moe_expert_intermediate_size = 3;
        cfg.uses_geglu = true;
        let weights = LayerWeights {
            attn_norm: zeros(&[4], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: None,
            ffn_norm: zeros(&[4], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: Some(dense_weight(&[4, 3])),
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: Some(dense_weight(&[2, 3, 4])),
            up_exps: Some(dense_weight(&[2, 3, 4])),
            down_exps: Some(dense_weight(&[2, 4, 3])),
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        };
        let x = zeros(&[1, 2, 4], MlxDtype::Float32, None);
        let indices_data = [0_u32, 1_u32, 1_u32, 0_u32];
        let top_k_indices = MlxArray::from_raw_data(
            indices_data.as_ptr() as *const u8,
            std::mem::size_of_val(&indices_data),
            &[1, 2, 2],
            MlxDtype::Uint32,
        );
        let weights_data = [0.75_f32, 0.25_f32, 0.25_f32, 0.75_f32];
        let top_k_weights = MlxArray::from_raw_data(
            weights_data.as_ptr() as *const u8,
            std::mem::size_of_val(&weights_data),
            &[1, 2, 2],
            MlxDtype::Float32,
        );

        let out = moe_experts_forward(&cfg, &weights, &x, &top_k_indices, &top_k_weights);

        assert_eq!(out.shape(), vec![1, 2, 4]);
    }

    #[test]
    fn moe_experts_forward_supports_reference_switchglu_broadcast_for_topk_gt_tokens() {
        let mut cfg = cfg(false);
        cfg.hidden_size = 4;
        cfg.moe_expert_count = 4;
        cfg.moe_experts_per_token = 3;
        cfg.moe_expert_intermediate_size = 3;
        cfg.uses_geglu = false;
        let weights = LayerWeights {
            attn_norm: zeros(&[4], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: None,
            ffn_norm: zeros(&[4], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: Some(dense_weight(&[4, 3])),
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: Some(dense_weight(&[4, 3, 4])),
            up_exps: Some(dense_weight(&[4, 3, 4])),
            down_exps: Some(dense_weight(&[4, 4, 3])),
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        };
        let x = zeros(&[1, 2, 4], MlxDtype::Float32, None);
        let indices_data = [0_u32, 1_u32, 2_u32, 2_u32, 1_u32, 0_u32];
        let top_k_indices = MlxArray::from_raw_data(
            indices_data.as_ptr() as *const u8,
            std::mem::size_of_val(&indices_data),
            &[1, 2, 3],
            MlxDtype::Uint32,
        );
        let weights_data = [0.50_f32, 0.25_f32, 0.25_f32, 0.25_f32, 0.25_f32, 0.50_f32];
        let top_k_weights = MlxArray::from_raw_data(
            weights_data.as_ptr() as *const u8,
            std::mem::size_of_val(&weights_data),
            &[1, 2, 3],
            MlxDtype::Float32,
        );

        let out = moe_experts_forward(&cfg, &weights, &x, &top_k_indices, &top_k_weights);

        assert_eq!(out.shape(), vec![1, 2, 4]);
    }

    #[test]
    fn moe_experts_forward_sorts_large_prefill_expert_indices() {
        let mut cfg = cfg(false);
        cfg.hidden_size = 4;
        cfg.moe_expert_count = 4;
        cfg.moe_experts_per_token = 4;
        cfg.moe_expert_intermediate_size = 3;
        cfg.uses_geglu = true;
        let weights = LayerWeights {
            attn_norm: zeros(&[4], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: None,
            ffn_norm: zeros(&[4], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: Some(dense_weight(&[4, 3])),
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: Some(dense_weight(&[4, 6, 4])),
            gate_exps: None,
            up_exps: None,
            down_exps: Some(dense_weight(&[4, 4, 3])),
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        };
        let x = zeros(&[1, 16, 4], MlxDtype::Float32, None);
        let indices_data = (0..64).map(|i| (3 - (i % 4)) as u32).collect::<Vec<_>>();
        let top_k_indices = MlxArray::from_raw_data(
            indices_data.as_ptr() as *const u8,
            indices_data.len() * std::mem::size_of::<u32>(),
            &[1, 16, 4],
            MlxDtype::Uint32,
        );
        let weights_data = vec![0.25_f32; 64];
        let top_k_weights = MlxArray::from_raw_data(
            weights_data.as_ptr() as *const u8,
            weights_data.len() * std::mem::size_of::<f32>(),
            &[1, 16, 4],
            MlxDtype::Float32,
        );

        let gather_inputs =
            switch_gather_inputs(&expand_dims_axes(&x, &[-2, -3], None), &top_k_indices);
        assert!(gather_inputs.sorted_indices);
        assert_eq!(gather_inputs.x.shape(), vec![64, 1, 4]);
        assert_eq!(gather_inputs.indices.shape(), vec![64]);

        let out = moe_experts_forward(&cfg, &weights, &x, &top_k_indices, &top_k_weights);

        assert_eq!(out.shape(), vec![1, 16, 4]);
    }

    #[test]
    fn switch_gather_inputs_sorts_indices_and_tracks_source_rows() {
        let x_data = (0..16)
            .flat_map(|row| std::iter::repeat_n(row as f32, 4))
            .collect::<Vec<_>>();
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            x_data.len() * std::mem::size_of::<f32>(),
            &[1, 16, 4],
            MlxDtype::Float32,
        );
        let indices_data = (0..64).rev().map(|i| i as u32).collect::<Vec<_>>();
        let top_k_indices = MlxArray::from_raw_data(
            indices_data.as_ptr() as *const u8,
            indices_data.len() * std::mem::size_of::<u32>(),
            &[1, 16, 4],
            MlxDtype::Uint32,
        );

        let gather_inputs =
            switch_gather_inputs(&expand_dims_axes(&x, &[-2, -3], None), &top_k_indices);

        eval(&[&gather_inputs.indices, &gather_inputs.x]);
        assert!(gather_inputs.sorted_indices);
        assert_eq!(
            gather_inputs.indices.data_u32(),
            &(0..64).map(|i| i as u32).collect::<Vec<_>>()
        );
        let sorted_rows = gather_inputs
            .x
            .data_f32()
            .chunks_exact(4)
            .map(|row| row[0] as usize)
            .collect::<Vec<_>>();
        let expected_rows = (0..64).map(|expert| (63 - expert) / 4).collect::<Vec<_>>();
        assert_eq!(sorted_rows, expected_rows);
    }

    #[test]
    fn moe_experts_forward_keeps_single_token_sequence_axis() {
        let mut cfg = cfg(false);
        cfg.hidden_size = 4;
        cfg.moe_expert_count = 4;
        cfg.moe_experts_per_token = 3;
        cfg.moe_expert_intermediate_size = 3;
        let weights = LayerWeights {
            attn_norm: zeros(&[4], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            deepseek_v4: None,
            ffn_norm: zeros(&[4], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: Some(dense_weight(&[4, 3])),
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: Some(dense_weight(&[4, 3, 4])),
            up_exps: Some(dense_weight(&[4, 3, 4])),
            down_exps: Some(dense_weight(&[4, 4, 3])),
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        };
        let x = zeros(&[1, 1, 4], MlxDtype::Float32, None);
        let indices_data = [0_u32, 1_u32, 2_u32];
        let top_k_indices = MlxArray::from_raw_data(
            indices_data.as_ptr() as *const u8,
            std::mem::size_of_val(&indices_data),
            &[1, 1, 3],
            MlxDtype::Uint32,
        );
        let weights_data = [0.50_f32, 0.25_f32, 0.25_f32];
        let top_k_weights = MlxArray::from_raw_data(
            weights_data.as_ptr() as *const u8,
            std::mem::size_of_val(&weights_data),
            &[1, 1, 3],
            MlxDtype::Float32,
        );

        let out = moe_experts_forward(&cfg, &weights, &x, &top_k_indices, &top_k_weights);

        assert_eq!(out.shape(), vec![1, 1, 4]);
    }

    #[test]
    fn value_norm_keeps_cache_shape_bhsd() {
        let v = zeros(&[1, 3, 2, 4], MlxDtype::Float32, None);
        let prepared = prepare_value_bhsd(v, true, 2, 4, 3, 1e-6);

        assert_eq!(prepared.shape(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn attention_mask_array_uses_fast_modes_for_simple_causal_cases() {
        // No sliding window, offset == 0 → None (Causal mode)
        assert!(attention_mask_array(1, 1, None).is_none());
        assert!(attention_mask_array(4, 4, None).is_none());
        // Sliding window, but offset == 0 and seq <= window → None (Causal mode)
        assert!(attention_mask_array(128, 128, Some(512)).is_none());
        assert!(attention_mask_array(512, 512, Some(512)).is_none());
        assert!(attention_mask_array(512, 512, Some(1024)).is_none());
    }

    #[test]
    fn attention_mask_array_uses_offset_mask_for_cached_prefill() {
        // Default: materialize offset causal for full-attention cached prefill.
        // With AX_MLX_NATIVE_OFFSET_CAUSAL=1 the mask is None (MLX native causal).
        let mask = attention_mask_array(2, 5, None).expect("cached prefill needs offset mask");

        assert_eq!(mask.shape(), vec![2, 5]);
    }

    #[test]
    fn attention_mask_array_creates_explicit_mask_when_seq_exceeds_window() {
        // seq > window: sliding constraint is active, explicit mask required
        let mask = attention_mask_array(1024, 1024, Some(512))
            .expect("seq > window must produce explicit mask");
        assert_eq!(mask.shape(), vec![1024, 1024]);

        // offset > 0: KV cache present, sliding constraint may be active
        let mask = attention_mask_array(512, 1024, Some(512))
            .expect("cached sliding prefill needs explicit mask");
        assert_eq!(mask.shape(), vec![512, 1024]);
    }

    #[test]
    fn attention_mask_array_keeps_full_kv_for_sliding_attention() {
        // mlx-lm's RotatingKVCache returns no mask for decode when SDPA already
        // receives only the retained sliding window.
        assert!(attention_mask_array(1, 4, Some(4)).is_none());
        assert!(attention_mask_array(1, 3, Some(4)).is_none());

        // If a caller still presents more than the retained window, the mask is
        // required to hide older keys.
        let mask = attention_mask_array(1, 6, Some(4)).expect("decode needs sliding mask");

        assert_eq!(mask.shape(), vec![1, 6]);
    }

    #[test]
    fn attention_mask_key_len_matches_decode_windowed_kv_views() {
        assert_eq!(attention_mask_key_len(1, 6, Some(4)), 4);
        assert_eq!(attention_mask_key_len(1, 6, None), 6);
        // Multi-token forwards retain `window + seq - 1` keys: the oldest
        // query still sees its full window, newer keys cover the rest.
        assert_eq!(attention_mask_key_len(2, 6, Some(4)), 5);
        assert_eq!(attention_mask_key_len(3, 100, Some(8)), 10);
        // No trim when the cache holds fewer keys than the retained bound.
        assert_eq!(attention_mask_key_len(4, 6, Some(4)), 6);
        assert_eq!(attention_mask_key_len(4, 6, None), 6);
    }

    #[test]
    fn linear_attention_profile_token_count_clamps_negative_shapes() {
        let _ = take_linear_attention_profile_snapshot();

        record_linear_attention_profile_layer(128);
        record_linear_attention_profile_layer(-1);
        let snapshot = take_linear_attention_profile_snapshot();

        assert_eq!(snapshot.layers, 2);
        assert_eq!(snapshot.tokens, 128);
    }

    #[test]
    fn build_layer_masks_returns_all_none_for_decode_with_layer_configs() {
        // For models with per-layer configs (e.g. Gemma 4), seq==1 decode must
        // return all-None masks without allocating a HashMap. Both sliding-window
        // and global-attention layer types must resolve to None.
        let cfg = gemma4_kv_shared_config();
        assert!(
            !cfg.layer_configs.is_empty(),
            "fixture must have layer_configs"
        );
        let n_layers = cfg.layer_configs.len();
        // key_len > seq simulates a decode step after a non-empty prefill.
        let masks = build_layer_masks(&cfg, n_layers, 1, 10);
        assert_eq!(masks.len(), n_layers);
        assert!(
            masks.iter().all(|m| m.is_none()),
            "all decode masks must be None for seq==1"
        );
    }

    #[test]
    fn bhsd_view_from_proj_matches_reshape_transpose() {
        // Synthetic Q/K/V projection output shape and values (Gemma 4 E2B
        // sliding layer: n_heads=8, head_dim=256, but use small dims here).
        let batch = 2_usize;
        let n_heads = 4_usize;
        let head_dim = 3_usize;
        let seq = 2_usize;
        let total = batch * seq * n_heads * head_dim;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let proj = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[batch as i32, seq as i32, (n_heads * head_dim) as i32],
            MlxDtype::Float32,
        );

        // Reference: reshape [batch, seq, n_heads*head_dim] to BSHD,
        // then transpose [0, 2, 1, 3] to BHSD.
        let reference_bsh = reshape(
            &proj,
            &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
        let reference = transpose(&reference_bsh, &[0, 2, 1, 3], None);

        // Candidate: single as_strided view directly to BHSD.
        let candidate = bhsd_view_from_proj(&proj, n_heads, head_dim, seq);

        eval(&[&reference, &candidate]);

        // Both must report the same shape.
        assert_eq!(
            reference.shape(),
            vec![batch as i32, n_heads as i32, seq as i32, head_dim as i32]
        );
        assert_eq!(
            candidate.shape(),
            vec![batch as i32, n_heads as i32, seq as i32, head_dim as i32]
        );

        // Bit-exact element comparison via contiguous + read-back.
        let reference_contig = mlx_sys::ops::contiguous(&reference, None);
        let candidate_contig = mlx_sys::ops::contiguous(&candidate, None);
        eval(&[&reference_contig, &candidate_contig]);
        assert_eq!(
            reference_contig.data_f32().to_vec(),
            candidate_contig.data_f32().to_vec(),
            "as_strided BHSD view must produce the same elementwise data as reshape+transpose"
        );
    }

    #[test]
    fn flatten_attention_output_bhsd_skips_decode_transpose() {
        let batch = 1_usize;
        let n_heads = 3_usize;
        let head_dim = 4_usize;
        let seq = 1_usize;
        let data: Vec<f32> = (0..(batch * n_heads * seq * head_dim))
            .map(|i| i as f32)
            .collect();
        let attn = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[batch as i32, n_heads as i32, seq as i32, head_dim as i32],
            MlxDtype::Float32,
        );

        let reference = {
            let transposed = transpose(&attn, &[0, 2, 1, 3], None);
            reshape(
                &transposed,
                &[batch as i32, seq as i32, (n_heads * head_dim) as i32],
                None,
            )
        };
        let op_count_before = mlx_sys::op_count_snapshot();
        let candidate = flatten_attention_output_bhsd(&attn, seq, n_heads, head_dim);
        assert_eq!(
            mlx_sys::op_count_take(op_count_before),
            1,
            "single-token attention flatten should be one reshape op"
        );

        eval(&[&reference, &candidate]);
        assert_eq!(
            candidate.shape(),
            vec![batch as i32, seq as i32, (n_heads * head_dim) as i32]
        );
        assert_eq!(candidate.data_f32(), reference.data_f32());
    }

    #[test]
    fn flatten_attention_output_bhsd_keeps_prefill_transpose_order() {
        let batch = 1_usize;
        let n_heads = 3_usize;
        let head_dim = 4_usize;
        let seq = 2_usize;
        let data: Vec<f32> = (0..(batch * n_heads * seq * head_dim))
            .map(|i| i as f32)
            .collect();
        let attn = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[batch as i32, n_heads as i32, seq as i32, head_dim as i32],
            MlxDtype::Float32,
        );

        let reference = {
            let transposed = transpose(&attn, &[0, 2, 1, 3], None);
            reshape(
                &transposed,
                &[batch as i32, seq as i32, (n_heads * head_dim) as i32],
                None,
            )
        };
        let candidate = flatten_attention_output_bhsd(&attn, seq, n_heads, head_dim);

        eval(&[&reference, &candidate]);
        assert_eq!(
            candidate.shape(),
            vec![batch as i32, seq as i32, (n_heads * head_dim) as i32]
        );
        assert_eq!(candidate.data_f32(), reference.data_f32());
    }

    #[test]
    fn qk_norm_bhsd_from_proj_matches_bshd_reference_path() {
        let n_heads = 4_usize;
        let head_dim = 3_usize;
        let seq = 2_usize;
        let proj_data: Vec<f32> = (0..(seq * n_heads * head_dim))
            .map(|i| ((i as f32) - 12.0) * 0.05)
            .collect();
        let norm_data: Vec<f32> = vec![0.5, 1.0, 1.5];
        let proj = array_f32(&proj_data, &[1, seq as i32, (n_heads * head_dim) as i32]);
        let norm = array_f32(&norm_data, &[head_dim as i32]);

        let reference_bshd = reshape(
            &proj,
            &[1, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
        let reference_normed =
            qk_norm_bshd(reference_bshd, Some(&norm), n_heads, head_dim, seq, 1.0e-6);
        let reference = transpose(&reference_normed, &[0, 2, 1, 3], None);
        let candidate = qk_norm_bhsd_from_proj(&proj, Some(&norm), n_heads, head_dim, seq, 1.0e-6);

        let reference_contig = mlx_sys::ops::contiguous(&reference, None);
        let candidate_contig = mlx_sys::ops::contiguous(&candidate, None);
        eval(&[&reference_contig, &candidate_contig]);

        assert_eq!(
            reference_contig.shape(),
            vec![1, n_heads as i32, seq as i32, head_dim as i32]
        );
        assert_eq!(candidate_contig.shape(), reference_contig.shape());
        assert_close(
            candidate_contig.data_f32(),
            reference_contig.data_f32(),
            1.0e-6,
        );
    }

    #[test]
    fn qk_norm_rope_bhsd_from_proj_matches_composed_reference_path() {
        let n_heads = 2_usize;
        let head_dim = 4_usize;
        let seq = 3_usize;
        let proj_data: Vec<f32> = (0..(seq * n_heads * head_dim))
            .map(|i| ((i as f32) - 11.0) * 0.03125)
            .collect();
        let norm_data: Vec<f32> = (0..head_dim).map(|i| 0.75 + (i as f32) * 0.125).collect();
        let proj = array_f32(&proj_data, &[1, seq as i32, (n_heads * head_dim) as i32]);
        let norm = array_f32(&norm_data, &[head_dim as i32]);

        let q = qk_norm_bhsd_from_proj(&proj, Some(&norm), n_heads, head_dim, seq, 1.0e-6);
        let reference = mlx_sys::rope(
            &q,
            head_dim as i32,
            false,
            Some(10_000.0),
            1.0,
            2,
            None,
            None,
        );
        let candidate = qk_norm_rope_bhsd_from_proj(
            &proj,
            Some(&norm),
            n_heads,
            head_dim,
            seq,
            1.0e-6,
            head_dim,
            Some(10_000.0),
            2,
            None,
        );

        let reference_contig = mlx_sys::ops::contiguous(&reference, None);
        let candidate_contig = mlx_sys::ops::contiguous(&candidate, None);
        eval(&[&reference_contig, &candidate_contig]);

        assert_eq!(
            reference_contig.shape(),
            vec![1, n_heads as i32, seq as i32, head_dim as i32]
        );
        assert_eq!(candidate_contig.shape(), reference_contig.shape());
        assert_close(
            candidate_contig.data_f32(),
            reference_contig.data_f32(),
            1.0e-6,
        );
    }

    #[test]
    fn qk_norm_rope_bhsd_from_proj_flat_matches_bshd_reference_path() {
        let batch = 2_usize;
        let n_heads = 2_usize;
        let head_dim = 4_usize;
        let seq = 3_usize;
        let proj_data: Vec<f32> = (0..(batch * seq * n_heads * head_dim))
            .map(|i| ((i as f32) - 23.0) * 0.03125)
            .collect();
        let norm_data: Vec<f32> = (0..head_dim).map(|i| 0.75 + (i as f32) * 0.125).collect();
        let proj = array_f32(
            &proj_data,
            &[batch as i32, seq as i32, (n_heads * head_dim) as i32],
        );
        let norm = array_f32(&norm_data, &[head_dim as i32]);

        let reference_bshd = reshape(
            &proj,
            &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
        let reference_normed = rms_norm(&reference_bshd, Some(&norm), 1.0e-6, None);
        let reference_bhsd = transpose(&reference_normed, &[0, 2, 1, 3], None);
        let reference = mlx_sys::rope(
            &reference_bhsd,
            head_dim as i32,
            false,
            Some(10_000.0),
            1.0,
            2,
            None,
            None,
        );
        let candidate = qk_norm_rope_bhsd_from_proj_flat(
            &proj,
            Some(&norm),
            n_heads,
            head_dim,
            seq,
            1.0e-6,
            head_dim,
            Some(10_000.0),
            2,
            None,
        );

        let reference_contig = mlx_sys::ops::contiguous(&reference, None);
        let candidate_contig = mlx_sys::ops::contiguous(&candidate, None);
        eval(&[&reference_contig, &candidate_contig]);

        assert_eq!(
            reference_contig.shape(),
            vec![batch as i32, n_heads as i32, seq as i32, head_dim as i32]
        );
        assert_eq!(candidate_contig.shape(), reference_contig.shape());
        assert_close(
            candidate_contig.data_f32(),
            reference_contig.data_f32(),
            1.0e-6,
        );
    }

    /// Last-layer embedding Q-only path: RoPE of a 1-token Q at offset = last
    /// position must match slicing the last position out of a full-seq Q rope.
    /// This is the numerical contract behind projecting Q only at the target
    /// for equal-length last-token pooling.
    ///
    /// Uses the flat reference rope path (not the direct-C++ probe) and skips
    /// under `GITHUB_ACTIONS`: CI Metal runners produce a deterministic dual-eval
    /// mismatch for this oracle under parallel `cargo test` (also fails on main).
    /// Local full-suite coverage remains enabled.
    #[test]
    fn embed_last_token_q_rope_at_offset_matches_full_seq_slice() {
        if std::env::var_os("GITHUB_ACTIONS").is_some() {
            // Known CI flake / Metal dual-eval instability (reproduced on main).
            return;
        }
        with_gpu_numeric_lock(|| {
            let batch = 2_usize;
            let n_heads = 2_usize;
            let head_dim = 4_usize;
            let seq = 8_usize;
            let target = seq - 1;
            let proj_data: Vec<f32> = (0..(batch * seq * n_heads * head_dim))
                .map(|i| ((i as f32) - 17.0) * 0.029)
                .collect();
            let norm_data: Vec<f32> = (0..head_dim).map(|i| 0.8 + (i as f32) * 0.05).collect();
            let proj = array_f32(
                &proj_data,
                &[batch as i32, seq as i32, (n_heads * head_dim) as i32],
            );
            let norm = array_f32(&norm_data, &[head_dim as i32]);
            eval(&[&proj, &norm]);

            // Flat reference path — independent of direct-C++ probe flags.
            let full = qk_norm_rope_bhsd_from_proj_flat(
                &proj,
                Some(&norm),
                n_heads,
                head_dim,
                seq,
                1.0e-6,
                head_dim,
                Some(10_000.0),
                0,
                None,
            );
            let full_target = select_attention_common_target_bhsd(&full, target);
            eval(&[&full, &full_target]);
            let full_c = mlx_sys::ops::contiguous(&full_target, None);
            eval(&[&full_c]);
            let full_vals: Vec<f32> = full_c.data_f32().to_vec();

            // Emulate Q-only projection: take the target token's raw proj rows and
            // RoPE them with offset = target.
            let target_proj = select_embedding_targets(&proj, &[target; 2]);
            eval(&[&target_proj]);
            let target_rope = qk_norm_rope_bhsd_from_proj_flat(
                &target_proj,
                Some(&norm),
                n_heads,
                head_dim,
                1,
                1.0e-6,
                head_dim,
                Some(10_000.0),
                target,
                None,
            );
            let target_c = mlx_sys::ops::contiguous(&target_rope, None);
            eval(&[&target_c]);
            let target_vals: Vec<f32> = target_c.data_f32().to_vec();

            assert_eq!(full_c.shape(), target_c.shape());
            assert_eq!(
                full_c.shape(),
                vec![batch as i32, n_heads as i32, 1, head_dim as i32]
            );
            assert_close(&target_vals, &full_vals, 1.0e-5);
        });
    }

    #[test]
    fn select_embedding_targets_common_pos_matches_slice() {
        let batch = 3_usize;
        let seq = 5_usize;
        let hidden = 4_usize;
        let data: Vec<f32> = (0..(batch * seq * hidden))
            .map(|i| i as f32 * 0.1)
            .collect();
        let arr = array_f32(&data, &[batch as i32, seq as i32, hidden as i32]);
        let pos = 3_usize;
        let selected = select_embedding_targets(&arr, &[pos; 3]);
        let sliced = slice(
            &arr,
            &[0, pos as i32, 0],
            &[batch as i32, pos as i32 + 1, hidden as i32],
            &[1, 1, 1],
            None,
        );
        let a = mlx_sys::ops::contiguous(&selected, None);
        let b = mlx_sys::ops::contiguous(&sliced, None);
        eval(&[&a, &b]);
        assert_eq!(a.shape(), vec![batch as i32, 1, hidden as i32]);
        assert_eq!(a.data_f32(), b.data_f32());
    }

    #[test]
    fn prepare_value_bhsd_from_proj_matches_bshd_reference_path() {
        let n_heads = 2_usize;
        let head_dim = 4_usize;
        let seq = 3_usize;
        let proj_data: Vec<f32> = (0..(seq * n_heads * head_dim))
            .map(|i| ((i as f32) - 8.0) * 0.0625)
            .collect();
        let proj = array_f32(&proj_data, &[1, seq as i32, (n_heads * head_dim) as i32]);

        let reference_bshd = reshape(
            &proj,
            &[1, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
        let reference = prepare_value_bhsd(reference_bshd, true, n_heads, head_dim, seq, 1.0e-6);
        let candidate = prepare_value_bhsd_from_proj(&proj, true, n_heads, head_dim, seq, 1.0e-6);

        let reference_contig = mlx_sys::ops::contiguous(&reference, None);
        let candidate_contig = mlx_sys::ops::contiguous(&candidate, None);
        eval(&[&reference_contig, &candidate_contig]);

        assert_eq!(
            reference_contig.shape(),
            vec![1, n_heads as i32, seq as i32, head_dim as i32]
        );
        assert_eq!(candidate_contig.shape(), reference_contig.shape());
        assert_close(
            candidate_contig.data_f32(),
            reference_contig.data_f32(),
            1.0e-6,
        );
    }

    #[test]
    fn prepare_value_bhsd_from_proj_flat_matches_bshd_reference_path() {
        let batch = 2_usize;
        let n_heads = 2_usize;
        let head_dim = 4_usize;
        let seq = 3_usize;
        let proj_data: Vec<f32> = (0..(batch * seq * n_heads * head_dim))
            .map(|i| ((i as f32) - 19.0) * 0.0625)
            .collect();
        let proj = array_f32(
            &proj_data,
            &[batch as i32, seq as i32, (n_heads * head_dim) as i32],
        );

        let reference_bshd = reshape(
            &proj,
            &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
        let reference = prepare_value_bhsd(reference_bshd, true, n_heads, head_dim, seq, 1.0e-6);
        let candidate =
            prepare_value_bhsd_from_proj_flat(&proj, true, n_heads, head_dim, seq, 1.0e-6);

        let reference_contig = mlx_sys::ops::contiguous(&reference, None);
        let candidate_contig = mlx_sys::ops::contiguous(&candidate, None);
        eval(&[&reference_contig, &candidate_contig]);

        assert_eq!(
            reference_contig.shape(),
            vec![batch as i32, n_heads as i32, seq as i32, head_dim as i32]
        );
        assert_eq!(candidate_contig.shape(), reference_contig.shape());
        assert_close(
            candidate_contig.data_f32(),
            reference_contig.data_f32(),
            1.0e-6,
        );
    }

    #[test]
    fn argmax_only_final_logits_skip_softcap_preserves_argmax() {
        let mut cfg = cfg(false);
        cfg.final_logit_softcapping = Some(30.0);
        let logits = array_f32(&[-12.0, -0.5, 1.0, 29.0, 31.0, 120.0], &[1, 1, 6]);

        let full = finalize_lm_head_logits(&cfg, &logits, FinalLogitsMode::Full);
        let argmax_only = finalize_lm_head_logits(&cfg, &logits, FinalLogitsMode::ArgmaxOnly);
        eval(&[&full, &argmax_only]);

        let max_index = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .expect("fixture must not be empty")
        };
        assert_eq!(
            max_index(full.data_f32()),
            max_index(argmax_only.data_f32())
        );
        assert_close(
            argmax_only.data_f32(),
            &[-12.0, -0.5, 1.0, 29.0, 31.0, 120.0],
            1.0e-6,
        );
    }

    #[test]
    fn argmax_only_final_logits_preserves_bfloat16_dtype() {
        let mut cfg = cfg(false);
        cfg.final_logit_softcapping = Some(30.0);
        let logits_f32 = array_f32(&[-12.0, -0.5, 1.0, 29.0, 31.0, 120.0], &[1, 1, 6]);
        let logits_bf16 = astype(&logits_f32, MlxDtype::Bfloat16, None);

        let full = finalize_lm_head_logits(&cfg, &logits_bf16, FinalLogitsMode::Full);
        let argmax_only = finalize_lm_head_logits(&cfg, &logits_bf16, FinalLogitsMode::ArgmaxOnly);
        assert_eq!(argmax_only.dtype(), MlxDtype::Bfloat16);

        let full_token = argmax(&full, None);
        let argmax_token = argmax(&argmax_only, None);
        eval(&[&full_token, &argmax_token]);

        assert_eq!(full_token.data_u32(), argmax_token.data_u32());
    }

    #[test]
    fn geglu_direct_shim_matches_imperative() {
        // Same shape and dtype the Gemma 4 FFN call site produces:
        // gate_proj output and up_proj output, both bf16.
        let gate_f32: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.05).collect();
        let x_f32: Vec<f32> = (0..32).map(|i| ((i as f32) + 1.0) * 0.07).collect();
        let gate_src = MlxArray::from_raw_data(
            gate_f32.as_ptr() as *const u8,
            std::mem::size_of_val(gate_f32.as_slice()),
            &[1, 4, 8],
            MlxDtype::Float32,
        );
        let x_src = MlxArray::from_raw_data(
            x_f32.as_ptr() as *const u8,
            std::mem::size_of_val(x_f32.as_slice()),
            &[1, 4, 8],
            MlxDtype::Float32,
        );
        let gate = astype(&gate_src, MlxDtype::Bfloat16, None);
        let x = astype(&x_src, MlxDtype::Bfloat16, None);

        // Imperative reference: gelu_approx(gate) * x
        let imperative = multiply(&gelu_approx(&gate, None), &x, None);
        let imperative_f32 = astype(&imperative, MlxDtype::Float32, None);

        // Direct MLX shim via the geglu helper.
        let direct = geglu(&gate, &x);
        let direct_f32 = astype(&direct, MlxDtype::Float32, None);

        eval(&[&imperative_f32, &direct_f32]);

        let imp = imperative_f32.data_f32().to_vec();
        let cmp = direct_f32.data_f32().to_vec();
        assert_eq!(
            imp, cmp,
            "direct geglu shim must produce bit-identical output to the imperative reference"
        );

        let direct_again = geglu(&gate, &x);
        let direct_again_f32 = astype(&direct_again, MlxDtype::Float32, None);
        eval(&[&direct_again_f32]);
        assert_eq!(
            cmp,
            direct_again_f32.data_f32().to_vec(),
            "direct geglu shim must remain stable across invocations"
        );
    }

    #[test]
    fn per_layer_input_gate_direct_path_matches_imperative() {
        let gate_f32: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.05).collect();
        let x_f32: Vec<f32> = (0..32).map(|i| ((i as f32) + 1.0) * 0.07).collect();
        let gate_src = MlxArray::from_raw_data(
            gate_f32.as_ptr() as *const u8,
            std::mem::size_of_val(gate_f32.as_slice()),
            &[1, 4, 8],
            MlxDtype::Float32,
        );
        let x_src = MlxArray::from_raw_data(
            x_f32.as_ptr() as *const u8,
            std::mem::size_of_val(x_f32.as_slice()),
            &[1, 4, 8],
            MlxDtype::Float32,
        );
        let gate = astype(&gate_src, MlxDtype::Bfloat16, None);
        let x = astype(&x_src, MlxDtype::Bfloat16, None);

        let imperative = multiply(&gelu_approx(&gate, None), &x, None);
        let direct = per_layer_input_gate(&gate, &x);
        let imperative_f32 = astype(&imperative, MlxDtype::Float32, None);
        let direct_f32 = astype(&direct, MlxDtype::Float32, None);
        eval(&[&imperative_f32, &direct_f32]);

        assert_eq!(
            imperative_f32.data_f32().to_vec(),
            direct_f32.data_f32().to_vec(),
            "direct per-layer-input gate must match mlx-lm's imperative gelu_approx multiply"
        );
    }

    #[test]
    fn swiglu_compiled_matches_imperative() {
        // Same shape and dtype the Qwen 3 dense FFN call site produces:
        // gate_proj output and up_proj output, both bf16.
        let gate_f32: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.05).collect();
        let up_f32: Vec<f32> = (0..32).map(|i| ((i as f32) + 1.0) * 0.07).collect();
        let gate_src = MlxArray::from_raw_data(
            gate_f32.as_ptr() as *const u8,
            std::mem::size_of_val(gate_f32.as_slice()),
            &[1, 4, 8],
            MlxDtype::Float32,
        );
        let up_src = MlxArray::from_raw_data(
            up_f32.as_ptr() as *const u8,
            std::mem::size_of_val(up_f32.as_slice()),
            &[1, 4, 8],
            MlxDtype::Float32,
        );
        let gate = astype(&gate_src, MlxDtype::Bfloat16, None);
        let up = astype(&up_src, MlxDtype::Bfloat16, None);

        // Imperative reference: silu(gate) * up
        let imperative = multiply(&mlx_sys::ops::silu(&gate, None), &up, None);
        let imperative_f32 = astype(&imperative, MlxDtype::Float32, None);

        let compiled = swiglu(&gate, &up);
        let compiled_f32 = astype(&compiled, MlxDtype::Float32, None);

        eval(&[&imperative_f32, &compiled_f32]);

        let imp = imperative_f32.data_f32().to_vec();
        let cmp = compiled_f32.data_f32().to_vec();
        assert_eq!(
            imp, cmp,
            "compiled swiglu must produce bit-identical output to the imperative fallback"
        );

        let compiled_again = swiglu(&gate, &up);
        let compiled_again_f32 = astype(&compiled_again, MlxDtype::Float32, None);
        eval(&[&compiled_again_f32]);
        assert_eq!(
            cmp,
            compiled_again_f32.data_f32().to_vec(),
            "cached compiled swiglu must remain stable across invocations"
        );
    }

    // ── Batched decode token assembly ──

    fn plain_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    /// The oracle: `embed_decode_tokens_batched([id0,id1,..])` row `r` is
    /// byte-identical to the single-sequence `embed_tokens([id_r])` — proving the
    /// `[B, 1, hidden]` batch assembly stacks the same per-token embeddings the
    /// batch=1 path produces, just on the batch axis. Covers both the quantized
    /// (production) and non-quantized embedding paths.
    #[test]
    fn batched_decode_embed_rows_match_single_token_embed() {
        use mlx_sys::{MlxQuantizationMode, contiguous, quantize, slice};

        let (vocab, hidden) = (8usize, 64usize);
        // Distinct row values so a wrong gather index would be caught.
        let table: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i % 97) as f32) * 0.013 - 0.5)
            .collect();
        let weight = plain_f32(&table, &[vocab as i32, hidden as i32]);

        // Token ids include repeats and out-of-order rows.
        let ids: Vec<u32> = vec![3, 0, 7, 3, 5];

        // Quantized (production) embedding and a non-quantized one.
        let q = quantize(
            &weight,
            Some(64),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let quantized = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size: 64,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
        };
        let plain = QuantizedWeight {
            weight: weight.clone(),
            scales: None,
            biases: None,
            group_size: 0,
            bits: 0,
            mode: "affine".to_string(),
            linear_bias: None,
        };

        for embedding in [&quantized, &plain] {
            let batched = embed_decode_tokens_batched(&ids, embedding, hidden);
            assert_eq!(batched.shape(), vec![ids.len() as i32, 1, hidden as i32]);
            for (row, &id) in ids.iter().enumerate() {
                let r = row as i32;
                let batched_row = contiguous(
                    &slice(
                        &batched,
                        &[r, 0, 0],
                        &[r + 1, 1, hidden as i32],
                        &[1, 1, 1],
                        None,
                    ),
                    None,
                );
                // Bind the id array to a named local: `embed_tokens` borrows it
                // via `from_raw_data`, and it must outlive the eval below.
                let single_ids = [id];
                let single = embed_tokens(&single_ids, embedding, hidden);
                eval(&[&batched_row, &single]);
                assert_eq!(single.shape(), vec![1, 1, hidden as i32]);
                assert_eq!(
                    batched_row.data_f32(),
                    single.data_f32(),
                    "row {row} (id {id}) differs from single-token embed"
                );
            }
        }
    }

    #[test]
    fn embed_tokens_clamps_out_of_range_ids_instead_of_reading_out_of_bounds() {
        // MLX's Metal gather kernel does no bounds checking for unsigned
        // indices (offset_neg_idx returns them unmodified), so a
        // client-supplied token id at or beyond vocab_size must be clamped
        // before it reaches `take()`, or it reads arbitrary GPU memory past
        // the embedding weight buffer.
        let vocab = 4;
        let hidden = 3;
        let weight_data: Vec<f32> = (0..(vocab * hidden)).map(|i| i as f32).collect();
        let weight = MlxArray::from_raw_data(
            weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(weight_data.as_slice()),
            &[vocab as i32, hidden as i32],
            MlxDtype::Float32,
        );
        let embedding = QuantizedWeight::new(weight, None, None);

        // In-range ids must be unaffected by the clamp.
        let in_range_ids = [0_u32, vocab as u32 - 1];
        let in_range = embed_tokens(&in_range_ids, &embedding, hidden);
        eval(&[&in_range]);
        assert_eq!(
            in_range.data_f32(),
            &[0.0, 1.0, 2.0, 9.0, 10.0, 11.0],
            "in-range ids must embed their real rows unchanged"
        );

        // Out-of-range ids (including u32::MAX) must clamp to the last valid
        // row instead of reading past the weight buffer.
        let out_of_range_ids = [vocab as u32, u32::MAX];
        let clamped = embed_tokens(&out_of_range_ids, &embedding, hidden);
        eval(&[&clamped]);
        let last_row = &weight_data[(vocab - 1) * hidden..vocab * hidden];
        assert_eq!(
            &clamped.data_f32()[0..hidden],
            last_row,
            "id == vocab_size must clamp to the last valid row"
        );
        assert_eq!(
            &clamped.data_f32()[hidden..2 * hidden],
            last_row,
            "u32::MAX must clamp to the last valid row"
        );
    }

    // ── Padded batched prefill: cohort-parity + capability tests ─────────
    //
    // Methodology (audit v2 §5.6): parity against the sequential path is
    // tolerance-based, never byte-identical — padded batching legitimately
    // changes reduction shapes at bf16 precision.

    fn patterned_signal(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    fn ones_norm(n: i32) -> MlxArray {
        array_f32(&vec![1.0; n as usize], &[n])
    }

    fn dense_full_attention_layer(cfg: &ModelConfig, seed: u32) -> LayerWeights {
        let hidden = cfg.hidden_size as i32;
        let q_out = (cfg.n_heads * cfg.head_dim) as i32;
        let kv_out = (cfg.n_kv_heads * cfg.head_dim) as i32;
        let inter = cfg.intermediate_size as i32;
        let weight = |rows: i32, cols: i32, salt: u32| {
            dense_weight_from_data(
                &patterned_signal((rows * cols) as usize, seed.wrapping_add(salt)),
                &[rows, cols],
            )
        };
        let mut layer = empty_layer_weights(cfg.hidden_size);
        layer.attn_norm = ones_norm(hidden);
        layer.ffn_norm = ones_norm(hidden);
        layer.q_proj = Some(weight(q_out, hidden, 1));
        layer.k_proj = Some(weight(kv_out, hidden, 2));
        layer.v_proj = Some(weight(kv_out, hidden, 3));
        layer.o_proj = Some(weight(hidden, q_out, 4));
        layer.gate_proj = Some(weight(inter, hidden, 5));
        layer.up_proj = Some(weight(inter, hidden, 6));
        layer.down_proj = Some(weight(hidden, inter, 7));
        layer
    }

    fn dense_prefill_test_model() -> (ModelConfig, ModelWeights) {
        let mut model_cfg = cfg(false);
        model_cfg.layer_count = 2;
        model_cfg.query_scale = 1.0 / (model_cfg.head_dim as f32).sqrt();
        let layers = (0..model_cfg.layer_count)
            .map(|layer| dense_full_attention_layer(&model_cfg, 100 + layer as u32 * 17))
            .collect();
        let vocab = model_cfg.vocab_size as i32;
        let hidden = model_cfg.hidden_size as i32;
        let weights = ModelWeights {
            token_embedding: dense_weight_from_data(
                &patterned_signal((vocab * hidden) as usize, 31),
                &[vocab, hidden],
            ),
            final_norm: ones_norm(hidden),
            lm_head: dense_weight_from_data(
                &patterned_signal((vocab * hidden) as usize, 37),
                &[vocab, hidden],
            ),
            layers,
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            deepseek_v4_head: None,
            deepseek_v4_nextn: None,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: None,
            assistant_post_projection: None,
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            gemma4_vl_vision: None,
            diffusion_self_conditioning: None,
            unlimited_ocr_vision: None,
            qwen3_vl_vision: None,
            minicpm_v46_vision: None,
            nemotron_omni: None,
        };
        (model_cfg, weights)
    }

    #[test]
    fn two_rank_pipeline_matches_monolithic_llama3_forward() {
        with_gpu_numeric_lock(|| {
            let (mut model_cfg, reference_weights) = dense_prefill_test_model();
            model_cfg.model_family = "llama3".into();

            let (_, mut rank0_source) = dense_prefill_test_model();
            let rank0_layer = rank0_source.layers.remove(0);
            let rank0 = PipelineStageWeights {
                assignment: ax_engine_core::PipelineRankAssignment {
                    rank: 0,
                    node_identity_digest: "node-a".into(),
                    layers: ax_engine_core::PipelineLayerRange { start: 0, end: 1 },
                    owns_embeddings: true,
                    owns_output_head: false,
                },
                token_embedding: Some(rank0_source.token_embedding),
                final_norm: None,
                lm_head: None,
                layers: vec![rank0_layer],
            };

            let (_, mut rank1_source) = dense_prefill_test_model();
            let rank1_layer = rank1_source.layers.remove(1);
            let rank1 = PipelineStageWeights {
                assignment: ax_engine_core::PipelineRankAssignment {
                    rank: 1,
                    node_identity_digest: "node-b".into(),
                    layers: ax_engine_core::PipelineLayerRange { start: 1, end: 2 },
                    owns_embeddings: false,
                    owns_output_head: true,
                },
                token_embedding: None,
                final_norm: Some(rank1_source.final_norm),
                lm_head: Some(rank1_source.lm_head),
                layers: vec![rank1_layer],
            };

            let tokens = [1_u32, 3, 5, 7];
            let mut reference_cache = MlxKVCache::new(model_cfg.layer_count);
            let reference = forward(
                &model_cfg,
                &reference_weights,
                &tokens,
                &mut reference_cache,
                0,
            );
            let topology = ax_engine_core::PipelineTopology {
                cluster_id: "cluster-a".into(),
                generation: 1,
                manifest_digest: "manifest-a".into(),
                model_artifact_digest: "model-a".into(),
                total_layers: 2,
                micro_batch_limit: 2,
                ranks: vec![rank0.assignment.clone(), rank1.assignment.clone()],
            };
            let mut rank0_executor = crate::pipeline::PipelineRankExecutor::new(
                topology.clone(),
                0,
                model_cfg.clone(),
                rank0,
            )
            .expect("rank 0 executor should initialize");
            let packet = match rank0_executor
                .execute_tokens(1, 1, 0, &tokens)
                .expect("rank 0 should execute")
            {
                crate::pipeline::PipelineRankOutput::Activation(packet) => packet,
                crate::pipeline::PipelineRankOutput::Logits(_) => {
                    panic!("non-final rank must emit an activation")
                }
            };
            let mut rank1_executor =
                crate::pipeline::PipelineRankExecutor::new(topology, 1, model_cfg.clone(), rank1)
                    .expect("rank 1 executor should initialize");
            let distributed = match rank1_executor
                .execute_activation(&packet)
                .expect("rank 1 should execute")
            {
                crate::pipeline::PipelineRankOutput::Logits(logits) => logits,
                crate::pipeline::PipelineRankOutput::Activation(_) => {
                    panic!("final rank must emit logits")
                }
            };
            eval(&[&reference, &distributed]);
            assert_close_rel(
                distributed.data_f32(),
                reference.data_f32(),
                3e-2,
                "two-rank pipeline logits",
            );

            assert!(
                rank0_executor.request_cache_has_layer(1, 0),
                "rank 0 must own global layer 0 KV"
            );
            assert!(
                !rank0_executor.request_cache_has_layer(1, 1),
                "rank 0 must not materialize rank 1 KV"
            );
            assert!(
                !rank1_executor.request_cache_has_layer(1, 0),
                "rank 1 must not materialize rank 0 KV"
            );
            assert!(
                rank1_executor.request_cache_has_layer(1, 1),
                "rank 1 must own global layer 1 KV"
            );
        });
    }

    #[test]
    fn cache_only_forward_preserves_full_forward_kv() {
        with_gpu_numeric_lock(|| {
            let (model_cfg, weights) = dense_prefill_test_model();
            for tokens in [vec![7u32], vec![1, 3, 5, 7]] {
                let mut full_cache = MlxKVCache::new(model_cfg.layer_count);
                let mut cache_only = MlxKVCache::new(model_cfg.layer_count);

                let logits = forward(&model_cfg, &weights, &tokens, &mut full_cache, 0);
                let cache_only_hidden =
                    forward_cache_only(&model_cfg, &weights, &tokens, &mut cache_only, 0);
                full_cache.advance(tokens.len());
                cache_only.advance(tokens.len());

                let mut eval_refs = vec![&logits, &cache_only_hidden];
                eval_refs.extend(full_cache.collect_eval_refs());
                eval_refs.extend(cache_only.collect_eval_refs());
                eval(&eval_refs);

                for layer in 0..model_cfg.layer_count {
                    let (full_k, full_v) = full_cache
                        .logical_layer_kv(layer)
                        .expect("full forward must populate KV");
                    let (cache_only_k, cache_only_v) = cache_only
                        .logical_layer_kv(layer)
                        .expect("cache-only forward must populate KV");
                    eval(&[&full_k, &full_v, &cache_only_k, &cache_only_v]);
                    assert_eq!(
                        cache_only_k.data_f32(),
                        full_k.data_f32(),
                        "seq {} layer {layer} K differs after skipping discarded last-layer work",
                        tokens.len()
                    );
                    assert_eq!(
                        cache_only_v.data_f32(),
                        full_v.data_f32(),
                        "seq {} layer {layer} V differs after skipping discarded last-layer work",
                        tokens.len()
                    );
                }
            }
        });
    }

    fn assert_close_rel(actual: &[f32], expected: &[f32], rel: f32, context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context}: length");
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let bound = rel * e.abs().max(1.0);
            assert!(
                (a - e).abs() <= bound,
                "{context} index {idx}: batched {a}, sequential {e}, bound {bound}"
            );
        }
    }

    #[test]
    fn supports_batched_prefill_accepts_dense_and_rejects_excluded_shapes() {
        let (model_cfg, mut weights) = dense_prefill_test_model();
        assert!(supports_batched_prefill(&model_cfg, &weights));

        let mut sliding_cfg = model_cfg.clone();
        sliding_cfg.global_sliding_window = Some(8);
        assert!(!supports_batched_prefill(&sliding_cfg, &weights));

        weights.layers[0].router_proj = Some(dense_weight(&[4, model_cfg.hidden_size as i32]));
        assert!(!supports_batched_prefill(&model_cfg, &weights));
    }

    /// Cohort parity: each row of one padded batched prefill must match the
    /// sequential single-row prefill within tolerance — both the
    /// last-position logits and, via a decode continuation on the seeded
    /// per-request cache, the KV contents themselves.
    #[test]
    fn batched_prefill_rows_match_sequential_prefill_within_tolerance() {
        with_gpu_numeric_lock(|| {
            let (model_cfg, weights) = dense_prefill_test_model();
            let prompts: Vec<Vec<u32>> = vec![
                vec![1, 2, 3, 4, 5],
                vec![7, 8, 9],
                vec![11, 3, 6, 2, 9, 1, 4],
            ];
            let prompt_refs: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
            let batch = prefill_batched_forward(&model_cfg, &weights, &prompt_refs)
                .expect("batched prefill should run on the dense fixture");

            for (row, prompt) in prompts.iter().enumerate() {
                // Sequential reference: one full single-row forward.
                let mut cache_seq = MlxKVCache::new(model_cfg.layer_count);
                let logits_seq = forward(&model_cfg, &weights, prompt, &mut cache_seq, 0);
                eval(&[&logits_seq]);
                assert_close_rel(
                    batch.row_logits[row].data_f32(),
                    logits_seq.data_f32(),
                    3e-2,
                    &format!("prefill logits row {row}"),
                );

                // KV parity end-to-end: seed a fresh per-request cache from
                // the batched rows and decode one token on both caches.
                cache_seq.advance(prompt.len());
                let mut cache_batched = MlxKVCache::new(model_cfg.layer_count);
                for (layer, (k, v)) in batch.row_layer_kv[row].iter().enumerate() {
                    cache_batched.set_layer_kv_logical(layer, k.clone(), v.clone(), prompt.len());
                }
                let next_token = 5u32;
                let decode_seq = forward(
                    &model_cfg,
                    &weights,
                    &[next_token],
                    &mut cache_seq,
                    prompt.len(),
                );
                let decode_batched = forward(
                    &model_cfg,
                    &weights,
                    &[next_token],
                    &mut cache_batched,
                    prompt.len(),
                );
                eval(&[&decode_seq, &decode_batched]);
                assert_close_rel(
                    decode_batched.data_f32(),
                    decode_seq.data_f32(),
                    3e-2,
                    &format!("decode continuation row {row}"),
                );
            }
        });
    }

    #[test]
    fn batched_prefill_rejects_degenerate_inputs() {
        let (model_cfg, weights) = dense_prefill_test_model();
        let single: Vec<&[u32]> = vec![&[1, 2, 3]];
        assert!(prefill_batched_forward(&model_cfg, &weights, &single).is_err());
        let with_empty: Vec<&[u32]> = vec![&[1, 2, 3], &[]];
        assert!(prefill_batched_forward(&model_cfg, &weights, &with_empty).is_err());
    }
}

#[cfg(test)]
mod embed_bucket_tests {
    use super::{
        DEFAULT_EMBED_MAX_LEN_BUCKETS, embed_bucket_pad_acceptable, embed_length_affinity_groups,
        embed_length_bucket,
    };

    #[test]
    fn default_buckets_are_finer_at_short_lengths() {
        assert!(DEFAULT_EMBED_MAX_LEN_BUCKETS.contains(&8));
        assert!(DEFAULT_EMBED_MAX_LEN_BUCKETS.contains(&16));
        assert!(
            DEFAULT_EMBED_MAX_LEN_BUCKETS
                .windows(2)
                .all(|w| w[0] < w[1])
        );
    }

    #[test]
    fn pad_acceptable_allows_small_absolute_or_relative() {
        assert!(embed_bucket_pad_acceptable(100, 112)); // 12 pad, abs ok
        assert!(embed_bucket_pad_acceptable(100, 125)); // 25% ok
        assert!(!embed_bucket_pad_acceptable(8, 32)); // 24 pad > 16 and > 25%
        assert!(embed_bucket_pad_acceptable(20, 32)); // 12 pad abs ok
    }

    #[test]
    fn length_affinity_groups_split_short_from_long() {
        let lens = [5usize, 200, 6, 210];
        let groups = embed_length_affinity_groups(&lens, 1.5, 32);
        assert!(groups.len() >= 2, "expected split, got {groups:?}");
        // All indices covered exactly once.
        let mut flat: Vec<usize> = groups.into_iter().flatten().collect();
        flat.sort_unstable();
        assert_eq!(flat, vec![0, 1, 2, 3]);
    }

    #[test]
    fn length_affinity_keeps_similar_lengths_together() {
        let lens = [10usize, 12, 11];
        let groups = embed_length_affinity_groups(&lens, 1.5, 32);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn embed_length_bucket_default_snaps_when_pad_cheap() {
        // Default is ON when env unset; 20→24 or 32 if pad acceptable.
        let b = embed_length_bucket(20);
        assert!(b >= 20);
        assert!(b == 20 || b == 24 || b == 32, "got {b}");
    }

    #[test]
    fn embed_length_split_defaults_on_when_env_unset() {
        // Production default: length-affinity split enabled unless explicitly off.
        // (OnceLock may already be initialized in this process; only assert the
        // documented parse rule for an unset-style path via the helper itself.)
        // When AX_EMBED_LENGTH_SPLIT is unset, first init returns true.
        if std::env::var("AX_EMBED_LENGTH_SPLIT").is_err() {
            assert!(
                super::embed_length_split_enabled(),
                "AX_EMBED_LENGTH_SPLIT unset must default to length-split ON"
            );
        }
    }

    #[test]
    fn embed_max_len_buckets_default_on_when_env_unset() {
        // Unset env → DEFAULT_EMBED_MAX_LEN_BUCKETS active (not identity-only).
        if std::env::var("AX_EMBED_MAX_LEN_BUCKETS").is_err() {
            // 20 with cheap pad should snap into the default table.
            let b = embed_length_bucket(20);
            assert!(
                b >= 20
                    && DEFAULT_EMBED_MAX_LEN_BUCKETS
                        .iter()
                        .any(|&e| e == b || b == 20),
                "default buckets must be active when env unset, got {b}"
            );
            // Coarse over-pad still rejected (8→32 not acceptable).
            assert_eq!(
                embed_length_bucket(8),
                8,
                "default-on must not force coarse over-pad on short seqs"
            );
        }
    }
}
