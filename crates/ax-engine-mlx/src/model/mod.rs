use mlx_sys::{
    MlxArray, MlxClosure, MlxDtype, MlxVectorArray, add, astype, broadcast_to, dequantize, reshape,
    rms_norm, rope, slice, split, take, take_along_axis, transpose,
};
use std::sync::Arc;
use std::time::Instant;

use crate::kv_cache::MlxKVCache;
use crate::weights::{LayerWeights, ModelWeights, QuantizedWeight};

pub(crate) mod profile;
pub use profile::{
    DecodeProfileSnapshot, Gemma4MoeProfileSnapshot, LinearAttentionProfileSnapshot,
    PrefillProfileSnapshot, take_decode_profile_snapshot, take_gemma4_moe_profile_snapshot,
    take_linear_attention_profile_snapshot, take_prefill_profile_snapshot,
};
use profile::{
    DecodeProfileStage, decode_profile_enabled, decode_profile_eval_elapsed,
    forward_profile_eval_elapsed, prefill_profile_enabled, record_decode_profile_step,
    record_prefill_profile_step,
};

mod turboquant_context;
pub use turboquant_context::{
    TurboQuantModelDecodeCandidate, TurboQuantModelDecodeCandidateStatus,
    TurboQuantModelDecodeContext,
};

mod config;
use config::layer_params;
pub use config::{
    GlmRouterConfig, LayerConfig, LinearAttentionConfig, MlaAttentionConfig, ModelConfig,
};

pub(crate) mod shared;
pub(crate) use shared::scale_hidden_pub;
use shared::*;

mod families;

#[cfg(test)]
use crate::attention_mask::create_causal_mask;
#[cfg(test)]
use crate::kv_cache::MlxKvCompressionDecodeOutcome;
#[cfg(test)]
use ax_engine_core::{KvCompressionConfig, NativeModelManifest, TurboQuantPreset};
#[cfg(test)]
use mlx_sys::expand_dims_axes;
#[cfg(test)]
use mlx_sys::{
    ScaledDotProductAttentionMask, gelu_approx, matmul, multiply,
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
    layer_forward_with_turboquant_context(
        cfg,
        w,
        hidden,
        cache,
        layer_idx,
        token_offset,
        per_layer_input,
        shared_mask,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn layer_forward_with_turboquant_context(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray, // [1, seq, hidden]
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    per_layer_input: Option<&MlxArray>, // [1, seq, per_layer_dim] or None
    shared_mask: Option<&Option<MlxArray>>,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    // Linear-attention layers dispatch before the family match because the same
    // model (qwen3_5 / qwen3_next) has both linear and full-attention layers
    // and the family string alone is not enough to disambiguate per-layer type.
    if cfg.is_linear_attention_layer(layer_idx) {
        return families::qwen3_linear::layer_forward(cfg, w, hidden, cache, layer_idx);
    }

    match cfg.model_family.as_str() {
        "gemma4" | "gemma3" | "qwen3" | "llama3" => families::standard::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            per_layer_input,
            shared_mask,
            turboquant_context,
            false,
        ),
        "llama4" => families::llama4::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            shared_mask,
            turboquant_context,
        ),
        "qwen3_5" | "qwen3_next" => families::standard::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            per_layer_input,
            shared_mask,
            turboquant_context,
            false,
        ),
        "glm4_moe_lite" => {
            families::glm4_moe_lite::layer_forward(cfg, w, hidden, cache, layer_idx, token_offset)
        }
        "deepseek_v3" | "deepseek_v32" => families::deepseek_v3::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            turboquant_context,
        ),
        "mistral3" => families::mistral3::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            shared_mask,
            turboquant_context,
        ),
        "mixtral" => families::mixtral::layer_forward(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            shared_mask,
            turboquant_context,
        ),
        f => panic!("unknown model_family in layer_forward: {f}"),
    }
}

/// Run a transformer layer with the "last-position-only after attention"
/// optimization enabled. Equivalent to
/// [`layer_forward_with_turboquant_context`] except that, for `seq > 1`,
/// the layer's pre-FFN slice happens **inside** the layer (between the
/// attention residual and the pre-FFN norm) instead of after the layer
/// returns. The MLP / MoE / per-layer-gate / layer-scalar steps then run
/// on `[1, 1, hidden]`, matching `mlx-lm`'s lazy-eval prune of the
/// terminal layer's post-attention work.
///
/// **Caller obligation:** only invoke this for the *last* transformer
/// layer of a prefill pass. Passing a 1-position hidden into the next
/// layer's attention would corrupt the KV cache (positions would no
/// longer align with `token_offset`).
///
/// Supported families: `gemma4`, `gemma3`, `qwen3`, `llama3`, `qwen3_5`,
/// `qwen3_next` (full-attention layers only). Linear-attention layers
/// and other families fall back to the normal `layer_forward` path; the
/// post-loop slice in [`forward_with_turboquant_context`] keeps
/// correctness while losing this specific perf win until those families
/// pick up the same optimization.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward_with_turboquant_context_last_only(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    per_layer_input: Option<&MlxArray>,
    shared_mask: Option<&Option<MlxArray>>,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    if cfg.is_linear_attention_layer(layer_idx) {
        // qwen3_linear layers haven't been extended yet; fall back to the
        // normal path. The outer post-loop slice in `forward()` still
        // ensures correctness for the final returned hidden tensor.
        return families::qwen3_linear::layer_forward(cfg, w, hidden, cache, layer_idx);
    }
    match cfg.model_family.as_str() {
        "gemma4" | "gemma3" | "qwen3" | "llama3" | "qwen3_5" | "qwen3_next" => {
            families::standard::layer_forward(
                cfg,
                w,
                hidden,
                cache,
                layer_idx,
                token_offset,
                per_layer_input,
                shared_mask,
                turboquant_context,
                true,
            )
        }
        // Other families: fall back to the unoptimized path. Correctness
        // is preserved by the post-loop slice in
        // `forward_with_turboquant_context`; perf gain is deferred until
        // the family is extended.
        _ => layer_forward_with_turboquant_context(
            cfg,
            w,
            hidden,
            cache,
            layer_idx,
            token_offset,
            per_layer_input,
            shared_mask,
            turboquant_context,
        ),
    }
}

/// Embed token IDs and return hidden states of shape [1, seq_len, hidden].
/// Embed tokens from a pre-built 1-D `[seq]` token-ID array.
///
/// Accepts lazy (unevaluated) arrays — all ops are lazy MLX graph nodes — so
/// this can be called with a GPU argmax result before it has been materialised.
/// Used internally by `embed_tokens` (materialized path) and by
/// `forward_lazy_single` (double-buffer pipelining path).
fn embed_tokens_arr(
    ids_1d: &MlxArray,
    embedding: &QuantizedWeight,
    hidden_size: usize,
) -> MlxArray {
    let seq = ids_1d.shape()[0]; // shape metadata is available without eval
    if let Some(scales) = &embedding.scales {
        let row_w = take(&embedding.weight, ids_1d, 0, None);
        let row_s = take(scales, ids_1d, 0, None);
        let row_b = embedding.biases.as_ref().map(|b| take(b, ids_1d, 0, None));
        let flat = dequantize(
            &row_w,
            &row_s,
            row_b.as_ref(),
            Some(embedding.group_size),
            Some(embedding.bits),
            None,
        );
        reshape(&flat, &[1, seq, hidden_size as i32], None)
    } else {
        let flat = take(&embedding.weight, ids_1d, 0, None);
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

/// Full forward pass: returns logits for the LAST token only — `[vocab_size]` f32.
pub fn forward(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    forward_with_turboquant_context(cfg, weights, token_ids, cache, token_offset, None)
}

pub fn forward_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    let profile_prefill = token_ids.len() > 1 && prefill_profile_enabled();

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
    let masks = build_layer_masks(cfg, weights.layers.len(), seq, token_offset + seq);
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
    // Last-position-only prefill optimization (matches mlx-lm's implicit
    // lazy-eval prune): for `seq > 1`, the final transformer layer slices
    // its attention-residual stream to the last position before running
    // the post-attention FFN / gate / scalar. Saves ~50% of the last
    // layer's wall time on long prompts (Gemma 4 E2B p=2048: ~8k → target
    // mlx-lm parity at ~16-18k tok/s prefill). Decode path (seq == 1)
    // never triggers the inner slice. See
    // `layer_forward_with_turboquant_context_last_only` for the contract.
    let last_layer_idx = weights.layers.len().saturating_sub(1);
    let use_last_layer_optimization = seq > 1;
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = if use_last_layer_optimization && li == last_layer_idx {
            layer_forward_with_turboquant_context_last_only(
                cfg,
                layer_w,
                &hidden,
                cache,
                li,
                token_offset,
                pli,
                Some(&masks[li]),
                turboquant_context,
            )
        } else {
            layer_forward_with_turboquant_context(
                cfg,
                layer_w,
                &hidden,
                cache,
                li,
                token_offset,
                pli,
                Some(&masks[li]),
                turboquant_context,
            )
        };
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
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    let logits = reshape(&logits_f32, &[cfg.vocab_size as i32], None);
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
    forward_all_positions_with_turboquant_context(
        cfg,
        weights,
        token_ids,
        cache,
        token_offset,
        None,
    )
}

pub fn forward_all_positions_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
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
    let masks = build_layer_masks(cfg, weights.layers.len(), seq, token_offset + seq);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &ids_1d, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward_with_turboquant_context(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
            turboquant_context,
        );
    }

    let seq = seq as i32;
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    reshape(&logits_f32, &[seq, cfg.vocab_size as i32], None)
}

/// Cache-free single transformer layer for dense embedding models.
///
/// Equivalent to the standard dense-attention path in `layer_forward`, but
/// skips all KV-cache writes (no `zeros` allocation, no `slice_update`).
/// Only valid for Qwen3/Gemma dense layers — no linear attention, no MLA,
/// no KV-source sharing, no MoE.
fn layer_forward_dense_embed(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray, // [batch, seq, hidden]
    layer_idx: usize,
) -> MlxArray {
    let (head_dim, rope_theta, rope_dims, _sliding_window, _kv_source, v_norm_no_scale) =
        layer_params(cfg, layer_idx);
    let batch = hidden.shape()[0] as usize;
    let seq = hidden.shape()[1] as usize;

    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);

    // 2-7. QKV projections, reshape, QK-norm, transpose, RoPE.
    let (q_raw, k_raw, v_raw, _attn_gate) = qkv_project(cfg, w, &normed, head_dim);
    let kv_heads = (k_raw.shape()[2] as usize)
        .checked_div(head_dim)
        .expect("k projection output must divide by head_dim");

    let q = reshape(
        &q_raw,
        &[
            batch as i32,
            seq as i32,
            cfg.n_heads as i32,
            head_dim as i32,
        ],
        None,
    );
    let k = reshape(
        &k_raw,
        &[batch as i32, seq as i32, kv_heads as i32, head_dim as i32],
        None,
    );
    let v = reshape(
        &v_raw,
        &[batch as i32, seq as i32, kv_heads as i32, head_dim as i32],
        None,
    );

    let q = qk_norm_bshd(
        q,
        w.q_norm.as_ref(),
        cfg.n_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    let k = qk_norm_bshd(
        k,
        w.k_norm.as_ref(),
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );

    let q = transpose(&q, &[0, 2, 1, 3], None);
    let k = transpose(&k, &[0, 2, 1, 3], None);
    let v = prepare_value_bhsd(
        v,
        v_norm_no_scale,
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );

    let q_rope = rope(
        &q,
        rope_dims as i32,
        false,
        Some(rope_theta),
        1.0,
        0,
        None,
        None,
    );
    let k_rope = rope(
        &k,
        rope_dims as i32,
        false,
        Some(rope_theta),
        1.0,
        0,
        None,
        None,
    );

    // 8. SDPA — k_rope/v used directly, no KV-cache writes.
    let mask_opt: Option<MlxArray> = None; // resolves to Causal in full_precision_attention
    let attn_sdpa = full_precision_attention(&q_rope, &k_rope, &v, cfg.query_scale, seq, &mask_opt);

    // 9-13. Transpose back, reshape, output projection, residual.
    let attn_out = transpose(&attn_sdpa, &[0, 2, 1, 3], None);
    let attn_flat = reshape(
        &attn_out,
        &[batch as i32, seq as i32, (cfg.n_heads * head_dim) as i32],
        None,
    );
    let attn_proj = attention_output_projection(
        &attn_flat,
        None,
        w.o_proj
            .as_ref()
            .expect("dense embed layer must have o_proj"),
    );
    let hidden = add(hidden, &attn_proj, None);

    // 14-17. Pre-FFN norm, dense SwiGLU, residual.
    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    let ffn_out = ffn_swiglu(cfg, w, &normed2);
    add(&hidden, &ffn_out, None)
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
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    // `dequantize` produces bf16 for bf16-scales quantized embeddings; only
    // insert the cast when the embed_tokens output is not already bf16
    // (e.g. unquantized fp16/fp32 embeddings). Skipping the redundant graph
    // node avoids one MLX FFI hop per call.
    if hidden.dtype() != MlxDtype::Bfloat16 {
        hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    }
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    forward_for_embedding_body(cfg, weights, hidden, target_position)
}

/// Body of `forward_for_embedding` starting from a pre-embedded bf16 hidden
/// state. Split out so the same logic can be wrapped in an `mlx_compile`
/// closure that fuses the per-layer dispatches into a single compiled graph.
fn forward_for_embedding_body(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    mut hidden: MlxArray,
    target_position: Option<usize>,
) -> MlxArray {
    for (li, layer_w) in weights.layers.iter().enumerate() {
        hidden = layer_forward_dense_embed(cfg, layer_w, &hidden, li);
    }
    let to_norm = match target_position {
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
    };
    rms_norm(&to_norm, Some(&weights.final_norm), cfg.rms_norm_eps, None)
}

/// Build an `mlx_compile`-wrapped closure that takes the pre-embedded bf16
/// hidden state and returns the final norm-output for Last/Cls pooling.
///
/// The closure captures `Arc<ModelWeights>` and `ModelConfig` by value so the
/// caller can drop them safely. The compiled graph is shape-specific (the
/// `seq_len` and `target_position` are baked into the trace), so callers
/// must cache one closure per `(seq_len, target_position)` shape.
///
/// Falling back to the imperative `forward_for_embedding_body` is always
/// correct; this path exists purely to amortize the per-op MLX C-API
/// dispatch cost across the ~28 ops/layer × N layers of the forward pass.
pub fn build_embedding_forward_closure(
    cfg: Arc<ModelConfig>,
    weights: Arc<ModelWeights>,
    target_position: Option<usize>,
) -> Result<MlxClosure, &'static str> {
    let body_closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
        if inputs.is_empty() {
            return vec![];
        }
        let hidden = inputs.get(0);
        let out = forward_for_embedding_body(&cfg, &weights, hidden, target_position);
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
    let actual_lens: Vec<usize> = batch_token_ids.iter().map(Vec::len).collect();
    let max_len = *actual_lens.iter().max().expect("non-empty batch");
    let batch = batch_token_ids.len();

    let hidden = build_embedding_batch_hidden(cfg, weights, batch_token_ids, batch, max_len);
    let out = forward_for_embedding_batch_body(cfg, weights, hidden, target_positions);
    (out, actual_lens)
}

/// Pre-embed a token-id batch + bf16 cast + (optional) hidden states scale.
/// Returns `[batch, max_len, hidden]` bf16. Split out so the same prelude
/// can be used by the imperative and compiled-closure batch paths.
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
    if hidden.dtype() != MlxDtype::Bfloat16 {
        hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    }
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    hidden
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
) -> MlxArray {
    let batch = hidden.shape()[0];
    for (li, layer_w) in weights.layers.iter().enumerate() {
        hidden = layer_forward_dense_embed(cfg, layer_w, &hidden, li);
    }
    // Extract per-sequence positions before the final norm when the caller only
    // needs one token per sequence (Last/Cls pooling). This avoids norming the
    // full padded [B, max_seq, H] tensor.
    let to_norm = match target_positions {
        Some(positions) => {
            let batch_size = batch;
            let hidden_size = hidden.shape()[2] as usize;
            // Fast path: when all sequences extract at the same position
            // (e.g. Cls pooling at index 0, or Last pooling on equal-length
            // batches) we can replace the gather with a zero-copy strided
            // slice — cheaper than take_along_axis(broadcast(...)).
            let common_pos = positions
                .first()
                .copied()
                .filter(|&first| positions.iter().all(|&p| p == first));
            if let Some(pos) = common_pos {
                let pos_i32 = pos as i32;
                let sliced = slice(
                    &hidden,
                    &[0, pos_i32, 0],
                    &[batch_size, pos_i32 + 1, hidden_size as i32],
                    &[1, 1, 1],
                    None,
                );
                reshape(&sliced, &[batch_size, hidden_size as i32], None) // [B, H]
            } else {
                let pos_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
                let idx_b11 = MlxArray::from_raw_data(
                    pos_u32.as_ptr() as *const u8,
                    pos_u32.len() * std::mem::size_of::<u32>(),
                    &[batch_size, 1_i32, 1_i32],
                    MlxDtype::Uint32,
                );
                let idx_broadcast =
                    broadcast_to(&idx_b11, &[batch_size, 1_i32, hidden_size as i32], None);
                let gathered = take_along_axis(&hidden, &idx_broadcast, 1, None); // [B, 1, H]
                reshape(&gathered, &[batch_size, hidden_size as i32], None) // [B, H]
            }
        }
        None => hidden,
    };
    rms_norm(&to_norm, Some(&weights.final_norm), cfg.rms_norm_eps, None)
}

/// Build an `mlx_compile`-wrapped closure for the batched embedding forward.
/// The closure takes the pre-embedded `[B, max_seq, H]` bf16 hidden state and
/// returns the final norm output. `target_positions` is baked into the trace,
/// so callers must cache one closure per `(batch_size, max_seq,
/// target_positions)` shape combination.
pub fn build_embedding_batch_forward_closure(
    cfg: Arc<ModelConfig>,
    weights: Arc<ModelWeights>,
    target_positions: Option<Vec<usize>>,
) -> Result<MlxClosure, &'static str> {
    let body_closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
        if inputs.is_empty() {
            return vec![];
        }
        let hidden = inputs.get(0);
        let out =
            forward_for_embedding_batch_body(&cfg, &weights, hidden, target_positions.as_deref());
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
    let max_len = *actual_lens.iter().max().expect("non-empty batch");
    let batch = batch_token_ids.len();
    let hidden = build_embedding_batch_hidden(cfg, weights, batch_token_ids, batch, max_len);
    (hidden, batch, max_len, actual_lens)
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
    forward_lazy_single_with_turboquant_context(cfg, weights, token_arr, cache, token_offset, None)
}

pub fn forward_lazy_single_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_arr: &MlxArray, // scalar or [1] u32; may be unevaluated (lazy)
    cache: &mut MlxKVCache,
    token_offset: usize,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    let profile_decode = decode_profile_enabled();

    // Normalise to [1] for embedding take (reshape is a no-op if already [1]).
    let tok_1d = reshape(token_arr, &[1_i32], None);

    let mut hidden = embed_tokens_arr(&tok_1d, &weights.token_embedding, cfg.hidden_size);
    if hidden.dtype() != MlxDtype::Bfloat16 {
        hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    }
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    let masks = build_layer_masks(cfg, weights.layers.len(), 1, token_offset + 1);

    let per_layer_started = profile_decode.then(Instant::now);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &tok_1d, &hidden);
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

    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward_with_turboquant_context(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
            turboquant_context,
        );
    }
    // Single token: hidden shape is [1, 1, hidden_size] — no sequence slice needed.
    // Return the logits as `[1, 1, vocab]`; the only consumer is `argmax(_, None)`
    // which operates on the last axis and produces `[1, 1]` (later read via
    // `data_u32().first()`). Skipping the flatten saves one reshape op per step
    // versus returning a 1-D `[vocab]` array.
    let lm_head_started = profile_decode.then(Instant::now);
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits = apply_final_logit_softcap(cfg, &logits_f32);
    if let Some(started) = lm_head_started {
        decode_profile_eval_elapsed(
            profile_decode,
            DecodeProfileStage::LmHead,
            started,
            &[&logits],
        );
    }
    if profile_decode {
        record_decode_profile_step(weights.layers.len() as u32);
    }
    logits
}

// ── private helpers ──────────────────────────────────────────────────────────

/// Compute per-layer input vectors for Gemma4 2B/4B models from a pre-built
/// 1-D `[seq]` token-ID array.  Accepts lazy (unevaluated) arrays.
///
/// Returns `Some(Vec<MlxArray>)` of length `num_layers`, each `[1, seq, per_layer_dim]`,
/// or `None` when the model does not use per-layer input gating.
///
/// Reference: Gemma4TextModel._get_per_layer_inputs + _project_per_layer_inputs.
fn compute_per_layer_inputs_arr(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ids_1d: &MlxArray, // [seq] u32, may be unevaluated
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
    let seq = ids_1d.shape()[0]; // shape metadata available without eval
    let dtype = MlxDtype::Bfloat16;

    // 1. Per-layer token embeddings: [1, seq, num_layers * per_layer_dim]
    //    embed_tokens_per_layer(input_ids) * sqrt(per_layer_dim)
    let embed_out = embed_tokens_arr(ids_1d, embed_w, num_layers * per_layer_dim);
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
    let combined_scale = GEMMA4_PER_LAYER_COMBINED_SCALE;
    let combined = add(&proj_out, &embed_out, None);
    let combined = scale_hidden(&combined, combined_scale);
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
        NativeLinearAttentionConfig, NativeMoeConfig, NativeRuntimeStatus, NativeTensorFormat,
    };
    use mlx_sys::{eval, zeros};
    use std::collections::BTreeMap;

    fn cfg(attn_output_gate: bool) -> ModelConfig {
        ModelConfig {
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
            gemma4_moe_router: false,
            uses_geglu: false,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            rms_norm_eps: 1e-6,
            rope_freqs: None,
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
        }
    }

    fn turboquant_decode_config() -> KvCompressionConfig {
        KvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..KvCompressionConfig::turboquant_fused_experimental()
        }
    }

    fn turboquant_dense_cfg() -> ModelConfig {
        let mut cfg = cfg(false);
        cfg.hidden_size = 256;
        cfg.n_heads = 2;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 128;
        cfg.rope_dims = 128;
        cfg.query_scale = 1.0 / (128.0_f32).sqrt();
        cfg
    }

    fn turboquant_cache_with_runtime_storage() -> MlxKVCache {
        let mut cache = MlxKVCache::new(1);
        let elements = 2 * 6 * 128;
        let k = zeros(&[1, 2, 6, 128], MlxDtype::Float32, None);
        let v_data = (0..elements)
            .map(|idx| ((idx % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let v = MlxArray::from_raw_data(
            v_data.as_ptr().cast(),
            v_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 128],
            MlxDtype::Float32,
        );
        let compression = turboquant_decode_config();
        cache.append(0, k, v);
        cache.seq_len = 6;
        cache.sync_turboquant_shadow_storage(&[None], compression, Some(&[true]));
        cache
    }

    fn turboquant_cache_with_runtime_storage_and_current_decode_token() -> MlxKVCache {
        turboquant_cache_with_runtime_storage_and_current_decode_token_for_config(
            turboquant_decode_config(),
        )
    }

    fn turboquant_cache_with_runtime_storage_and_current_decode_token_for_config(
        compression: KvCompressionConfig,
    ) -> MlxKVCache {
        let mut cache = MlxKVCache::new(1);
        let initial_elements = 2 * 6 * 128;
        let initial_k = zeros(&[1, 2, 6, 128], MlxDtype::Float32, None);
        let initial_v_data = (0..initial_elements)
            .map(|idx| ((idx % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let initial_v = MlxArray::from_raw_data(
            initial_v_data.as_ptr().cast(),
            initial_v_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 128],
            MlxDtype::Float32,
        );
        cache.append(0, initial_k, initial_v);
        cache.seq_len = 6;
        cache.sync_turboquant_shadow_storage(&[None], compression, Some(&[true]));

        let current_elements = 2 * 128;
        let current_k = zeros(&[1, 2, 1, 128], MlxDtype::Float32, None);
        let current_v_data = (0..current_elements)
            .map(|idx| ((idx % 11) as f32 - 5.0) / 13.0)
            .collect::<Vec<_>>();
        let current_v = MlxArray::from_raw_data(
            current_v_data.as_ptr().cast(),
            current_v_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 1, 128],
            MlxDtype::Float32,
        );
        cache.append(0, current_k, current_v);
        cache
    }

    #[test]
    fn turboquant_model_decode_context_gates_candidate_layers() {
        let cfg = turboquant_dense_cfg();
        let compression = turboquant_decode_config();
        let context = TurboQuantModelDecodeContext {
            config: compression,
            layer_eligible: &[true],
        };
        let mut cache = MlxKVCache::new(1);
        cache.seq_len = 6;

        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 2, 128, 2, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::PrefillOnly
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 128, 2, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::MissingRuntimeStorage
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 256, 1, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::MissingRuntimeStorage
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 128, 2, Some(128), None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::SlidingWindowLayer
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 128, 2, None, Some(0), false)
                .status,
            TurboQuantModelDecodeCandidateStatus::KvSharedLayer
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 64, 2, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::UnsupportedHeadDim
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 128, 3, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::GroupedQueryAttention
        );

        let context = TurboQuantModelDecodeContext {
            config: compression,
            layer_eligible: &[false],
        };
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 128, 2, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::IneligibleLayer
        );
    }

    #[test]
    fn turboquant_model_decode_context_marks_runtime_storage_ready() {
        let cfg = turboquant_dense_cfg();
        let cache = turboquant_cache_with_runtime_storage();
        let context = TurboQuantModelDecodeContext {
            config: turboquant_decode_config(),
            layer_eligible: &[true],
        };

        let candidate = context.decode_candidate(&cfg, &cache, 0, 1, 128, 2, None, None, false);

        assert_eq!(
            candidate.status,
            TurboQuantModelDecodeCandidateStatus::Ready
        );
        assert_eq!(candidate.cold_tokens, 4);
        assert_eq!(candidate.hot_tokens, 3);
    }

    #[test]
    fn turboquant_decode_attention_experimental_prefers_metal_runtime_storage() {
        let cache = turboquant_cache_with_runtime_storage_and_current_decode_token();
        let q_data = (0..(2 * 128))
            .map(|idx| ((idx % 19) as f32 - 9.0) / 31.0)
            .collect::<Vec<_>>();
        let q_rope = MlxArray::from_raw_data(
            q_data.as_ptr().cast(),
            q_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 1, 128],
            MlxDtype::Float32,
        );
        let expected_queries = q_data
            .chunks_exact(128)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let expected = cache
            .debug_turboquant_shadow_decode_attention_for_layer_with_total_tokens(
                0,
                &expected_queries,
                turboquant_decode_config().hot_window_tokens,
                7,
            )
            .expect("runtime storage should decode cold and hot partitions");

        let actual = turboquant_decode_attention_experimental(
            &cache,
            0,
            &q_rope,
            1,
            2,
            128,
            (128.0_f32).sqrt().recip(),
        )
        .expect("ready TurboQuant decoder should decode from runtime storage");
        assert_eq!(actual.outcome, MlxKvCompressionDecodeOutcome::Metal);
        eval(&[&actual.attention]);

        assert_eq!(actual.attention.shape(), vec![1, 2, 1, 128]);
        let actual_data = actual.attention.data_f32();
        let expected_data = expected.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(actual_data.len(), expected_data.len());
        for (actual, expected) in actual_data.iter().zip(expected_data) {
            assert!((actual - expected).abs() <= 0.05);
        }
    }

    #[test]
    fn turboquant_decode_attention_experimental_applies_model_query_scale() {
        let cache = turboquant_cache_with_runtime_storage_and_current_decode_token();
        let q_data = (0..(2 * 128))
            .map(|idx| ((idx % 23) as f32 - 11.0) / 37.0)
            .collect::<Vec<_>>();
        let q_rope = MlxArray::from_raw_data(
            q_data.as_ptr().cast(),
            q_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 1, 128],
            MlxDtype::Float32,
        );
        let base_scale = (128.0_f32).sqrt().recip();
        let query_scale = base_scale * 2.0;
        let expected_queries = q_data
            .chunks_exact(128)
            .map(|chunk| chunk.iter().map(|value| value * 2.0).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let expected = cache
            .debug_turboquant_shadow_decode_attention_for_layer_with_total_tokens(
                0,
                &expected_queries,
                turboquant_decode_config().hot_window_tokens,
                7,
            )
            .expect("runtime storage should decode scaled queries");

        let actual =
            turboquant_decode_attention_experimental(&cache, 0, &q_rope, 1, 2, 128, query_scale)
                .expect("ready TurboQuant decoder should accept model-specific query scale");
        assert_eq!(actual.outcome, MlxKvCompressionDecodeOutcome::Metal);
        eval(&[&actual.attention]);

        let actual_data = actual.attention.data_f32();
        let expected_data = expected.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(actual_data.len(), expected_data.len());
        for (actual, expected) in actual_data.iter().zip(expected_data) {
            assert!((actual - expected).abs() <= 0.05);
        }
    }

    #[test]
    fn turboquant_decode_attention_experimental_does_not_use_cpu_oracle_in_runtime_path() {
        let mut compression = turboquant_decode_config();
        compression.preset = TurboQuantPreset::K4V4;
        let cache =
            turboquant_cache_with_runtime_storage_and_current_decode_token_for_config(compression);
        let q_data = (0..(2 * 128))
            .map(|idx| ((idx % 29) as f32 - 14.0) / 41.0)
            .collect::<Vec<_>>();
        let q_rope = MlxArray::from_raw_data(
            q_data.as_ptr().cast(),
            q_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 1, 128],
            MlxDtype::Float32,
        );
        let queries = q_data
            .chunks_exact(128)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        cache
            .debug_turboquant_shadow_decode_attention_for_layer_with_total_tokens(
                0,
                &queries,
                compression.hot_window_tokens,
                7,
            )
            .expect("CPU oracle remains available for debug comparisons");

        let actual = turboquant_decode_attention_experimental(
            &cache,
            0,
            &q_rope,
            1,
            2,
            128,
            (128.0_f32).sqrt().recip(),
        );

        assert!(
            actual.is_none(),
            "runtime path should fall back to full-precision SDPA instead of CPU oracle"
        );
    }

    #[test]
    fn turboquant_attention_output_array_from_flat_skips_float32_cast() {
        let output = vec![0.25, -0.5, 0.75, 1.0];
        let actual =
            turboquant_attention_output_array_from_flat(output.clone(), 2, 2, MlxDtype::Float32)
                .expect("flat output should become attention array");
        eval(&[&actual]);

        assert_eq!(actual.shape(), vec![1, 2, 1, 2]);
        assert_eq!(actual.dtype(), MlxDtype::Float32);
        assert_eq!(actual.data_f32(), output.as_slice());
    }

    #[test]
    fn turboquant_query_readback_array_borrows_float32_input() {
        let q_data = vec![0.25, -0.5, 0.75, 1.0];
        let q_rope = MlxArray::from_raw_data(
            q_data.as_ptr().cast(),
            q_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 1, 2],
            MlxDtype::Float32,
        );

        let readback = turboquant_query_readback_array(&q_rope);
        assert!(matches!(
            readback,
            TurboQuantQueryReadbackArray::Borrowed(_)
        ));
        let readback_array = readback.as_array();
        eval(&[readback_array]);

        assert_eq!(readback_array.shape(), vec![1, 2, 1, 2]);
        assert_eq!(readback_array.dtype(), MlxDtype::Float32);
        assert_eq!(readback_array.data_f32(), q_data.as_slice());
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
            weight_sanitize: ax_engine_core::WeightSanitize::None,
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
            weight_sanitize: ax_engine_core::WeightSanitize::None,
            tensors: Vec::new(),
        }
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
            weight_sanitize: ax_engine_core::WeightSanitize::None,
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
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
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
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
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
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
            rotation_smoothing_inverse: None,
        }
    }

    #[test]
    fn qkv_slices_dense_attention_without_gate() {
        assert_eq!(
            qkv_slices(&cfg(false), 8),
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
            qkv_slices(&cfg(true), 8),
            QkvSlices {
                q: (0, 16),
                gate: Some((16, 32)),
                k: (32, 40),
                v: (40, 48),
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
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
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

        let out = ffn_swiglu(&cfg, &weights, &x);

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
        assert_eq!(cfg.layer_configs[1].head_dim, 512);
        assert_eq!(cfg.layer_configs[1].rope_theta, 1_000_000.0);
        assert_eq!(cfg.layer_configs[1].rope_dims, 128);
        assert_eq!(cfg.layer_configs[1].sliding_window, None);
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
            None,
            /* last_position_only_after_attention */ false,
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
            None,
            /* last_position_only_after_attention */ true,
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

        cache.seq_len = 2;
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

        cache.seq_len = 2;
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
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: Some(dense_weight(&[2, 6, 4])),
            gate_exps: None,
            up_exps: None,
            down_exps: Some(dense_weight(&[2, 4, 3])),
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
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: Some(dense_weight(&[2, 6, 4])),
            gate_exps: None,
            up_exps: None,
            down_exps: Some(dense_weight(&[2, 4, 3])),
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
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: Some(dense_weight(&[2, 3, 4])),
            up_exps: Some(dense_weight(&[2, 3, 4])),
            down_exps: Some(dense_weight(&[2, 4, 3])),
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
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: Some(dense_weight(&[4, 3, 4])),
            up_exps: Some(dense_weight(&[4, 3, 4])),
            down_exps: Some(dense_weight(&[4, 4, 3])),
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
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: Some(dense_weight(&[4, 6, 4])),
            gate_exps: None,
            up_exps: None,
            down_exps: Some(dense_weight(&[4, 4, 3])),
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
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: Some(dense_weight(&[4, 3, 4])),
            up_exps: Some(dense_weight(&[4, 3, 4])),
            down_exps: Some(dense_weight(&[4, 4, 3])),
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
        // decode (seq=1) with cache offset — offset > 0 so explicit mask still built
        let mask = attention_mask_array(1, 6, Some(4)).expect("decode needs sliding mask");

        assert_eq!(mask.shape(), vec![1, 6]);
    }

    #[test]
    fn attention_mask_key_len_matches_decode_windowed_kv_views() {
        assert_eq!(attention_mask_key_len(1, 6, Some(4)), 4);
        assert_eq!(attention_mask_key_len(1, 6, None), 6);
        assert_eq!(attention_mask_key_len(2, 6, Some(4)), 6);
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
        let n_heads = 4_usize;
        let head_dim = 3_usize;
        let seq = 2_usize;
        let total = seq * n_heads * head_dim;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let proj = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, seq as i32, (n_heads * head_dim) as i32],
            MlxDtype::Float32,
        );

        // Reference: reshape [1, seq, n_heads*head_dim] -> [1, seq, n_heads, head_dim]
        // then transpose [0, 2, 1, 3] -> [1, n_heads, seq, head_dim].
        let reference_bsh = reshape(
            &proj,
            &[1, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
        let reference = transpose(&reference_bsh, &[0, 2, 1, 3], None);

        // Candidate: single as_strided view directly to BHSD.
        let candidate = bhsd_view_from_proj(&proj, n_heads, head_dim, seq);

        eval(&[&reference, &candidate]);

        // Both must report the same shape.
        assert_eq!(
            reference.shape(),
            vec![1, n_heads as i32, seq as i32, head_dim as i32]
        );
        assert_eq!(
            candidate.shape(),
            vec![1, n_heads as i32, seq as i32, head_dim as i32]
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
    fn geglu_compiled_matches_imperative() {
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

        // Compiled fusion via the geglu helper.
        let compiled = geglu(&gate, &x);
        let compiled_f32 = astype(&compiled, MlxDtype::Float32, None);

        eval(&[&imperative_f32, &compiled_f32]);

        let imp = imperative_f32.data_f32().to_vec();
        let cmp = compiled_f32.data_f32().to_vec();
        assert_eq!(
            imp, cmp,
            "compiled geglu must produce bit-identical output to the imperative fallback"
        );

        // Second invocation hits the cached compiled closure and must still match.
        let compiled_again = geglu(&gate, &x);
        let compiled_again_f32 = astype(&compiled_again, MlxDtype::Float32, None);
        eval(&[&compiled_again_f32]);
        assert_eq!(
            cmp,
            compiled_again_f32.data_f32().to_vec(),
            "cached compiled geglu must remain stable across invocations"
        );
    }

    #[test]
    fn per_layer_input_gate_compile_cache_is_independent_from_ffn_geglu() {
        // Regression for W5: the per-layer gate compile path must not reuse
        // the FFN GeGLU closure compiled on the same thread with a different
        // last dimension.
        let ffn_gate_f32: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.05).collect();
        let ffn_up_f32: Vec<f32> = (0..32).map(|i| ((i as f32) + 1.0) * 0.07).collect();
        let ffn_gate = astype(
            &array_f32(&ffn_gate_f32, &[1, 4, 8]),
            MlxDtype::Bfloat16,
            None,
        );
        let ffn_up = astype(
            &array_f32(&ffn_up_f32, &[1, 4, 8]),
            MlxDtype::Bfloat16,
            None,
        );
        let _ = geglu(&ffn_gate, &ffn_up);

        let gate_f32: Vec<f32> = (0..12).map(|i| ((i as f32) - 6.0) * 0.04).collect();
        let pli_f32: Vec<f32> = (0..12).map(|i| ((i as f32) + 1.0) * 0.03).collect();
        let gate = astype(&array_f32(&gate_f32, &[1, 4, 3]), MlxDtype::Bfloat16, None);
        let pli = astype(&array_f32(&pli_f32, &[1, 4, 3]), MlxDtype::Bfloat16, None);

        let imperative = multiply(&gelu_approx(&gate, None), &pli, None);
        let imperative_f32 = astype(&imperative, MlxDtype::Float32, None);
        let compiled = per_layer_input_gate_compiled(&gate, &pli)
            .expect("small per-layer gate shape should be compile-eligible");
        let compiled_f32 = astype(&compiled, MlxDtype::Float32, None);

        eval(&[&imperative_f32, &compiled_f32]);
        assert_eq!(
            imperative_f32.data_f32().to_vec(),
            compiled_f32.data_f32().to_vec(),
            "per-layer gate compile path must match the imperative fallback"
        );

        let gate_short = astype(
            &array_f32(&gate_f32[..6], &[1, 2, 3]),
            MlxDtype::Bfloat16,
            None,
        );
        let pli_short = astype(
            &array_f32(&pli_f32[..6], &[1, 2, 3]),
            MlxDtype::Bfloat16,
            None,
        );
        let expected_short = astype(
            &multiply(&gelu_approx(&gate_short, None), &pli_short, None),
            MlxDtype::Float32,
            None,
        );
        let compiled_short = astype(
            &per_layer_input_gate_compiled(&gate_short, &pli_short)
                .expect("same last-dim per-layer gate shape should be compile-eligible"),
            MlxDtype::Float32,
            None,
        );

        eval(&[&expected_short, &compiled_short]);
        assert_eq!(
            expected_short.data_f32().to_vec(),
            compiled_short.data_f32().to_vec(),
            "same-last-dim cache entry must tolerate sequence-length changes"
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
}
