use mlx_sys::{
    add, argmax, argsort_axis, astype, concatenate, eval, multiply, reshape, rms_norm,
    scaled_dot_product_attention_with_mask, sigmoid, slice, MlxArray, MlxDtype,
    ScaledDotProductAttentionMask,
};

use crate::kv_cache::MlxKVCache;
use crate::model::shared::{
    apply_final_logit_softcap, ffn_swiglu, flatten_attention_output_bhsd, moe_experts_forward,
    moe_router_deepseek_v3, moe_router_glm, moe_router_qwen3, prepare_value_bhsd_from_proj,
    qk_norm_rope_bhsd_from_proj, qw, shared_expert_forward,
};
use crate::model::{embed_tokens_arr, ModelConfig};
use crate::sampling::{
    full_vocab_token_logprob, sample_categorical_with_logprob_and_distribution, TokenDistribution,
    Xorshift64,
};
use crate::weights::{ModelWeights, MtpWeights};

/// Number of top-k candidate tokens pre-selected from the verify-step logits for
/// filtered draft LM head computation.  The draft argmax is restricted to these
/// candidates; the remaining vocabulary is never read from VRAM.  k=4096 at 4-bit
/// reads 10MB per draft pass vs 640MB for full lm_head; the extra coverage vs k=2048
/// reduces complete-miss rate on diverse code prompts without measurable bandwidth cost.
pub const MTP_DRAFT_TOP_K: usize = 4096;

/// Greedy argmax over a `[vocab_size]` f32 logit array; syncs GPU and returns token id.
fn greedy_sample_logits(logits: &MlxArray) -> u32 {
    let vocab = logits.shape()[0];
    let logits_2d = reshape(logits, &[1_i32, vocab], None);
    let idx = argmax(&logits_2d, None);
    eval(&[&idx]);
    idx.data_u32()[0]
}

/// Run one recurrent MTP head forward pass for a single decode step.
///
/// Returns new hidden state `[1, 1, hidden_size]`.  Caller applies
/// `rms_norm(h, mtp_norm) @ lm_head` to get draft logits.
///
/// * `head`           — shared MTP weights (reused across all depth levels).
/// * `main_hidden`    — post-norm hidden from the main model (hidden_variant="post_norm")
///   or output from a preceding MTP head call, shape `[1, 1, hidden_size]`.
/// * `prev_token_arr` — token ID as a GPU uint32 array, shape `[1]`.  May be a lazy
///   argmax result; no CPU sync is required before calling.
/// * `weights`        — main model weights (for the shared token embedding).
/// * `cache`          — shared 1-layer KV cache for this head (grows by 1 per call).
///
/// RoPE offset is taken from `cache.seq_len` before appending, matching the
/// mlx-lm `cache.offset` convention: position 0 for the first MTP call, 1 for
/// the second, etc.  Callers must NOT pass absolute sequence positions.
pub fn mtp_head_forward(
    head: &MtpWeights,
    main_hidden: &MlxArray,
    prev_token_arr: &MlxArray,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    cfg: &ModelConfig,
) -> MlxArray {
    // Use the MTP KV-cache length as the RoPE offset (matches mlx-lm cache.offset).
    let token_offset = cache.seq_len;
    // 1. Embed prev_token → [1, 1, hidden_size] in bf16.
    let embed = embed_tokens_arr(prev_token_arr, &weights.token_embedding, cfg.hidden_size);
    let embed = astype(&embed, MlxDtype::Bfloat16, None);

    // 2. Combined input: fc(cat([enorm(embed), hnorm(hidden)])).
    //    concat_order = "embedding_hidden" → [enorm, hnorm] along last dim.
    let enormed = rms_norm(
        &embed,
        Some(&head.pre_fc_norm_embedding),
        cfg.rms_norm_eps,
        None,
    );
    let hnormed = rms_norm(
        main_hidden,
        Some(&head.pre_fc_norm_hidden),
        cfg.rms_norm_eps,
        None,
    );
    let combined = concatenate(&[&enormed, &hnormed], -1, None);
    let mut h = qw(&combined, &head.fc);

    // 3. Attention sub-layer (Qwen3NextAttention).
    //
    // q_proj output = [1, 1, n_heads * head_dim * 2] with per-head interleaving:
    //   [h0_query(head_dim), h0_gate(head_dim), h1_query(head_dim), h1_gate(head_dim), ...]
    // This matches mlx-lm: `mx.split(q_proj_out.reshape(B,L,n_heads,-1), 2, axis=-1)`.
    // We must reshape to [1, 1, n_heads, 2*head_dim] and then slice the last dim —
    // NOT a simple first-half / second-half slice (which mixes heads).
    // Output = o_proj(sdpa_out * sigmoid(gate)), then residual.
    {
        let normed = rms_norm(&h, Some(&head.attn_norm), cfg.rms_norm_eps, None);

        // Reshape q_proj output to expose per-head query/gate layout.
        let n_h = head.n_heads as i32;
        let hd = head.head_dim as i32;
        let q_half = n_h * hd;
        let qg_raw = qw(&normed, &head.q_proj); // [1, 1, n_heads * head_dim * 2]
        let qg_heads = reshape(&qg_raw, &[1_i32, 1, n_h, 2 * hd], None); // [1, 1, n_heads, 2*hd]
        let q_raw = reshape(
            &slice(
                &qg_heads,
                &[0, 0, 0, 0],
                &[1, 1, n_h, hd],
                &[1, 1, 1, 1],
                None,
            ),
            &[1_i32, 1, q_half],
            None,
        );
        let gate = reshape(
            &slice(
                &qg_heads,
                &[0, 0, 0, hd],
                &[1, 1, n_h, 2 * hd],
                &[1, 1, 1, 1],
                None,
            ),
            &[1_i32, 1, q_half],
            None,
        );

        let k_raw = qw(&normed, &head.k_proj);
        let v_raw = qw(&normed, &head.v_proj);

        let v = prepare_value_bhsd_from_proj(
            &v_raw,
            false,
            head.n_kv_heads,
            head.head_dim,
            1,
            cfg.rms_norm_eps,
        );

        let (rope_base, rope_freqs_ref) = if let Some(freqs) = cfg.rope_freqs.as_ref() {
            (None, Some(freqs))
        } else {
            (Some(cfg.rope_theta), None)
        };

        let q_rope = qk_norm_rope_bhsd_from_proj(
            &q_raw,
            head.q_norm.as_ref(),
            head.n_heads,
            head.head_dim,
            1,
            cfg.rms_norm_eps,
            cfg.rope_dims,
            rope_base,
            token_offset,
            rope_freqs_ref,
        );
        let k_rope = qk_norm_rope_bhsd_from_proj(
            &k_raw,
            head.k_norm.as_ref(),
            head.n_kv_heads,
            head.head_dim,
            1,
            cfg.rms_norm_eps,
            cfg.rope_dims,
            rope_base,
            token_offset,
            rope_freqs_ref,
        );

        let (cached_k, cached_v) = cache.append(0, k_rope, v);
        cache.seq_len += 1;

        let query_scale = 1.0 / (head.head_dim as f32).sqrt();
        let attn_out = scaled_dot_product_attention_with_mask(
            &q_rope,
            &cached_k,
            &cached_v,
            query_scale,
            ScaledDotProductAttentionMask::None,
            None,
        );

        // Flatten [1, n_heads, 1, head_dim] → [1, 1, n_heads * head_dim].
        let attn_flat = flatten_attention_output_bhsd(&attn_out, 1, head.n_heads, head.head_dim);

        // Apply sigmoid gating: o_proj(attn_flat * sigmoid(gate)).
        let gated = multiply(&attn_flat, &sigmoid(&gate, None), None);
        let attn_proj = qw(&gated, &head.o_proj);
        h = add(&h, &attn_proj, None);
    }

    // 4. FFN sub-layer (SwiGLU).
    {
        let normed = rms_norm(&h, Some(&head.ffn_norm), cfg.rms_norm_eps, None);
        let ffn_out = if head.ffn_layer.router_proj.is_some() {
            let (top_k_indices, top_k_weights) = if cfg.glm_router.is_some() {
                moe_router_glm(cfg, &head.ffn_layer, &normed)
            } else if cfg.moe_sigmoid_routing {
                moe_router_deepseek_v3(cfg, &head.ffn_layer, &normed)
            } else {
                moe_router_qwen3(cfg, &head.ffn_layer, &normed)
            };
            let mut out = moe_experts_forward(
                cfg,
                &head.ffn_layer,
                &normed,
                &top_k_indices,
                &top_k_weights,
            );
            if head.ffn_layer.shared_gate_proj.is_some() {
                out = add(
                    &out,
                    &shared_expert_forward(cfg, &head.ffn_layer, &normed),
                    None,
                );
            }
            out
        } else {
            ffn_swiglu(cfg, &head.ffn_layer, &normed, None)
        };
        h = add(&h, &ffn_out, None);
    }

    h
}

/// Apply `rms_norm(hidden, mtp_norm) @ main_model.lm_head` to produce draft logits.
///
/// Returns f32 logits `[vocab_size]` ready for argmax / sampling.
pub fn mtp_hidden_to_logits(
    hidden: &MlxArray,
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
) -> MlxArray {
    let normed = mtp_hidden_post_norm(hidden, head, cfg);
    mtp_post_norm_to_logits(&normed, weights, cfg)
}

fn mtp_hidden_post_norm(hidden: &MlxArray, head: &MtpWeights, cfg: &ModelConfig) -> MlxArray {
    rms_norm(hidden, Some(&head.mtp_norm), cfg.rms_norm_eps, None)
}

fn mtp_post_norm_to_logits(
    post_norm_hidden: &MlxArray,
    weights: &ModelWeights,
    cfg: &ModelConfig,
) -> MlxArray {
    use mlx_sys::reshape as mlx_reshape;
    let logits = qw(post_norm_hidden, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    // [1, 1, vocab] → [vocab]
    mlx_reshape(&logits_f32, &[cfg.vocab_size as i32], None)
}

/// Compute the top-k candidate token indices from a logit row.
///
/// `logit_row` may have shape `[vocab]` or `[1, vocab]` (e.g. a slice with a
/// retained batch dimension); either is normalised to `[vocab]` internally.
///
/// Returns a `uint32` MlxArray of shape `[k]` containing the indices of the
/// k highest-logit tokens.  The array is lazily computed; callers must eval it
/// before passing to `take`.
pub fn mtp_top_k_candidates(logit_row: &MlxArray, k: usize) -> MlxArray {
    // Normalise to 1-D: take the last-dim size as vocab and flatten.
    let ndim = logit_row.ndim();
    let vocab = logit_row.shape()[ndim - 1];
    let flat = if ndim > 1 {
        reshape(logit_row, &[vocab], None)
    } else {
        logit_row.clone()
    };
    // argsort ascending on -logits gives descending-logit order.
    let m1 = MlxArray::from_f32(-1.0);
    let neg = multiply(&flat, &m1, None);
    let sorted = argsort_axis(&neg, 0, None); // [vocab] uint32, ascending = descending logit
    slice(&sorted, &[0], &[k as i32], &[1], None) // [k] top-k indices
}

/// Draft up to `head.max_depth` tokens by applying the MTP head recurrently.
///
/// Returns `(draft_tokens, draft_log_probs, draft_count)`:
/// * `draft_tokens[i]`    — greedy argmax draft token at depth `i+1`.
/// * `draft_log_probs[i]` — log-probability of `draft_tokens[i]` under the
///   temperature-scaled draft distribution. When `draft_sampling.temperature == 0`
///   (greedy), log_prob is 0.0 (point-mass convention).
/// * `draft_count`        — how many entries were appended to `cache`.
///
/// When `temperature > 0`, tokens are sampled from the configured draft sampler and
/// log_probs record the proposal probability used by rejection-sampling acceptance
/// in `run_mtp_decode`.
/// When `temperature == 0`, tokens are greedy argmax and acceptance falls back to exact
/// argmax comparison.
///
/// `candidates` — optional evaluated `uint32 [k]` array of top-k token indices from the
/// previous verify step.  When present, the lm_head computation is restricted to those k
/// tokens at ALL depths (cascaded filtered approach).  Sampled mode samples from the
/// configured draft sampler over this filtered set and records the proposal log-prob.
///
/// Returns `(draft_tokens, draft_log_probs, draft_distributions, added, top2_margins)`.
/// `top2_margins[d]` is the rank-1 minus rank-2 logit gap at depth `d` from the filtered
/// path's partial logits (0.0 for the full-vocab fallback path).  Used by ADR-009 EV gate.
///
/// Gracefully handles `weights.mtp = None` by returning empty.
#[allow(clippy::too_many_arguments)]
pub fn mtp_draft_tokens(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth_cap: Option<usize>,
    rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let Some(head) = weights.mtp.as_ref() else {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    };
    let max_depth = max_depth_cap.unwrap_or(head.max_depth).min(head.max_depth);
    if max_depth == 0 {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    }

    let use_temperature = head.draft_sampling.temperature > 0.0;
    let mut draft_tokens = Vec::with_capacity(max_depth);
    let mut draft_log_probs = Vec::with_capacity(max_depth);
    let mut draft_distributions = Vec::with_capacity(max_depth);

    let mut prev_hidden = first_hidden.clone();
    // Wrap first_token as a GPU uint32 [1] array.
    let first_token_data = [first_token];
    let mut prev_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        4,
        &[1_i32],
        MlxDtype::Uint32,
    );

    for _ in 0..max_depth {
        let new_hidden = mtp_head_forward(head, &prev_hidden, &prev_token_arr, weights, cache, cfg);
        let post_norm_hidden = mtp_hidden_post_norm(&new_hidden, head, cfg);

        // Always use full-vocab logits for correct rejection sampling.
        // The filtered path (top-k candidates) was an optimization to reduce memory
        // bandwidth, but it computed draft probabilities within a restricted set,
        // causing a probability space mismatch with the target model's full-vocab
        // probabilities.  This led to ~40 pp lower acceptance vs mtplx (which uses
        // full-vocab softmax for both draft and target).
        //
        // Full-vocab logits ensure p_draft(d) is computed from the true draft
        // distribution, matching mtplx's approach and maximizing acceptance rate.
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, weights, cfg);
        let draft_token = if use_temperature {
            eval(&[&logits]);
            let logits_cpu = logits.data_f32();
            let (draft_token, _filtered_log_prob, distribution) =
                sample_categorical_with_logprob_and_distribution(logits_cpu, head.draft_sampling, rng);
            // Use full-vocab log-prob for rejection sampling: filtered-distribution
            // probabilities are renormalized over the top-k/top-p subset, making
            // p_draft_filtered > p_draft_full for in-set tokens.  Dividing full-vocab
            // p_target by the inflated p_draft_filtered causes systematic over-rejection.
            let log_prob =
                full_vocab_token_logprob(logits_cpu, draft_token, head.draft_sampling.temperature);
            draft_log_probs.push(log_prob);
            if let Some(distribution) = distribution {
                draft_distributions.push(distribution);
            }
            draft_token
        } else {
            let draft_token = greedy_sample_logits(&logits); // syncs GPU
            draft_log_probs.push(0.0);
            draft_token
        };
        let tok_data = [draft_token];
        let tok_arr = MlxArray::from_raw_data(
            tok_data.as_ptr() as *const u8,
            4,
            &[1_i32],
            MlxDtype::Uint32,
        );
        draft_tokens.push(draft_token);
        // Chain through post-norm hidden: MTPLX trains depth>1 heads expecting
        // mtp_norm(prev_output) as input, matching depth-1's post-norm convention.
        // Passing raw new_hidden (no mtp_norm applied) causes ~50% reject rate.
        prev_hidden = post_norm_hidden;
        prev_token_arr = tok_arr;
    }

    // top2_margins is no longer computed (filtered path removed).
    // Return zeros for API compatibility; telemetry can be updated later if needed.
    let top2_margins = [0.0f32; 3];

    let added = draft_tokens.len();
    (
        draft_tokens,
        draft_log_probs,
        draft_distributions,
        added,
        top2_margins,
    )
}
