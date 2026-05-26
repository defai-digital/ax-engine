use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, add, argpartition_axis, astype, concatenate,
    eval, multiply, reshape, rms_norm, scaled_dot_product_attention_with_mask, sigmoid, silu_mul,
    slice, softmax, take, take_along_axis,
};

use crate::kv_cache::MlxKVCache;
use crate::model::shared::{
    apply_final_logit_softcap, flatten_attention_output_bhsd, prepare_value_bhsd_from_proj,
    qk_norm_rope_bhsd_from_proj, qw,
};
use crate::model::{ModelConfig, embed_tokens};
use crate::sampling::{MlxSamplingParams, Xorshift64, indexed_token_logprob};
use crate::weights::{ModelWeights, MtpWeights};

/// Run one recurrent MTP head forward pass for a single decode step.
///
/// Returns new hidden state `[1, 1, hidden_size]`.  Caller applies
/// `rms_norm(h, mtp_norm) @ lm_head` to get draft logits.
///
/// * `head`        — shared MTP weights (reused across all depth levels).
/// * `main_hidden` — post-norm hidden from the main model (hidden_variant="post_norm")
///   or output from a preceding MTP head call, shape `[1, 1, hidden_size]`.
/// * `prev_token`  — token ID predicted at the previous level.
/// * `weights`     — main model weights (for the shared token embedding).
/// * `cache`       — shared 1-layer KV cache for this head (grows by 1 per call).
///
/// RoPE offset is taken from `cache.seq_len` before appending, matching the
/// mlx-lm `cache.offset` convention: position 0 for the first MTP call, 1 for
/// the second, etc.  Callers must NOT pass absolute sequence positions.
pub fn mtp_head_forward(
    head: &MtpWeights,
    main_hidden: &MlxArray,
    prev_token: u32,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    cfg: &ModelConfig,
) -> MlxArray {
    // Use the MTP KV-cache length as the RoPE offset (matches mlx-lm cache.offset).
    let token_offset = cache.seq_len;
    // 1. Embed prev_token → [1, 1, hidden_size] in bf16.
    let embed = embed_tokens(&[prev_token], &weights.token_embedding, cfg.hidden_size);
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
        let gate = qw(&normed, &head.gate_proj);
        let up = qw(&normed, &head.up_proj);
        let ffn_hidden = silu_mul(&gate, &up, None);
        let ffn_out = qw(&ffn_hidden, &head.down_proj);
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
    use mlx_sys::reshape as mlx_reshape;
    let normed = rms_norm(hidden, Some(&head.mtp_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    // [1, 1, vocab] → [vocab]
    mlx_reshape(&logits_f32, &[cfg.vocab_size as i32], None)
}

/// Draft up to `head.max_depth` tokens by applying the MTP head recurrently.
///
/// Returns `(draft_tokens, draft_log_probs, draft_count)`:
/// * `draft_tokens[i]`    — greedy argmax draft token at depth `i+1`.
/// * `draft_log_probs[i]` — log-probability of `draft_tokens[i]` under the draft distribution.
///   When `draft_sampling.temperature == 0` (greedy), log_prob is 0.0 (point-mass convention).
/// * `draft_count`        — how many entries were appended to `cache`.
///
/// When `temperature > 0`, tokens are still selected by argmax for throughput; the returned
/// log_probs use temperature-scaled full-vocab probabilities and enable rejection-sampling
/// acceptance in `run_mtp_decode`. top-k/top-p are not applied on this fast path.
/// When `temperature == 0`, tokens are greedy argmax and acceptance falls back to exact
/// argmax comparison.
///
/// Gracefully handles `weights.mtp = None` by returning empty.
pub fn mtp_draft_tokens(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth_cap: Option<usize>,
    _rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, usize) {
    let Some(head) = weights.mtp.as_ref() else {
        return (vec![], vec![], 0);
    };
    let max_depth = max_depth_cap.unwrap_or(head.max_depth).min(head.max_depth);
    if max_depth == 0 {
        return (vec![], vec![], 0);
    }

    let use_temperature = head.draft_sampling.temperature > 0.0;
    let use_topk_logprob = use_temperature && head.draft_sampling.top_k > 0;
    let mut draft_tokens = Vec::with_capacity(max_depth);
    let mut draft_log_probs = Vec::with_capacity(max_depth);
    // Lazy arrays holding p(draft_token) under the temperature-scaled draft distribution.
    // Evaluated in one batch after all depth iterations to avoid per-step GPU sync.
    let mut lazy_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut lazy_topk_logprobs: Vec<Option<(MlxArray, MlxArray)>> = Vec::with_capacity(max_depth);

    let mut prev_hidden = first_hidden.clone();
    let mut prev_token = first_token;

    // inv_temp scalar, computed once outside the loop.
    let inv_temp_arr = if use_temperature {
        Some(MlxArray::from_f32(1.0 / head.draft_sampling.temperature))
    } else {
        None
    };

    for _ in 0..max_depth {
        let new_hidden = mtp_head_forward(head, &prev_hidden, prev_token, weights, cache, cfg);
        let logits = mtp_hidden_to_logits(&new_hidden, head, weights, cfg);

        let draft_token = {
            // Greedy argmax: transfers only 1 u32. The argmax eval also materialises
            // `logits` on GPU, which is required before we compute the lazy softmax below.
            let draft_token = greedy_sample_logits(&logits);
            if use_topk_logprob {
                lazy_topk_logprobs.push(greedy_topk_logprob_arrays(
                    &logits,
                    head.draft_sampling,
                    cfg.vocab_size,
                ));
            } else if let Some(ref inv_t) = inv_temp_arr {
                // Lazily compute p(draft_token) from the temperature-scaled distribution.
                // No eval here — all these lazy arrays are batched-eval'd after the loop.
                let scaled = multiply(&logits, inv_t, None);
                let probs = softmax(&scaled, -1, None); // [vocab]
                let idx_data = [draft_token as i32];
                let idx_arr = MlxArray::from_raw_data(
                    idx_data.as_ptr() as *const u8,
                    4,
                    &[1_i32],
                    MlxDtype::Int32,
                );
                let prob_d = take(&probs, &idx_arr, 0, None); // [1]
                lazy_probs.push(prob_d);
            } else {
                draft_log_probs.push(0.0);
            }
            draft_token
        };

        draft_tokens.push(draft_token);
        prev_hidden = new_hidden;
        prev_token = draft_token;
    }

    // Batch-eval all lazy prob arrays (one GPU sync for all depths, not one per depth).
    if use_topk_logprob && !lazy_topk_logprobs.is_empty() {
        let mut refs: Vec<&MlxArray> = Vec::with_capacity(lazy_topk_logprobs.len() * 2);
        for (logits, indices) in lazy_topk_logprobs.iter().flatten() {
            refs.push(logits);
            refs.push(indices);
        }
        if !refs.is_empty() {
            eval(&refs);
        }
        draft_log_probs.extend(lazy_topk_logprobs.iter().enumerate().map(|(i, arrays)| {
            if let Some((logits, indices)) = arrays {
                let Some(&token) = draft_tokens.get(i) else {
                    return -30.0;
                };
                indexed_token_logprob(
                    logits.data_f32(),
                    indices.data_u32(),
                    token,
                    head.draft_sampling,
                )
                .unwrap_or(-30.0)
            } else {
                -30.0
            }
        }));
    } else if !use_topk_logprob && use_temperature && !lazy_probs.is_empty() {
        let refs: Vec<&MlxArray> = lazy_probs.iter().collect();
        eval(&refs);
        draft_log_probs.extend(lazy_probs.iter().map(|arr| {
            let p = arr.data_f32()[0].max(0.0_f32);
            if p > 0.0 {
                p.ln().max(-30.0)
            } else {
                -30.0_f32
            }
        }));
    }
    if draft_log_probs.len() < draft_tokens.len() {
        draft_log_probs.extend(std::iter::repeat_n(
            0.0_f32,
            draft_tokens.len() - draft_log_probs.len(),
        ));
    }
    let draft_log_probs: Vec<f32> = draft_log_probs
        .into_iter()
        .take(draft_tokens.len())
        .collect();

    let added = draft_tokens.len();
    (draft_tokens, draft_log_probs, added)
}

fn greedy_topk_logprob_arrays(
    logits: &MlxArray,
    sampling: MlxSamplingParams,
    vocab_size: usize,
) -> Option<(MlxArray, MlxArray)> {
    let vocab = vocab_size as i32;
    let k = (sampling.top_k as i32).min(vocab);
    if k <= 0 || sampling.temperature <= 0.0 {
        return None;
    }
    let row = reshape(logits, &[1_i32, vocab], None);
    let part_indices = argpartition_axis(&row, -k, -1, None);
    let top_indices = slice(&part_indices, &[0, vocab - k], &[1, vocab], &[1, 1], None);
    let top_logits = take_along_axis(&row, &top_indices, -1, None);
    let top_indices = astype(&top_indices, MlxDtype::Uint32, None);
    Some((top_logits, top_indices))
}

/// Greedy argmax over a `[vocab_size]` f32 logit array.
fn greedy_sample_logits(logits: &MlxArray) -> u32 {
    use mlx_sys::{argmax, eval, reshape};
    let vocab = logits.shape()[0];
    let logits_2d = reshape(logits, &[1_i32, vocab], None);
    let idx = argmax(&logits_2d, None);
    eval(&[&idx]);
    idx.data_u32()[0]
}
