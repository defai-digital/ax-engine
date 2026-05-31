use mlx_sys::{
    add, argmax, argsort_axis, astype, concatenate, eval, multiply, reshape, rms_norm,
    scaled_dot_product_attention_with_mask, sigmoid, slice, softmax, take, MlxArray, MlxDtype,
    ScaledDotProductAttentionMask,
};

use crate::kv_cache::MlxKVCache;
use crate::model::shared::{
    apply_final_logit_softcap, ffn_swiglu, flatten_attention_output_bhsd, moe_experts_forward,
    moe_router_deepseek_v3, moe_router_glm, moe_router_qwen3, prepare_value_bhsd_from_proj,
    qk_norm_rope_bhsd_from_proj, qw, shared_expert_forward,
};
use crate::model::{embed_tokens_arr, ModelConfig};
use crate::sampling::Xorshift64;
use crate::weights::{ModelWeights, MtpWeights, QuantizedWeight};

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

/// Filtered draft logits: compute lm_head only for the `candidates` token subset.
///
/// Returns f32 logits `[k]` where k = candidates.shape[0].  Indices are LOCAL
/// (0..k); callers must map back to global vocab ids via `candidates`.
///
/// Dramatically cheaper than the full lm_head: reads ~k×hidden bytes instead of
/// vocab×hidden bytes (121x less for k=2048, vocab=248320).
fn mtp_post_norm_to_logits_filtered(
    post_norm_hidden: &MlxArray,
    candidates: &MlxArray,
    weights: &ModelWeights,
    cfg: &ModelConfig,
) -> MlxArray {
    use mlx_sys::reshape as mlx_reshape;
    let k = candidates.shape()[0];
    // Build a partial QuantizedWeight by selecting only the candidate rows.
    let partial = QuantizedWeight {
        weight: take(&weights.lm_head.weight, candidates, 0, None),
        scales: weights
            .lm_head
            .scales
            .as_ref()
            .map(|s| take(s, candidates, 0, None)),
        biases: weights
            .lm_head
            .biases
            .as_ref()
            .map(|b| take(b, candidates, 0, None)),
        group_size: weights.lm_head.group_size,
        bits: weights.lm_head.bits,
    };
    let logits = qw(post_norm_hidden, &partial);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    // [1, 1, k] → [k]
    mlx_reshape(&logits_f32, &[k], None)
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
/// When `temperature > 0`, tokens are selected by argmax for throughput; the returned
/// log_probs use temperature-scaled probabilities and enable rejection-sampling
/// acceptance in `run_mtp_decode`.
/// When `temperature == 0`, tokens are greedy argmax and acceptance falls back to exact
/// argmax comparison.
///
/// `candidates` — optional evaluated `uint32 [k]` array of top-k token indices from the
/// previous verify step.  When present, the lm_head computation is restricted to those k
/// tokens at ALL depths (cascaded filtered approach).  Draft tokens are selected by greedy
/// argmax over the filtered set; log_probs are left empty, triggering greedy argmax comparison
/// in acceptance.  This yields ~89% accept rate but reduces per-draft bandwidth from 640MB
/// to 5MB, producing ~68 tok/s vs ~49 tok/s for the mixed filtered/full approach.
///
/// Returns `(draft_tokens, draft_log_probs, added, top2_margins)`.
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
    candidates: Option<&MlxArray>,
    _rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, usize, [f32; 3]) {
    let Some(head) = weights.mtp.as_ref() else {
        return (vec![], vec![], 0, [0.0; 3]);
    };
    let max_depth = max_depth_cap.unwrap_or(head.max_depth).min(head.max_depth);
    if max_depth == 0 {
        return (vec![], vec![], 0, [0.0; 3]);
    }

    let use_temperature = head.draft_sampling.temperature > 0.0;
    let mut draft_tokens = Vec::with_capacity(max_depth);
    let mut draft_log_probs = Vec::with_capacity(max_depth);
    // Lazy arrays holding p(draft_token) under the temperature-scaled draft distribution.
    // Evaluated in one batch after all depth iterations to avoid per-step GPU sync.
    let mut lazy_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut lazy_draft_tok_arrs: Vec<MlxArray> = Vec::with_capacity(max_depth);
    // Partial logits saved per depth for top2_margin extraction after the batch eval.
    let mut partial_logits_arrs: Vec<MlxArray> = Vec::with_capacity(max_depth);

    let mut prev_hidden = first_hidden.clone();
    // Wrap first_token as a GPU uint32 [1] array.
    let first_token_data = [first_token];
    let mut prev_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        4,
        &[1_i32],
        MlxDtype::Uint32,
    );

    // inv_temp scalar, computed once outside the loop.
    let inv_temp_arr = if use_temperature {
        Some(MlxArray::from_f32(1.0 / head.draft_sampling.temperature))
    } else {
        None
    };

    // Per-depth refined candidate sets: after each filtered lm_head pass, re-rank the
    // K candidates by the MTP head's logit score at that depth and use the top-K/2 as
    // candidates for the next depth.  The verify-bonus candidates are a good proxy for
    // depth 0 (same position), but for depths 1 and 2 the MTP head's own ranking (which
    // reflects the draft context) is more aligned with the true distribution.
    let mut refined_candidates: Option<MlxArray> = None;

    for depth_idx in 0..max_depth {
        let new_hidden = mtp_head_forward(head, &prev_hidden, &prev_token_arr, weights, cache, cfg);
        let post_norm_hidden = mtp_hidden_post_norm(&new_hidden, head, cfg);

        // Depth 0: use verify-bonus candidates (correct position N).
        // Depths 1+: use per-depth refined candidates derived from the previous depth's logits.
        // Take ownership from refined_candidates so the borrow does not extend into the
        // assignment at the end of the filtered branch.
        let cands_owned: Option<MlxArray> = if depth_idx == 0 {
            candidates.cloned()
        } else {
            refined_candidates.take()
        };

        if let Some(ref cands) = cands_owned {
            let partial_logits =
                mtp_post_norm_to_logits_filtered(&post_norm_hidden, cands, weights, cfg);
            let k = partial_logits.shape()[0];

            let logits_2d = reshape(&partial_logits, &[1_i32, k], None);
            let local_idx_arr = argmax(&logits_2d, None);

            let next_depth_cands = if depth_idx + 1 < max_depth {
                let half_k = k / 2;
                let m1 = MlxArray::from_f32(-1.0);
                let neg = multiply(&partial_logits, &m1, None);
                let local_sorted = argsort_axis(&neg, 0, None);
                let local_top = slice(&local_sorted, &[0], &[half_k], &[1], None);
                Some(take(cands, &local_top, 0, None))
            } else {
                None
            };

            let local_idx_1d = reshape(&local_idx_arr, &[1_i32], None);
            let draft_tok_lazy = take(cands, &local_idx_1d, 0, None);

            if use_temperature {
                let inv_t = inv_temp_arr.as_ref().unwrap();
                let scaled = multiply(&partial_logits, inv_t, None);
                let probs = softmax(&scaled, -1, None);
                let prob_d = take(&probs, &local_idx_1d, 0, None);
                lazy_probs.push(prob_d);
            }

            partial_logits_arrs.push(partial_logits);
            lazy_draft_tok_arrs.push(draft_tok_lazy.clone());
            refined_candidates = next_depth_cands;
            prev_hidden = post_norm_hidden;
            prev_token_arr = draft_tok_lazy;
        } else {
            // Fallback full-vocab path (first draft step before candidates are available).
            let logits = mtp_post_norm_to_logits(&post_norm_hidden, weights, cfg);
            let draft_token = greedy_sample_logits(&logits); // syncs GPU
            if let Some(ref inv_t) = inv_temp_arr {
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
    }

    // Batch-eval all lazy filtered-path arrays in one GPU dispatch.
    // Evaluating draft_tok_arrs materialises their transitive deps (partial_logits,
    // intermediate refined_candidates).  Including partial_logits_arrs explicitly
    // ensures data_f32() is available for top2_margin extraction after this eval.
    if !lazy_draft_tok_arrs.is_empty() {
        let mut eval_targets: Vec<&MlxArray> = Vec::with_capacity(
            lazy_draft_tok_arrs.len() + partial_logits_arrs.len() + lazy_probs.len(),
        );
        for arr in &lazy_draft_tok_arrs {
            eval_targets.push(arr);
        }
        for pla in &partial_logits_arrs {
            eval_targets.push(pla);
        }
        for lp in &lazy_probs {
            eval_targets.push(lp);
        }
        eval(&eval_targets);
        for arr in &lazy_draft_tok_arrs {
            draft_tokens.push(arr.data_u32()[0]);
        }
    } else if !lazy_probs.is_empty() {
        let refs: Vec<&MlxArray> = lazy_probs.iter().collect();
        eval(&refs);
    }

    let mut top2_margins = [0.0f32; 3];
    for (d, pla) in partial_logits_arrs.iter().enumerate().take(3) {
        top2_margins[d] = top2_margin_from_slice(pla.data_f32());
    }

    if !lazy_probs.is_empty() {
        draft_log_probs.extend(lazy_probs.iter().map(|arr| {
            let p = arr.data_f32()[0].max(0.0_f32);
            if p > 0.0 {
                p.ln().max(-30.0)
            } else {
                -30.0_f32
            }
        }));
    }
    if !draft_log_probs.is_empty() && draft_log_probs.len() < draft_tokens.len() {
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
    (draft_tokens, draft_log_probs, added, top2_margins)
}

/// Compute the rank-1 minus rank-2 logit gap from a partial logit slice.
/// O(n) scan, no allocation.  Returns 0.0 when fewer than 2 values are finite.
fn top2_margin_from_slice(data: &[f32]) -> f32 {
    let mut m1 = f32::NEG_INFINITY;
    let mut m2 = f32::NEG_INFINITY;
    for &v in data {
        if v > m1 {
            m2 = m1;
            m1 = v;
        } else if v > m2 {
            m2 = v;
        }
    }
    if m2.is_finite() {
        m1 - m2
    } else {
        0.0
    }
}
