use mlx_sys::{
    add, argmax, astype, concatenate, eval, multiply, reshape, rms_norm,
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
use crate::sampling::{
    sample_categorical_with_logprob_and_distribution, TokenDistribution, Xorshift64,
};
use crate::weights::{ModelWeights, MtpWeights};

/// Lazy argmax over a `[vocab_size]` f32 logit array.
///
/// Returns a lazy `[1]` uint32 array — caller must `eval` it to materialise.
/// This avoids a per-depth GPU sync barrier in the draft loop, allowing
/// all depth levels to build their compute graphs before a single batch eval.
fn lazy_argmax_logits(logits: &MlxArray) -> MlxArray {
    let vocab = logits.shape()[0];
    let logits_2d = reshape(logits, &[1_i32, vocab], None);
    argmax(&logits_2d, None)
}

/// Compute `log(softmax(logits / temperature)[token])` on GPU.
///
/// Uses the same GPU softmax path as the target-model probability
/// computation (`compute_mtp_target_probs_lazy` in runner.rs), so
/// `p_draft` and `p_target` share the same numerical basis and the
/// rejection-sampling acceptance ratio `min(1, p_target/p_draft)` is
/// not biased by different reduction orders or intermediate dtypes.
///
/// Returns a lazy `[1]` f32 array.  Caller must include it in the batch
/// `eval` and read back with `data_f32()`.
fn gpu_draft_log_prob(logits: &MlxArray, token: u32, temperature: f32, vocab: i32) -> MlxArray {
    use mlx_sys::log as mlx_log;
    // logits is [vocab] f32.  Scale by 1/T and softmax on GPU.
    let logits_2d = reshape(logits, &[1_i32, vocab], None);
    let inv_temp = MlxArray::from_f32(1.0 / temperature);
    let scaled = multiply(&logits_2d, &inv_temp, None);
    let probs = softmax(&scaled, -1, None); // [1, vocab] lazy

    // Gather the probability of the sampled token.
    let idx_arr =
        MlxArray::from_raw_data([token].as_ptr() as *const u8, 4, &[1_i32], MlxDtype::Uint32);
    let prob = take(&probs, &idx_arr, 1, None); // [1] lazy

    // log(p) — clamp to a small floor to avoid -inf.
    let log_prob = mlx_log(&prob, None);
    let floor = MlxArray::from_f32(-30.0f32);
    mlx_sys::maximum(&log_prob, &floor, None)
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

/// Draft up to `head.max_depth` tokens by applying the MTP head recurrently.
///
/// Returns `(draft_tokens, draft_log_probs, draft_distributions, added, top2_margins)`.
/// Draft log-probs are full-vocab softmax probabilities for rejection-sampling acceptance.
/// `top2_margins` is always `[0.0; 3]` (retained for API compatibility).
///
/// Gracefully handles `weights.mtp = None` by returning empty.
///
/// ## Performance design
///
/// **Greedy mode (temperature == 0):** Chains lazy `argmax` across all depth
/// levels without per-depth GPU sync barriers, then materialises all tokens in
/// a single `eval`.  This eliminates 2–3 synchronous GPU round-trips per draft
/// step compared to the previous depth-by-depth `eval` + `data_u32` pattern,
/// allowing the GPU to execute the full multi-depth graph as one fused batch.
///
/// **Temperature mode (temperature > 0):** CPU-side sampling requires the
/// logits at each depth level (to apply top-k/top-p with a per-request RNG),
/// so a per-depth GPU sync is unavoidable.  However, the draft log-probability
/// is now computed on GPU using the same softmax path as the target-model
/// probability (`softmax(logits / T)` → `take`), eliminating the numerical
/// mismatch between CPU-side `full_vocab_token_logprob` (f32 reduction) and
/// GPU-side target probs that caused ~20 pp lower acceptance rates on complex
/// code prompts.
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
    let vocab = cfg.vocab_size as i32;

    if use_temperature {
        mtp_draft_tokens_sampled(
            head,
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            max_depth,
            vocab,
            rng,
        )
    } else {
        mtp_draft_tokens_greedy(
            head,
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            max_depth,
            vocab,
        )
    }
}

/// Greedy draft path: build full lazy graph across all depths, eval once.
///
/// Eliminates per-depth GPU sync barriers by passing lazy `argmax` results
/// directly as the next depth's token input.  `mtp_head_forward` already
/// supports lazy `prev_token_arr` (embedding lookup works on unevaluated
/// arrays), so the entire multi-depth computation builds a single fused graph
/// that MLX can execute in one GPU dispatch batch.
#[allow(clippy::too_many_arguments)]
fn mtp_draft_tokens_greedy(
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    _vocab: i32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut prev_hidden = first_hidden.clone();
    let first_token_data = [first_token];
    let mut prev_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        4,
        &[1_i32],
        MlxDtype::Uint32,
    );

    // Build the full multi-depth lazy graph: no GPU syncs.
    for _ in 0..max_depth {
        let new_hidden = mtp_head_forward(head, &prev_hidden, &prev_token_arr, weights, cache, cfg);
        let post_norm_hidden = mtp_hidden_post_norm(&new_hidden, head, cfg);
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, weights, cfg);

        // Lazy argmax — NOT evaluated yet.
        let lazy_tok = lazy_argmax_logits(&logits);
        lazy_tokens.push(lazy_tok.clone());

        prev_hidden = post_norm_hidden;
        prev_token_arr = lazy_tok;
    }

    // Single batch eval for all depth levels at once.
    let refs: Vec<&MlxArray> = lazy_tokens.iter().collect();
    eval(&refs);

    // Read back materialised tokens.
    let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
    let draft_log_probs = vec![0.0f32; draft_tokens.len()];

    let added = draft_tokens.len();
    (
        draft_tokens,
        draft_log_probs,
        vec![], // no distributions for greedy
        added,
        [0.0f32; 3],
    )
}

/// Sampled draft path: per-depth CPU sampling with GPU-side log-probs.
///
/// Temperature/top-k/top-p sampling requires per-depth CPU work (RNG-based
/// categorical), so each depth must eval its logits.  However, the draft
/// log-probability is computed on GPU using the same `softmax(logits/T)`
/// path as the target-model's acceptance probability, eliminating the
/// numerical mismatch that caused lower acceptance rates.
#[allow(clippy::too_many_arguments)]
fn mtp_draft_tokens_sampled(
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    vocab: i32,
    rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let temperature = head.draft_sampling.temperature;
    let mut draft_tokens = Vec::with_capacity(max_depth);
    let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut draft_distributions = Vec::with_capacity(max_depth);

    let mut prev_hidden = first_hidden.clone();
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
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, weights, cfg);

        // Per-depth eval + CPU sampling (required for top-k/top-p RNG).
        eval(&[&logits]);
        let logits_cpu = logits.data_f32();
        let (draft_token, _filtered_log_prob, distribution) =
            sample_categorical_with_logprob_and_distribution(logits_cpu, head.draft_sampling, rng);

        // GPU-side log-prob using same softmax path as target model.
        // Collected as lazy arrays and batch-evaluated after the loop to
        // avoid a second per-depth GPU sync (N evals instead of 2N).
        let lazy_lp = gpu_draft_log_prob(&logits, draft_token, temperature, vocab);
        lazy_log_probs.push(lazy_lp);

        if let Some(distribution) = distribution {
            draft_distributions.push(distribution);
        }

        let tok_data = [draft_token];
        let tok_arr = MlxArray::from_raw_data(
            tok_data.as_ptr() as *const u8,
            4,
            &[1_i32],
            MlxDtype::Uint32,
        );
        draft_tokens.push(draft_token);
        prev_hidden = post_norm_hidden;
        prev_token_arr = tok_arr;
    }

    // Batch-eval all log-probs in one GPU sync instead of per-depth.
    let lp_refs: Vec<&MlxArray> = lazy_log_probs.iter().collect();
    eval(&lp_refs);
    let draft_log_probs: Vec<f32> = lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();

    let added = draft_tokens.len();
    (
        draft_tokens,
        draft_log_probs,
        draft_distributions,
        added,
        [0.0f32; 3],
    )
}
