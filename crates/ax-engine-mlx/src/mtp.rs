use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, add, astype, concatenate, multiply,
    reshape, rms_norm, scaled_dot_product_attention_with_mask, sigmoid, silu_mul, slice,
};

use crate::kv_cache::MlxKVCache;
use crate::model::{ModelConfig, embed_tokens};
use crate::model::shared::{
    apply_final_logit_softcap, flatten_attention_output_bhsd, prepare_value_bhsd_from_proj,
    qk_norm_rope_bhsd_from_proj, qw,
};
use crate::weights::{MtpWeights, ModelWeights};

/// Run one recurrent MTP head forward pass for a single decode step.
///
/// Returns new hidden state `[1, 1, hidden_size]`.  Caller applies
/// `rms_norm(h, mtp_norm) @ lm_head` to get draft logits.
///
/// * `head`         — shared MTP weights (reused across all depth levels).
/// * `main_hidden`  — post-norm hidden from the main model (hidden_variant="post_norm")
///   or output from a preceding MTP head call, shape `[1, 1, hidden_size]`.
/// * `prev_token`   — token ID predicted at the previous level.
/// * `weights`      — main model weights (for the shared token embedding).
/// * `cache`        — shared 1-layer KV cache for this head (grows by 1 per call).
/// * `token_offset` — tokens already in `cache` (= calls made so far).
pub fn mtp_head_forward(
    head: &MtpWeights,
    main_hidden: &MlxArray,
    prev_token: u32,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    token_offset: usize,
    cfg: &ModelConfig,
) -> MlxArray {
    // 1. Embed prev_token → [1, 1, hidden_size] in bf16.
    let embed = embed_tokens(&[prev_token], &weights.token_embedding, cfg.hidden_size);
    let embed = astype(&embed, MlxDtype::Bfloat16, None);

    // 2. Combined input: fc(cat([enorm(embed), hnorm(hidden)])).
    //    concat_order = "embedding_hidden" → [enorm, hnorm] along last dim.
    let enormed = rms_norm(&embed, Some(&head.pre_fc_norm_embedding), cfg.rms_norm_eps, None);
    let hnormed = rms_norm(main_hidden, Some(&head.pre_fc_norm_hidden), cfg.rms_norm_eps, None);
    let combined = concatenate(&[&enormed, &hnormed], -1, None);
    let mut h = qw(&combined, &head.fc);

    // 3. Attention sub-layer (Qwen3NextAttention).
    //
    // q_proj output = [1, 1, n_heads * head_dim * 2]:
    //   first half  = queries  [1, 1, n_heads * head_dim]
    //   second half = gate     [1, 1, n_heads * head_dim]
    // Output = o_proj(sdpa_out) * sigmoid(gate), then residual.
    {
        let normed = rms_norm(&h, Some(&head.attn_norm), cfg.rms_norm_eps, None);

        // Split q_proj output into queries and gate.
        let qg_raw = qw(&normed, &head.q_proj); // [1, 1, n_heads * head_dim * 2]
        let q_half = (head.n_heads * head.head_dim) as i32;
        let q_raw = slice(&qg_raw, &[0, 0, 0], &[1, 1, q_half], &[1, 1, 1], None);
        let gate  = slice(&qg_raw, &[0, 0, q_half], &[1, 1, q_half * 2], &[1, 1, 1], None);

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
        let gate_sig = sigmoid(&gate, None);
        let gated = multiply(&attn_flat, &gate_sig, None);
        // Flatten gate back to same shape as attn_flat for element-wise multiply.
        let gated_flat = reshape(&gated, &[1_i32, 1, q_half], None);
        let attn_proj = qw(&gated_flat, &head.o_proj);
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
/// Returns `(draft_tokens, draft_count)`:
/// * `draft_tokens[i]` — greedy-sampled draft token at depth `i+1`.
/// * `draft_count`     — how many entries were appended to `cache` (= draft_tokens.len()).
///
/// `start_offset` is the current number of tokens already in `cache` (used for RoPE).
///
/// Gracefully handles `weights.mtp = None` by returning empty.
pub fn mtp_draft_tokens(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    start_offset: usize,
) -> (Vec<u32>, usize) {
    let Some(head) = weights.mtp.as_ref() else {
        return (vec![], 0);
    };
    if head.max_depth == 0 {
        return (vec![], 0);
    }

    let mut draft_tokens = Vec::with_capacity(head.max_depth);

    let mut prev_hidden = first_hidden.clone();
    let mut prev_token = first_token;

    for i in 0..head.max_depth {
        let new_hidden = mtp_head_forward(
            head,
            &prev_hidden,
            prev_token,
            weights,
            cache,
            start_offset + i,
            cfg,
        );

        let logits = mtp_hidden_to_logits(&new_hidden, head, weights, cfg);
        let draft_token = greedy_sample_logits(&logits);

        draft_tokens.push(draft_token);
        prev_hidden = new_hidden;
        prev_token = draft_token;
    }

    let added = draft_tokens.len();
    (draft_tokens, added)
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
