use mlx_sys::{
    MlxArray, ScaledDotProductAttentionMask, concatenate, expand_dims, matmul, quantized_matmul,
    reshape, rms_norm, rope, scaled_dot_product_attention_with_mask, slice_last_dim, transpose,
};
use std::time::Instant;

use super::super::config::ModelConfig;
use super::super::profile::{
    DecodeProfileStage, decode_profile_enabled, forward_profile_eval_elapsed,
    prefill_profile_enabled,
};
use super::mlp::attention_output_projection;
use super::utils::qw;
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

pub(crate) struct GlmMlaProjectedInputs {
    /// `[1, n_heads, seq, qk_nope_head_dim]`.
    pub q_nope: MlxArray,
    /// `[1, n_heads, seq, qk_rope_head_dim]`, with RoPE applied.
    pub q_pe: MlxArray,
    /// `[1, 1, seq, kv_lora_rank]`.
    pub kv_latent: MlxArray,
    /// `[1, 1, seq, qk_rope_head_dim]`, with RoPE applied.
    pub k_pe: MlxArray,
}

pub(crate) struct GlmMlaCachedInputs {
    /// `[1, n_heads, new_seq, qk_nope_head_dim]`.
    pub q_nope: MlxArray,
    /// `[1, n_heads, new_seq, qk_rope_head_dim]`, with RoPE applied.
    pub q_pe: MlxArray,
    /// `[1, 1, total_seq, kv_lora_rank]`.
    pub kv_latent: MlxArray,
    /// `[1, 1, total_seq, qk_rope_head_dim]`, with RoPE applied.
    pub k_pe: MlxArray,
}

/// GLM4MoELite MLA input projection up to the latent-cache boundary.
pub(crate) fn glm_mla_project_inputs(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    token_offset: usize,
) -> GlmMlaProjectedInputs {
    let mla_cfg = cfg
        .mla_attention
        .as_ref()
        .expect("GLM MLA attention config");
    let mla_w = w.glm_mla_attn.as_ref().expect("GLM MLA attention weights");
    let seq = x.shape()[1];

    // Single matmul replaces separate q_a_proj and kv_a_proj launches.
    // Output: [seq, q_lora_rank + kv_lora_rank + qk_rope_head_dim].
    let qa_kva = qw(x, &mla_w.qa_kva_fused);
    let q_split = mla_cfg.q_lora_rank as i32;
    let kva_end = (mla_cfg.q_lora_rank + mla_cfg.kv_lora_rank + mla_cfg.qk_rope_head_dim) as i32;
    let q_a = slice_last_dim(&qa_kva, 0, q_split, None);
    let q_a = rms_norm(&q_a, Some(&mla_w.q_a_norm), cfg.rms_norm_eps, None);
    let q = qw(&q_a, &mla_w.q_b_proj);
    let q = reshape(
        &q,
        &[1, seq, cfg.n_heads as i32, mla_cfg.q_head_dim as i32],
        None,
    );
    let q = transpose(&q, &[0, 2, 1, 3], None);
    let q_nope = slice_last_dim(&q, 0, mla_cfg.qk_nope_head_dim as i32, None);
    let q_pe = slice_last_dim(
        &q,
        mla_cfg.qk_nope_head_dim as i32,
        mla_cfg.q_head_dim as i32,
        None,
    );
    let q_pe = rope(
        &q_pe,
        mla_cfg.qk_rope_head_dim as i32,
        true,
        Some(cfg.rope_theta),
        1.0,
        token_offset as i32,
        None,
        None,
    );

    let compressed_kv = slice_last_dim(&qa_kva, q_split, kva_end, None);
    let kv_latent_raw = slice_last_dim(&compressed_kv, 0, mla_cfg.kv_lora_rank as i32, None);
    let k_pe = slice_last_dim(
        &compressed_kv,
        mla_cfg.kv_lora_rank as i32,
        (mla_cfg.kv_lora_rank + mla_cfg.qk_rope_head_dim) as i32,
        None,
    );
    let kv_latent = rms_norm(
        &kv_latent_raw,
        Some(&mla_w.kv_a_norm),
        cfg.rms_norm_eps,
        None,
    );
    let kv_latent = expand_dims(&kv_latent, 1, None);
    let k_pe = reshape(&k_pe, &[1, seq, 1, mla_cfg.qk_rope_head_dim as i32], None);
    let k_pe = transpose(&k_pe, &[0, 2, 1, 3], None);
    let k_pe = rope(
        &k_pe,
        mla_cfg.qk_rope_head_dim as i32,
        true,
        Some(cfg.rope_theta),
        1.0,
        token_offset as i32,
        None,
        None,
    );

    GlmMlaProjectedInputs {
        q_nope,
        q_pe,
        kv_latent,
        k_pe,
    }
}

/// GLM4MoELite MLA projection plus latent cache update/fetch.
pub(crate) fn glm_mla_project_and_cache_inputs(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
) -> GlmMlaCachedInputs {
    let projected = glm_mla_project_inputs(cfg, w, x, token_offset);
    let (kv_latent, k_pe) = cache.append_glm_mla(layer_idx, projected.kv_latent, projected.k_pe);

    GlmMlaCachedInputs {
        q_nope: projected.q_nope,
        q_pe: projected.q_pe,
        kv_latent,
        k_pe,
    }
}

/// GLM4MoELite MLA attention up to the output projection boundary.
///
/// Uses packed SDPA matching Swift's GLM4MoELite implementation:
/// Q = concat([embed_q(q_nope), q_pe], axis=-1)  [n_heads, seq, kv_lora_rank+rope_dim]
/// K = concat([kv_latent, k_pe], axis=-1)         [1, key_len, kv_lora_rank+rope_dim]
/// V = kv_latent                                   [1, key_len, kv_lora_rank]
///
/// scale * Q @ K.T = scale * (embed_q(q_nope) @ kv_latent.T + q_pe @ k_pe.T), which is
/// identical to the old separate pe_scores additive mask approach — but avoids the
/// expensive embed_q_prefill(kv_latent) quantized matmul during prefill.
pub(crate) fn glm_mla_attention_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
) -> MlxArray {
    let mla_cfg = cfg
        .mla_attention
        .as_ref()
        .expect("GLM MLA attention config");
    let seq = x.shape()[1] as usize;

    let profile_decode_layer = seq == 1 && decode_profile_enabled();
    let profile_prefill_layer = seq > 1 && prefill_profile_enabled();
    let profile_forward_layer = profile_decode_layer || profile_prefill_layer;

    let qkv_started = profile_forward_layer.then(Instant::now);
    let cached = glm_mla_project_and_cache_inputs(cfg, w, x, cache, layer_idx, token_offset);
    if let Some(started) = qkv_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::PreSdpaQkvProj,
            started,
            &[
                &cached.q_nope,
                &cached.q_pe,
                &cached.kv_latent,
                &cached.k_pe,
            ],
        );
    }

    let rope_kv_started = profile_forward_layer.then(Instant::now);
    // embed_q maps q_nope [n_heads, seq, nope_dim] → [n_heads, seq, kv_lora_rank].
    // Same quantized op as decode; works for any seq length.
    let q_nope_proj = glm_mla_embed_q_decode(cfg, w, &cached.q_nope);

    // Pack rope components into Q and K so a single SDPA fuses both score contributions.
    let q_packed = concatenate(&[&q_nope_proj, &cached.q_pe], -1, None);
    let k_packed = concatenate(&[&cached.kv_latent, &cached.k_pe], -1, None);

    let mask = if seq > 1 {
        ScaledDotProductAttentionMask::Causal
    } else {
        ScaledDotProductAttentionMask::None
    };
    if let Some(started) = rope_kv_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::PreSdpaRopeKv,
            started,
            &[&q_packed, &k_packed],
        );
    }

    let sdpa_started = profile_forward_layer.then(Instant::now);
    // V stays as kv_latent [1, key_len, kv_lora_rank]; unembed_out maps output → value space.
    let out = scaled_dot_product_attention_with_mask(
        &q_packed,
        &k_packed,
        &cached.kv_latent,
        mla_cfg.query_scale,
        mask,
        None,
    );
    if let Some(started) = sdpa_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::Sdpa,
            started,
            &[&out],
        );
    }

    let oproj_started = profile_forward_layer.then(Instant::now);
    let attn_out = glm_mla_unembed_out(cfg, w, &out);

    let attn_out = transpose(&attn_out, &[0, 2, 1, 3], None);
    let attn_flat = reshape(
        &attn_out,
        &[1, seq as i32, (cfg.n_heads * mla_cfg.value_head_dim) as i32],
        None,
    );
    let result = attention_output_projection(
        &attn_flat,
        None,
        w.o_proj.as_ref().expect("GLM MLA layer must have o_proj"),
    );
    if let Some(started) = oproj_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::PostAttnOutputProj,
            started,
            &[&result],
        );
    }
    result
}

/// Prefill path: `kv_latent [B,1,seq,kv_lora_rank] @ embed_q [n_heads,kv_lora_rank,qk_nope_head_dim]`
/// → `[B,n_heads,seq,qk_nope_head_dim]`.  Uses quantized_matmul (transpose=false) so the 4-bit
/// packed weight is never materialized as float — equivalent to mlx-lm QuantizedMultiLinear(transpose=False).
#[cfg(test)]
pub(crate) fn glm_mla_embed_q_prefill(
    cfg: &ModelConfig,
    w: &LayerWeights,
    kv_latent: &MlxArray,
) -> MlxArray {
    let mla_cfg = cfg
        .mla_attention
        .as_ref()
        .expect("GLM MLA attention config");
    let mla_w = w.glm_mla_attn.as_ref().expect("GLM MLA attention weights");
    let embed_q = &mla_w.embed_q;
    if let Some(scales) = &embed_q.scales {
        quantized_matmul(
            kv_latent,
            &embed_q.weight,
            scales,
            embed_q.biases.as_ref(),
            false,
            Some(embed_q.group_size),
            Some(embed_q.bits),
            None,
        )
    } else {
        let weight = reshape(
            &embed_q.weight,
            &[
                cfg.n_heads as i32,
                mla_cfg.kv_lora_rank as i32,
                mla_cfg.qk_nope_head_dim as i32,
            ],
            None,
        );
        matmul(kv_latent, &weight, None)
    }
}

/// Decode path: `q_nope [B,n_heads,1,qk_nope_head_dim] @ embed_q.T [n_heads,qk_nope_head_dim,kv_lora_rank]`
/// → `[B,n_heads,1,kv_lora_rank]`.  Uses quantized_matmul (transpose=true) — equivalent to
/// mlx-lm QuantizedMultiLinear(transpose=True).
pub(crate) fn glm_mla_embed_q_decode(
    cfg: &ModelConfig,
    w: &LayerWeights,
    q_nope: &MlxArray,
) -> MlxArray {
    let mla_cfg = cfg
        .mla_attention
        .as_ref()
        .expect("GLM MLA attention config");
    let mla_w = w.glm_mla_attn.as_ref().expect("GLM MLA attention weights");
    let embed_q = &mla_w.embed_q;
    if let Some(scales) = &embed_q.scales {
        quantized_matmul(
            q_nope,
            &embed_q.weight,
            scales,
            embed_q.biases.as_ref(),
            true,
            Some(embed_q.group_size),
            Some(embed_q.bits),
            None,
        )
    } else {
        let weight = reshape(
            &embed_q.weight,
            &[
                cfg.n_heads as i32,
                mla_cfg.kv_lora_rank as i32,
                mla_cfg.qk_nope_head_dim as i32,
            ],
            None,
        );
        let weight = transpose(&weight, &[0, 2, 1], None);
        matmul(q_nope, &weight, None)
    }
}

/// `latent @ unembed_out.T`: maps latent KV space → value head space.
/// Uses quantized_matmul (transpose=true) — equivalent to mlx-lm QuantizedMultiLinear(transpose=True).
pub(crate) fn glm_mla_unembed_out(
    cfg: &ModelConfig,
    w: &LayerWeights,
    latent: &MlxArray,
) -> MlxArray {
    let mla_cfg = cfg
        .mla_attention
        .as_ref()
        .expect("GLM MLA attention config");
    let mla_w = w.glm_mla_attn.as_ref().expect("GLM MLA attention weights");
    let unembed_out = &mla_w.unembed_out;
    if let Some(scales) = &unembed_out.scales {
        quantized_matmul(
            latent,
            &unembed_out.weight,
            scales,
            unembed_out.biases.as_ref(),
            true,
            Some(unembed_out.group_size),
            Some(unembed_out.bits),
            None,
        )
    } else {
        let weight = reshape(
            &unembed_out.weight,
            &[
                cfg.n_heads as i32,
                mla_cfg.value_head_dim as i32,
                mla_cfg.kv_lora_rank as i32,
            ],
            None,
        );
        let weight = transpose(&weight, &[0, 2, 1], None);
        matmul(latent, &weight, None)
    }
}
