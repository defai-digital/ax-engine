use mlx_sys::{
    MlxArray, MlxDtype, add, argpartition_axis, astype, dequantize, expand_dims, gather_mm,
    gather_qmm, gelu, matmul, multiply, quantized_matmul, reshape, rms_norm, rope,
    scaled_dot_product_attention, slice, slice_last_dim, softmax, sum_axis, take, take_along_axis,
    tanh, transpose,
};

use ax_engine_core::NativeModelManifest;

use crate::kv_cache::MlxKVCache;
use crate::weights::{LayerWeights, ModelWeights, QuantizedWeight};

/// Per-layer hyperparameters for interleaved-SWA models (Gemma4).
#[derive(Clone, Debug)]
pub struct LayerConfig {
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rope_dims: usize,
    /// None = global causal attention; Some(n) = sliding-window attention.
    pub sliding_window: Option<usize>,
    /// None = compute own K/V; Some(src) = reuse K/V from layer `src`.
    pub kv_source_layer: Option<usize>,
    /// Apply no-scale RMSNorm to V before caching (Gemma4 non-KV-shared layers).
    pub v_norm_no_scale: bool,
}

/// Hyperparameters extracted from the manifest.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub layer_count: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rope_dims: usize,
    pub attn_output_gate: bool,
    pub query_scale: f32,
    pub final_logit_softcapping: Option<f32>,
    // MoE (0 means dense-only model).
    pub moe_expert_count: usize,
    pub moe_experts_per_token: usize,
    pub moe_expert_intermediate_size: usize,
    /// Per-layer config (non-empty only for interleaved SWA models like Gemma4).
    pub layer_configs: Vec<LayerConfig>,
    /// True → Gemma4 dual-path MoE routing (rms_norm → proj → softmax).
    /// False → Qwen3 MoE routing (proj → softmax, no rms_norm).
    pub gemma4_moe_router: bool,
    /// Use GELU (Gemma4) instead of SiLU (Qwen3) for FFN gate activation.
    pub uses_geglu: bool,
    /// Scale hidden states after embedding (Gemma4: sqrt(hidden_size)).
    pub hidden_states_scale: Option<f32>,
    /// Normalise top-k MoE routing weights to sum to 1 (Qwen3 MoE norm_topk_prob).
    pub moe_norm_topk_prob: bool,
}

impl ModelConfig {
    pub fn from_manifest(m: &NativeModelManifest) -> Self {
        let head_dim = m.attention_head_dim as usize;
        let rope_dims = m
            .partial_rotary_factor
            .map(|f| ((head_dim as f32 * f) as usize).next_multiple_of(2))
            .unwrap_or(head_dim);
        let query_scale = m
            .query_pre_attn_scalar
            .map(|s| 1.0 / (s as f32).sqrt())
            .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
        let intermediate_size = if m.intermediate_size > 0 {
            m.intermediate_size as usize
        } else {
            (m.hidden_size as usize * 8 / 3).next_multiple_of(256)
        };
        let rope_theta = m.rope_theta.map(|t| t as f32).unwrap_or(10000.0);
        let layer_configs = build_layer_configs(m, head_dim, rope_theta, rope_dims);
        let is_gemma4 = m.model_family == "gemma4";

        Self {
            layer_count: m.layer_count as usize,
            hidden_size: m.hidden_size as usize,
            intermediate_size,
            n_heads: m.attention_head_count as usize,
            n_kv_heads: m.kv_head_count as usize,
            head_dim,
            vocab_size: m.vocab_size as usize,
            rope_theta,
            rope_dims,
            attn_output_gate: m.attn_output_gate,
            query_scale,
            final_logit_softcapping: m.final_logit_softcapping,
            moe_expert_count: m.moe.expert_count.unwrap_or(0) as usize,
            moe_experts_per_token: m.moe.experts_per_token.unwrap_or(0) as usize,
            moe_expert_intermediate_size: m.moe.expert_intermediate_size.unwrap_or(0) as usize,
            layer_configs,
            gemma4_moe_router: is_gemma4,
            uses_geglu: is_gemma4,
            hidden_states_scale: m.hidden_states_scale,
            moe_norm_topk_prob: m.moe_norm_topk_prob,
        }
    }
}

fn build_layer_configs(
    m: &NativeModelManifest,
    default_head_dim: usize,
    default_rope_theta: f32,
    default_rope_dims: usize,
) -> Vec<LayerConfig> {
    if m.layer_types.is_empty() {
        return Vec::new();
    }
    let swa_theta = m.rope_theta_swa.map(|t| t as f32).unwrap_or(10000.0);
    let full_head_dim = m.global_head_dim.unwrap_or(m.attention_head_dim) as usize;
    let full_rope_dims = m
        .partial_rotary_factor
        .map(|f| ((full_head_dim as f32 * f) as usize).next_multiple_of(2))
        .unwrap_or(full_head_dim);
    let sliding_window = m.sliding_window_size.map(|w| w as usize);

    m.layer_types
        .iter()
        .enumerate()
        .map(|(i, lt)| {
            let kv_source_layer = m
                .kv_shared_source_layers
                .get(&(i as u32))
                .map(|&s| s as usize);
            let v_norm_no_scale = m.attention_v_norm_no_scale_layers.contains(&(i as u32));
            if lt == "full_attention" {
                LayerConfig {
                    head_dim: full_head_dim,
                    rope_theta: default_rope_theta,
                    rope_dims: full_rope_dims,
                    sliding_window: None,
                    kv_source_layer,
                    v_norm_no_scale,
                }
            } else {
                LayerConfig {
                    head_dim: default_head_dim,
                    rope_theta: swa_theta,
                    rope_dims: default_rope_dims,
                    sliding_window,
                    kv_source_layer,
                    v_norm_no_scale,
                }
            }
        })
        .collect()
}

/// Resolve per-layer params: (head_dim, rope_theta, rope_dims, sliding_window, kv_source, v_norm_no_scale).
fn layer_params(
    cfg: &ModelConfig,
    layer_idx: usize,
) -> (usize, f32, usize, Option<usize>, Option<usize>, bool) {
    if let Some(lc) = cfg.layer_configs.get(layer_idx) {
        (
            lc.head_dim,
            lc.rope_theta,
            lc.rope_dims,
            lc.sliding_window,
            lc.kv_source_layer,
            lc.v_norm_no_scale,
        )
    } else {
        (
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_dims,
            None,
            None,
            false,
        )
    }
}

/// Forward pass for one transformer layer.
///
/// Returns updated hidden states.
pub fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray, // [1, seq, hidden]
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
) -> MlxArray {
    let (head_dim, rope_theta, rope_dims, sliding_window, kv_source, v_norm_no_scale) =
        layer_params(cfg, layer_idx);

    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), 1e-6, None);

    let seq = hidden.shape()[1] as usize;

    // 2-7. QKV projections + RoPE. KV-shared layers skip K/V and borrow from source.
    let (q_rope, cached_k, cached_v, attn_gate) = if let Some(src_layer) = kv_source {
        // KV-shared layer (Gemma4 layers 24-41): compute Q only.
        let q_raw = qw(
            &normed,
            w.q_proj.as_ref().expect("KV-shared layer must have q_proj"),
        );
        let q = reshape(
            &q_raw,
            &[1, seq as i32, cfg.n_heads as i32, head_dim as i32],
            None,
        );
        let q = qk_norm_bshd(q, w.q_norm.as_ref(), cfg.n_heads, head_dim, seq);
        let q = transpose(&q, &[0, 2, 1, 3], None);
        let q_rope = rope(
            &q,
            rope_dims as i32,
            false,
            Some(rope_theta),
            1.0,
            token_offset as i32,
            None,
            None,
        );
        let (ck, cv) = cache.peek_source_kv(src_layer, seq);
        (q_rope, ck, cv, None)
    } else {
        // Normal layer: compute Q, K, V from own projections.
        let (q_raw, k_raw, v_raw, attn_gate_raw) = qkv_project(cfg, w, &normed, head_dim);

        let q = reshape(
            &q_raw,
            &[1, seq as i32, cfg.n_heads as i32, head_dim as i32],
            None,
        );
        let k = reshape(
            &k_raw,
            &[1, seq as i32, cfg.n_kv_heads as i32, head_dim as i32],
            None,
        );
        let v = reshape(
            &v_raw,
            &[1, seq as i32, cfg.n_kv_heads as i32, head_dim as i32],
            None,
        );

        let q = qk_norm_bshd(q, w.q_norm.as_ref(), cfg.n_heads, head_dim, seq);
        let k = qk_norm_bshd(k, w.k_norm.as_ref(), cfg.n_kv_heads, head_dim, seq);

        let q = transpose(&q, &[0, 2, 1, 3], None);
        let k = transpose(&k, &[0, 2, 1, 3], None);
        // V: transpose to BSHD then apply no-scale RMSNorm if required (Gemma4).
        let v = transpose(&v, &[0, 2, 1, 3], None);
        let v = if v_norm_no_scale {
            // RMSNormNoScale: normalize by RMS without learnable scale.
            qk_norm_bshd(v, None, cfg.n_kv_heads, head_dim, seq)
        } else {
            v
        };

        let q_rope = rope(
            &q,
            rope_dims as i32,
            false,
            Some(rope_theta),
            1.0,
            token_offset as i32,
            None,
            None,
        );
        let k_rope = rope(
            &k,
            rope_dims as i32,
            false,
            Some(rope_theta),
            1.0,
            token_offset as i32,
            None,
            None,
        );

        let (ck, cv) = cache.append(layer_idx, k_rope, v);
        (q_rope, ck, cv, attn_gate_raw)
    };

    // 8. Optional sliding-window KV slice (trim cached KV to the last `window` tokens).
    let (cached_k, cached_v) = if let Some(window) = sliding_window {
        let total = cached_k.shape()[2] as usize;
        if total > window {
            let start = (total - window) as i32;
            let end = total as i32;
            let nkv = cfg.n_kv_heads as i32;
            let hd = head_dim as i32;
            (
                slice(
                    &cached_k,
                    &[0, 0, start, 0],
                    &[1, nkv, end, hd],
                    &[1, 1, 1, 1],
                    None,
                ),
                slice(
                    &cached_v,
                    &[0, 0, start, 0],
                    &[1, nkv, end, hd],
                    &[1, 1, 1, 1],
                    None,
                ),
            )
        } else {
            (cached_k, cached_v)
        }
    } else {
        (cached_k, cached_v)
    };

    // 9. SDPA (GQA: MLX broadcasts KV heads internally). Use per-layer head_dim scale.
    let causal = seq > 1;
    let query_scale = 1.0 / (head_dim as f32).sqrt();
    let attn_sdpa =
        scaled_dot_product_attention(&q_rope, &cached_k, &cached_v, query_scale, causal, None);

    // 10. Transpose back: [1, n_heads, seq, head_dim] → [1, seq, n_heads, head_dim].
    let attn_out = transpose(&attn_sdpa, &[0, 2, 1, 3], None);

    // 11. Reshape to [1, seq, hidden].
    let attn_flat = reshape(
        &attn_out,
        &[1, seq as i32, (cfg.n_heads * head_dim) as i32],
        None,
    );

    // 12. Output projection.
    let attn_proj = qw(
        &attn_flat,
        w.o_proj
            .as_ref()
            .expect("full-attention layer must have o_proj"),
    );

    // 13. Optional attention output gate (Qwen3.5): sigmoid(gate) * o_proj(attn).
    let attn_proj = if let Some(gate) = attn_gate {
        multiply(&mlx_sys::ops::sigmoid(&gate, None), &attn_proj, None)
    } else {
        attn_proj
    };

    // 14. Optional post-attention layernorm (Gemma4): applied BEFORE residual add.
    let attn_proj = if let Some(post_norm) = &w.attn_post_norm {
        rms_norm(&attn_proj, Some(post_norm), 1e-6, None)
    } else {
        attn_proj
    };

    // 15. Residual.
    let hidden = add(hidden, &attn_proj, None);

    // 16. Pre-FFN norm.
    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), 1e-6, None);

    // 17. FFN: MoE or dense.
    let ffn_out = if w.router_proj.is_some() {
        if cfg.gemma4_moe_router {
            // Gemma4 dual-path: dense sub-block + expert sub-block.
            let h1 = ffn_swiglu(cfg, w, &normed2);
            let h1 = rms_norm_opt(&h1, w.ffn_post_norm1.as_ref());
            let h2_normed = if let Some(n2) = &w.ffn_norm2 {
                rms_norm(&hidden, Some(n2), 1e-6, None)
            } else {
                normed2
            };
            let (top_k_indices, top_k_weights) = moe_router_gemma4(cfg, w, &hidden);
            let h2 = moe_experts_forward(cfg, w, &h2_normed, &top_k_indices, &top_k_weights);
            let h2 = rms_norm_opt(&h2, w.ffn_post_norm2.as_ref());
            let combined = add(&h1, &h2, None);
            rms_norm_opt(&combined, w.ffn_post_norm.as_ref())
        } else {
            // Qwen3 MoE: router (proj → softmax → top-k) + expert forward.
            let (top_k_indices, top_k_weights) = moe_router_qwen3(cfg, w, &normed2);
            let out = moe_experts_forward(cfg, w, &normed2, &top_k_indices, &top_k_weights);
            rms_norm_opt(&out, w.ffn_post_norm.as_ref())
        }
    } else {
        // Dense path (Qwen3, Gemma4 non-MoE layers).
        let out = ffn_swiglu(cfg, w, &normed2);
        rms_norm_opt(&out, w.ffn_post_norm.as_ref())
    };

    // 18. Residual.
    add(&hidden, &ffn_out, None)
}

/// Embed token IDs and return hidden states of shape [1, seq_len, hidden].
pub fn embed_tokens(
    token_ids: &[u32],
    embedding: &QuantizedWeight,
    hidden_size: usize,
) -> MlxArray {
    let seq = token_ids.len() as i32;
    let ids_shape = [seq];
    let ids_arr = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &ids_shape,
        MlxDtype::Uint32,
    );
    if let Some(scales) = &embedding.scales {
        let row_w = take(&embedding.weight, &ids_arr, 0, None);
        let row_s = take(scales, &ids_arr, 0, None);
        let row_b = embedding
            .biases
            .as_ref()
            .map(|b| take(b, &ids_arr, 0, None));
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
        let flat = take(&embedding.weight, &ids_arr, 0, None);
        reshape(&flat, &[1, seq, hidden_size as i32], None)
    }
}

/// Full forward pass: returns logits for the LAST token only — `[vocab_size]` f32.
pub fn forward(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    for (li, layer_w) in weights.layers.iter().enumerate() {
        hidden = layer_forward(cfg, layer_w, &hidden, cache, li, token_offset);
    }

    // Slice to last token only: [1, seq, hidden] → [1, 1, hidden].
    let last_hidden = if token_ids.len() > 1 {
        let last_idx: u32 = (token_ids.len() - 1) as u32;
        let idx_arr = MlxArray::from_raw_data(
            &last_idx as *const u32 as *const u8,
            std::mem::size_of_val(&last_idx),
            &[1_i32],
            MlxDtype::Uint32,
        );
        take(&hidden, &idx_arr, 1, None)
    } else {
        hidden
    };

    let normed = rms_norm(&last_hidden, Some(&weights.final_norm), 1e-6, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    reshape(&logits_f32, &[cfg.vocab_size as i32], None)
}

/// Forward pass returning logits for ALL token positions — `[seq, vocab_size]` f32.
///
/// Used by speculative decode verification to check all draft tokens in one pass.
pub fn forward_all_positions(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    for (li, layer_w) in weights.layers.iter().enumerate() {
        hidden = layer_forward(cfg, layer_w, &hidden, cache, li, token_offset);
    }

    let seq = token_ids.len() as i32;
    let normed = rms_norm(&hidden, Some(&weights.final_norm), 1e-6, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    reshape(&logits_f32, &[seq, cfg.vocab_size as i32], None)
}

// ── private helpers ──────────────────────────────────────────────────────────

/// Apply optional per-head RMS norm in BSHD [1, seq, n_heads, head_dim] space.
fn qk_norm_bshd(
    x: MlxArray,
    norm: Option<&MlxArray>,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
) -> MlxArray {
    let Some(n) = norm else { return x };
    let flat = reshape(&x, &[(n_heads * seq) as i32, head_dim as i32], None);
    let normed = rms_norm(&flat, Some(n), 1e-6, None);
    reshape(
        &normed,
        &[1, seq as i32, n_heads as i32, head_dim as i32],
        None,
    )
}

/// Apply optional RMS norm; pass `x` through if `norm` is None.
fn rms_norm_opt(x: &MlxArray, norm: Option<&MlxArray>) -> MlxArray {
    if let Some(n) = norm {
        rms_norm(x, Some(n), 1e-6, None)
    } else {
        x.clone()
    }
}

fn qkv_project(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    let slices = qkv_slices(cfg, head_dim);
    if let Some(packed) = &w.qkv_packed {
        let out = qw(x, packed);
        let q = mlx_slice_last_dim(&out, slices.q.0, slices.q.1);
        let gate = slices
            .gate
            .map(|(start, end)| mlx_slice_last_dim(&out, start, end));
        let k = mlx_slice_last_dim(&out, slices.k.0, slices.k.1);
        let v = mlx_slice_last_dim(&out, slices.v.0, slices.v.1);
        (q, k, v, gate)
    } else {
        let q_full = qw(x, w.q_proj.as_ref().unwrap());
        let q = mlx_slice_last_dim(&q_full, slices.q.0, slices.q.1);
        let gate = slices
            .gate
            .map(|(start, end)| mlx_slice_last_dim(&q_full, start, end));
        let k = qw(x, w.k_proj.as_ref().unwrap());
        let v = qw(x, w.v_proj.as_ref().unwrap());
        (q, k, v, gate)
    }
}

#[derive(Debug, Eq, PartialEq)]
struct QkvSlices {
    q: (i32, i32),
    gate: Option<(i32, i32)>,
    k: (i32, i32),
    v: (i32, i32),
}

fn qkv_slices(cfg: &ModelConfig, head_dim: usize) -> QkvSlices {
    let q_size = (cfg.n_heads * head_dim) as i32;
    let kv_size = (cfg.n_kv_heads * head_dim) as i32;
    let gate = cfg.attn_output_gate.then_some((q_size, q_size * 2));
    let kv_start = if cfg.attn_output_gate {
        q_size * 2
    } else {
        q_size
    };
    QkvSlices {
        q: (0, q_size),
        gate,
        k: (kv_start, kv_start + kv_size),
        v: (kv_start + kv_size, kv_start + kv_size * 2),
    }
}

fn qw(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    if let Some(scales) = &qw.scales {
        quantized_matmul(
            x,
            &qw.weight,
            scales,
            qw.biases.as_ref(),
            true,
            Some(qw.group_size),
            Some(qw.bits),
            None,
        )
    } else {
        let wt = transpose(&qw.weight, &[1, 0], None);
        matmul(x, &wt, None)
    }
}

fn ffn_swiglu(cfg: &ModelConfig, w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let (gate_out, up_out) = if let Some(packed) = &w.gate_up_packed {
        let out = qw(x, packed);
        let half = cfg.intermediate_size as i32;
        let gate = mlx_slice_last_dim(&out, 0, half);
        let up = mlx_slice_last_dim(&out, half, half * 2);
        (gate, up)
    } else {
        let gate = qw(x, w.gate_proj.as_ref().unwrap());
        let up = qw(x, w.up_proj.as_ref().unwrap());
        (gate, up)
    };

    // Gemma4 uses GEGLU (GELU gate); Qwen3 uses SwiGLU (SiLU gate).
    let gate_act = if cfg.uses_geglu {
        gelu(&gate_out, None)
    } else {
        mlx_sys::ops::silu(&gate_out, None)
    };
    let ffn_hidden = multiply(&gate_act, &up_out, None);
    qw(&ffn_hidden, &w.down_proj)
}

fn mlx_slice_last_dim(x: &MlxArray, start: i32, end: i32) -> MlxArray {
    slice_last_dim(x, start, end, None)
}

fn scale_hidden(hidden: &MlxArray, scale: f32) -> MlxArray {
    let dtype = hidden.dtype();
    let s_arr = MlxArray::from_raw_data(
        &scale as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    let s_arr = astype(&s_arr, dtype, None);
    multiply(hidden, &s_arr, None)
}

fn apply_final_logit_softcap(cfg: &ModelConfig, logits: &MlxArray) -> MlxArray {
    let Some(cap) = cfg.final_logit_softcapping.filter(|cap| *cap > 0.0) else {
        return logits.clone();
    };
    let inv_cap = 1.0_f32 / cap;
    let inv_cap_arr = MlxArray::from_raw_data(
        &inv_cap as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    let cap_arr = MlxArray::from_raw_data(
        &cap as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    let scaled = multiply(logits, &inv_cap_arr, None);
    multiply(&tanh(&scaled, None), &cap_arr, None)
}

/// Gemma4 MoE router: rms_norm(scale * hidden) → proj → argpartition → softmax.
fn moe_router_gemma4(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
) -> (MlxArray, MlxArray) {
    let router_proj = w.router_proj.as_ref().unwrap();
    let router_scale = w.router_scale.as_ref().unwrap();

    let root_factor = 1.0_f32 / (cfg.hidden_size as f32).sqrt();
    let scale_arr = MlxArray::from_raw_data(
        &root_factor as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    let scale_arr = astype(&scale_arr, MlxDtype::Bfloat16, None);
    let combined_scale = multiply(router_scale, &scale_arr, None);
    let normed = rms_norm(hidden, Some(&combined_scale), 1e-6, None);

    let expert_scores = qw(&normed, router_proj);
    top_k_by_argpartition(
        &expert_scores,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        true,
    )
}

/// Qwen3 MoE router: proj → softmax → pick top-k by weight value (no rms_norm).
fn moe_router_qwen3(
    cfg: &ModelConfig,
    w: &LayerWeights,
    normed: &MlxArray,
) -> (MlxArray, MlxArray) {
    let router_proj = w.router_proj.as_ref().unwrap();
    let logits = qw(normed, router_proj);
    let last_axis = logits.ndim() as i32 - 1;
    let weights_all = softmax(&logits, last_axis, None);
    let (top_k_indices, top_k_weights) = top_k_by_argpartition(
        &weights_all,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        false,
    );
    // norm_topk_prob: renormalise top-k weights to sum to 1.
    let top_k_weights = if cfg.moe_norm_topk_prob {
        let sum = sum_axis(&top_k_weights, last_axis, true, None);
        mlx_sys::ops::divide(&top_k_weights, &sum, None)
    } else {
        top_k_weights
    };
    (top_k_indices, top_k_weights)
}

/// Pick top-k elements via argpartition and optionally re-apply softmax.
fn top_k_by_argpartition(
    scores: &MlxArray,
    num_experts: usize,
    top_k: usize,
    resoftmax: bool,
) -> (MlxArray, MlxArray) {
    let last_axis = scores.ndim() as i32 - 1;
    let part_indices = argpartition_axis(scores, -(top_k as i32), last_axis, None);
    let top_k_indices = slice_last_dim(
        &part_indices,
        (num_experts - top_k) as i32,
        num_experts as i32,
        None,
    );
    let top_k_raw = take_along_axis(scores, &top_k_indices, last_axis, None);
    let top_k_weights = if resoftmax {
        softmax(&top_k_raw, last_axis, None)
    } else {
        top_k_raw
    };
    (top_k_indices, top_k_weights)
}

/// Expert forward: applies selected experts to `x` and returns the weighted sum.
///
/// x: [1, seq, hidden] (already pre-normed via pre_feedforward_layernorm_2)
/// top_k_indices: [1, seq, top_k]   expert assignments (uint32)
/// top_k_weights: [1, seq, top_k]   softmax-normalised weights (bf16)
fn moe_experts_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
) -> MlxArray {
    // Expand x for broadcast over top_k: [1, seq, hidden] → [1, seq, 1, hidden]
    let x_exp = expand_dims(x, x.ndim() as i32 - 1, None);

    let down_exps = w.down_exps.as_ref().unwrap();

    let (gate_out, up_out) = if let Some(packed) = &w.gate_up_exps_packed {
        let out = qw_gather(&x_exp, packed, top_k_indices);
        let half = cfg.moe_expert_intermediate_size as i32;
        (
            mlx_slice_last_dim(&out, 0, half),
            mlx_slice_last_dim(&out, half, half * 2),
        )
    } else {
        let gate_exps = w.gate_exps.as_ref().unwrap();
        let up_exps = w.up_exps.as_ref().unwrap();
        (
            qw_gather(&x_exp, gate_exps, top_k_indices),
            qw_gather(&x_exp, up_exps, top_k_indices),
        )
    };

    // Gemma4 experts use GEGLU (GELU gate); Qwen3 uses SwiGLU (SiLU gate).
    let gate_act = if cfg.uses_geglu {
        gelu(&gate_out, None)
    } else {
        mlx_sys::ops::silu(&gate_out, None)
    };
    let hidden = multiply(&gate_act, &up_out, None);

    // Down projection: [1, seq, top_k, hidden]
    let down_out = qw_gather(&hidden, down_exps, top_k_indices);

    // Weighted sum over top_k dimension → [1, seq, hidden]
    let seq_dim = down_out.ndim() as i32;
    let top_k_axis = seq_dim - 2; // second-to-last dim
    let scores_exp = expand_dims(top_k_weights, seq_dim - 1, None); // [1,seq,top_k,1]
    let weighted = multiply(&down_out, &scores_exp, None);
    sum_axis(&weighted, top_k_axis, false, None)
}

/// Gather-matmul for expert weights (quantized or dense).
///
/// `x`: [..., hidden], `qw.weight`: [num_experts, expert_size, hidden] (or packed).
/// `indices`: [..., top_k].  Returns [..., top_k, out_size].
fn qw_gather(x: &MlxArray, qw: &QuantizedWeight, indices: &MlxArray) -> MlxArray {
    if let Some(scales) = &qw.scales {
        gather_qmm(
            x,
            &qw.weight,
            scales,
            qw.biases.as_ref(),
            indices,
            true,
            Some(qw.group_size),
            Some(qw.bits),
            false,
            None,
        )
    } else {
        // Dense experts: weight shape [N, out, in] → need [N, in, out] for gather_mm.
        let ndim = qw.weight.ndim();
        let mut axes: Vec<i32> = (0..ndim as i32).collect();
        let last = axes.len() - 1;
        axes.swap(last - 1, last);
        let wt = transpose(&qw.weight, &axes, None);
        gather_mm(x, &wt, indices, false, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(attn_output_gate: bool) -> ModelConfig {
        ModelConfig {
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
            gemma4_moe_router: false,
            uses_geglu: false,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
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
}
