use mlx_sys::{
    MlxArray, MlxDtype, add, argpartition_axis, astype, dequantize, expand_dims, gather_mm,
    gather_qmm, matmul, multiply, quantized_matmul, reshape, rms_norm, rope,
    scaled_dot_product_attention, slice_last_dim, softmax, sum_axis, take, take_along_axis, tanh,
    transpose,
};

use ax_engine_core::NativeModelManifest;

use crate::kv_cache::MlxKVCache;
use crate::weights::{LayerWeights, ModelWeights, QuantizedWeight};

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
        Self {
            layer_count: m.layer_count as usize,
            hidden_size: m.hidden_size as usize,
            intermediate_size,
            n_heads: m.attention_head_count as usize,
            n_kv_heads: m.kv_head_count as usize,
            head_dim,
            vocab_size: m.vocab_size as usize,
            rope_theta: m.rope_theta.map(|t| t as f32).unwrap_or(10000.0),
            rope_dims,
            attn_output_gate: m.attn_output_gate,
            query_scale,
            final_logit_softcapping: m.final_logit_softcapping,
            moe_expert_count: m.moe.expert_count.unwrap_or(0) as usize,
            moe_experts_per_token: m.moe.experts_per_token.unwrap_or(0) as usize,
            moe_expert_intermediate_size: m.moe.expert_intermediate_size.unwrap_or(0) as usize,
        }
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
    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), 1e-6, None);

    // 2. Q, K, V projections.
    let (q_raw, k_raw, v_raw, attn_gate) = qkv_project(cfg, w, &normed);

    let seq = hidden.shape()[1] as usize;

    // 3. Reshape to BSHD [1, seq, heads, head_dim].
    let q = reshape(
        &q_raw,
        &[1, seq as i32, cfg.n_heads as i32, cfg.head_dim as i32],
        None,
    );
    let k = reshape(
        &k_raw,
        &[1, seq as i32, cfg.n_kv_heads as i32, cfg.head_dim as i32],
        None,
    );
    let v = reshape(
        &v_raw,
        &[1, seq as i32, cfg.n_kv_heads as i32, cfg.head_dim as i32],
        None,
    );

    // 4. Optional per-head Q/K norms (Qwen3) applied in BSHD space before transpose.
    let q = if let Some(qn) = &w.q_norm {
        let q_f = reshape(&q, &[(cfg.n_heads * seq) as i32, cfg.head_dim as i32], None);
        let q_n = rms_norm(&q_f, Some(qn), 1e-6, None);
        reshape(
            &q_n,
            &[1, seq as i32, cfg.n_heads as i32, cfg.head_dim as i32],
            None,
        )
    } else {
        q
    };

    let k = if let Some(kn) = &w.k_norm {
        let k_f = reshape(
            &k,
            &[(cfg.n_kv_heads * seq) as i32, cfg.head_dim as i32],
            None,
        );
        let k_n = rms_norm(&k_f, Some(kn), 1e-6, None);
        reshape(
            &k_n,
            &[1, seq as i32, cfg.n_kv_heads as i32, cfg.head_dim as i32],
            None,
        )
    } else {
        k
    };

    // 5. Transpose BSHD → BHSD [1, heads, seq, head_dim].
    let q = transpose(&q, &[0, 2, 1, 3], None);
    let k = transpose(&k, &[0, 2, 1, 3], None);
    let v = transpose(&v, &[0, 2, 1, 3], None);

    // 6. RoPE on [1, heads, seq, head_dim].
    let q_rope = rope(
        &q,
        cfg.rope_dims as i32,
        false,
        Some(cfg.rope_theta),
        1.0,
        token_offset as i32,
        None,
        None,
    );
    let k_rope = rope(
        &k,
        cfg.rope_dims as i32,
        false,
        Some(cfg.rope_theta),
        1.0,
        token_offset as i32,
        None,
        None,
    );

    // 7. Update KV cache and get full K/V: [1, n_kv_heads, kv_seq, head_dim].
    let (cached_k, cached_v) = cache.append(layer_idx, k_rope, v);

    // 8. SDPA with native GQA support (n_heads may differ from n_kv_heads).
    //    MLX broadcasts K/V heads internally — no manual expand_kv_heads needed.
    let causal = seq > 1;
    let attn_sdpa =
        scaled_dot_product_attention(&q_rope, &cached_k, &cached_v, cfg.query_scale, causal, None);

    // 9. Transpose back: [1, n_heads, seq, head_dim] → [1, seq, n_heads, head_dim].
    let attn_out = transpose(&attn_sdpa, &[0, 2, 1, 3], None);

    // 10. Reshape back to [1, seq, hidden].
    let attn_flat = reshape(
        &attn_out,
        &[1, seq as i32, (cfg.n_heads * cfg.head_dim) as i32],
        None,
    );

    // 11. Output projection.
    let attn_proj = qw(&attn_flat, &w.o_proj);

    // 12. Optional attention output gate.
    let attn_out = if let Some(gate) = attn_gate {
        multiply(&mlx_sys::ops::sigmoid(&gate, None), &attn_proj, None)
    } else {
        attn_proj
    };
    let attn_out = if let Some(post_norm) = &w.attn_post_norm {
        rms_norm(&attn_out, Some(post_norm), 1e-6, None)
    } else {
        attn_out
    };

    // 13. Residual.
    let hidden = add(hidden, &attn_out, None);

    // 14. Pre-FFN norm (pre_feedforward_layernorm).
    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), 1e-6, None);

    // 15. FFN: MoE or dense.
    let ffn_out = if w.router_proj.is_some() {
        // MoE path (Gemma4 MoE layers).
        // Dense sub-block:
        let h1 = ffn_swiglu(cfg, w, &normed2);
        let h1 = rms_norm(&h1, w.ffn_post_norm1.as_ref(), 1e-6, None);
        // Expert sub-block:
        let h2_normed = rms_norm(&hidden, w.ffn_norm2.as_ref(), 1e-6, None);
        let (top_k_indices, top_k_weights) = moe_router_forward(cfg, w, &hidden);
        let h2 = moe_experts_forward(cfg, w, &h2_normed, &top_k_indices, &top_k_weights);
        let h2 = rms_norm(&h2, w.ffn_post_norm2.as_ref(), 1e-6, None);
        // Combine and apply shared post norm.
        let combined = add(&h1, &h2, None);
        rms_norm(&combined, w.ffn_post_norm.as_ref(), 1e-6, None)
    } else {
        // Dense path (Qwen3, Gemma4 non-MoE).
        let out = ffn_swiglu(cfg, w, &normed2);
        // post_feedforward_layernorm (Gemma4 dense; optional for Qwen3).
        if let Some(pn) = &w.ffn_post_norm {
            rms_norm(&out, Some(pn), 1e-6, None)
        } else {
            out
        }
    };

    // 16. Residual.
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

fn qkv_project(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    let slices = qkv_slices(cfg);
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

fn qkv_slices(cfg: &ModelConfig) -> QkvSlices {
    let q_size = (cfg.n_heads * cfg.head_dim) as i32;
    let kv_size = (cfg.n_kv_heads * cfg.head_dim) as i32;
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

    let gate_act = mlx_sys::ops::silu(&gate_out, None);
    let ffn_hidden = multiply(&gate_act, &up_out, None);
    qw(&ffn_hidden, &w.down_proj)
}

fn mlx_slice_last_dim(x: &MlxArray, start: i32, end: i32) -> MlxArray {
    slice_last_dim(x, start, end, None)
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

/// Router forward: returns (top_k_indices [1,seq,k], top_k_weights [1,seq,k]).
///
/// Mirrors Gemma4TextRouter.__call__:
///   rms_norm(x, scale * (1/sqrt(hidden))) → proj → argpartition → softmax
fn moe_router_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
) -> (MlxArray, MlxArray) {
    let router_proj = w.router_proj.as_ref().unwrap();
    let router_scale = w.router_scale.as_ref().unwrap();

    // Combined rms_norm weight: router.scale * (1 / sqrt(hidden_size))
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

    // Expert logit projection: [1, seq, num_experts]
    let expert_scores = qw(&normed, router_proj);

    let num_experts = cfg.moe_expert_count;
    let top_k = cfg.moe_experts_per_token;
    let ndim = expert_scores.ndim() as i32;
    let last_axis = ndim - 1;

    // argpartition to get top-k indices (last top_k slots are the top-k elements)
    let part_indices = argpartition_axis(&expert_scores, -(top_k as i32), last_axis, None);
    let top_k_indices = slice_last_dim(
        &part_indices,
        (num_experts - top_k) as i32,
        num_experts as i32,
        None,
    );

    // Gather corresponding logits and softmax
    let top_k_raw = take_along_axis(&expert_scores, &top_k_indices, last_axis, None);
    let top_k_weights = softmax(&top_k_raw, last_axis, None);

    (top_k_indices, top_k_weights)
}

/// Expert forward: applies selected experts to `x` and returns the weighted sum.
///
/// x: [1, seq, hidden] (already pre-normed via pre_feedforward_layernorm_2)
/// top_k_indices: [1, seq, top_k]   expert assignments (uint32)
/// top_k_weights: [1, seq, top_k]   softmax-normalised weights (bf16)
fn moe_experts_forward(
    _cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
) -> MlxArray {
    // Expand x for broadcast over top_k: [1, seq, hidden] → [1, seq, 1, hidden]
    let x_exp = expand_dims(x, x.ndim() as i32 - 1, None);

    // Gate and up projections using gather_qmm (quantized) or gather_mm (fp).
    let down_exps = w.down_exps.as_ref().unwrap();

    let (gate_out, up_out) = if let Some(packed) = &w.gate_up_exps_packed {
        let out = qw_gather(&x_exp, packed, top_k_indices);
        let half = _cfg.moe_expert_intermediate_size as i32;
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

    // SwiGLU activation: silu(gate) * up → [1, seq, top_k, expert_size]
    let gate_act = mlx_sys::ops::silu(&gate_out, None);
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
        }
    }

    #[test]
    fn qkv_slices_dense_attention_without_gate() {
        assert_eq!(
            qkv_slices(&cfg(false)),
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
            qkv_slices(&cfg(true)),
            QkvSlices {
                q: (0, 16),
                gate: Some((16, 32)),
                k: (32, 40),
                v: (40, 48),
            }
        );
    }
}
