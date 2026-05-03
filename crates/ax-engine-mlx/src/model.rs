use mlx_sys::{
    MlxArray, MlxDtype,
    add, astype, dequantize, matmul, multiply, quantized_matmul,
    reshape, slice_last_dim, take, transpose, rms_norm, rope, scaled_dot_product_attention,
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
}

impl ModelConfig {
    pub fn from_manifest(m: &NativeModelManifest) -> Self {
        let head_dim = m.attention_head_dim as usize;
        let rope_dims = m.partial_rotary_factor
            .map(|f| ((head_dim as f32 * f) as usize).next_multiple_of(2))
            .unwrap_or(head_dim);
        let query_scale = m.query_pre_attn_scalar
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
        }
    }
}

/// Forward pass for one transformer layer.
///
/// Returns updated hidden states.
pub fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,   // [1, seq, hidden]
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
) -> MlxArray {
    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), 1e-6, None);

    // 2. Q, K, V projections.
    let (q_raw, k_raw, v_raw) = qkv_project(cfg, w, &normed);

    let seq = hidden.shape()[1] as usize;

    // 3. Reshape to BSHD [1, seq, heads, head_dim].
    let q = reshape(&q_raw, &[1, seq as i32, cfg.n_heads as i32, cfg.head_dim as i32], None);
    let k = reshape(&k_raw, &[1, seq as i32, cfg.n_kv_heads as i32, cfg.head_dim as i32], None);
    let v = reshape(&v_raw, &[1, seq as i32, cfg.n_kv_heads as i32, cfg.head_dim as i32], None);

    // 4. Optional per-head Q/K norms (Qwen3) applied in BSHD space before transpose.
    let q = if let Some(qn) = &w.q_norm {
        let q_f = reshape(&q, &[(cfg.n_heads * seq) as i32, cfg.head_dim as i32], None);
        let q_n = rms_norm(&q_f, Some(qn), 1e-6, None);
        reshape(&q_n, &[1, seq as i32, cfg.n_heads as i32, cfg.head_dim as i32], None)
    } else { q };

    let k = if let Some(kn) = &w.k_norm {
        let k_f = reshape(&k, &[(cfg.n_kv_heads * seq) as i32, cfg.head_dim as i32], None);
        let k_n = rms_norm(&k_f, Some(kn), 1e-6, None);
        reshape(&k_n, &[1, seq as i32, cfg.n_kv_heads as i32, cfg.head_dim as i32], None)
    } else { k };

    // 5. Transpose BSHD → BHSD [1, heads, seq, head_dim].
    let q = transpose(&q, &[0, 2, 1, 3], None);
    let k = transpose(&k, &[0, 2, 1, 3], None);
    let v = transpose(&v, &[0, 2, 1, 3], None);

    // 6. RoPE on [1, heads, seq, head_dim].
    let q_rope = rope(&q, cfg.rope_dims as i32, false, Some(cfg.rope_theta), 1.0, token_offset as i32, None, None);
    let k_rope = rope(&k, cfg.rope_dims as i32, false, Some(cfg.rope_theta), 1.0, token_offset as i32, None, None);

    // 7. Update KV cache and get full K/V: [1, n_kv_heads, kv_seq, head_dim].
    let (cached_k, cached_v) = cache.append(layer_idx, k_rope, v);

    // 8. SDPA with native GQA support (n_heads may differ from n_kv_heads).
    //    MLX broadcasts K/V heads internally — no manual expand_kv_heads needed.
    let causal = seq > 1;
    let attn_sdpa = scaled_dot_product_attention(&q_rope, cached_k, cached_v, cfg.query_scale, causal, None);

    // 9. Transpose back: [1, n_heads, seq, head_dim] → [1, seq, n_heads, head_dim].
    let attn_out = transpose(&attn_sdpa, &[0, 2, 1, 3], None);

    // 10. Reshape back to [1, seq, hidden].
    let attn_flat = reshape(&attn_out, &[1, seq as i32, (cfg.n_heads * cfg.head_dim) as i32], None);

    // 11. Output projection.
    let attn_proj = qw(&attn_flat, &w.o_proj);

    // 12. Optional attention output gate (Qwen3).
    let attn_out = if cfg.attn_output_gate {
        let gate = mlx_sys::ops::sigmoid(&attn_proj, None);
        multiply(&attn_proj, &gate, None)
    } else {
        attn_proj
    };

    // 13. Residual.
    let hidden = add(hidden, &attn_out, None);

    // 14. FFN norm.
    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), 1e-6, None);

    // 15. SwiGLU FFN.
    let ffn_out = ffn_swiglu(cfg, w, &normed2);

    // 16. Residual.
    add(&hidden, &ffn_out, None)
}

/// Embed token IDs and return hidden states of shape [1, seq_len, hidden].
pub fn embed_tokens(token_ids: &[u32], embedding: &QuantizedWeight, hidden_size: usize) -> MlxArray {
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
        let row_b = embedding.biases.as_ref().map(|b| take(b, &ids_arr, 0, None));
        let flat = dequantize(
            &row_w, &row_s, row_b.as_ref(),
            Some(embedding.group_size), Some(embedding.bits), None,
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
    reshape(&logits_f32, &[seq, cfg.vocab_size as i32], None)
}

// ── private helpers ──────────────────────────────────────────────────────────

fn qkv_project(cfg: &ModelConfig, w: &LayerWeights, x: &MlxArray) -> (MlxArray, MlxArray, MlxArray) {
    if let Some(packed) = &w.qkv_packed {
        let out = qw(x, packed);
        let q_size = cfg.n_heads * cfg.head_dim;
        let kv_size = cfg.n_kv_heads * cfg.head_dim;
        let seq = x.shape()[1];
        let q = mlx_slice_last_dim(&out, 0, q_size as i32, seq);
        let k = mlx_slice_last_dim(&out, q_size as i32, (q_size + kv_size) as i32, seq);
        let v = mlx_slice_last_dim(&out, (q_size + kv_size) as i32, (q_size + 2 * kv_size) as i32, seq);
        (q, k, v)
    } else {
        let q = qw(x, w.q_proj.as_ref().unwrap());
        let k = qw(x, w.k_proj.as_ref().unwrap());
        let v = qw(x, w.v_proj.as_ref().unwrap());
        (q, k, v)
    }
}

fn qw(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    if let Some(scales) = &qw.scales {
        quantized_matmul(x, &qw.weight, scales, qw.biases.as_ref(), true, Some(qw.group_size), Some(qw.bits), None)
    } else {
        let wt = transpose(&qw.weight, &[1, 0], None);
        matmul(x, &wt, None)
    }
}

fn ffn_swiglu(cfg: &ModelConfig, w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let (gate_out, up_out) = if let Some(packed) = &w.gate_up_packed {
        let out = qw(x, packed);
        let half = cfg.intermediate_size as i32;
        let gate = mlx_slice_last_dim(&out, 0, half, x.shape()[1]);
        let up = mlx_slice_last_dim(&out, half, half * 2, x.shape()[1]);
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

fn mlx_slice_last_dim(x: &MlxArray, start: i32, end: i32, _seq: i32) -> MlxArray {
    slice_last_dim(x, start, end, None)
}
