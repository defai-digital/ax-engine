use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, add, argpartition_axis, argsort_axis,
    astype, concatenate, dequantize, divide, expand_dims, expand_dims_axes, gather_mm, gather_qmm,
    gelu, gelu_approx, matmul, multiply, quantized_matmul, reshape, rms_norm, rope,
    scaled_dot_product_attention_with_mask, slice_last_dim, softmax, sum_axis, take,
    take_along_axis, tanh, transpose, zeros,
};

use ax_engine_core::NativeModelManifest;

use crate::attention_mask::create_causal_mask;
use crate::kv_cache::MlxKVCache;
use crate::linear_attention::{
    compute_gated_delta_g, gated_delta_kernel, linear_attention_conv1d,
    normalize_linear_attention_qk, rms_norm_gated, split_linear_attention_qkv,
};
use crate::weights::{LayerWeights, ModelWeights, QuantizedWeight};

const SWITCH_GLU_SORT_THRESHOLD: usize = 64;

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

/// Hyperparameters for Qwen3.5 gated-delta linear-attention layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearAttentionConfig {
    pub full_attention_interval: usize,
    pub num_value_heads: usize,
    pub num_key_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_dim: usize,
}

impl LinearAttentionConfig {
    fn from_manifest(m: &NativeModelManifest) -> Option<Self> {
        let cfg = &m.linear_attention;
        if !cfg.is_enabled() {
            return None;
        }
        Some(Self {
            full_attention_interval: cfg
                .resolved_full_attention_interval(&m.model_family)
                .expect("validated linear_attention.full_attention_interval")
                as usize,
            num_value_heads: cfg
                .num_value_heads
                .expect("validated linear_attention.num_value_heads")
                as usize,
            num_key_heads: cfg
                .num_key_heads
                .expect("validated linear_attention.num_key_heads")
                as usize,
            key_head_dim: cfg
                .key_head_dim
                .expect("validated linear_attention.key_head_dim")
                as usize,
            value_head_dim: cfg
                .value_head_dim
                .expect("validated linear_attention.value_head_dim")
                as usize,
            conv_kernel_dim: cfg
                .conv_kernel_dim
                .expect("validated linear_attention.conv_kernel_dim")
                as usize,
        })
    }

    fn is_linear_layer(&self, layer_idx: usize) -> bool {
        !(layer_idx + 1).is_multiple_of(self.full_attention_interval)
    }

    pub fn key_dim(&self) -> usize {
        self.num_key_heads * self.key_head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.num_value_heads * self.value_head_dim
    }

    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

/// GLM4MoELite MLA attention dimensions extracted from the manifest.
#[derive(Clone, Debug, PartialEq)]
pub struct GlmMlaAttentionConfig {
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub value_head_dim: usize,
    pub q_head_dim: usize,
    pub query_scale: f32,
}

impl GlmMlaAttentionConfig {
    fn from_manifest(m: &NativeModelManifest) -> Option<Self> {
        let cfg = &m.mla_attention;
        if !cfg.is_enabled() {
            return None;
        }

        let q_lora_rank = cfg
            .q_lora_rank
            .expect("validated mla_attention.q_lora_rank") as usize;
        let kv_lora_rank = cfg
            .kv_lora_rank
            .expect("validated mla_attention.kv_lora_rank") as usize;
        let qk_nope_head_dim =
            cfg.qk_nope_head_dim
                .expect("validated mla_attention.qk_nope_head_dim") as usize;
        let qk_rope_head_dim =
            cfg.qk_rope_head_dim
                .expect("validated mla_attention.qk_rope_head_dim") as usize;
        let value_head_dim = cfg
            .value_head_dim
            .expect("validated mla_attention.value_head_dim") as usize;
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

        Some(Self {
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            q_head_dim,
            query_scale: 1.0 / (q_head_dim as f32).sqrt(),
        })
    }

    pub fn latent_kv_cache_width(&self) -> usize {
        self.kv_lora_rank
    }

    pub fn rope_key_cache_width(&self) -> usize {
        self.qk_rope_head_dim
    }
}

/// GLM4MoELite router contract extracted from mlx-lm/glm4_moe_lite.py.
#[derive(Clone, Debug, PartialEq)]
pub struct GlmRouterConfig {
    pub first_dense_layer_count: usize,
    pub routed_scaling_factor: f32,
    pub has_shared_experts: bool,
}

impl GlmRouterConfig {
    fn from_manifest(m: &NativeModelManifest) -> Option<Self> {
        let cfg = &m.glm_router;
        if !cfg.is_enabled() {
            return None;
        }

        Some(Self {
            first_dense_layer_count: cfg
                .first_dense_layer_count
                .expect("validated glm_router.first_dense_layer_count")
                as usize,
            routed_scaling_factor: cfg
                .routed_scaling_factor
                .expect("validated glm_router.routed_scaling_factor"),
            has_shared_experts: cfg.has_shared_experts,
        })
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.first_dense_layer_count
    }
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
    /// Dimension of per-layer token embeddings (Gemma4 2B/4B); 0 = disabled.
    pub hidden_size_per_layer_input: usize,
    /// Qwen3.5 gated-delta linear-attention config, when present.
    pub linear_attention: Option<LinearAttentionConfig>,
    /// GLM4MoELite MLA attention config, when present.
    pub glm_mla_attention: Option<GlmMlaAttentionConfig>,
    /// GLM4MoELite sigmoid router config, when present.
    pub glm_router: Option<GlmRouterConfig>,
}

impl ModelConfig {
    pub fn from_manifest(m: &NativeModelManifest) -> Self {
        let head_dim = m.attention_head_dim as usize;
        let rope_dims = m
            .partial_rotary_factor
            .map(|f| ((head_dim as f32 * f) as usize).next_multiple_of(2))
            .unwrap_or(head_dim);
        let intermediate_size = if m.intermediate_size > 0 {
            m.intermediate_size as usize
        } else {
            (m.hidden_size as usize * 8 / 3).next_multiple_of(256)
        };
        let rope_theta = m.rope_theta.map(|t| t as f32).unwrap_or(10000.0);
        let layer_configs = build_layer_configs(m, head_dim, rope_theta, rope_dims);
        let is_gemma4 = m.model_family == "gemma4";
        let query_scale = if is_gemma4 {
            1.0
        } else {
            m.query_pre_attn_scalar
                .map(|s| 1.0 / (s as f32).sqrt())
                .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt())
        };

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
            hidden_size_per_layer_input: m.hidden_size_per_layer_input as usize,
            linear_attention: LinearAttentionConfig::from_manifest(m),
            glm_mla_attention: GlmMlaAttentionConfig::from_manifest(m),
            glm_router: GlmRouterConfig::from_manifest(m),
        }
    }

    pub fn is_linear_attention_layer(&self, layer_idx: usize) -> bool {
        self.linear_attention
            .as_ref()
            .is_some_and(|linear| linear.is_linear_layer(layer_idx))
    }

    pub fn is_glm_moe_layer(&self, layer_idx: usize) -> bool {
        self.glm_router
            .as_ref()
            .is_some_and(|router| router.is_moe_layer(layer_idx))
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
    let sliding_rope_dims = if m.model_family == "gemma4" {
        // Gemma4's partial_rotary_factor belongs to full_attention's
        // proportional RoPE. sliding_attention uses default RoPE over the full
        // sliding head_dim.
        default_head_dim
    } else {
        default_rope_dims
    };
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
                    rope_dims: sliding_rope_dims,
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
    per_layer_input: Option<&MlxArray>, // [1, seq, per_layer_dim] or None
) -> MlxArray {
    let (head_dim, rope_theta, rope_dims, sliding_window, kv_source, v_norm_no_scale) =
        layer_params(cfg, layer_idx);

    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), 1e-6, None);

    let seq = hidden.shape()[1] as usize;

    let attn_proj = if cfg.is_linear_attention_layer(layer_idx) {
        linear_attention_forward(cfg, w, &normed, cache, layer_idx)
    } else {
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
            let kv_heads = (k_raw.shape()[2] as usize)
                .checked_div(head_dim)
                .expect("k projection output must divide by head_dim");
            let k = reshape(
                &k_raw,
                &[1, seq as i32, kv_heads as i32, head_dim as i32],
                None,
            );
            let v = reshape(
                &v_raw,
                &[1, seq as i32, kv_heads as i32, head_dim as i32],
                None,
            );

            let q = qk_norm_bshd(q, w.q_norm.as_ref(), cfg.n_heads, head_dim, seq);
            let k = qk_norm_bshd(k, w.k_norm.as_ref(), kv_heads, head_dim, seq);

            let q = transpose(&q, &[0, 2, 1, 3], None);
            let k = transpose(&k, &[0, 2, 1, 3], None);
            let v = prepare_value_bhsd(v, v_norm_no_scale, kv_heads, head_dim, seq);

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

        // 8. SDPA (GQA: MLX broadcasts KV heads internally). Sliding attention is
        // enforced by an array mask, not by truncating K/V, matching mlx-lm.
        let key_len = cached_k.shape()[2] as usize;
        let mask_array = attention_mask_array(seq, key_len, sliding_window);
        let mask = match mask_array.as_ref() {
            Some(mask) => ScaledDotProductAttentionMask::Array(mask),
            None if seq > 1 => ScaledDotProductAttentionMask::Causal,
            None => ScaledDotProductAttentionMask::None,
        };
        let attn_sdpa = scaled_dot_product_attention_with_mask(
            &q_rope,
            &cached_k,
            &cached_v,
            cfg.query_scale,
            mask,
            None,
        );

        // 10. Transpose back: [1, n_heads, seq, head_dim] → [1, seq, n_heads, head_dim].
        let attn_out = transpose(&attn_sdpa, &[0, 2, 1, 3], None);

        // 11. Reshape to [1, seq, hidden].
        let attn_flat = reshape(
            &attn_out,
            &[1, seq as i32, (cfg.n_heads * head_dim) as i32],
            None,
        );

        // 12-13. Optional Qwen3.5 output gate, then output projection.
        let attn_proj = attention_output_projection(
            &attn_flat,
            attn_gate.as_ref(),
            w.o_proj
                .as_ref()
                .expect("full-attention layer must have o_proj"),
        );

        // 14. Optional post-attention layernorm (Gemma4): applied BEFORE residual add.
        if let Some(post_norm) = &w.attn_post_norm {
            rms_norm(&attn_proj, Some(post_norm), 1e-6, None)
        } else {
            attn_proj
        }
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
            let mut out = moe_experts_forward(cfg, w, &normed2, &top_k_indices, &top_k_weights);
            if w.shared_expert_gate.is_some() {
                out = add(&out, &shared_expert_forward(w, &normed2), None);
            }
            rms_norm_opt(&out, w.ffn_post_norm.as_ref())
        }
    } else {
        // Dense path (Qwen3, Gemma4 non-MoE layers).
        let out = ffn_swiglu(cfg, w, &normed2);
        rms_norm_opt(&out, w.ffn_post_norm.as_ref())
    };

    // 18. Residual.
    let mut out = add(&hidden, &ffn_out, None);

    // 19. Per-layer input gating (Gemma4 2B/4B): gate(h) * per_layer_embed → proj → norm + h.
    if let (Some(gate_w), Some(proj_w), Some(post_norm), Some(pli)) = (
        w.per_layer_gate.as_ref(),
        w.per_layer_proj_w.as_ref(),
        w.per_layer_post_norm.as_ref(),
        per_layer_input,
    ) {
        let gate = gelu_approx(&qw(&out, gate_w), None);
        let gated = multiply(&gate, pli, None);
        let projected = qw(&gated, proj_w);
        let normed = rms_norm(&projected, Some(post_norm), 1e-6, None);
        out = add(&out, &normed, None);
    }

    // 20. Optional layer scalar (Gemma4): h = h * layer_scalar.
    if let Some(scalar) = &w.layer_scalar {
        multiply(&out, scalar, None)
    } else {
        out
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
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    let per_layer_inputs = compute_per_layer_inputs(cfg, weights, token_ids, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward(cfg, layer_w, &hidden, cache, li, token_offset, pli);
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
/// Used by draft verification to check all draft tokens in one pass.
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

    let per_layer_inputs = compute_per_layer_inputs(cfg, weights, token_ids, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward(cfg, layer_w, &hidden, cache, li, token_offset, pli);
    }

    let seq = token_ids.len() as i32;
    let normed = rms_norm(&hidden, Some(&weights.final_norm), 1e-6, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    reshape(&logits_f32, &[seq, cfg.vocab_size as i32], None)
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
) -> MlxArray {
    let mut cache = MlxKVCache::new(weights.layers.len());
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    let per_layer_inputs = compute_per_layer_inputs(cfg, weights, token_ids, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward(cfg, layer_w, &hidden, &mut cache, li, 0, pli);
    }
    rms_norm(&hidden, Some(&weights.final_norm), 1e-6, None)
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
    // Normalise to [1] for embedding take (reshape is a no-op if already [1]).
    let tok_1d = reshape(token_arr, &[1_i32], None);

    let mut hidden = embed_tokens_arr(&tok_1d, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &tok_1d, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward(cfg, layer_w, &hidden, cache, li, token_offset, pli);
    }
    // Single token: hidden shape is [1, 1, hidden_size] — no sequence slice needed.
    let normed = rms_norm(&hidden, Some(&weights.final_norm), 1e-6, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    reshape(&logits_f32, &[cfg.vocab_size as i32], None)
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
    let embed_out = astype(&embed_out, dtype, None);
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
    let proj_out = rms_norm(&proj_out, Some(proj_norm_w), 1e-6, None);

    // 3. Combine: (proj + embed) * 2^(-0.5)
    let combined_scale = 2.0_f32.powf(-0.5);
    let combined = add(&proj_out, &embed_out, None);
    let combined = scale_hidden(&combined, combined_scale);
    // combined shape: [1, seq, num_layers, per_layer_dim]

    // 4. Split per layer using take on axis 2
    let per_layer = (0..num_layers)
        .map(|li| {
            let idx_arr = MlxArray::from_raw_data(
                &(li as u32) as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
                &[1_i32],
                MlxDtype::Uint32,
            );
            // take on axis 2: [1, seq, 1, per_layer_dim] → squeeze → [1, seq, per_layer_dim]
            let slice = take(&combined, &idx_arr, 2, None);
            reshape(&slice, &[1, seq, per_layer_dim as i32], None)
        })
        .collect();

    Some(per_layer)
}

fn compute_per_layer_inputs(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    hidden: &MlxArray,
) -> Option<Vec<MlxArray>> {
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    compute_per_layer_inputs_arr(cfg, weights, &ids_1d, hidden)
}

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

/// Apply no-scale per-head RMS norm in BSHD [1, seq, n_heads, head_dim] space.
fn rms_norm_no_scale_bshd(x: MlxArray, n_heads: usize, head_dim: usize, seq: usize) -> MlxArray {
    let flat = reshape(&x, &[(n_heads * seq) as i32, head_dim as i32], None);
    let normed = rms_norm(&flat, None, 1e-6, None);
    reshape(
        &normed,
        &[1, seq as i32, n_heads as i32, head_dim as i32],
        None,
    )
}

/// Apply optional V RMSNorm in BSHD, then convert to BHSD for attention/KV cache.
fn prepare_value_bhsd(
    v: MlxArray,
    v_norm_no_scale: bool,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
) -> MlxArray {
    let v = if v_norm_no_scale {
        rms_norm_no_scale_bshd(v, n_heads, head_dim, seq)
    } else {
        v
    };
    transpose(&v, &[0, 2, 1, 3], None)
}

/// Build the array mask only when the fast causal/none modes cannot express it.
fn attention_mask_array(
    seq_len: usize,
    key_len: usize,
    sliding_window: Option<usize>,
) -> Option<MlxArray> {
    if seq_len == 0 {
        return None;
    }

    let offset = key_len.saturating_sub(seq_len);
    if let Some(window) = sliding_window {
        return Some(create_causal_mask(seq_len, offset, Some(window)));
    }
    if offset > 0 && seq_len > 1 {
        return Some(create_causal_mask(seq_len, offset, None));
    }
    None
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
        let (q, gate) = if slices.gate.is_some() {
            // attn_output_gate=true: q_proj output is [B, L, n_heads, 2*head_dim] interleaved.
            // Split by reshaping to [B, L, n_heads, 2*head_dim] and slicing last dim,
            // matching mlx-lm's `mx.split(q_proj_out.reshape(B, L, n_heads, -1), 2, axis=-1)`.
            let seq = q_full.shape()[1];
            let q_heads = reshape(
                &q_full,
                &[1, seq, cfg.n_heads as i32, 2 * head_dim as i32],
                None,
            );
            let q = reshape(
                &slice_last_dim(&q_heads, 0, head_dim as i32, None),
                &[1, seq, (cfg.n_heads * head_dim) as i32],
                None,
            );
            let gate = reshape(
                &slice_last_dim(&q_heads, head_dim as i32, 2 * head_dim as i32, None),
                &[1, seq, (cfg.n_heads * head_dim) as i32],
                None,
            );
            (q, Some(gate))
        } else {
            let q = mlx_slice_last_dim(&q_full, slices.q.0, slices.q.1);
            (q, None)
        };
        let k = qw(x, w.k_proj.as_ref().unwrap());
        let v = w
            .v_proj
            .as_ref()
            .map(|v_proj| qw(x, v_proj))
            .unwrap_or_else(|| k.clone());
        (q, k, v, gate)
    }
}

fn attention_output_projection(
    attn_flat: &MlxArray,
    attn_gate: Option<&MlxArray>,
    o_proj: &QuantizedWeight,
) -> MlxArray {
    let gated = if let Some(gate) = attn_gate {
        multiply(attn_flat, &mlx_sys::ops::sigmoid(gate, None), None)
    } else {
        attn_flat.clone()
    };
    qw(&gated, o_proj)
}

fn linear_attention_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
) -> MlxArray {
    let linear_cfg = cfg
        .linear_attention
        .as_ref()
        .expect("linear attention layer requires linear_attention config");
    let linear_w = w
        .linear_attn
        .as_ref()
        .expect("linear attention layer requires linear attention weights");
    let seq = x.shape()[1];

    let (qkv, z, a, b) = linear_attention_inputs(linear_cfg, linear_w, x, seq);

    let (conv_state, recurrent_state) = cache.linear_state(layer_idx);
    let conv_weight = linear_conv_weight(&linear_w.conv1d);
    let (conv_out, new_conv_state) =
        linear_attention_conv1d(linear_cfg, &qkv, &conv_weight, conv_state, None);
    let qkv = split_linear_attention_qkv(linear_cfg, &conv_out);
    let (q, k) = normalize_linear_attention_qk(linear_cfg, &qkv.q, &qkv.k);
    let beta = mlx_sys::ops::sigmoid(&b, None);
    let g = compute_gated_delta_g(&linear_w.a_log, &a, &linear_w.dt_bias);
    let state = recurrent_state.cloned().unwrap_or_else(|| {
        zeros(
            &[
                1,
                linear_cfg.num_value_heads as i32,
                linear_cfg.value_head_dim as i32,
                linear_cfg.key_head_dim as i32,
            ],
            q.dtype(),
            None,
        )
    });
    let (out, new_recurrent_state) = gated_delta_kernel(&q, &k, &qkv.v, &g, &beta, &state);
    cache.set_linear_state(layer_idx, new_conv_state, new_recurrent_state);

    let out = rms_norm_gated(&out, &z, &linear_w.norm);
    let flat = reshape(&out, &[1, seq, linear_cfg.value_dim() as i32], None);
    qw(&flat, &linear_w.out_proj)
}

fn linear_attention_inputs(
    cfg: &LinearAttentionConfig,
    w: &crate::weights::LinearAttentionWeights,
    x: &MlxArray,
    seq: i32,
) -> (MlxArray, MlxArray, MlxArray, MlxArray) {
    if let (Some(qkvz_w), Some(ba_w)) = (&w.in_proj_qkvz, &w.in_proj_ba) {
        let mixed_qkvz = qw(x, qkvz_w);
        let value_heads_per_key = cfg.num_value_heads / cfg.num_key_heads;
        let value_dim_per_key = value_heads_per_key * cfg.value_head_dim;
        let qkvz_per_key = cfg.key_head_dim * 2 + value_dim_per_key * 2;
        let mixed_qkvz = reshape(
            &mixed_qkvz,
            &[1, seq, cfg.num_key_heads as i32, qkvz_per_key as i32],
            None,
        );
        let q = slice_last_dim(&mixed_qkvz, 0, cfg.key_head_dim as i32, None);
        let k = slice_last_dim(
            &mixed_qkvz,
            cfg.key_head_dim as i32,
            (cfg.key_head_dim * 2) as i32,
            None,
        );
        let v = slice_last_dim(
            &mixed_qkvz,
            (cfg.key_head_dim * 2) as i32,
            (cfg.key_head_dim * 2 + value_dim_per_key) as i32,
            None,
        );
        let z = slice_last_dim(
            &mixed_qkvz,
            (cfg.key_head_dim * 2 + value_dim_per_key) as i32,
            qkvz_per_key as i32,
            None,
        );
        let qkv = concatenate(
            &[
                &reshape(&q, &[1, seq, cfg.key_dim() as i32], None),
                &reshape(&k, &[1, seq, cfg.key_dim() as i32], None),
                &reshape(&v, &[1, seq, cfg.value_dim() as i32], None),
            ],
            2,
            None,
        );
        let z = reshape(
            &z,
            &[
                1,
                seq,
                cfg.num_value_heads as i32,
                cfg.value_head_dim as i32,
            ],
            None,
        );

        let mixed_ba = qw(x, ba_w);
        let ba = reshape(
            &mixed_ba,
            &[
                1,
                seq,
                cfg.num_key_heads as i32,
                (value_heads_per_key * 2) as i32,
            ],
            None,
        );
        let b = reshape(
            &slice_last_dim(&ba, 0, value_heads_per_key as i32, None),
            &[1, seq, cfg.num_value_heads as i32],
            None,
        );
        let a = reshape(
            &slice_last_dim(
                &ba,
                value_heads_per_key as i32,
                (value_heads_per_key * 2) as i32,
                None,
            ),
            &[1, seq, cfg.num_value_heads as i32],
            None,
        );
        return (qkv, z, a, b);
    }

    let qkv = qw(
        x,
        w.in_proj_qkv
            .as_ref()
            .expect("split linear attention must have qkv projection"),
    );
    let z = reshape(
        &qw(
            x,
            w.in_proj_z
                .as_ref()
                .expect("split linear attention must have z projection"),
        ),
        &[
            1,
            seq,
            cfg.num_value_heads as i32,
            cfg.value_head_dim as i32,
        ],
        None,
    );
    let a = qw(
        x,
        w.in_proj_a
            .as_ref()
            .expect("split linear attention must have a projection"),
    );
    let b = qw(
        x,
        w.in_proj_b
            .as_ref()
            .expect("split linear attention must have b projection"),
    );
    (qkv, z, a, b)
}

fn linear_conv_weight(weight: &QuantizedWeight) -> MlxArray {
    if let Some(scales) = &weight.scales {
        dequantize(
            &weight.weight,
            scales,
            weight.biases.as_ref(),
            Some(weight.group_size),
            Some(weight.bits),
            None,
        )
    } else {
        weight.weight.clone()
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
    qw(
        &ffn_hidden,
        w.down_proj
            .as_ref()
            .expect("dense FFN layer must have down_proj"),
    )
}

fn shared_expert_forward(w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let gate = qw(
        x,
        w.shared_gate_proj
            .as_ref()
            .expect("shared expert must have gate projection"),
    );
    let up = qw(
        x,
        w.shared_up_proj
            .as_ref()
            .expect("shared expert must have up projection"),
    );
    let hidden = multiply(&mlx_sys::ops::silu(&gate, None), &up, None);
    let shared = qw(
        &hidden,
        w.shared_down_proj
            .as_ref()
            .expect("shared expert must have down projection"),
    );
    let shared_gate = qw(
        x,
        w.shared_expert_gate
            .as_ref()
            .expect("shared expert must have gate input projection"),
    );
    multiply(&mlx_sys::ops::sigmoid(&shared_gate, None), &shared, None)
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
    let (top_k_indices, mut top_k_weights) = top_k_by_argpartition(
        &expert_scores,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        true,
    );
    // Apply per-expert output scale (initialized to ones; fine-tuned checkpoints may differ).
    if let Some(pes) = &w.router_expert_scale {
        let gathered = take(pes, &top_k_indices, 0, None);
        top_k_weights = multiply(&top_k_weights, &gathered, None);
    }
    (top_k_indices, top_k_weights)
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
        let sum = sum_axis(&top_k_weights, last_axis, false, None);
        let shape = top_k_weights.shape();
        let sum = reshape(&sum, &[shape[0], shape[1], 1], None);
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
    // Match MLX SwitchGLU: [batch, seq, hidden] → [batch, seq, 1, 1, hidden].
    // The extra singleton before top_k is required by gather_mm/gather_qmm broadcasting.
    let x_exp = expand_dims_axes(x, &[-2, -3], None);
    let gather_inputs = switch_gather_inputs(&x_exp, top_k_indices);
    let down_exps = w.down_exps.as_ref().unwrap();

    let (gate_out, up_out) = if let Some(packed) = &w.gate_up_exps_packed {
        let out = qw_gather(
            &gather_inputs.x,
            packed,
            &gather_inputs.indices,
            gather_inputs.sorted_indices,
        );
        let half = cfg.moe_expert_intermediate_size as i32;
        (
            mlx_slice_last_dim(&out, 0, half),
            mlx_slice_last_dim(&out, half, half * 2),
        )
    } else {
        let gate_exps = w.gate_exps.as_ref().unwrap();
        let up_exps = w.up_exps.as_ref().unwrap();
        (
            qw_gather(
                &gather_inputs.x,
                gate_exps,
                &gather_inputs.indices,
                gather_inputs.sorted_indices,
            ),
            qw_gather(
                &gather_inputs.x,
                up_exps,
                &gather_inputs.indices,
                gather_inputs.sorted_indices,
            ),
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
    let down_out = squeeze_switch_singleton(&qw_gather(
        &hidden,
        down_exps,
        &gather_inputs.indices,
        gather_inputs.sorted_indices,
    ));
    let down_out = gather_inputs.unsort(down_out);

    // Weighted sum over top_k dimension → [1, seq, hidden]
    let seq_dim = down_out.ndim() as i32;
    let top_k_axis = seq_dim - 2; // second-to-last dim
    let scores_exp = expand_dims(top_k_weights, top_k_weights.ndim() as i32, None);
    let weighted = multiply(&down_out, &scores_exp, None);
    sum_axis(&weighted, top_k_axis, false, None)
}

struct SwitchGatherInputs {
    x: MlxArray,
    indices: MlxArray,
    sorted_indices: bool,
    inv_order: Option<MlxArray>,
    original_indices_shape: Vec<i32>,
}

impl SwitchGatherInputs {
    fn unsort(&self, x: MlxArray) -> MlxArray {
        let Some(inv_order) = &self.inv_order else {
            return x;
        };
        let unsorted = take(&x, inv_order, 0, None);
        let mut shape = self.original_indices_shape.clone();
        let hidden = *x
            .shape()
            .last()
            .expect("expert output must have hidden dim");
        shape.push(hidden);
        reshape(&unsorted, &shape, None)
    }
}

fn switch_gather_inputs(x_expanded: &MlxArray, indices: &MlxArray) -> SwitchGatherInputs {
    let indices_shape = indices.shape();
    let selection_count = shape_element_count(&indices_shape);
    let top_k = indices_shape.last().copied().unwrap_or(1).max(1) as usize;
    if selection_count < SWITCH_GLU_SORT_THRESHOLD {
        return SwitchGatherInputs {
            x: x_expanded.clone(),
            indices: indices.clone(),
            sorted_indices: false,
            inv_order: None,
            original_indices_shape: indices_shape,
        };
    }

    let flat_indices = reshape(indices, &[selection_count as i32], None);
    let order = argsort_axis(&flat_indices, -1, None);
    let inv_order = argsort_axis(&order, -1, None);
    let sorted_indices = take(&flat_indices, &order, 0, None);

    let x_shape = x_expanded.shape();
    let hidden = *x_shape
        .last()
        .expect("SwitchGLU input must include hidden dim");
    let rows = selection_count / top_k;
    let x_flat = reshape(x_expanded, &[rows as i32, 1, hidden], None);
    let top_k_scalar = MlxArray::from_raw_data(
        &(top_k as u32) as *const u32 as *const u8,
        std::mem::size_of::<u32>(),
        &[1],
        MlxDtype::Uint32,
    );
    let row_indices = astype(&divide(&order, &top_k_scalar, None), MlxDtype::Uint32, None);
    let x_sorted = take(&x_flat, &row_indices, 0, None);

    SwitchGatherInputs {
        x: x_sorted,
        indices: sorted_indices,
        sorted_indices: true,
        inv_order: Some(inv_order),
        original_indices_shape: indices_shape,
    }
}

fn shape_element_count(shape: &[i32]) -> usize {
    shape
        .iter()
        .map(|dim| usize::try_from(*dim).expect("MLX shape dims must be non-negative"))
        .product()
}

fn squeeze_switch_singleton(x: &MlxArray) -> MlxArray {
    let mut shape = x.shape();
    let ndim = shape.len();
    if ndim >= 2 && shape[ndim - 2] == 1 {
        shape.remove(ndim - 2);
        reshape(x, &shape, None)
    } else {
        x.clone()
    }
}

/// Gather-matmul for expert weights (quantized or dense).
///
/// `x`: [..., hidden], `qw.weight`: [num_experts, expert_size, hidden] (or packed).
/// `indices`: [..., top_k].  Returns [..., top_k, out_size].
fn qw_gather(
    x: &MlxArray,
    qw: &QuantizedWeight,
    indices: &MlxArray,
    sorted_indices: bool,
) -> MlxArray {
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
            sorted_indices,
            None,
        )
    } else {
        // Dense experts: weight shape [N, out, in] → need [N, in, out] for gather_mm.
        let ndim = qw.weight.ndim();
        let mut axes: Vec<i32> = (0..ndim as i32).collect();
        let last = axes.len() - 1;
        axes.swap(last - 1, last);
        let wt = transpose(&qw.weight, &axes, None);
        gather_mm(x, &wt, indices, sorted_indices, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::LinearAttentionWeights;
    use ax_engine_core::model::{NativeGlmRouterConfig, NativeMlaAttentionConfig};
    use ax_engine_core::{
        NativeLinearAttentionConfig, NativeMoeConfig, NativeRuntimeStatus, NativeTensorFormat,
    };
    use mlx_sys::{eval, zeros};
    use std::collections::BTreeMap;

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
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            glm_mla_attention: None,
            glm_router: None,
        }
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
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: Some(0.25),
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
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: true,
            partial_rotary_factor: Some(0.25),
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
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
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
            },
            glm_router: NativeGlmRouterConfig {
                first_dense_layer_count: Some(1),
                routed_scaling_factor: Some(1.8),
                has_shared_experts: true,
            },
            tensors: Vec::new(),
        }
    }

    fn dense_weight(shape: &[i32]) -> QuantizedWeight {
        QuantizedWeight::new(zeros(shape, MlxDtype::Float32, None), None, None)
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
                conv1d: dense_weight(&[cfg.conv_dim() as i32, cfg.conv_kernel_dim as i32, 1_i32]),
                dt_bias: zeros(&[cfg.num_value_heads as i32], MlxDtype::Float32, None),
                a_log: zeros(&[cfg.num_value_heads as i32], MlxDtype::Float32, None),
                norm: zeros(&[cfg.value_head_dim as i32], MlxDtype::Float32, None),
                out_proj: dense_weight(&[hidden_size as i32, cfg.value_dim() as i32]),
            }),
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
            router_scale: None,
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
            router_scale: None,
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
        };
        let x = zeros(&[1, 2, 4], MlxDtype::Float32, None);

        let (_q, k, v, gate) = qkv_project(&cfg, &weights, &x, 4);

        assert!(gate.is_none());
        assert_eq!(k.shape(), vec![1, 2, 4]);
        assert_eq!(v.shape(), vec![1, 2, 4]);
    }

    #[test]
    fn gemma4_layer_configs_keep_sliding_rope_at_full_head_dim() {
        let cfg = ModelConfig::from_manifest(&gemma4_interleaved_manifest());

        assert_eq!(cfg.query_scale, 1.0);
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
    fn qwen35_linear_attention_config_matches_reference_interval() {
        let cfg = ModelConfig::from_manifest(&qwen35_linear_manifest());
        let linear = cfg
            .linear_attention
            .as_ref()
            .expect("linear attention config");

        assert!(cfg.glm_mla_attention.is_none());
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
    fn glm_mla_attention_config_matches_reference_shape_contract() {
        let cfg = ModelConfig::from_manifest(&glm4_moe_lite_manifest());
        let mla = cfg
            .glm_mla_attention
            .as_ref()
            .expect("GLM MLA attention config");

        assert_eq!(mla.q_lora_rank, 768);
        assert_eq!(mla.kv_lora_rank, 512);
        assert_eq!(mla.qk_nope_head_dim, 192);
        assert_eq!(mla.qk_rope_head_dim, 64);
        assert_eq!(mla.value_head_dim, 256);
        assert_eq!(mla.q_head_dim, 256);
        assert_eq!(mla.latent_kv_cache_width(), 512);
        assert_eq!(mla.rope_key_cache_width(), 64);
        assert!((mla.query_scale - (1.0 / 256_f32.sqrt())).abs() < f32::EPSILON);
        assert_eq!(cfg.query_scale, mla.query_scale);
    }

    #[test]
    fn glm_router_config_matches_reference_dense_moe_split() {
        let cfg = ModelConfig::from_manifest(&glm4_moe_lite_manifest());
        let router = cfg.glm_router.as_ref().expect("GLM router config");

        assert_eq!(router.first_dense_layer_count, 1);
        assert!((router.routed_scaling_factor - 1.8).abs() < f32::EPSILON);
        assert!(router.has_shared_experts);
        assert!(!cfg.is_glm_moe_layer(0));
        assert!(cfg.is_glm_moe_layer(1));
        assert!(cfg.is_glm_moe_layer(2));
    }

    #[test]
    fn linear_attention_forward_returns_hidden_shape_and_updates_cache() {
        let mut cfg = cfg(true);
        cfg.hidden_size = 8;
        cfg.linear_attention = Some(LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 1,
            num_key_heads: 1,
            key_head_dim: 32,
            value_head_dim: 4,
            conv_kernel_dim: 4,
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
            router_scale: None,
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
            router_scale: None,
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
            router_scale: None,
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
            router_scale: None,
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
            router_scale: None,
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
            router_scale: None,
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
        let prepared = prepare_value_bhsd(v, true, 2, 4, 3);

        assert_eq!(prepared.shape(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn attention_mask_array_uses_fast_modes_for_simple_causal_cases() {
        assert!(attention_mask_array(1, 1, None).is_none());
        assert!(attention_mask_array(4, 4, None).is_none());
    }

    #[test]
    fn attention_mask_array_uses_offset_mask_for_cached_prefill() {
        let mask = attention_mask_array(2, 5, None).expect("cached prefill needs offset mask");

        assert_eq!(mask.shape(), vec![2, 5]);
    }

    #[test]
    fn attention_mask_array_keeps_full_kv_for_sliding_attention() {
        let mask = attention_mask_array(1, 6, Some(4)).expect("decode needs sliding mask");

        assert_eq!(mask.shape(), vec![1, 6]);
    }
}
