use ax_engine_core::NativeModelManifest;
use mlx_sys::MlxArray;

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
#[derive(Clone, Debug, PartialEq)]
pub struct LinearAttentionConfig {
    pub full_attention_interval: usize,
    pub num_value_heads: usize,
    pub num_key_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_dim: usize,
    /// q_scale = key_head_dim^(-1); precomputed at load time to avoid per-step powf calls.
    pub q_scale: f32,
    /// k_scale = key_head_dim^(-0.5); precomputed at load time to avoid per-step powf calls.
    pub k_scale: f32,
}

impl LinearAttentionConfig {
    pub(super) fn from_manifest(m: &NativeModelManifest) -> Option<Self> {
        let cfg = &m.linear_attention;
        if !cfg.is_enabled() {
            return None;
        }
        let key_head_dim = cfg
            .key_head_dim
            .expect("validated linear_attention.key_head_dim") as usize;
        let (q_scale, k_scale) =
            crate::linear_attention_ops::linear_attention_qk_scale(key_head_dim);
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
            key_head_dim,
            value_head_dim: cfg
                .value_head_dim
                .expect("validated linear_attention.value_head_dim")
                as usize,
            conv_kernel_dim: cfg
                .conv_kernel_dim
                .expect("validated linear_attention.conv_kernel_dim")
                as usize,
            q_scale,
            k_scale,
        })
    }

    pub(super) fn is_linear_layer(&self, layer_idx: usize) -> bool {
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
pub struct MlaAttentionConfig {
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub value_head_dim: usize,
    pub q_head_dim: usize,
    pub query_scale: f32,
}

impl MlaAttentionConfig {
    pub(super) fn from_manifest(m: &NativeModelManifest) -> Option<Self> {
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
            // GLM MLA scales scores by the original query head width
            // (qk_nope_head_dim + qk_rope_head_dim), not by the packed
            // SDPA key width (kv_lora_rank + qk_rope_head_dim).
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
    pub n_group: usize,
    pub topk_group: usize,
    pub has_shared_experts: bool,
}

impl GlmRouterConfig {
    pub(super) fn from_manifest(m: &NativeModelManifest) -> Option<Self> {
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
            n_group: cfg.n_group.expect("validated glm_router.n_group") as usize,
            topk_group: cfg.topk_group.expect("validated glm_router.topk_group") as usize,
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
    /// Model family string from the manifest (e.g. "gemma4", "qwen3", "llama3").
    /// Used for named dispatch in `layer_forward_with_turboquant_context`.
    pub model_family: String,
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
    /// Per-layer config (non-empty only for interleaved SWA models like Gemma4/Gemma3).
    pub layer_configs: Vec<LayerConfig>,
    /// Uniform sliding-window size for families where every layer uses the same
    /// window (Mistral3, Mixtral). `None` for families with no SWA or interleaved
    /// SWA (which use `layer_configs` instead).
    pub global_sliding_window: Option<usize>,
    /// True → Gemma4 dual-path MoE routing (rms_norm → proj → softmax).
    /// False → Qwen3 MoE routing (proj → softmax, no rms_norm).
    pub gemma4_moe_router: bool,
    /// Use GELU (Gemma4/Gemma3) instead of SiLU (Qwen3/LLaMA) for FFN gate activation.
    pub uses_geglu: bool,
    /// Scale hidden states after embedding (Gemma4/Gemma3: sqrt(hidden_size)).
    pub hidden_states_scale: Option<f32>,
    /// Normalise top-k MoE routing weights to sum to 1 (Qwen3 MoE norm_topk_prob).
    pub moe_norm_topk_prob: bool,
    /// Dimension of per-layer token embeddings (Gemma4 2B/4B); 0 = disabled.
    pub hidden_size_per_layer_input: usize,
    /// Qwen3.5 gated-delta linear-attention config, when present.
    pub linear_attention: Option<LinearAttentionConfig>,
    /// GLM4MoELite MLA attention config, when present.
    pub mla_attention: Option<MlaAttentionConfig>,
    /// GLM4MoELite sigmoid router config, when present.
    pub glm_router: Option<GlmRouterConfig>,
    /// Epsilon for all RMSNorm operations (1e-6 for Qwen/Gemma, 1e-5 for GLM/LLaMA/Mistral).
    pub rms_norm_eps: f32,
    /// Precomputed LLaMA-3 corrected RoPE frequencies `[dims/2]`.
    /// `None` means standard RoPE (compute freqs from `rope_theta` at runtime).
    /// `Some(freqs)` is passed directly to `mlx_sys::rope` as the `freqs` arg.
    pub rope_freqs: Option<MlxArray>,
    /// LLaMA-4 iRoPE interval: every N-th layer has no RoPE. 0 = all layers use RoPE.
    pub no_rope_layer_interval: usize,
    /// LLaMA-4 attention temperature floor scale (positions / floor → log scale).
    pub attn_temperature_floor: f32,
    /// LLaMA-4 attention temperature scale multiplier.
    pub attn_temperature_scale: f32,
    /// Dense (non-MoE) FFN intermediate size for LLaMA4.
    /// 0 means use `intermediate_size` for both dense and MoE layers.
    pub intermediate_size_mlp: usize,
    /// MoE every N layers (DeepSeek V3: `moe_layer_freq`). 0 = use GlmRouter dispatch.
    pub moe_layer_freq: usize,
    /// First K layers use dense FFN, rest use MoE (DeepSeek V3: `first_k_dense_replace`).
    pub moe_first_dense_layers: usize,
    /// Number of always-active shared experts (DeepSeek V3: `n_shared_experts`).
    pub moe_shared_expert_count: usize,
    /// Use sigmoid routing (DeepSeek V3). False → softmax (Qwen3/GLM).
    pub moe_sigmoid_routing: bool,
    /// Scale applied to selected expert weights (DeepSeek V3: 2.5, others: 1.0).
    pub moe_routed_scaling_factor: f32,
    /// Number of expert groups for group-based top-k (DeepSeek V3: 8, others: 1).
    pub moe_n_group: usize,
    /// Number of groups retained after group scoring (DeepSeek V3: 4, others: 1).
    pub moe_topk_group: usize,
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
        let uses_geglu = matches!(m.model_family.as_str(), "gemma4" | "gemma3");
        let query_scale = if is_gemma4 {
            1.0
        } else {
            m.query_pre_attn_scalar
                .map(|s| 1.0 / (s as f32).sqrt())
                .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt())
        };

        // Uniform SWA: used by families where every layer has the same window
        // (e.g. Mistral3, Mixtral). Set only when layer_types is empty (no
        // interleaved pattern) and sliding_window_size is present.
        let global_sliding_window = if layer_configs.is_empty() {
            m.sliding_window_size.map(|w| w as usize)
        } else {
            None
        };

        // LLaMA-3 corrected RoPE frequencies, precomputed once at model load.
        let rope_freqs = if m.rope_scaling_type.as_deref() == Some("llama3") {
            let factor = m.rope_scaling_factor.unwrap_or(8.0);
            let low_ff = m.rope_low_freq_factor.unwrap_or(1.0);
            let high_ff = m.rope_high_freq_factor.unwrap_or(4.0);
            let orig_ctx = m.rope_original_context_len.unwrap_or(8192);
            Some(super::shared::build_llama3_rope_freqs(
                rope_dims, rope_theta, factor, low_ff, high_ff, orig_ctx,
            ))
        } else {
            None
        };

        Self {
            model_family: m.model_family.clone(),
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
            global_sliding_window,
            gemma4_moe_router: is_gemma4,
            uses_geglu,
            hidden_states_scale: m.hidden_states_scale,
            moe_norm_topk_prob: m.moe_norm_topk_prob,
            hidden_size_per_layer_input: m.hidden_size_per_layer_input as usize,
            linear_attention: LinearAttentionConfig::from_manifest(m),
            mla_attention: MlaAttentionConfig::from_manifest(m),
            glm_router: GlmRouterConfig::from_manifest(m),
            rms_norm_eps: m
                .rms_norm_eps
                .unwrap_or_else(|| default_rms_norm_eps(&m.model_family)),
            rope_freqs,
            no_rope_layer_interval: m.no_rope_layer_interval as usize,
            attn_temperature_floor: m.attn_temperature_floor.unwrap_or(8192) as f32,
            attn_temperature_scale: m.attn_temperature_scale.unwrap_or(0.1),
            intermediate_size_mlp: m.intermediate_size_mlp as usize,
            moe_layer_freq: m.moe.layer_freq.unwrap_or(1) as usize,
            moe_first_dense_layers: m.moe.first_dense_layers.unwrap_or(0) as usize,
            moe_shared_expert_count: m.moe.shared_expert_count.unwrap_or(0) as usize,
            moe_sigmoid_routing: m.moe.sigmoid_routing,
            moe_routed_scaling_factor: m.moe.routed_scaling_factor.unwrap_or(1.0),
            moe_n_group: m.moe.n_group.unwrap_or(1) as usize,
            moe_topk_group: m.moe.topk_group.unwrap_or(1) as usize,
        }
    }

    pub fn is_linear_attention_layer(&self, layer_idx: usize) -> bool {
        self.linear_attention
            .as_ref()
            .is_some_and(|linear| linear.is_linear_layer(layer_idx))
    }

    /// True when the layer is a MoE layer for DeepSeek V3:
    /// `layer_idx >= first_dense_layers && layer_idx % moe_layer_freq == 0`.
    pub fn is_deepseek_moe_layer(&self, layer_idx: usize) -> bool {
        self.moe_expert_count > 0
            && self.moe_layer_freq > 0
            && layer_idx >= self.moe_first_dense_layers
            && layer_idx.is_multiple_of(self.moe_layer_freq)
    }

    pub fn is_glm_moe_layer(&self, layer_idx: usize) -> bool {
        self.glm_router
            .as_ref()
            .is_some_and(|router| router.is_moe_layer(layer_idx))
    }
}

fn default_rms_norm_eps(model_family: &str) -> f32 {
    if model_family.starts_with("qwen") || model_family.starts_with("gemma") {
        1e-6
    } else {
        1e-5
    }
}

pub(super) fn build_layer_configs(
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
pub(super) fn layer_params(
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
            cfg.global_sliding_window,
            None,
            false,
        )
    }
}
