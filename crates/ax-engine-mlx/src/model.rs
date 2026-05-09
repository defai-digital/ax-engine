use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, add, argpartition_axis, argsort_axis,
    astype, broadcast_to, concatenate, dequantize, divide, eval, expand_dims, expand_dims_axes,
    gather_mm, gather_qmm, gelu_approx, matmul, multiply, put_along_axis, quantized_matmul,
    reshape, rms_norm, rope, scaled_dot_product_attention_with_mask, slice, slice_last_dim,
    softmax, sum_axis, take, take_along_axis, tanh, transpose, where_cond, zeros,
};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use ax_engine_core::{MlxKvCompressionConfig, MlxTurboQuantPreset, NativeModelManifest};

use crate::attention_mask::create_causal_mask;
use crate::kv_cache::{MlxKVCache, MlxKvCompressionDecodeCandidate, MlxKvCompressionDecodeOutcome};
use crate::linear_attention::{
    gated_delta_kernel, linear_attention_conv1d, normalize_linear_attention_qk, rms_norm_gated,
    split_linear_attention_qkv,
};
use crate::turboquant::turboquant_fused_decode_head_dim_supported;
use crate::weights::{LayerWeights, ModelWeights, QuantizedWeight};

const SWITCH_GLU_SORT_THRESHOLD: usize = 64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Gemma4MoeProfileSnapshot {
    pub enabled: u32,
    pub decode_layers: u32,
    pub topk_selections: u32,
    pub sorted_gather_layers: u32,
    pub unsorted_gather_layers: u32,
    pub attention_wall_us: u32,
    pub dense_wall_us: u32,
    pub router_wall_us: u32,
    pub expert_wall_us: u32,
    pub post_wall_us: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LinearAttentionProfileSnapshot {
    pub enabled: u32,
    pub layers: u32,
    pub tokens: u32,
    pub projection_wall_us: u32,
    pub conv_wall_us: u32,
    pub qk_norm_wall_us: u32,
    pub recurrent_wall_us: u32,
    pub output_wall_us: u32,
}

#[derive(Clone, Copy)]
enum Gemma4MoeProfileStage {
    Attention,
    Dense,
    Router,
    Expert,
    Post,
}

#[derive(Clone, Copy)]
enum LinearAttentionProfileStage {
    Projection,
    Conv,
    QkNorm,
    Recurrent,
    Output,
}

static GEMMA4_MOE_PROFILE: OnceLock<Mutex<Gemma4MoeProfileSnapshot>> = OnceLock::new();
static LINEAR_ATTENTION_PROFILE: OnceLock<Mutex<LinearAttentionProfileSnapshot>> = OnceLock::new();

fn gemma4_moe_profile_enabled() -> bool {
    matches!(
        std::env::var("AX_MLX_GEMMA4_MOE_PROFILE").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn linear_attention_profile_enabled() -> bool {
    matches!(
        std::env::var("AX_MLX_LINEAR_ATTENTION_PROFILE").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn gemma4_moe_profile() -> &'static Mutex<Gemma4MoeProfileSnapshot> {
    GEMMA4_MOE_PROFILE.get_or_init(|| Mutex::new(Gemma4MoeProfileSnapshot::default()))
}

fn linear_attention_profile() -> &'static Mutex<LinearAttentionProfileSnapshot> {
    LINEAR_ATTENTION_PROFILE.get_or_init(|| Mutex::new(LinearAttentionProfileSnapshot::default()))
}

fn saturating_profile_us(started: Instant) -> u32 {
    started.elapsed().as_micros().min(u32::MAX as u128) as u32
}

fn record_gemma4_moe_decode_layer(topk_selections: usize, sorted_gather: bool) {
    let mut profile = gemma4_moe_profile().lock().unwrap();
    profile.enabled = 1;
    profile.decode_layers = profile.decode_layers.saturating_add(1);
    profile.topk_selections = profile
        .topk_selections
        .saturating_add(topk_selections.min(u32::MAX as usize) as u32);
    if sorted_gather {
        profile.sorted_gather_layers = profile.sorted_gather_layers.saturating_add(1);
    } else {
        profile.unsorted_gather_layers = profile.unsorted_gather_layers.saturating_add(1);
    }
}

fn record_gemma4_moe_profile_stage(stage: Gemma4MoeProfileStage, wall_us: u32) {
    let mut profile = gemma4_moe_profile().lock().unwrap();
    profile.enabled = 1;
    let target = match stage {
        Gemma4MoeProfileStage::Attention => &mut profile.attention_wall_us,
        Gemma4MoeProfileStage::Dense => &mut profile.dense_wall_us,
        Gemma4MoeProfileStage::Router => &mut profile.router_wall_us,
        Gemma4MoeProfileStage::Expert => &mut profile.expert_wall_us,
        Gemma4MoeProfileStage::Post => &mut profile.post_wall_us,
    };
    *target = target.saturating_add(wall_us);
}

fn record_linear_attention_profile_layer(tokens: i32) {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.enabled = 1;
    profile.layers = profile.layers.saturating_add(1);
    profile.tokens = profile
        .tokens
        .saturating_add(tokens.max(0).min(u32::MAX as i32) as u32);
}

fn record_linear_attention_profile_stage(stage: LinearAttentionProfileStage, wall_us: u32) {
    let mut profile = linear_attention_profile().lock().unwrap();
    profile.enabled = 1;
    let target = match stage {
        LinearAttentionProfileStage::Projection => &mut profile.projection_wall_us,
        LinearAttentionProfileStage::Conv => &mut profile.conv_wall_us,
        LinearAttentionProfileStage::QkNorm => &mut profile.qk_norm_wall_us,
        LinearAttentionProfileStage::Recurrent => &mut profile.recurrent_wall_us,
        LinearAttentionProfileStage::Output => &mut profile.output_wall_us,
    };
    *target = target.saturating_add(wall_us);
}

fn profile_eval_elapsed(
    enabled: bool,
    stage: Gemma4MoeProfileStage,
    started: Instant,
    targets: &[&MlxArray],
) {
    if enabled {
        eval(targets);
        record_gemma4_moe_profile_stage(stage, saturating_profile_us(started));
    }
}

fn linear_attention_profile_eval_elapsed(
    enabled: bool,
    stage: LinearAttentionProfileStage,
    started: Instant,
    targets: &[&MlxArray],
) {
    if enabled {
        eval(targets);
        record_linear_attention_profile_stage(stage, saturating_profile_us(started));
    }
}

pub fn take_gemma4_moe_profile_snapshot() -> Gemma4MoeProfileSnapshot {
    let mut profile = gemma4_moe_profile().lock().unwrap();
    let snapshot = *profile;
    *profile = Gemma4MoeProfileSnapshot::default();
    snapshot
}

pub fn take_linear_attention_profile_snapshot() -> LinearAttentionProfileSnapshot {
    let mut profile = linear_attention_profile().lock().unwrap();
    let snapshot = *profile;
    *profile = LinearAttentionProfileSnapshot::default();
    snapshot
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantModelDecodeCandidateStatus {
    Disabled,
    PrefillOnly,
    LinearAttentionLayer,
    GlmMlaLayer,
    SlidingWindowLayer,
    KvSharedLayer,
    IneligibleLayer,
    UnsupportedPreset,
    UnsupportedHeadDim,
    GroupedQueryAttention,
    MissingRuntimeStorage,
    Ready,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantModelDecodeCandidate {
    pub status: TurboQuantModelDecodeCandidateStatus,
    pub cold_tokens: usize,
    pub hot_tokens: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TurboQuantModelDecodeContext<'a> {
    pub config: MlxKvCompressionConfig,
    pub layer_eligible: &'a [bool],
}

impl TurboQuantModelDecodeCandidate {
    pub const fn disabled() -> Self {
        Self {
            status: TurboQuantModelDecodeCandidateStatus::Disabled,
            cold_tokens: 0,
            hot_tokens: 0,
        }
    }

    pub const fn telemetry_status(self) -> MlxKvCompressionDecodeCandidate {
        match self.status {
            TurboQuantModelDecodeCandidateStatus::Disabled => {
                MlxKvCompressionDecodeCandidate::Disabled
            }
            TurboQuantModelDecodeCandidateStatus::PrefillOnly => {
                MlxKvCompressionDecodeCandidate::PrefillOnly
            }
            TurboQuantModelDecodeCandidateStatus::LinearAttentionLayer
            | TurboQuantModelDecodeCandidateStatus::GlmMlaLayer
            | TurboQuantModelDecodeCandidateStatus::SlidingWindowLayer
            | TurboQuantModelDecodeCandidateStatus::KvSharedLayer => {
                MlxKvCompressionDecodeCandidate::AttentionKind
            }
            TurboQuantModelDecodeCandidateStatus::IneligibleLayer => {
                MlxKvCompressionDecodeCandidate::IneligibleLayer
            }
            TurboQuantModelDecodeCandidateStatus::UnsupportedPreset => {
                MlxKvCompressionDecodeCandidate::UnsupportedPreset
            }
            TurboQuantModelDecodeCandidateStatus::UnsupportedHeadDim => {
                MlxKvCompressionDecodeCandidate::UnsupportedHeadDim
            }
            TurboQuantModelDecodeCandidateStatus::GroupedQueryAttention => {
                MlxKvCompressionDecodeCandidate::GroupedQueryAttention
            }
            TurboQuantModelDecodeCandidateStatus::MissingRuntimeStorage => {
                MlxKvCompressionDecodeCandidate::MissingRuntimeStorage
            }
            TurboQuantModelDecodeCandidateStatus::Ready => MlxKvCompressionDecodeCandidate::Ready,
        }
    }
}

impl<'a> TurboQuantModelDecodeContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn decode_candidate(
        self,
        cfg: &ModelConfig,
        cache: &MlxKVCache,
        layer_idx: usize,
        seq: usize,
        head_dim: usize,
        kv_heads: usize,
        sliding_window: Option<usize>,
        kv_source: Option<usize>,
        has_glm_mla_attention: bool,
    ) -> TurboQuantModelDecodeCandidate {
        let status = if !self.config.requests_fused_decode() {
            TurboQuantModelDecodeCandidateStatus::Disabled
        } else if seq != 1 {
            TurboQuantModelDecodeCandidateStatus::PrefillOnly
        } else if cfg.is_linear_attention_layer(layer_idx) {
            TurboQuantModelDecodeCandidateStatus::LinearAttentionLayer
        } else if has_glm_mla_attention {
            TurboQuantModelDecodeCandidateStatus::GlmMlaLayer
        } else if sliding_window.is_some() {
            TurboQuantModelDecodeCandidateStatus::SlidingWindowLayer
        } else if kv_source.is_some() {
            TurboQuantModelDecodeCandidateStatus::KvSharedLayer
        } else if !self.layer_eligible.get(layer_idx).copied().unwrap_or(false) {
            TurboQuantModelDecodeCandidateStatus::IneligibleLayer
        } else if self.config.preset != MlxTurboQuantPreset::K8V4 {
            TurboQuantModelDecodeCandidateStatus::UnsupportedPreset
        } else if !turboquant_fused_decode_head_dim_supported(head_dim) {
            TurboQuantModelDecodeCandidateStatus::UnsupportedHeadDim
        } else if kv_heads == 0 || !cfg.n_heads.is_multiple_of(kv_heads) {
            TurboQuantModelDecodeCandidateStatus::GroupedQueryAttention
        } else if cache
            .turboquant_shadow_storage_cold_tokens(layer_idx)
            .unwrap_or(0)
            == 0
        {
            TurboQuantModelDecodeCandidateStatus::MissingRuntimeStorage
        } else {
            TurboQuantModelDecodeCandidateStatus::Ready
        };

        let cold_tokens = if status == TurboQuantModelDecodeCandidateStatus::Ready {
            cache
                .turboquant_shadow_storage_cold_tokens(layer_idx)
                .unwrap_or(0)
        } else {
            0
        };
        let total_tokens = cache.seq_len.saturating_add(seq);
        TurboQuantModelDecodeCandidate {
            status,
            cold_tokens,
            hot_tokens: total_tokens.saturating_sub(cold_tokens),
        }
    }
}

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
    /// Epsilon for all RMSNorm operations (1e-6 for Qwen/Gemma, 1e-5 for GLM).
    pub rms_norm_eps: f32,
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
            rms_norm_eps: if m.model_family.starts_with("qwen")
                || m.model_family.starts_with("gemma")
            {
                1e-6
            } else {
                1e-5
            },
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
/// `shared_mask`: pre-computed SDPA mask for this layer — `None` computes it
/// internally from `seq`, `key_len`, and `sliding_window`.  Pass `Some(&m)`
/// from `build_layer_masks` in `forward*` to avoid creating identical mask
/// graphs for every layer of the same attention type.
///
/// Returns updated hidden states.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray, // [1, seq, hidden]
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    per_layer_input: Option<&MlxArray>, // [1, seq, per_layer_dim] or None
    shared_mask: Option<&Option<MlxArray>>,
) -> MlxArray {
    layer_forward_with_turboquant_context(
        cfg,
        w,
        hidden,
        cache,
        layer_idx,
        token_offset,
        per_layer_input,
        shared_mask,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn layer_forward_with_turboquant_context(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray, // [1, seq, hidden]
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    per_layer_input: Option<&MlxArray>, // [1, seq, per_layer_dim] or None
    shared_mask: Option<&Option<MlxArray>>,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    let (head_dim, rope_theta, rope_dims, sliding_window, kv_source, v_norm_no_scale) =
        layer_params(cfg, layer_idx);

    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);

    let seq = hidden.shape()[1] as usize;
    let profile_gemma4_moe_decode =
        cfg.gemma4_moe_router && seq == 1 && gemma4_moe_profile_enabled();
    let attention_started = profile_gemma4_moe_decode.then(Instant::now);

    let attn_proj = if cfg.is_linear_attention_layer(layer_idx) {
        linear_attention_forward(cfg, w, &normed, cache, layer_idx)
    } else if w.glm_mla_attn.is_some() {
        let attn_proj = glm_mla_attention_forward(cfg, w, &normed, cache, layer_idx, token_offset);
        if let Some(post_norm) = &w.attn_post_norm {
            rms_norm(&attn_proj, Some(post_norm), cfg.rms_norm_eps, None)
        } else {
            attn_proj
        }
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
            let q = qk_norm_bshd(
                q,
                w.q_norm.as_ref(),
                cfg.n_heads,
                head_dim,
                seq,
                cfg.rms_norm_eps,
            );
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

            let q = qk_norm_bshd(
                q,
                w.q_norm.as_ref(),
                cfg.n_heads,
                head_dim,
                seq,
                cfg.rms_norm_eps,
            );
            let k = qk_norm_bshd(
                k,
                w.k_norm.as_ref(),
                kv_heads,
                head_dim,
                seq,
                cfg.rms_norm_eps,
            );

            let q = transpose(&q, &[0, 2, 1, 3], None);
            let k = transpose(&k, &[0, 2, 1, 3], None);
            let v = prepare_value_bhsd(
                v,
                v_norm_no_scale,
                kv_heads,
                head_dim,
                seq,
                cfg.rms_norm_eps,
            );

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

            let (ck, cv) = if seq == 1 {
                cache.append_with_retained_window(layer_idx, k_rope, v, sliding_window)
            } else {
                cache.append(layer_idx, k_rope, v)
            };
            (q_rope, ck, cv, attn_gate_raw)
        };
        let turboquant_candidate = turboquant_context
            .map(|context| {
                context.decode_candidate(
                    cfg,
                    cache,
                    layer_idx,
                    seq,
                    head_dim,
                    cached_k.shape()[1] as usize,
                    sliding_window,
                    kv_source,
                    false,
                )
            })
            .unwrap_or_else(TurboQuantModelDecodeCandidate::disabled);
        cache.record_turboquant_decode_candidate(turboquant_candidate.telemetry_status());

        // 8. SDPA (GQA: MLX broadcasts KV heads internally). For decode, Gemma
        // sliding-window layers present only the retained window to SDPA, matching
        // mlx_lm/mlx-swift-lm rotating-cache behavior.
        let key_len = cached_k.shape()[2] as usize;
        let local_mask: Option<MlxArray>;
        let mask_opt: &Option<MlxArray> = if let Some(m) = shared_mask {
            m
        } else {
            local_mask = attention_mask_array(seq, key_len, sliding_window);
            &local_mask
        };
        let attn_sdpa =
            if turboquant_candidate.status == TurboQuantModelDecodeCandidateStatus::Ready {
                let turboquant_out = turboquant_decode_attention_experimental(
                    cache,
                    layer_idx,
                    &q_rope,
                    seq,
                    cfg.n_heads,
                    head_dim,
                    cfg.query_scale,
                    turboquant_context
                        .expect("ready TurboQuant candidate requires context")
                        .config
                        .hot_window_tokens,
                );
                let outcome = turboquant_out
                    .as_ref()
                    .map(|output| output.outcome)
                    .unwrap_or(MlxKvCompressionDecodeOutcome::Fallback);
                cache.record_turboquant_fused_decode_attempt(outcome);
                turboquant_out
                    .map(|output| output.attention)
                    .unwrap_or_else(|| {
                        full_precision_attention(
                            &q_rope,
                            &cached_k,
                            &cached_v,
                            cfg.query_scale,
                            seq,
                            mask_opt,
                        )
                    })
            } else {
                full_precision_attention(
                    &q_rope,
                    &cached_k,
                    &cached_v,
                    cfg.query_scale,
                    seq,
                    mask_opt,
                )
            };

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
            rms_norm(&attn_proj, Some(post_norm), cfg.rms_norm_eps, None)
        } else {
            attn_proj
        }
    };
    if let Some(started) = attention_started {
        profile_eval_elapsed(
            profile_gemma4_moe_decode,
            Gemma4MoeProfileStage::Attention,
            started,
            &[&attn_proj],
        );
    }

    // 15. Residual.
    let hidden = add(hidden, &attn_proj, None);

    // 16. Pre-FFN norm.
    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);

    // 17. FFN: MoE or dense.
    let ffn_out = if w.router_proj.is_some() {
        if cfg.gemma4_moe_router {
            // Gemma4 dual-path: dense sub-block + expert sub-block.
            let dense_started = profile_gemma4_moe_decode.then(Instant::now);
            let h1 = ffn_swiglu(cfg, w, &normed2);
            let h1 = rms_norm_opt(&h1, w.ffn_post_norm1.as_ref(), cfg.rms_norm_eps);
            if let Some(started) = dense_started {
                profile_eval_elapsed(
                    profile_gemma4_moe_decode,
                    Gemma4MoeProfileStage::Dense,
                    started,
                    &[&h1],
                );
            }
            let h2_norm = w
                .ffn_norm2
                .as_ref()
                .expect("validated Gemma4 MoE layer must include ffn_norm_2");
            let h2_normed = rms_norm(&hidden, Some(h2_norm), cfg.rms_norm_eps, None);
            let router_started = profile_gemma4_moe_decode.then(Instant::now);
            let (top_k_indices, top_k_weights) = moe_router_gemma4(cfg, w, &hidden);
            if let Some(started) = router_started {
                profile_eval_elapsed(
                    profile_gemma4_moe_decode,
                    Gemma4MoeProfileStage::Router,
                    started,
                    &[&top_k_indices, &top_k_weights],
                );
            }
            if profile_gemma4_moe_decode {
                let topk_selections = shape_element_count(&top_k_indices.shape());
                record_gemma4_moe_decode_layer(
                    topk_selections,
                    topk_selections >= SWITCH_GLU_SORT_THRESHOLD,
                );
            }
            let expert_started = profile_gemma4_moe_decode.then(Instant::now);
            let h2 = moe_experts_forward(cfg, w, &h2_normed, &top_k_indices, &top_k_weights);
            if let Some(started) = expert_started {
                profile_eval_elapsed(
                    profile_gemma4_moe_decode,
                    Gemma4MoeProfileStage::Expert,
                    started,
                    &[&h2],
                );
            }
            let post_started = profile_gemma4_moe_decode.then(Instant::now);
            let h2 = rms_norm_opt(&h2, w.ffn_post_norm2.as_ref(), cfg.rms_norm_eps);
            let combined = add(&h1, &h2, None);
            let out = rms_norm_opt(&combined, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps);
            if let Some(started) = post_started {
                profile_eval_elapsed(
                    profile_gemma4_moe_decode,
                    Gemma4MoeProfileStage::Post,
                    started,
                    &[&out],
                );
            }
            out
        } else {
            let (top_k_indices, top_k_weights) = if cfg.glm_router.is_some() {
                moe_router_glm(cfg, w, &normed2)
            } else {
                // Qwen3 MoE: router (proj → softmax → top-k) + expert forward.
                moe_router_qwen3(cfg, w, &normed2)
            };
            let mut out = moe_experts_forward(cfg, w, &normed2, &top_k_indices, &top_k_weights);
            if w.shared_gate_proj.is_some() {
                out = add(&out, &shared_expert_forward(w, &normed2), None);
            }
            rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps)
        }
    } else {
        // Dense path (Qwen3, Gemma4 non-MoE layers).
        let out = ffn_swiglu(cfg, w, &normed2);
        rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps)
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
        let normed = rms_norm(&projected, Some(post_norm), cfg.rms_norm_eps, None);
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
    forward_with_turboquant_context(cfg, weights, token_ids, cache, token_offset, None)
}

pub fn forward_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    let seq = token_ids.len();
    let masks = build_layer_masks(cfg, weights.layers.len(), seq, token_offset + seq);
    let per_layer_inputs = compute_per_layer_inputs(cfg, weights, token_ids, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward_with_turboquant_context(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
            turboquant_context,
        );
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

    let normed = rms_norm(
        &last_hidden,
        Some(&weights.final_norm),
        cfg.rms_norm_eps,
        None,
    );
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
    forward_all_positions_with_turboquant_context(
        cfg,
        weights,
        token_ids,
        cache,
        token_offset,
        None,
    )
}

pub fn forward_all_positions_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut MlxKVCache,
    token_offset: usize,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }

    let seq = token_ids.len();
    let masks = build_layer_masks(cfg, weights.layers.len(), seq, token_offset + seq);
    let per_layer_inputs = compute_per_layer_inputs(cfg, weights, token_ids, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward_with_turboquant_context(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
            turboquant_context,
        );
    }

    let seq = seq as i32;
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    reshape(&logits_f32, &[seq, cfg.vocab_size as i32], None)
}

/// Cache-free single transformer layer for dense embedding models.
///
/// Equivalent to the standard dense-attention path in `layer_forward`, but
/// skips all KV-cache writes (no `zeros` allocation, no `slice_update`).
/// Only valid for Qwen3/Gemma dense layers — no linear attention, no MLA,
/// no KV-source sharing, no MoE.
fn layer_forward_dense_embed(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray, // [batch, seq, hidden]
    layer_idx: usize,
) -> MlxArray {
    let (head_dim, rope_theta, rope_dims, _sliding_window, _kv_source, v_norm_no_scale) =
        layer_params(cfg, layer_idx);
    let batch = hidden.shape()[0] as usize;
    let seq = hidden.shape()[1] as usize;

    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);

    // 2-7. QKV projections, reshape, QK-norm, transpose, RoPE.
    let (q_raw, k_raw, v_raw, _attn_gate) = qkv_project(cfg, w, &normed, head_dim);
    let kv_heads = (k_raw.shape()[2] as usize)
        .checked_div(head_dim)
        .expect("k projection output must divide by head_dim");

    let q = reshape(
        &q_raw,
        &[
            batch as i32,
            seq as i32,
            cfg.n_heads as i32,
            head_dim as i32,
        ],
        None,
    );
    let k = reshape(
        &k_raw,
        &[batch as i32, seq as i32, kv_heads as i32, head_dim as i32],
        None,
    );
    let v = reshape(
        &v_raw,
        &[batch as i32, seq as i32, kv_heads as i32, head_dim as i32],
        None,
    );

    let q = qk_norm_bshd(
        q,
        w.q_norm.as_ref(),
        cfg.n_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    let k = qk_norm_bshd(
        k,
        w.k_norm.as_ref(),
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );

    let q = transpose(&q, &[0, 2, 1, 3], None);
    let k = transpose(&k, &[0, 2, 1, 3], None);
    let v = prepare_value_bhsd(
        v,
        v_norm_no_scale,
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );

    let q_rope = rope(
        &q,
        rope_dims as i32,
        false,
        Some(rope_theta),
        1.0,
        0,
        None,
        None,
    );
    let k_rope = rope(
        &k,
        rope_dims as i32,
        false,
        Some(rope_theta),
        1.0,
        0,
        None,
        None,
    );

    // 8. SDPA — k_rope/v used directly, no KV-cache writes.
    let mask_opt: Option<MlxArray> = None; // resolves to Causal in full_precision_attention
    let attn_sdpa = full_precision_attention(&q_rope, &k_rope, &v, cfg.query_scale, seq, &mask_opt);

    // 9-13. Transpose back, reshape, output projection, residual.
    let attn_out = transpose(&attn_sdpa, &[0, 2, 1, 3], None);
    let attn_flat = reshape(
        &attn_out,
        &[batch as i32, seq as i32, (cfg.n_heads * head_dim) as i32],
        None,
    );
    let attn_proj = attention_output_projection(
        &attn_flat,
        None,
        w.o_proj
            .as_ref()
            .expect("dense embed layer must have o_proj"),
    );
    let hidden = add(hidden, &attn_proj, None);

    // 14-17. Pre-FFN norm, dense SwiGLU, residual.
    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    let ffn_out = ffn_swiglu(cfg, w, &normed2);
    add(&hidden, &ffn_out, None)
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
    target_position: Option<usize>,
) -> MlxArray {
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    for (li, layer_w) in weights.layers.iter().enumerate() {
        hidden = layer_forward_dense_embed(cfg, layer_w, &hidden, li);
    }
    // For Last/Cls pooling the caller knows which position it needs. Extract it
    // before the final norm so we norm [1, 1, H] instead of [1, seq, H].
    let to_norm = match target_position {
        Some(pos) => {
            let pos_u32 = pos as u32;
            let idx = MlxArray::from_raw_data(
                &pos_u32 as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
                &[1_i32],
                MlxDtype::Uint32,
            );
            take(&hidden, &idx, 1, None)
        }
        None => hidden,
    };
    rms_norm(&to_norm, Some(&weights.final_norm), cfg.rms_norm_eps, None)
}

/// Embed a flat token-id array and reshape to [batch, max_seq, hidden_size].
fn embed_tokens_batched(
    ids_flat: &MlxArray, // [batch * max_seq] u32
    embedding: &QuantizedWeight,
    hidden_size: usize,
    batch: usize,
    max_seq: usize,
) -> MlxArray {
    // Reuse single-sequence path; it produces [1, batch*max_seq, hidden].
    let flat_hidden = embed_tokens_arr(ids_flat, embedding, hidden_size);
    reshape(
        &flat_hidden,
        &[batch as i32, max_seq as i32, hidden_size as i32],
        None,
    )
}

/// Batch stateless forward pass for dense-embedding extraction.
///
/// Pads `batch_token_ids` to the longest sequence length, runs a single
/// transformer forward pass for all sequences, and returns normalized hidden
/// states along with the actual (un-padded) length of each sequence.
///
/// When `target_positions` is `Some`, each sequence's hidden state is extracted
/// at its specified position *before* the final norm, so the norm runs on
/// `[B, hidden_size]` instead of `[B, max_seq, hidden_size]`. Pass it for
/// Last/Cls pooling; pass `None` for Mean pooling (which needs all positions).
/// The returned array shape is `[B, hidden_size]` when positions are given, or
/// `[B, max_seq, hidden_size]` otherwise.
pub fn forward_for_embedding_batch(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    batch_token_ids: &[Vec<u32>],
    target_positions: Option<&[usize]>,
) -> (MlxArray, Vec<usize>) {
    let actual_lens: Vec<usize> = batch_token_ids.iter().map(Vec::len).collect();
    let max_len = *actual_lens.iter().max().expect("non-empty batch");
    let batch = batch_token_ids.len();

    let mut flat_ids = vec![0u32; batch * max_len];
    for (i, ids) in batch_token_ids.iter().enumerate() {
        flat_ids[i * max_len..i * max_len + ids.len()].copy_from_slice(ids);
    }
    // mlx_array_new_data copies the buffer immediately; flat_ids can be dropped.
    let ids_flat = MlxArray::from_raw_data(
        flat_ids.as_ptr() as *const u8,
        flat_ids.len() * std::mem::size_of::<u32>(),
        &[(batch * max_len) as i32],
        MlxDtype::Uint32,
    );

    let mut hidden = embed_tokens_batched(
        &ids_flat,
        &weights.token_embedding,
        cfg.hidden_size,
        batch,
        max_len,
    );
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    for (li, layer_w) in weights.layers.iter().enumerate() {
        hidden = layer_forward_dense_embed(cfg, layer_w, &hidden, li);
    }
    // Extract per-sequence positions before the final norm when the caller only
    // needs one token per sequence (Last/Cls pooling). This avoids norming the
    // full padded [B, max_seq, H] tensor.
    let to_norm = match target_positions {
        Some(positions) => {
            let batch_size = batch as i32;
            let hidden_size = hidden.shape()[2] as usize;
            let pos_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
            let idx_b11 = MlxArray::from_raw_data(
                pos_u32.as_ptr() as *const u8,
                pos_u32.len() * std::mem::size_of::<u32>(),
                &[batch_size, 1_i32, 1_i32],
                MlxDtype::Uint32,
            );
            let idx_broadcast =
                broadcast_to(&idx_b11, &[batch_size, 1_i32, hidden_size as i32], None);
            let gathered = take_along_axis(&hidden, &idx_broadcast, 1, None); // [B, 1, H]
            reshape(&gathered, &[batch_size, hidden_size as i32], None) // [B, H]
        }
        None => hidden,
    };
    let out = rms_norm(&to_norm, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    (out, actual_lens)
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
    forward_lazy_single_with_turboquant_context(cfg, weights, token_arr, cache, token_offset, None)
}

pub fn forward_lazy_single_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_arr: &MlxArray, // scalar or [1] u32; may be unevaluated (lazy)
    cache: &mut MlxKVCache,
    token_offset: usize,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    // Normalise to [1] for embedding take (reshape is a no-op if already [1]).
    let tok_1d = reshape(token_arr, &[1_i32], None);

    let mut hidden = embed_tokens_arr(&tok_1d, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden(&hidden, scale);
    }
    let masks = build_layer_masks(cfg, weights.layers.len(), 1, token_offset + 1);
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &tok_1d, &hidden);
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward_with_turboquant_context(
            cfg,
            layer_w,
            &hidden,
            cache,
            li,
            token_offset,
            pli,
            Some(&masks[li]),
            turboquant_context,
        );
    }
    // Single token: hidden shape is [1, 1, hidden_size] — no sequence slice needed.
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
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
    let proj_out = rms_norm(&proj_out, Some(proj_norm_w), cfg.rms_norm_eps, None);

    // 3. Combine: (proj + embed) * 2^(-0.5)
    let combined_scale = 2.0_f32.powf(-0.5);
    let combined = add(&proj_out, &embed_out, None);
    let combined = scale_hidden(&combined, combined_scale);
    // combined shape: [1, seq, num_layers, per_layer_dim]

    // 4. Split per layer using direct slice on axis 2.
    // combined: [1, seq, num_layers, per_layer_dim] — contiguous after add + scale.
    // slice(li..li+1) creates a strided view ([1, seq, 1, per_layer_dim]) with no GPU gather
    // dispatch, matching Python's arr[:, :, i, :] behaviour. The 35 gather+upload overhead
    // from take(MlxArray::from_raw_data(li)) adds ~0.5 ms/step on fast small models (E2B).
    let per_layer = (0..num_layers)
        .map(|li| {
            let s = slice(
                &combined,
                &[0, 0, li as i32, 0],
                &[1, seq, li as i32 + 1, per_layer_dim as i32],
                &[1, 1, 1, 1],
                None,
            );
            reshape(&s, &[1, seq, per_layer_dim as i32], None)
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
    eps: f32,
) -> MlxArray {
    let Some(n) = norm else { return x };
    if use_flat_qk_norm_path() {
        let batch = x.shape()[0] as usize;
        let flat = reshape(&x, &[(batch * n_heads * seq) as i32, head_dim as i32], None);
        let normed = rms_norm(&flat, Some(n), eps, None);
        return reshape(
            &normed,
            &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
    }
    rms_norm(&x, Some(n), eps, None)
}

/// Apply no-scale per-head RMS norm in BSHD [batch, seq, n_heads, head_dim] space.
fn rms_norm_no_scale_bshd(
    x: MlxArray,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
) -> MlxArray {
    if use_flat_qk_norm_path() {
        let batch = x.shape()[0] as usize;
        let flat = reshape(&x, &[(batch * n_heads * seq) as i32, head_dim as i32], None);
        let normed = rms_norm(&flat, None, eps, None);
        return reshape(
            &normed,
            &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
    }
    rms_norm(&x, None, eps, None)
}

fn use_flat_qk_norm_path() -> bool {
    static USE_FLAT: OnceLock<bool> = OnceLock::new();
    *USE_FLAT.get_or_init(|| std::env::var("AX_MLX_QK_NORM_FLAT").as_deref() == Ok("1"))
}

/// Apply optional V RMSNorm in BSHD, then convert to BHSD for attention/KV cache.
fn prepare_value_bhsd(
    v: MlxArray,
    v_norm_no_scale: bool,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
) -> MlxArray {
    let v = if v_norm_no_scale {
        rms_norm_no_scale_bshd(v, n_heads, head_dim, seq, eps)
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

fn attention_mask_key_len(seq_len: usize, key_len: usize, sliding_window: Option<usize>) -> usize {
    if seq_len == 1
        && let Some(window) = sliding_window.filter(|window| *window > 0)
    {
        return key_len.min(window);
    }
    key_len
}

fn full_precision_attention(
    q_rope: &MlxArray,
    cached_k: &MlxArray,
    cached_v: &MlxArray,
    query_scale: f32,
    seq: usize,
    mask_opt: &Option<MlxArray>,
) -> MlxArray {
    let mask = match mask_opt.as_ref() {
        Some(mask) => ScaledDotProductAttentionMask::Array(mask),
        None if seq > 1 => ScaledDotProductAttentionMask::Causal,
        None => ScaledDotProductAttentionMask::None,
    };
    scaled_dot_product_attention_with_mask(q_rope, cached_k, cached_v, query_scale, mask, None)
}

struct TurboQuantExperimentalDecodeOutput {
    attention: MlxArray,
    outcome: MlxKvCompressionDecodeOutcome,
}

#[allow(clippy::too_many_arguments)]
fn turboquant_decode_attention_experimental(
    cache: &MlxKVCache,
    layer_idx: usize,
    q_rope: &MlxArray,
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    query_scale: f32,
    hot_window_tokens: usize,
) -> Option<TurboQuantExperimentalDecodeOutput> {
    if seq != 1 {
        return None;
    }
    let expected_scale = (head_dim as f32).sqrt().recip();
    if !query_scale.is_finite() || query_scale <= 0.0 {
        return None;
    }

    let q_f32 = astype(q_rope, MlxDtype::Float32, None);
    eval(&[&q_f32]);
    let q_data = q_f32.data_f32();
    if q_data.len() != n_heads.saturating_mul(head_dim) {
        return None;
    }
    let mut queries = q_data
        .chunks_exact(head_dim)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    if queries.len() != n_heads {
        return None;
    }
    let query_multiplier = query_scale / expected_scale;
    if (query_multiplier - 1.0).abs() > 1.0e-6 {
        for query in &mut queries {
            for value in query {
                *value *= query_multiplier;
            }
        }
    }

    let total_tokens = cache.seq_len.saturating_add(seq);
    if let Ok(outputs) = cache
        .debug_turboquant_shadow_decode_attention_metal_for_layer_with_total_tokens(
            layer_idx,
            &queries,
            total_tokens,
        )
    {
        return turboquant_attention_output_array(outputs, n_heads, head_dim, q_rope.dtype()).map(
            |attention| TurboQuantExperimentalDecodeOutput {
                attention,
                outcome: MlxKvCompressionDecodeOutcome::Metal,
            },
        );
    }

    let outputs = cache
        .debug_turboquant_shadow_decode_attention_for_layer_with_total_tokens(
            layer_idx,
            &queries,
            hot_window_tokens,
            total_tokens,
        )
        .ok()?;
    turboquant_attention_output_array(outputs, n_heads, head_dim, q_rope.dtype()).map(|attention| {
        TurboQuantExperimentalDecodeOutput {
            attention,
            outcome: MlxKvCompressionDecodeOutcome::CpuOracle,
        }
    })
}

fn turboquant_attention_output_array(
    outputs: Vec<Vec<f32>>,
    n_heads: usize,
    head_dim: usize,
    dtype: MlxDtype,
) -> Option<MlxArray> {
    if outputs.len() != n_heads || outputs.iter().any(|head| head.len() != head_dim) {
        return None;
    }

    let flat = outputs.into_iter().flatten().collect::<Vec<_>>();
    let out = MlxArray::from_raw_data(
        flat.as_ptr().cast(),
        flat.len() * std::mem::size_of::<f32>(),
        &[1, n_heads as i32, 1, head_dim as i32],
        MlxDtype::Float32,
    );
    Some(astype(&out, dtype, None))
}

/// Pre-compute one SDPA mask per unique sliding-window size before the layer
/// loop.  Mirrors Python mlx_lm's `_make_masks` and Swift's `maskByType`:
/// all layers of the same attention type share one mask object, avoiding
/// N redundant `create_causal_mask` calls (= N × ~7 MLX graph nodes) per
/// forward pass.
fn build_layer_masks(
    cfg: &ModelConfig,
    n_layers: usize,
    seq: usize,
    key_len: usize,
) -> Vec<Option<MlxArray>> {
    if cfg.layer_configs.is_empty() {
        let m = attention_mask_array(seq, key_len, None);
        (0..n_layers).map(|_| m.clone()).collect()
    } else {
        let mut memo: std::collections::HashMap<(Option<usize>, usize), Option<MlxArray>> =
            std::collections::HashMap::new();
        cfg.layer_configs
            .iter()
            .map(|lc| {
                let mask_key_len = attention_mask_key_len(seq, key_len, lc.sliding_window);
                // For decode (seq==1) with sliding window, key_len is already truncated
                // to ≤ window by attention_mask_key_len. The single query can attend to
                // all retained keys, so no mask is needed. This matches mlx_lm's behavior
                // where N==1 → None mask for all layers (base.py create_attention_mask).
                if seq == 1 && lc.sliding_window.is_some() {
                    return None;
                }
                memo.entry((lc.sliding_window, mask_key_len))
                    .or_insert_with(|| attention_mask_array(seq, mask_key_len, lc.sliding_window))
                    .clone()
            })
            .collect()
    }
}

/// Apply optional RMS norm; pass `x` through if `norm` is None.
fn rms_norm_opt(x: &MlxArray, norm: Option<&MlxArray>, eps: f32) -> MlxArray {
    if let Some(n) = norm {
        rms_norm(x, Some(n), eps, None)
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
            // q_proj output is exactly [B, L, n_heads * head_dim] — no slice needed.
            (q_full, None)
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

struct GlmMlaProjectedInputs {
    /// `[1, n_heads, seq, qk_nope_head_dim]`.
    q_nope: MlxArray,
    /// `[1, n_heads, seq, qk_rope_head_dim]`, with RoPE applied.
    q_pe: MlxArray,
    /// `[1, 1, seq, kv_lora_rank]`.
    kv_latent: MlxArray,
    /// `[1, 1, seq, qk_rope_head_dim]`, with RoPE applied.
    k_pe: MlxArray,
}

struct GlmMlaCachedInputs {
    /// `[1, n_heads, new_seq, qk_nope_head_dim]`.
    q_nope: MlxArray,
    /// `[1, n_heads, new_seq, qk_rope_head_dim]`, with RoPE applied.
    q_pe: MlxArray,
    /// `[1, 1, total_seq, kv_lora_rank]`.
    kv_latent: MlxArray,
    /// `[1, 1, total_seq, qk_rope_head_dim]`, with RoPE applied.
    k_pe: MlxArray,
}

/// GLM4MoELite MLA input projection up to the latent-cache boundary.
fn glm_mla_project_inputs(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    token_offset: usize,
) -> GlmMlaProjectedInputs {
    let mla_cfg = cfg
        .glm_mla_attention
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
fn glm_mla_project_and_cache_inputs(
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

/// GLM4MoELite RoPE score bias: `(q_pe * scale) @ k_pe.T`.
#[allow(dead_code)] // Reference for the split positional-score path replaced by packed SDPA.
fn glm_mla_positional_scores(cfg: &ModelConfig, q_pe: &MlxArray, k_pe: &MlxArray) -> MlxArray {
    let mla_cfg = cfg
        .glm_mla_attention
        .as_ref()
        .expect("GLM MLA attention config");
    let q_pe = scale_hidden(q_pe, mla_cfg.query_scale);
    matmul(&q_pe, &transpose(k_pe, &[0, 1, 3, 2], None), None)
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
fn glm_mla_attention_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
) -> MlxArray {
    let mla_cfg = cfg
        .glm_mla_attention
        .as_ref()
        .expect("GLM MLA attention config");
    let seq = x.shape()[1] as usize;
    let cached = glm_mla_project_and_cache_inputs(cfg, w, x, cache, layer_idx, token_offset);

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

    // V stays as kv_latent [1, key_len, kv_lora_rank]; unembed_out maps output → value space.
    let out = scaled_dot_product_attention_with_mask(
        &q_packed,
        &k_packed,
        &cached.kv_latent,
        mla_cfg.query_scale,
        mask,
        None,
    );
    let attn_out = glm_mla_unembed_out(cfg, w, &out);

    let attn_out = transpose(&attn_out, &[0, 2, 1, 3], None);
    let attn_flat = reshape(
        &attn_out,
        &[1, seq as i32, (cfg.n_heads * mla_cfg.value_head_dim) as i32],
        None,
    );
    attention_output_projection(
        &attn_flat,
        None,
        w.o_proj.as_ref().expect("GLM MLA layer must have o_proj"),
    )
}

#[allow(dead_code)] // Reference for the split positional-score causal mask replaced by packed SDPA.
fn glm_mla_causal_positional_scores(
    seq: usize,
    key_len: usize,
    positional_scores: &MlxArray,
) -> MlxArray {
    if seq == 1 {
        return positional_scores.clone();
    }
    let offset = key_len
        .checked_sub(seq)
        .expect("GLM MLA key length must include current sequence");
    let mask = create_causal_mask(seq, offset, None);
    let mask = expand_dims_axes(&mask, &[0, 1], None);
    let masked_value = scalar_like(-1.0e30, positional_scores.dtype());
    where_cond(&mask, positional_scores, &masked_value, None)
}

/// Prefill path: `kv_latent [B,1,seq,kv_lora_rank] @ embed_q [n_heads,kv_lora_rank,qk_nope_head_dim]`
/// → `[B,n_heads,seq,qk_nope_head_dim]`.  Uses quantized_matmul (transpose=false) so the 4-bit
/// packed weight is never materialized as float — equivalent to mlx-lm QuantizedMultiLinear(transpose=False).
#[allow(dead_code)] // Reference for the old split prefill path; packed SDPA uses decode embedding.
fn glm_mla_embed_q_prefill(cfg: &ModelConfig, w: &LayerWeights, kv_latent: &MlxArray) -> MlxArray {
    let mla_cfg = cfg
        .glm_mla_attention
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
fn glm_mla_embed_q_decode(cfg: &ModelConfig, w: &LayerWeights, q_nope: &MlxArray) -> MlxArray {
    let mla_cfg = cfg
        .glm_mla_attention
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
fn glm_mla_unembed_out(cfg: &ModelConfig, w: &LayerWeights, latent: &MlxArray) -> MlxArray {
    let mla_cfg = cfg
        .glm_mla_attention
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
    let profile_enabled = linear_attention_profile_enabled();
    if profile_enabled {
        record_linear_attention_profile_layer(seq);
    }

    let profile_started = Instant::now();
    let (qkv, z, a, b) = linear_attention_inputs(linear_cfg, linear_w, x, seq);
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Projection,
        profile_started,
        &[&qkv, &z, &a, &b],
    );

    let (conv_state, recurrent_state) = cache.linear_state(layer_idx);
    let profile_started = Instant::now();
    let conv_weight = linear_conv_weight(&linear_w.conv1d);
    let (conv_out, new_conv_state) =
        linear_attention_conv1d(linear_cfg, &qkv, &conv_weight, conv_state, None);
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Conv,
        profile_started,
        &[&conv_out, &new_conv_state],
    );
    let qkv = split_linear_attention_qkv(linear_cfg, &conv_out);
    let profile_started = Instant::now();
    let (q, k) = normalize_linear_attention_qk(linear_cfg, &qkv.q, &qkv.k);
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::QkNorm,
        profile_started,
        &[&q, &k],
    );
    // Cast a_log and dt_bias to float32: mlx_lm preserves A_log as float32 and
    // computes g in float32 precision.  dt_bias may be BF16 in quantized models.
    let profile_started = Instant::now();
    let a_log_f32 = astype(&linear_w.a_log, MlxDtype::Float32, None);
    let dt_bias_f32 = astype(&linear_w.dt_bias, MlxDtype::Float32, None);
    // State is always float32: mlx_lm initialises state as mx.zeros(..., dtype=mx.float32).
    let state = recurrent_state.cloned().unwrap_or_else(|| {
        zeros(
            &[
                1,
                linear_cfg.num_value_heads as i32,
                linear_cfg.value_head_dim as i32,
                linear_cfg.key_head_dim as i32,
            ],
            MlxDtype::Float32,
            None,
        )
    });
    // g and beta are computed inside the Metal kernel (fused) instead of as separate
    // lazy MLX ops, eliminating ~8 kernel dispatches per layer.
    let (out, new_recurrent_state) =
        gated_delta_kernel(&q, &k, &qkv.v, &a_log_f32, &a, &dt_bias_f32, &b, &state);
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Recurrent,
        profile_started,
        &[&out, &new_recurrent_state],
    );
    cache.set_linear_state(layer_idx, new_conv_state, new_recurrent_state);

    let profile_started = Instant::now();
    let out = rms_norm_gated(&out, &z, &linear_w.norm);
    let flat = reshape(&out, &[1, seq, linear_cfg.value_dim() as i32], None);
    let out = qw(&flat, &linear_w.out_proj);
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Output,
        profile_started,
        &[&out],
    );
    out
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

    // Gemma4 uses GEGLU with fast-approx GELU gate (matches mlx_lm's `nn.gelu_approx`).
    // Qwen3 uses SwiGLU (SiLU gate).
    let gate_act = if cfg.uses_geglu {
        gelu_approx(&gate_out, None)
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
    if let Some(shared_expert_gate) = &w.shared_expert_gate {
        let shared_gate = qw(x, shared_expert_gate);
        multiply(&mlx_sys::ops::sigmoid(&shared_gate, None), &shared, None)
    } else {
        shared
    }
}

fn mlx_slice_last_dim(x: &MlxArray, start: i32, end: i32) -> MlxArray {
    slice_last_dim(x, start, end, None)
}

fn scale_hidden(hidden: &MlxArray, scale: f32) -> MlxArray {
    let dtype = hidden.dtype();
    let s_arr = scalar_like(scale, dtype);
    multiply(hidden, &s_arr, None)
}

fn scalar_like(value: f32, dtype: MlxDtype) -> MlxArray {
    let scalar = MlxArray::from_raw_data(
        &value as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    astype(&scalar, dtype, None)
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
    let combined_scale = if let Some(precomputed) = &w.router_combined_scale {
        precomputed.clone()
    } else {
        let router_scale = w.router_scale.as_ref().unwrap();
        let root_factor = 1.0_f32 / (cfg.hidden_size as f32).sqrt();
        let scale_arr = MlxArray::from_raw_data(
            &root_factor as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1_i32],
            MlxDtype::Float32,
        );
        let scale_arr = astype(&scale_arr, MlxDtype::Bfloat16, None);
        multiply(router_scale, &scale_arr, None)
    };
    let normed = rms_norm(hidden, Some(&combined_scale), cfg.rms_norm_eps, None);

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

fn moe_router_glm(cfg: &ModelConfig, w: &LayerWeights, normed: &MlxArray) -> (MlxArray, MlxArray) {
    let logits = qw(
        normed,
        w.router_proj
            .as_ref()
            .expect("GLM MoE layer must have router projection"),
    );
    let correction_bias = w
        .router_correction_bias
        .as_ref()
        .expect("GLM MoE layer must have router correction bias");
    moe_router_glm_from_logits(cfg, &logits, correction_bias)
}

/// GLM4MoELite router: sigmoid logits + correction bias selects top-k;
/// gathered weights come from the original sigmoid scores.
pub(crate) fn moe_router_glm_from_logits(
    cfg: &ModelConfig,
    logits: &MlxArray,
    correction_bias: &MlxArray,
) -> (MlxArray, MlxArray) {
    let router = cfg.glm_router.as_ref().expect("GLM router config");
    let last_axis = logits.ndim() as i32 - 1;
    let scores = mlx_sys::ops::sigmoid(&astype(logits, MlxDtype::Float32, None), None);
    let selection_scores = add(&scores, correction_bias, None);
    let selection_scores = glm_router_apply_group_selection(cfg, router, &selection_scores);
    let (top_k_indices, _) = top_k_by_argpartition(
        &selection_scores,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        false,
    );
    let top_k_weights = take_along_axis(&scores, &top_k_indices, last_axis, None);
    let top_k_weights = if cfg.moe_experts_per_token > 1 && cfg.moe_norm_topk_prob {
        let denominator = sum_axis(&top_k_weights, last_axis, true, None);
        let epsilon = 1e-20_f32;
        let epsilon = MlxArray::from_raw_data(
            &epsilon as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1_i32],
            MlxDtype::Float32,
        );
        divide(&top_k_weights, &add(&denominator, &epsilon, None), None)
    } else {
        top_k_weights
    };
    (
        top_k_indices,
        scale_hidden(&top_k_weights, router.routed_scaling_factor),
    )
}

fn glm_router_apply_group_selection(
    cfg: &ModelConfig,
    router: &GlmRouterConfig,
    selection_scores: &MlxArray,
) -> MlxArray {
    if router.n_group <= 1 {
        return selection_scores.clone();
    }

    assert!(
        cfg.moe_expert_count.is_multiple_of(router.n_group),
        "GLM expert count must divide evenly across router groups"
    );
    assert!(
        router.topk_group <= router.n_group,
        "GLM topk_group must be <= n_group"
    );
    let zero_group_count = router.n_group - router.topk_group;
    if zero_group_count == 0 {
        return selection_scores.clone();
    }

    let shape = selection_scores.shape();
    assert_eq!(
        shape.len(),
        3,
        "GLM router scores must be [batch, seq, experts]"
    );
    let batch = shape[0];
    let seq = shape[1];
    let experts_per_group = cfg.moe_expert_count / router.n_group;
    assert!(
        experts_per_group >= 2,
        "GLM grouped router requires at least two experts per group"
    );

    let grouped = reshape(
        selection_scores,
        &[batch, seq, router.n_group as i32, experts_per_group as i32],
        None,
    );
    let (_, group_top2) = top_k_by_argpartition(&grouped, experts_per_group, 2, false);
    let group_scores = sum_axis(&group_top2, -1, true, None);
    let group_axis = group_scores.ndim() as i32 - 2;
    let group_idx = argpartition_axis(
        &group_scores,
        (zero_group_count as i32) - 1,
        group_axis,
        None,
    );
    let group_idx = slice(
        &group_idx,
        &[0, 0, 0, 0],
        &[batch, seq, zero_group_count as i32, 1],
        &[1, 1, 1, 1],
        None,
    );
    let group_idx = broadcast_to(
        &group_idx,
        &[
            batch,
            seq,
            zero_group_count as i32,
            experts_per_group as i32,
        ],
        None,
    );
    let zero = scalar_like(0.0, grouped.dtype());
    let masked = put_along_axis(&grouped, &group_idx, &zero, group_axis, None);
    reshape(&masked, &[batch, seq, cfg.moe_expert_count as i32], None)
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

    // Gemma4 experts use GEGLU with fast-approx GELU (matches mlx_lm's `nn.gelu_approx`).
    // Qwen3 uses SwiGLU (SiLU gate).
    let gate_act = if cfg.uses_geglu {
        gelu_approx(&gate_out, None)
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
    let out = sum_axis(&weighted, top_k_axis, false, None);
    // Cast back to the input dtype. GLM scores are f32 (sigmoid over astype→f32),
    // so without this the weighted sum is f32 and contaminates all downstream
    // residuals and projections. Python's MoE does `.astype(y.dtype)` here.
    astype(&out, x.dtype(), None)
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
    use crate::weights::{GlmMlaAttentionWeights, LinearAttentionWeights};
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
            rms_norm_eps: 1e-6,
        }
    }

    fn turboquant_decode_config() -> MlxKvCompressionConfig {
        MlxKvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..MlxKvCompressionConfig::turboquant_fused_experimental()
        }
    }

    fn turboquant_dense_cfg() -> ModelConfig {
        let mut cfg = cfg(false);
        cfg.hidden_size = 256;
        cfg.n_heads = 2;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 128;
        cfg.rope_dims = 128;
        cfg.query_scale = 1.0 / (128.0_f32).sqrt();
        cfg
    }

    fn turboquant_cache_with_runtime_storage() -> MlxKVCache {
        let mut cache = MlxKVCache::new(1);
        let elements = 2 * 6 * 128;
        let k = zeros(&[1, 2, 6, 128], MlxDtype::Float32, None);
        let v_data = (0..elements)
            .map(|idx| ((idx % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let v = MlxArray::from_raw_data(
            v_data.as_ptr().cast(),
            v_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 128],
            MlxDtype::Float32,
        );
        let compression = turboquant_decode_config();
        cache.append(0, k, v);
        cache.seq_len = 6;
        cache.sync_turboquant_shadow_storage(&[None], compression, Some(&[true]));
        cache
    }

    fn turboquant_cache_with_runtime_storage_and_current_decode_token() -> MlxKVCache {
        let mut cache = MlxKVCache::new(1);
        let initial_elements = 2 * 6 * 128;
        let initial_k = zeros(&[1, 2, 6, 128], MlxDtype::Float32, None);
        let initial_v_data = (0..initial_elements)
            .map(|idx| ((idx % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let initial_v = MlxArray::from_raw_data(
            initial_v_data.as_ptr().cast(),
            initial_v_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 128],
            MlxDtype::Float32,
        );
        cache.append(0, initial_k, initial_v);
        cache.seq_len = 6;
        cache.sync_turboquant_shadow_storage(&[None], turboquant_decode_config(), Some(&[true]));

        let current_elements = 2 * 128;
        let current_k = zeros(&[1, 2, 1, 128], MlxDtype::Float32, None);
        let current_v_data = (0..current_elements)
            .map(|idx| ((idx % 11) as f32 - 5.0) / 13.0)
            .collect::<Vec<_>>();
        let current_v = MlxArray::from_raw_data(
            current_v_data.as_ptr().cast(),
            current_v_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 1, 128],
            MlxDtype::Float32,
        );
        cache.append(0, current_k, current_v);
        cache
    }

    #[test]
    fn turboquant_model_decode_context_gates_candidate_layers() {
        let cfg = turboquant_dense_cfg();
        let compression = turboquant_decode_config();
        let context = TurboQuantModelDecodeContext {
            config: compression,
            layer_eligible: &[true],
        };
        let mut cache = MlxKVCache::new(1);
        cache.seq_len = 6;

        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 2, 128, 2, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::PrefillOnly
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 128, 2, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::MissingRuntimeStorage
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 256, 1, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::MissingRuntimeStorage
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 128, 2, Some(128), None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::SlidingWindowLayer
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 128, 2, None, Some(0), false)
                .status,
            TurboQuantModelDecodeCandidateStatus::KvSharedLayer
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 64, 2, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::UnsupportedHeadDim
        );
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 128, 3, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::GroupedQueryAttention
        );

        let context = TurboQuantModelDecodeContext {
            config: compression,
            layer_eligible: &[false],
        };
        assert_eq!(
            context
                .decode_candidate(&cfg, &cache, 0, 1, 128, 2, None, None, false)
                .status,
            TurboQuantModelDecodeCandidateStatus::IneligibleLayer
        );
    }

    #[test]
    fn turboquant_model_decode_context_marks_runtime_storage_ready() {
        let cfg = turboquant_dense_cfg();
        let cache = turboquant_cache_with_runtime_storage();
        let context = TurboQuantModelDecodeContext {
            config: turboquant_decode_config(),
            layer_eligible: &[true],
        };

        let candidate = context.decode_candidate(&cfg, &cache, 0, 1, 128, 2, None, None, false);

        assert_eq!(
            candidate.status,
            TurboQuantModelDecodeCandidateStatus::Ready
        );
        assert_eq!(candidate.cold_tokens, 4);
        assert_eq!(candidate.hot_tokens, 3);
    }

    #[test]
    fn turboquant_decode_attention_experimental_prefers_metal_runtime_storage() {
        let cache = turboquant_cache_with_runtime_storage_and_current_decode_token();
        let q_data = (0..(2 * 128))
            .map(|idx| ((idx % 19) as f32 - 9.0) / 31.0)
            .collect::<Vec<_>>();
        let q_rope = MlxArray::from_raw_data(
            q_data.as_ptr().cast(),
            q_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 1, 128],
            MlxDtype::Float32,
        );
        let expected_queries = q_data
            .chunks_exact(128)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let expected = cache
            .debug_turboquant_shadow_decode_attention_for_layer_with_total_tokens(
                0,
                &expected_queries,
                turboquant_decode_config().hot_window_tokens,
                7,
            )
            .expect("runtime storage should decode cold and hot partitions");

        let actual = turboquant_decode_attention_experimental(
            &cache,
            0,
            &q_rope,
            1,
            2,
            128,
            (128.0_f32).sqrt().recip(),
            turboquant_decode_config().hot_window_tokens,
        )
        .expect("ready TurboQuant decoder should decode from runtime storage");
        assert_eq!(actual.outcome, MlxKvCompressionDecodeOutcome::Metal);
        eval(&[&actual.attention]);

        assert_eq!(actual.attention.shape(), vec![1, 2, 1, 128]);
        let actual_data = actual.attention.data_f32();
        let expected_data = expected.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(actual_data.len(), expected_data.len());
        for (actual, expected) in actual_data.iter().zip(expected_data) {
            assert!((actual - expected).abs() <= 0.05);
        }
    }

    #[test]
    fn turboquant_decode_attention_experimental_applies_model_query_scale() {
        let cache = turboquant_cache_with_runtime_storage_and_current_decode_token();
        let q_data = (0..(2 * 128))
            .map(|idx| ((idx % 23) as f32 - 11.0) / 37.0)
            .collect::<Vec<_>>();
        let q_rope = MlxArray::from_raw_data(
            q_data.as_ptr().cast(),
            q_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 1, 128],
            MlxDtype::Float32,
        );
        let base_scale = (128.0_f32).sqrt().recip();
        let query_scale = base_scale * 2.0;
        let expected_queries = q_data
            .chunks_exact(128)
            .map(|chunk| chunk.iter().map(|value| value * 2.0).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let expected = cache
            .debug_turboquant_shadow_decode_attention_for_layer_with_total_tokens(
                0,
                &expected_queries,
                turboquant_decode_config().hot_window_tokens,
                7,
            )
            .expect("runtime storage should decode scaled queries");

        let actual = turboquant_decode_attention_experimental(
            &cache,
            0,
            &q_rope,
            1,
            2,
            128,
            query_scale,
            turboquant_decode_config().hot_window_tokens,
        )
        .expect("ready TurboQuant decoder should accept model-specific query scale");
        assert_eq!(actual.outcome, MlxKvCompressionDecodeOutcome::Metal);
        eval(&[&actual.attention]);

        let actual_data = actual.attention.data_f32();
        let expected_data = expected.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(actual_data.len(), expected_data.len());
        for (actual, expected) in actual_data.iter().zip(expected_data) {
            assert!((actual - expected).abs() <= 0.05);
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
                n_group: Some(1),
                topk_group: Some(1),
                has_shared_experts: true,
            },
            tensors: Vec::new(),
        }
    }

    fn dense_weight(shape: &[i32]) -> QuantizedWeight {
        QuantizedWeight::new(zeros(shape, MlxDtype::Float32, None), None, None)
    }

    fn quantized_zero_weight(packed_shape: &[i32], scale_shape: &[i32]) -> QuantizedWeight {
        QuantizedWeight {
            weight: zeros(packed_shape, MlxDtype::Uint32, None),
            scales: Some(zeros(scale_shape, MlxDtype::Float32, None)),
            biases: Some(zeros(scale_shape, MlxDtype::Float32, None)),
            group_size: 64,
            bits: 4,
        }
    }

    fn glm_mla_layer_weights(cfg: &ModelConfig) -> LayerWeights {
        let mla = cfg.glm_mla_attention.as_ref().expect("GLM MLA config");
        LayerWeights {
            attn_norm: zeros(&[cfg.hidden_size as i32], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: Some(dense_weight(&[
                cfg.hidden_size as i32,
                (cfg.n_heads * mla.value_head_dim) as i32,
            ])),
            linear_attn: None,
            glm_mla_attn: Some(GlmMlaAttentionWeights {
                qa_kva_fused: dense_weight(&[
                    (mla.q_lora_rank + mla.kv_lora_rank + mla.qk_rope_head_dim) as i32,
                    cfg.hidden_size as i32,
                ]),
                q_a_norm: zeros(&[mla.q_lora_rank as i32], MlxDtype::Float32, None),
                q_b_proj: dense_weight(&[
                    (cfg.n_heads * mla.q_head_dim) as i32,
                    mla.q_lora_rank as i32,
                ]),
                kv_a_norm: zeros(&[mla.kv_lora_rank as i32], MlxDtype::Float32, None),
                embed_q: dense_weight(&[
                    (cfg.n_heads * mla.kv_lora_rank) as i32,
                    mla.qk_nope_head_dim as i32,
                ]),
                unembed_out: dense_weight(&[
                    (cfg.n_heads * mla.value_head_dim) as i32,
                    mla.kv_lora_rank as i32,
                ]),
            }),
            ffn_norm: zeros(&[cfg.hidden_size as i32], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: None,
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
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

    fn glm_mla_quantized_multilinear_layer_weights(cfg: &ModelConfig) -> LayerWeights {
        let mla = cfg.glm_mla_attention.as_ref().expect("GLM MLA config");
        let mut weights = glm_mla_layer_weights(cfg);
        weights.glm_mla_attn = Some(GlmMlaAttentionWeights {
            qa_kva_fused: dense_weight(&[
                (mla.q_lora_rank + mla.kv_lora_rank + mla.qk_rope_head_dim) as i32,
                cfg.hidden_size as i32,
            ]),
            q_a_norm: zeros(&[mla.q_lora_rank as i32], MlxDtype::Float32, None),
            q_b_proj: dense_weight(&[
                (cfg.n_heads * mla.q_head_dim) as i32,
                mla.q_lora_rank as i32,
            ]),
            kv_a_norm: zeros(&[mla.kv_lora_rank as i32], MlxDtype::Float32, None),
            embed_q: quantized_zero_weight(
                &[cfg.n_heads as i32, mla.kv_lora_rank as i32, 8],
                &[cfg.n_heads as i32, mla.kv_lora_rank as i32, 1],
            ),
            unembed_out: quantized_zero_weight(
                &[cfg.n_heads as i32, mla.value_head_dim as i32, 8],
                &[cfg.n_heads as i32, mla.value_head_dim as i32, 1],
            ),
        });
        weights
    }

    fn attach_dense_ffn(weights: &mut LayerWeights, cfg: &ModelConfig) {
        weights.gate_proj = Some(dense_weight(&[
            cfg.intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.up_proj = Some(dense_weight(&[
            cfg.intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.down_proj = Some(dense_weight(&[
            cfg.hidden_size as i32,
            cfg.intermediate_size as i32,
        ]));
    }

    fn attach_glm_moe_ffn(weights: &mut LayerWeights, cfg: &ModelConfig) {
        weights.router_proj = Some(dense_weight(&[
            cfg.moe_expert_count as i32,
            cfg.hidden_size as i32,
        ]));
        weights.router_correction_bias = Some(zeros(
            &[cfg.moe_expert_count as i32],
            MlxDtype::Float32,
            None,
        ));
        weights.gate_exps = Some(dense_weight(&[
            cfg.moe_expert_count as i32,
            cfg.moe_expert_intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.up_exps = Some(dense_weight(&[
            cfg.moe_expert_count as i32,
            cfg.moe_expert_intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.down_exps = Some(dense_weight(&[
            cfg.moe_expert_count as i32,
            cfg.hidden_size as i32,
            cfg.moe_expert_intermediate_size as i32,
        ]));
        weights.shared_expert_gate = Some(dense_weight(&[1, cfg.hidden_size as i32]));
        weights.shared_gate_proj = Some(dense_weight(&[
            cfg.moe_expert_intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.shared_up_proj = Some(dense_weight(&[
            cfg.moe_expert_intermediate_size as i32,
            cfg.hidden_size as i32,
        ]));
        weights.shared_down_proj = Some(dense_weight(&[
            cfg.hidden_size as i32,
            cfg.moe_expert_intermediate_size as i32,
        ]));
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
            glm_mla_attn: None,
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
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
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
            glm_mla_attn: None,
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
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
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
        assert_eq!(mla.kv_lora_rank + mla.qk_rope_head_dim, 576);
        assert_eq!(mla.latent_kv_cache_width(), 512);
        assert_eq!(mla.rope_key_cache_width(), 64);
        assert!((mla.query_scale - (1.0 / 256_f32.sqrt())).abs() < f32::EPSILON);
        assert_ne!(mla.query_scale, 1.0 / 576_f32.sqrt());
        assert_eq!(cfg.query_scale, mla.query_scale);
    }

    #[test]
    fn glm_router_config_matches_reference_dense_moe_split() {
        let cfg = ModelConfig::from_manifest(&glm4_moe_lite_manifest());
        let router = cfg.glm_router.as_ref().expect("GLM router config");

        assert_eq!(router.first_dense_layer_count, 1);
        assert!((router.routed_scaling_factor - 1.8).abs() < f32::EPSILON);
        assert_eq!(router.n_group, 1);
        assert_eq!(router.topk_group, 1);
        assert!(router.has_shared_experts);
        assert!(!cfg.is_glm_moe_layer(0));
        assert!(cfg.is_glm_moe_layer(1));
        assert!(cfg.is_glm_moe_layer(2));
    }

    #[test]
    fn glm_router_uses_correction_bias_for_selection_and_sigmoid_for_weights() {
        let mut cfg = ModelConfig::from_manifest(&glm4_moe_lite_manifest());
        cfg.moe_expert_count = 4;
        cfg.moe_experts_per_token = 2;
        cfg.moe_norm_topk_prob = true;
        let logits_data = [0.0_f32, 0.0, 0.0, 0.0];
        let logits = MlxArray::from_raw_data(
            logits_data.as_ptr() as *const u8,
            std::mem::size_of_val(&logits_data),
            &[1, 1, 4],
            MlxDtype::Float32,
        );
        let bias_data = [0.0_f32, 10.0, 0.0, 5.0];
        let bias = MlxArray::from_raw_data(
            bias_data.as_ptr() as *const u8,
            std::mem::size_of_val(&bias_data),
            &[1, 1, 4],
            MlxDtype::Float32,
        );

        let (indices, weights) = moe_router_glm_from_logits(&cfg, &logits, &bias);
        eval(&[&indices, &weights]);

        let mut selected = indices.data_u32().to_vec();
        selected.sort_unstable();
        assert_eq!(selected, vec![1, 3]);
        assert_eq!(weights.shape(), vec![1, 1, 2]);
        for weight in weights.data_f32() {
            assert!((*weight - 0.9).abs() < 1e-5, "{weight}");
        }
    }

    #[test]
    fn glm_router_group_selection_masks_unselected_groups() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.moe.expert_count = Some(4);
        manifest.moe.experts_per_token = Some(2);
        manifest.glm_router.n_group = Some(2);
        manifest.glm_router.topk_group = Some(1);
        let cfg = ModelConfig::from_manifest(&manifest);
        let logits_data = [0.0_f32, 0.0, 0.0, 0.0];
        let logits = MlxArray::from_raw_data(
            logits_data.as_ptr() as *const u8,
            std::mem::size_of_val(&logits_data),
            &[1, 1, 4],
            MlxDtype::Float32,
        );
        let bias_data = [10.0_f32, 10.0, 0.0, 0.0];
        let bias = MlxArray::from_raw_data(
            bias_data.as_ptr() as *const u8,
            std::mem::size_of_val(&bias_data),
            &[1, 1, 4],
            MlxDtype::Float32,
        );

        let (indices, weights) = moe_router_glm_from_logits(&cfg, &logits, &bias);
        eval(&[&indices, &weights]);

        let mut selected = indices.data_u32().to_vec();
        selected.sort_unstable();
        assert_eq!(selected, vec![0, 1]);
        for weight in weights.data_f32() {
            assert!((*weight - 0.9).abs() < 1e-5, "{weight}");
        }
    }

    #[test]
    fn glm_mla_projection_matches_reference_cache_shapes() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_layer_weights(&cfg);
        let hidden = zeros(&[1, 3, cfg.hidden_size as i32], MlxDtype::Float32, None);

        let projected = glm_mla_project_inputs(&cfg, &weights, &hidden, 5);
        eval(&[
            &projected.q_nope,
            &projected.q_pe,
            &projected.kv_latent,
            &projected.k_pe,
        ]);

        assert_eq!(projected.q_nope.shape(), vec![1, 2, 3, 2]);
        assert_eq!(projected.q_pe.shape(), vec![1, 2, 3, 2]);
        assert_eq!(projected.kv_latent.shape(), vec![1, 1, 3, 4]);
        assert_eq!(projected.k_pe.shape(), vec![1, 1, 3, 2]);
    }

    #[test]
    fn glm_mla_projection_updates_latent_cache_and_rope_keys() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_layer_weights(&cfg);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let cached = glm_mla_project_and_cache_inputs(&cfg, &weights, &hidden, &mut cache, 0, 0);
        eval(&[
            &cached.q_nope,
            &cached.q_pe,
            &cached.kv_latent,
            &cached.k_pe,
        ]);
        assert_eq!(cached.q_nope.shape(), vec![1, 2, 2, 2]);
        assert_eq!(cached.kv_latent.shape(), vec![1, 1, 2, 4]);
        assert_eq!(cached.k_pe.shape(), vec![1, 1, 2, 2]);

        cache.seq_len = 2;
        let hidden = zeros(&[1, 1, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let cached = glm_mla_project_and_cache_inputs(&cfg, &weights, &hidden, &mut cache, 0, 2);
        eval(&[
            &cached.q_nope,
            &cached.q_pe,
            &cached.kv_latent,
            &cached.k_pe,
        ]);

        assert_eq!(cached.q_nope.shape(), vec![1, 2, 1, 2]);
        assert_eq!(cached.kv_latent.shape(), vec![1, 1, 3, 4]);
        assert_eq!(cached.k_pe.shape(), vec![1, 1, 3, 2]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn glm_mla_multilinear_matches_prefill_and_decode_shape_contracts() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_layer_weights(&cfg);

        let kv_latent = zeros(&[1, 1, 3, 4], MlxDtype::Float32, None);
        let prefill_k = glm_mla_embed_q_prefill(&cfg, &weights, &kv_latent);
        let prefill_v = glm_mla_unembed_out(&cfg, &weights, &kv_latent);
        eval(&[&prefill_k, &prefill_v]);
        assert_eq!(prefill_k.shape(), vec![1, 2, 3, 2]);
        assert_eq!(prefill_v.shape(), vec![1, 2, 3, 3]);

        let q_nope = zeros(&[1, 2, 1, 2], MlxDtype::Float32, None);
        let decode_q = glm_mla_embed_q_decode(&cfg, &weights, &q_nope);
        let decode_out = glm_mla_unembed_out(&cfg, &weights, &decode_q);
        eval(&[&decode_q, &decode_out]);
        assert_eq!(decode_q.shape(), vec![1, 2, 1, 4]);
        assert_eq!(decode_out.shape(), vec![1, 2, 1, 3]);
    }

    #[test]
    fn glm_mla_quantized_multilinear_dequantizes_to_prefill_and_decode_contracts() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 66;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(64);
        manifest.mla_attention.qk_nope_head_dim = Some(64);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(64);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_quantized_multilinear_layer_weights(&cfg);

        let kv_latent = zeros(&[1, 1, 3, 64], MlxDtype::Float32, None);
        let prefill_k = glm_mla_embed_q_prefill(&cfg, &weights, &kv_latent);
        let prefill_v = glm_mla_unembed_out(&cfg, &weights, &kv_latent);
        eval(&[&prefill_k, &prefill_v]);
        assert_eq!(prefill_k.shape(), vec![1, 2, 3, 64]);
        assert_eq!(prefill_v.shape(), vec![1, 2, 3, 64]);

        let q_nope = zeros(&[1, 2, 1, 64], MlxDtype::Float32, None);
        let decode_q = glm_mla_embed_q_decode(&cfg, &weights, &q_nope);
        let decode_out = glm_mla_unembed_out(&cfg, &weights, &decode_q);
        eval(&[&decode_q, &decode_out]);
        assert_eq!(decode_q.shape(), vec![1, 2, 1, 64]);
        assert_eq!(decode_out.shape(), vec![1, 2, 1, 64]);
    }

    #[test]
    fn glm_mla_attention_forward_returns_hidden_shape_and_updates_cache() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_layer_weights(&cfg);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let out = glm_mla_attention_forward(&cfg, &weights, &hidden, &mut cache, 0, 0);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);

        cache.seq_len = 2;
        let hidden = zeros(&[1, 1, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let out = glm_mla_attention_forward(&cfg, &weights, &hidden, &mut cache, 0, 2);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 1, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn glm_mla_attention_forward_accepts_quantized_multilinear_weights() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 66;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(64);
        manifest.mla_attention.qk_nope_head_dim = Some(64);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(64);
        let cfg = ModelConfig::from_manifest(&manifest);
        let weights = glm_mla_quantized_multilinear_layer_weights(&cfg);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let out = glm_mla_attention_forward(&cfg, &weights, &hidden, &mut cache, 0, 0);
        eval(&[&out]);

        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn layer_forward_routes_glm_mla_without_standard_qkv_weights() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.intermediate_size = 6;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        let cfg = ModelConfig::from_manifest(&manifest);
        let mut weights = glm_mla_layer_weights(&cfg);
        weights.attn_post_norm = Some(zeros(&[cfg.hidden_size as i32], MlxDtype::Float32, None));
        attach_dense_ffn(&mut weights, &cfg);
        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let out = layer_forward(&cfg, &weights, &hidden, &mut cache, 0, 0, None, None);
        eval(&[&out]);

        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn layer_forward_routes_glm_moe_with_correction_bias_and_shared_expert() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.intermediate_size = 6;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        manifest.moe.expert_count = Some(4);
        manifest.moe.experts_per_token = Some(2);
        manifest.moe.expert_intermediate_size = Some(3);
        manifest.glm_router.first_dense_layer_count = Some(0);
        let cfg = ModelConfig::from_manifest(&manifest);
        let mut weights = glm_mla_layer_weights(&cfg);
        attach_glm_moe_ffn(&mut weights, &cfg);
        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let out = layer_forward(&cfg, &weights, &hidden, &mut cache, 0, 0, None, None);
        eval(&[&out]);

        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn layer_forward_routes_glm_moe_with_ungated_shared_expert() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.intermediate_size = 6;
        manifest.layer_count = 1;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        manifest.moe.expert_count = Some(4);
        manifest.moe.experts_per_token = Some(2);
        manifest.moe.expert_intermediate_size = Some(3);
        manifest.glm_router.first_dense_layer_count = Some(0);
        let cfg = ModelConfig::from_manifest(&manifest);
        let mut weights = glm_mla_layer_weights(&cfg);
        attach_glm_moe_ffn(&mut weights, &cfg);
        weights.shared_expert_gate = None;
        let hidden = zeros(&[1, 2, cfg.hidden_size as i32], MlxDtype::Float32, None);
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let out = layer_forward(&cfg, &weights, &hidden, &mut cache, 0, 0, None, None);
        eval(&[&out]);

        assert_eq!(out.shape(), vec![1, 2, cfg.hidden_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn glm_full_forward_spans_dense_and_moe_layers() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.hidden_size = 8;
        manifest.intermediate_size = 6;
        manifest.layer_count = 2;
        manifest.vocab_size = 16;
        manifest.attention_head_count = 2;
        manifest.kv_head_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention.q_lora_rank = Some(4);
        manifest.mla_attention.kv_lora_rank = Some(4);
        manifest.mla_attention.qk_nope_head_dim = Some(2);
        manifest.mla_attention.qk_rope_head_dim = Some(2);
        manifest.mla_attention.value_head_dim = Some(3);
        manifest.moe.expert_count = Some(4);
        manifest.moe.experts_per_token = Some(2);
        manifest.moe.expert_intermediate_size = Some(3);
        manifest.glm_router.first_dense_layer_count = Some(1);
        let cfg = ModelConfig::from_manifest(&manifest);

        let mut dense_layer = glm_mla_layer_weights(&cfg);
        attach_dense_ffn(&mut dense_layer, &cfg);
        let mut moe_layer = glm_mla_layer_weights(&cfg);
        attach_glm_moe_ffn(&mut moe_layer, &cfg);
        let weights = ModelWeights {
            token_embedding: dense_weight(&[cfg.vocab_size as i32, cfg.hidden_size as i32]),
            final_norm: zeros(&[cfg.hidden_size as i32], MlxDtype::Float32, None),
            lm_head: dense_weight(&[cfg.vocab_size as i32, cfg.hidden_size as i32]),
            layers: vec![dense_layer, moe_layer],
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
        };
        let mut cache = MlxKVCache::new(cfg.layer_count);

        let logits = forward_all_positions(&cfg, &weights, &[1, 2], &mut cache, 0);
        eval(&[&logits]);

        assert_eq!(logits.shape(), vec![2, cfg.vocab_size as i32]);
        assert_eq!(cache.collect_eval_refs().len(), 4);
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
            glm_mla_attn: None,
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
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
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
            glm_mla_attn: None,
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
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
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
            glm_mla_attn: None,
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
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
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
            glm_mla_attn: None,
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
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
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
            glm_mla_attn: None,
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
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
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
            glm_mla_attn: None,
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
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
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
        let prepared = prepare_value_bhsd(v, true, 2, 4, 3, 1e-6);

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

    #[test]
    fn attention_mask_key_len_matches_decode_windowed_kv_views() {
        assert_eq!(attention_mask_key_len(1, 6, Some(4)), 4);
        assert_eq!(attention_mask_key_len(1, 6, None), 6);
        assert_eq!(attention_mask_key_len(2, 6, Some(4)), 6);
    }
}
