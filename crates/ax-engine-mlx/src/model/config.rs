use ax_engine_core::NativeModelManifest;
use mlx_sys::MlxArray;

/// Per-layer hyperparameters for interleaved-SWA models (Gemma4).
#[derive(Clone, Debug)]
pub struct LayerConfig {
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rope_dims: usize,
    /// Optional per-layer RoPE frequency denominators passed to `mlx.fast.rope`.
    pub rope_freqs: Option<MlxArray>,
    /// None = global causal attention; Some(n) = sliding-window attention.
    pub sliding_window: Option<usize>,
    /// None = compute own K/V; Some(src) = reuse K/V from layer `src`.
    pub kv_source_layer: Option<usize>,
    /// Apply no-scale RMSNorm to V before caching (Gemma4 non-KV-shared layers).
    pub v_norm_no_scale: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4AssistantSharedKvLayers {
    pub full_attention_layer: Option<usize>,
    pub sliding_attention_layer: Option<usize>,
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
    pub(crate) fn from_manifest(m: &NativeModelManifest) -> Option<Self> {
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

/// Sampling strategy for DiffusionGemma denoising steps.
///
/// The choice of sampler dominates denoise throughput: confidence-threshold
/// avoids argsort/cumsum/inverse-sort and is 4–5× faster than entropy-bound
/// with equivalent output quality (per mlx-optiq benchmarks on Apple Silicon).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiffusionSampler {
    /// Entropy-bound: sort by entropy ascending, accept greedily within budget.
    EntropyBound,
    /// Confidence-threshold: accept when peak softmax prob >= threshold.
    ConfidenceThreshold,
}

/// Temperature schedule shape for DiffusionGemma denoising.
///
/// Controls how quickly the sampler cools from exploration (`temp_start`)
/// to exploitation (`temp_end`). Exponential decay drops temperature faster
/// in early steps, which can reduce denoise iterations by 1–3 steps on
/// in-distribution prompts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiffusionTemperatureSchedule {
    /// `temp = start + (end - start) * (step / max_steps)`
    Linear,
    /// `temp = start * (end / start) ^ (step / max_steps)`
    Exponential,
}

/// Diffusion decoding hyperparameters for DiffusionGemma.
#[derive(Clone, Debug)]
pub struct DiffusionConfig {
    /// Number of tokens generated per diffusion block (default 256).
    pub canvas_size: usize,
    /// Maximum denoising steps per block before forced convergence (default 48).
    pub max_denoise_steps: usize,
    /// Entropy bound for position acceptance during denoising (default 0.1).
    pub entropy_bound: f32,
    /// Mean entropy threshold for convergence detection (default 0.02).
    pub entropy_threshold: f32,
    /// Consecutive stable argmax steps required for convergence (default 2).
    pub convergence_steps: usize,
    /// Temperature schedule start (high, for exploration; default 0.8).
    pub temp_start: f32,
    /// Temperature schedule end (low, for locking final tokens; default 0.4).
    pub temp_end: f32,
    /// Enable self-conditioning feedback between denoising steps (default true).
    pub self_conditioning: bool,
    /// Steps between convergence checks (default 2). Non-check steps skip
    /// argmax stability and mean-entropy materialisation to reduce GPU→CPU syncs.
    pub convergence_check_interval: usize,
    /// Update-rate threshold for adaptive convergence (default 0.075 = 7.5%).
    /// `acceptance_rate` tracks positions kept from the current canvas, so
    /// convergence fires when fewer than this fraction still update.
    pub acceptance_rate_threshold: f32,
    /// Entropy plateau delta for convergence detection (default 0.005).
    /// When the absolute change in mean entropy between consecutive check
    /// steps falls below this value after step 8, plateau convergence fires.
    pub entropy_plateau_delta: f32,
    /// Sampling strategy for denoising acceptance (default: ConfidenceThreshold).
    pub sampler: DiffusionSampler,
    /// Confidence threshold for ConfidenceThreshold sampler (default 0.9).
    pub confidence_threshold: f32,
    /// Temperature schedule shape (default: Linear).
    pub temperature_schedule: DiffusionTemperatureSchedule,
    /// Acceptance rate above which self-conditioning matmul is skipped.
    /// When the canvas is mostly stable (>95% positions accepted), the
    /// self-conditioning signal barely changes and the expensive
    /// `prob × embed_table` matmul can be skipped to save ~5% per step.
    /// Default: 0.95.
    pub sc_skip_acceptance_rate: f32,
}

impl DiffusionConfig {
    pub(super) fn from_manifest(m: &NativeModelManifest) -> Option<Self> {
        let cfg = &m.diffusion;
        if !cfg.is_enabled() {
            return None;
        }
        // Reject canvas_size=0 — a malformed manifest must not reach MLX execution.
        // Treat explicit Some(0) as disabled diffusion rather than crashing later
        // when argmax is called on a zero-length canvas tensor.
        let canvas_size = cfg.canvas_size.unwrap_or(256) as usize;
        if canvas_size == 0 {
            return None;
        }
        // Reject convergence_check_interval=0 — used as divisor in
        // `step.is_multiple_of(convergence_check_interval)` which panics on 0.
        //
        // Default 1 (check every step): the per-step scalar eval is negligible
        // (A/B: intervals 4/8 are within noise of 2), but a coarser grid
        // *overshoots* the true convergence step to the next multiple, wasting a
        // full ~179 ms denoise pass. Checking every step stops exactly at
        // convergence — measured +5% (512-token) / +7% (2048-token) first-block
        // decode with byte-identical or 1-token output.
        let convergence_check_interval = cfg.convergence_check_interval.unwrap_or(1) as usize;
        if convergence_check_interval == 0 {
            return None;
        }
        // Reject max_denoise_steps=0 — a denoise loop with zero iterations
        // produces degenerate output (empty canvas committed as tokens).
        let max_denoise_steps = cfg.max_denoise_steps.unwrap_or(48) as usize;
        if max_denoise_steps == 0 {
            return None;
        }
        // Reject convergence_steps=0 — would cause instant convergence trigger
        // (stable_count >= 0 is trivially true), producing incorrect output.
        let convergence_steps = cfg.convergence_steps.unwrap_or(2) as usize;
        if convergence_steps == 0 {
            return None;
        }
        let sampler = match cfg.sampler.unwrap_or_default() {
            ax_engine_core::model::NativeDiffusionSampler::EntropyBound => {
                DiffusionSampler::EntropyBound
            }
            ax_engine_core::model::NativeDiffusionSampler::ConfidenceThreshold => {
                DiffusionSampler::ConfidenceThreshold
            }
        };
        let mut dc = Self {
            canvas_size,
            max_denoise_steps,
            entropy_bound: cfg.entropy_bound.unwrap_or(0.1),
            entropy_threshold: cfg.entropy_threshold.unwrap_or(0.02),
            convergence_steps,
            temp_start: cfg.temperature_start.unwrap_or(0.8),
            temp_end: cfg.temperature_end.unwrap_or(0.4),
            self_conditioning: cfg.self_conditioning.unwrap_or(true),
            convergence_check_interval,
            acceptance_rate_threshold: cfg.acceptance_rate_threshold.unwrap_or(0.075),
            entropy_plateau_delta: 0.005,
            sampler,
            confidence_threshold: cfg.confidence_threshold.unwrap_or(0.9),
            temperature_schedule: DiffusionTemperatureSchedule::Linear,
            sc_skip_acceptance_rate: 0.95,
        };
        // Apply env-var overrides for benchmark sweep campaigns.
        if let Some(v) = crate::fastpath::diffusion_entropy_threshold() {
            dc.entropy_threshold = v;
        }
        if let Some(v) = crate::fastpath::diffusion_acceptance_rate_threshold() {
            dc.acceptance_rate_threshold = v;
        }
        if let Some(v) = crate::fastpath::diffusion_entropy_plateau_delta() {
            dc.entropy_plateau_delta = v;
        }
        if let Some(v) = crate::fastpath::diffusion_max_steps() {
            dc.max_denoise_steps = v;
        }
        // Env-var sampler override: AX_DIFFUSION_SAMPLER=confidence_threshold
        if let Some(v) = crate::fastpath::diffusion_sampler() {
            dc.sampler = match v.as_str() {
                "confidence_threshold" | "confidence" => DiffusionSampler::ConfidenceThreshold,
                _ => DiffusionSampler::EntropyBound,
            };
        }
        if let Some(v) = crate::fastpath::diffusion_confidence_threshold() {
            dc.confidence_threshold = v;
        }
        if let Some(v) = crate::fastpath::diffusion_check_interval() {
            dc.convergence_check_interval = v;
        }
        // Env-var temperature schedule override.
        if let Some(v) = crate::fastpath::diffusion_temperature_schedule() {
            dc.temperature_schedule = match v.as_str() {
                "exponential" | "exp" => DiffusionTemperatureSchedule::Exponential,
                _ => DiffusionTemperatureSchedule::Linear,
            };
        }
        if let Some(v) = crate::fastpath::diffusion_sc_skip_acceptance_rate() {
            dc.sc_skip_acceptance_rate = v;
        }
        Some(dc)
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
    /// Token ID that opens a `<think>` block (Qwen3 family: 151668).
    /// When `Some`, n-gram and MTP n-gram stacking gate drafting to inside `<think>`.
    pub think_start_token_id: Option<u32>,
    /// Token ID that closes a `</think>` block (Qwen3 family: 151669).
    pub think_end_token_id: Option<u32>,
    /// Diffusion decoding config (DiffusionGemma). `None` = standard AR decoding.
    pub diffusion: Option<DiffusionConfig>,
    /// GPT-OSS uses MXFP4 quantized expert weights (dequantized to BF16 at load time).
    /// When true, the MoE forward path uses `mxfp4_gate_up_exps` / `mxfp4_down_exps` instead
    /// of the standard `gate_exps`/`up_exps`/`down_exps` QuantizedWeight path.
    pub gpt_oss_uses_mxfp4_experts: bool,
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
        let is_gemma4 = matches!(
            m.model_family.as_str(),
            "gemma4" | "gemma4_assistant" | "diffusion_gemma"
        );
        let uses_geglu = matches!(
            m.model_family.as_str(),
            "gemma4" | "gemma4_assistant" | "diffusion_gemma" | "gemma3" | "embeddinggemma"
        );
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

        let moe_norm_topk_prob = if m.model_family == "qwen3_5" && m.moe.is_enabled() {
            // mlx_lm.models.qwen3_5.TextModelArgs defaults norm_topk_prob to
            // true. Older AX manifests emitted false when config.json omitted
            // the field, which makes Qwen3.6 35B A3B route MoE experts with
            // the wrong weights. Keep the loader compatible with those cached
            // manifests while the converter now emits the correct default.
            true
        } else {
            m.moe_norm_topk_prob
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
            moe_norm_topk_prob,
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
            think_start_token_id: think_token_ids_from_manifest(m).0,
            think_end_token_id: think_token_ids_from_manifest(m).1,
            diffusion: DiffusionConfig::from_manifest(m),
            gpt_oss_uses_mxfp4_experts: m.model_family == "gpt_oss" && m.moe.is_enabled(),
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

    pub fn gemma4_assistant_shared_kv_layers(&self) -> Gemma4AssistantSharedKvLayers {
        let mut full_attention_layer = None;
        let mut sliding_attention_layer = None;
        for (idx, layer) in self.layer_configs.iter().enumerate() {
            let source = layer.kv_source_layer.unwrap_or(idx);
            if layer.sliding_window.is_some() {
                sliding_attention_layer = Some(source);
            } else {
                full_attention_layer = Some(source);
            }
        }
        Gemma4AssistantSharedKvLayers {
            full_attention_layer,
            sliding_attention_layer,
        }
    }
}

/// Return `(think_start_token_id, think_end_token_id)` for a model manifest.
///
/// Explicit manifest fields take precedence over family-derived defaults.
/// Returns `(None, None)` for families without think-block tokens.
fn think_token_ids_from_manifest(m: &NativeModelManifest) -> (Option<u32>, Option<u32>) {
    if m.think_start_token_id.is_some() || m.think_end_token_id.is_some() {
        return (m.think_start_token_id, m.think_end_token_id);
    }
    // Qwen ships two tokenizer generations with different <think> special
    // token ids: the original Qwen3 tokenizer (vocab ~151k) uses
    // 151668/151669, while the Qwen3.6 248k tokenizer moved them to
    // 248068/248069 (verified against the mlx-community Qwen3.6-27B and
    // 35B-A3B `tokenizer.json` added_tokens). Manifests converted before the
    // converter learned to record these ids carry `None`, so pick the
    // generation by vocab width. qwen3_next is reserved for future variants.
    // qwen3_5 linear-attention models also emit <think> when reasoning mode
    // is enabled.
    match m.model_family.as_str() {
        "qwen3" | "qwen3_5" | "qwen3_next" => {
            if m.vocab_size >= 200_000 {
                (Some(248_068), Some(248_069))
            } else {
                (Some(151_668), Some(151_669))
            }
        }
        _ => (None, None),
    }
}

fn default_rms_norm_eps(model_family: &str) -> f32 {
    if model_family.starts_with("qwen")
        || model_family.starts_with("gemma")
        || model_family == "diffusion_gemma"
    {
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
    // The Gemma4 assistant drafter reuses gemma4's RoPE geometry (proportional
    // full-attention RoPE + full-width sliding RoPE). It attends to the target's
    // cached K, so its Q rotation must match the target's exactly — gate the
    // gemma4-specific RoPE on the whole family, not just the dense target.
    let is_gemma4_family = matches!(
        m.model_family.as_str(),
        "gemma4" | "gemma4_assistant" | "diffusion_gemma"
    );
    let full_head_dim = m.global_head_dim.unwrap_or(m.attention_head_dim) as usize;
    let full_rope_dims = m
        .partial_rotary_factor
        .map(|f| ((full_head_dim as f32 * f) as usize).next_multiple_of(2))
        .unwrap_or(full_head_dim);
    let full_rope_freqs = if is_gemma4_family && full_rope_dims < full_head_dim {
        Some(build_gemma4_proportional_rope_freqs(
            full_head_dim,
            full_rope_dims,
            default_rope_theta,
            m.rope_scaling_factor.unwrap_or(1.0),
        ))
    } else {
        None
    };
    let sliding_rope_dims = if is_gemma4_family {
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
                    rope_dims: if full_rope_freqs.is_some() {
                        full_head_dim
                    } else {
                        full_rope_dims
                    },
                    rope_freqs: full_rope_freqs.clone(),
                    sliding_window: None,
                    kv_source_layer,
                    v_norm_no_scale,
                }
            } else {
                LayerConfig {
                    head_dim: default_head_dim,
                    rope_theta: swa_theta,
                    rope_dims: sliding_rope_dims,
                    rope_freqs: None,
                    sliding_window,
                    kv_source_layer,
                    v_norm_no_scale,
                }
            }
        })
        .collect()
}

/// Resolve per-layer params:
/// (head_dim, rope_theta, rope_dims, rope_freqs, sliding_window, kv_source, v_norm_no_scale).
pub(super) fn layer_params(
    cfg: &ModelConfig,
    layer_idx: usize,
) -> (
    usize,
    f32,
    usize,
    Option<&MlxArray>,
    Option<usize>,
    Option<usize>,
    bool,
) {
    if let Some(lc) = cfg.layer_configs.get(layer_idx) {
        (
            lc.head_dim,
            lc.rope_theta,
            lc.rope_dims,
            lc.rope_freqs.as_ref(),
            lc.sliding_window,
            lc.kv_source_layer,
            lc.v_norm_no_scale,
        )
    } else {
        (
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_dims,
            cfg.rope_freqs.as_ref(),
            cfg.global_sliding_window,
            None,
            false,
        )
    }
}

fn build_gemma4_proportional_rope_freqs(
    head_dim: usize,
    rotated_dims: usize,
    theta: f32,
    factor: f32,
) -> MlxArray {
    let rotated_pairs = rotated_dims / 2;
    let total_pairs = head_dim / 2;
    let freqs: Vec<f32> = (0..total_pairs)
        .map(|i| {
            if i < rotated_pairs {
                factor * theta.powf((2 * i) as f32 / head_dim as f32)
            } else {
                f32::INFINITY
            }
        })
        .collect();
    MlxArray::from_f32_slice(&freqs)
}
