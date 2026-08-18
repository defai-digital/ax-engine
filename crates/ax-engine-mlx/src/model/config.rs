use std::sync::atomic::{AtomicU64, Ordering};

use ax_engine_core::{GenerationKind, NativeModelManifest};
use mlx_sys::MlxArray;

static NEXT_COMPILE_CACHE_IDENTITY: AtomicU64 = AtomicU64::new(1);

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

/// YaRN rope_scaling params retained for the DeepSeek V4 **compress-layer**
/// freqs: compress layers rotate with `compress_rope_theta` as the YaRN base
/// plus these shared scaling params (llama.cpp deepseek4.cpp
/// `build_attention_impl`: `freq_base_l = dsv4_compress_rope_base` with
/// `freq_scale = 1/factor`, `beta_fast`/`beta_slow`, `n_ctx_orig`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepseekV4CompressRopeScaling {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_context_len: u32,
}

/// DeepSeek V4 (Flash) architecture parameters extracted from the manifest.
///
/// V4 drops the V3 MLA keys: a fused `wkv` projection feeds per-head K/V
/// (single KV head), the output projection is a grouped LoRA pair
/// (`wo_a`/`wo_b`), and every layer is MoE with hyper-connection tensors.
/// Routing is `scoring_func`-based (e.g. "sqrtsoftplus"), **not** the V3
/// sigmoid routing.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepseekV4Config {
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub index_topk: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub compress_rope_theta: f32,
    /// YaRN scaling for the compress-layer freqs; `None` = plain
    /// `compress_rope_theta` base (no rope_scaling in the manifest).
    pub compress_rope_scaling: Option<DeepseekV4CompressRopeScaling>,
    /// V4 attention layers carry a learned per-head attention sink.
    pub has_attn_sinks: bool,
    /// Per-layer compressor ratios (0 / 4 / 128; 0 = uncompressed).
    pub compress_ratios: Vec<u32>,
    /// Hyper-connection stream multiplier.
    pub hc_mult: usize,
    /// Sinkhorn iterations for the HC mixing coefficients.
    pub hc_sinkhorn_iters: usize,
    /// Epsilon for the HC Sinkhorn normalisation.
    pub hc_eps: f32,
    /// Leading MoE layers that route via the `tid2eid` hash table.
    pub num_hash_layers: usize,
    /// MTP (nextn) predictor layers stacked after the main layers.
    pub num_nextn_predict_layers: usize,
    /// Routing scoring function (e.g. "sqrtsoftplus").
    pub scoring_func: Option<String>,
    /// SwiGLU clamp limit applied in the expert/shared-expert FFNs.
    pub swiglu_limit: f32,
}

impl DeepseekV4Config {
    pub(crate) fn from_manifest(m: &NativeModelManifest) -> Option<Self> {
        let cfg = &m.deepseek_v4;
        if !cfg.is_enabled() {
            return None;
        }
        let attention = &cfg.attention;

        Some(Self {
            head_dim: attention
                .head_dim
                .expect("validated deepseek_v4.attention.head_dim") as usize,
            qk_rope_head_dim: attention
                .qk_rope_head_dim
                .expect("validated deepseek_v4.attention.qk_rope_head_dim")
                as usize,
            q_lora_rank: attention
                .q_lora_rank
                .expect("validated deepseek_v4.attention.q_lora_rank")
                as usize,
            o_lora_rank: attention
                .o_lora_rank
                .expect("validated deepseek_v4.attention.o_lora_rank")
                as usize,
            o_groups: attention
                .o_groups
                .expect("validated deepseek_v4.attention.o_groups") as usize,
            index_topk: attention
                .index_topk
                .expect("validated deepseek_v4.attention.index_topk")
                as usize,
            index_n_heads: attention
                .index_n_heads
                .expect("validated deepseek_v4.attention.index_n_heads")
                as usize,
            index_head_dim: attention
                .index_head_dim
                .expect("validated deepseek_v4.attention.index_head_dim")
                as usize,
            compress_rope_theta: attention
                .compress_rope_theta
                .expect("validated deepseek_v4.attention.compress_rope_theta")
                as f32,
            // Same yarn family gate as the standard freqs below; beta
            // fast/slow fall back to the mlx-lm YarnRoPE defaults (32/1)
            // when the manifest omits them.
            compress_rope_scaling: match m.rope_scaling_type.as_deref() {
                Some("yarn") | Some("deepseek_yarn") | Some("telechat3-yarn") => {
                    Some(DeepseekV4CompressRopeScaling {
                        factor: m.rope_scaling_factor.unwrap_or(1.0),
                        beta_fast: m.rope_beta_fast.unwrap_or(32.0),
                        beta_slow: m.rope_beta_slow.unwrap_or(1.0),
                        original_context_len: m.rope_original_context_len.unwrap_or(4096),
                    })
                }
                _ => None,
            },
            has_attn_sinks: attention.has_attn_sinks,
            compress_ratios: cfg.compress_ratios.clone(),
            hc_mult: cfg.hc_mult.expect("validated deepseek_v4.hc_mult") as usize,
            hc_sinkhorn_iters: cfg
                .hc_sinkhorn_iters
                .expect("validated deepseek_v4.hc_sinkhorn_iters")
                as usize,
            hc_eps: cfg.hc_eps.expect("validated deepseek_v4.hc_eps"),
            num_hash_layers: cfg
                .num_hash_layers
                .expect("validated deepseek_v4.num_hash_layers")
                as usize,
            num_nextn_predict_layers: cfg.num_nextn_predict_layers.unwrap_or(0) as usize,
            scoring_func: cfg.scoring_func.clone(),
            swiglu_limit: cfg
                .swiglu_limit
                .expect("validated deepseek_v4.swiglu_limit"),
        })
    }

    /// Compressor ratio for a layer (0 = uncompressed).
    pub fn compress_ratio(&self, layer_idx: usize) -> u32 {
        self.compress_ratios.get(layer_idx).copied().unwrap_or(0)
    }

    /// Whether a MoE layer routes via the `tid2eid` hash table.
    pub fn is_hash_routed_layer(&self, layer_idx: usize) -> bool {
        layer_idx < self.num_hash_layers
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

/// Per-layer KV-cache quantization parameters lifted from the manifest's
/// `kv_cache_quantization` table (Phase 3a: plumbed only; the runtime
/// quantization path lands in Phase 3b).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvQuantSpec {
    pub bits: u32,
    pub group_size: u32,
}

/// Hyperparameters extracted from the manifest.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// Unique for each model-config construction and retained by clones.
    /// Per-layer compiled closures include this identity in their cache key so
    /// a blocking-pool thread cannot reuse an old model's captured graph schema
    /// after hot-swap.
    pub compile_cache_identity: u64,
    /// Model family string from the manifest (e.g. "gemma4", "qwen3", "llama3").
    /// Used for named dispatch in `layer_forward`.
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
    /// Scalar multiplied into logits after lm_head, before the softcap
    /// (Muse Glimmer `output_multiplier`). `None` = no scaling.
    pub final_logits_scale: Option<f32>,
    /// RMSNorm eps for the post-attention / post-feedforward sandwich norms.
    /// Defaults to `rms_norm_eps` when the manifest carries no override
    /// (Muse Glimmer: 1e-8 vs 1e-6).
    pub post_norm_eps: f32,
    /// Apply a weightless RMSNorm to token embeddings before the first layer
    /// (Muse Glimmer `embed_norm`; eps = `rms_norm_eps`).
    pub embed_norm_no_weight: bool,
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
    /// Decode-only ring window that retains the complete prefill as a protected
    /// prefix. Unlimited-OCR uses this instead of ordinary uniform SWA.
    pub protected_prefix_sliding_window: Option<usize>,
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
    /// DeepSeek V4 (Flash) architecture config, when present.
    pub deepseek_v4: Option<DeepseekV4Config>,
    /// Epsilon for all RMSNorm operations (1e-6 for Qwen/Gemma, 1e-5 for GLM/LLaMA/Mistral).
    pub rms_norm_eps: f32,
    /// Precomputed LLaMA-3 / YaRN corrected RoPE frequencies `[dims/2]`.
    /// `None` means standard RoPE (compute freqs from `rope_theta` at runtime).
    /// `Some(freqs)` is passed directly to `mlx_sys::rope` as the `freqs` arg.
    pub rope_freqs: Option<MlxArray>,
    /// YaRN attention mscale applied to Q/K before RoPE (1.0 = no scale).
    /// Matches mlx-lm `YarnRoPE.mscale` for GPT-OSS and other yarn models.
    pub rope_mscale: f32,
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
    /// Generation paradigm derived from the manifest (ADR-038). Prefer this over
    /// family-string checks when gating diffusion / embed / AR behavior.
    pub generation_kind: GenerationKind,
    /// Per-layer KV-cache quantization from the manifest's
    /// `kv_cache_quantization` table. `None` per layer = full precision
    /// (manifest bits 16, or no table at all). Length == `layer_count`.
    pub kv_cache_quant: Vec<Option<KvQuantSpec>>,
}

impl ModelConfig {
    pub fn from_manifest(m: &NativeModelManifest) -> Self {
        let head_dim = m.attention_head_dim as usize;
        // DeepSeek V4 (MLA) rotates only the `qk_rope_head_dim` pe slice, not
        // the full head dim — building YaRN freqs at head_dim would give
        // freqs of len head_dim/2 and mlx_fast_rope would reject them.
        let rope_dims = if m.deepseek_v4.is_enabled() {
            m.deepseek_v4
                .attention
                .qk_rope_head_dim
                .expect("validated deepseek_v4.attention.qk_rope_head_dim") as usize
        } else {
            m.partial_rotary_factor
                .map(|f| ((head_dim as f32 * f) as usize).next_multiple_of(2))
                .unwrap_or(head_dim)
        };
        let intermediate_size = if m.intermediate_size > 0 {
            m.intermediate_size as usize
        } else {
            (m.hidden_size as usize * 8 / 3).next_multiple_of(256)
        };
        let rope_theta = m.rope_theta.map(|t| t as f32).unwrap_or(10000.0);
        let layer_configs = build_layer_configs(m, head_dim, rope_theta, rope_dims);
        // gemma4_vl / gemma4_unified are separate family labels for vision
        // capability gating; the language tower is still standard Gemma 4
        // (GeGLU, query_scale=1.0). DI-W1-001: include gemma4_unified so
        // convert-emitted unified packages do not fall through to generic
        // SwiGLU / non-Gemma RoPE geometry.
        let is_gemma4 = matches!(
            m.model_family.as_str(),
            "gemma4" | "gemma4_vl" | "gemma4_unified" | "gemma4_assistant" | "diffusion_gemma"
        );
        let uses_geglu = matches!(
            m.model_family.as_str(),
            "gemma4"
                | "gemma4_vl"
                | "gemma4_unified"
                | "gemma4_assistant"
                | "diffusion_gemma"
                | "gemma3"
                | "embeddinggemma"
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
        let protected_prefix_sliding_window = if m.model_family == "unlimited_ocr" {
            m.sliding_window_size.map(|w| w as usize)
        } else {
            None
        };
        let global_sliding_window =
            if layer_configs.is_empty() && protected_prefix_sliding_window.is_none() {
                m.sliding_window_size.map(|w| w as usize)
            } else {
                None
            };

        // Scaled RoPE frequencies, precomputed once at model load.
        // llama3: LLaMA-3 smooth wavelength correction.
        // yarn / deepseek_yarn: YaRN (GPT-OSS); also yields attention mscale.
        let (rope_freqs, rope_mscale) = match m.rope_scaling_type.as_deref() {
            Some("llama3") => {
                let factor = m.rope_scaling_factor.unwrap_or(8.0);
                let low_ff = m.rope_low_freq_factor.unwrap_or(1.0);
                let high_ff = m.rope_high_freq_factor.unwrap_or(4.0);
                let orig_ctx = m.rope_original_context_len.unwrap_or(8192);
                (
                    Some(super::shared::build_llama3_rope_freqs(
                        rope_dims, rope_theta, factor, low_ff, high_ff, orig_ctx,
                    )),
                    1.0,
                )
            }
            Some("yarn") | Some("deepseek_yarn") | Some("telechat3-yarn") => {
                let factor = m.rope_scaling_factor.unwrap_or(1.0);
                let orig_ctx = m.rope_original_context_len.unwrap_or(4096);
                // Manifest fields from convert (`rope_scaling.beta_fast/slow`);
                // mlx-lm YarnRoPE / openai gpt-oss defaults are 32 / 1 when omitted.
                // DeepSeek V4 compress rope already honors the same fields — keep
                // the primary YaRN path consistent so non-default bounds land.
                let beta_fast = m.rope_beta_fast.unwrap_or(32.0);
                let beta_slow = m.rope_beta_slow.unwrap_or(1.0);
                let (freqs, mscale) = super::shared::build_yarn_rope_freqs(
                    rope_dims, rope_theta, factor, orig_ctx, beta_fast, beta_slow, 1.0, 0.0,
                );
                (Some(freqs), mscale)
            }
            _ => (None, 1.0),
        };

        let moe_norm_topk_prob =
            if matches!(m.model_family.as_str(), "qwen3_5" | "qwen3_next") && m.moe.is_enabled() {
                // mlx_lm / Transformers default norm_topk_prob to true for Qwen MoE
                // hybrids (qwen3_5 MoE and qwen3_next / Qwen3.6-35B-A3B). Older AX
                // manifests emitted false when config.json omitted the field, which
                // routes experts with the wrong weights. Keep the loader compatible
                // with those cached manifests while the converter emits the correct
                // default for both families.
                true
            } else {
                m.moe_norm_topk_prob
            };

        Self {
            compile_cache_identity: NEXT_COMPILE_CACHE_IDENTITY.fetch_add(1, Ordering::Relaxed),
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
            query_scale: query_scale * m.attention_scale_multiplier.unwrap_or(1.0),
            final_logit_softcapping: m.final_logit_softcapping,
            final_logits_scale: m.final_logits_scale,
            post_norm_eps: m.post_norm_eps.unwrap_or_else(|| {
                m.rms_norm_eps
                    .unwrap_or_else(|| default_rms_norm_eps(&m.model_family))
            }),
            embed_norm_no_weight: m.model_family == "muse_glimmer",
            moe_expert_count: m.moe.expert_count.unwrap_or(0) as usize,
            moe_experts_per_token: m.moe.experts_per_token.unwrap_or(0) as usize,
            moe_expert_intermediate_size: m.moe.expert_intermediate_size.unwrap_or(0) as usize,
            layer_configs,
            global_sliding_window,
            protected_prefix_sliding_window,
            gemma4_moe_router: is_gemma4,
            uses_geglu,
            hidden_states_scale: m.hidden_states_scale,
            moe_norm_topk_prob,
            hidden_size_per_layer_input: m.hidden_size_per_layer_input as usize,
            linear_attention: LinearAttentionConfig::from_manifest(m),
            mla_attention: MlaAttentionConfig::from_manifest(m),
            glm_router: GlmRouterConfig::from_manifest(m),
            deepseek_v4: DeepseekV4Config::from_manifest(m),
            rms_norm_eps: m
                .rms_norm_eps
                .unwrap_or_else(|| default_rms_norm_eps(&m.model_family)),
            rope_freqs,
            rope_mscale,
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
            generation_kind: GenerationKind::from_manifest(m),
            kv_cache_quant: kv_cache_quant_from_manifest(m),
        }
    }

    /// True when this model uses block-diffusion generation (ADR-038).
    #[inline]
    pub fn is_block_diffusion(&self) -> bool {
        matches!(self.generation_kind, GenerationKind::BlockDiffusion) || self.diffusion.is_some()
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
    // Fully explicit pair always wins.
    if m.think_start_token_id.is_some() && m.think_end_token_id.is_some() {
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
    //
    // DeepSeek uses the same `<think>`/`</think>` content strings, but the
    // special-token ids differ across tokenizer generations (verified against
    // official `tokenizer.json` added_tokens):
    //   - deepseek_v3 / deepseek_v32 / R1: 128798 / 128799
    //   - deepseek_v4 (Flash + Pro):         128821 / 128822
    // Without family defaults, manifests converted without a present
    // tokenizer.json leave think IDs unset, so `ngram_in_think` never
    // transitions and DeepSeek V4 think-aware MTP draft temperature is inert
    // (DI-DS-A001).
    let family_defaults = match m.model_family.as_str() {
        "qwen3" | "qwen3_5" | "qwen3_next" | "minicpmv4_6" => {
            if m.vocab_size >= 200_000 {
                (Some(248_068), Some(248_069))
            } else {
                (Some(151_668), Some(151_669))
            }
        }
        "deepseek_v4" => (Some(128_821), Some(128_822)),
        "deepseek_v3" | "deepseek_v32" => (Some(128_798), Some(128_799)),
        _ => (None, None),
    };
    // Partial explicit fields (legacy / hand-edited manifests) must not leave
    // an unclosable think block: fill the missing side from family defaults.
    (
        m.think_start_token_id.or(family_defaults.0),
        m.think_end_token_id.or(family_defaults.1),
    )
}

/// Map the manifest's `kv_cache_quantization` table to per-layer specs.
/// Bits 16 marks a full-precision layer (`None`); absent table → all `None`.
fn kv_cache_quant_from_manifest(m: &NativeModelManifest) -> Vec<Option<KvQuantSpec>> {
    let layer_count = m.layer_count as usize;
    let Some(table) = &m.kv_cache_quantization else {
        return vec![None; layer_count];
    };
    table
        .layer_bits
        .iter()
        .zip(table.layer_group_sizes.iter())
        .map(|(&bits, &group_size)| {
            if bits == 16 {
                None
            } else {
                Some(KvQuantSpec { bits, group_size })
            }
        })
        .collect()
}

fn default_rms_norm_eps(model_family: &str) -> f32 {
    // DeepSeek-V2/V3 and Unlimited-OCR use 1e-6 (configuration_deepseek_v2 /
    // DeepseekV3Config). Qwen and Gemma share the same default.
    if model_family.starts_with("qwen")
        || model_family == "minicpmv4_6"
        || model_family.starts_with("gemma")
        || model_family == "diffusion_gemma"
        || model_family == "unlimited_ocr"
        || model_family == "muse_glimmer"
        || model_family.starts_with("deepseek")
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
        "gemma4" | "gemma4_vl" | "gemma4_unified" | "gemma4_assistant" | "diffusion_gemma"
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
                    // Muse Glimmer full-attention layers are NoPE (iRoPE):
                    // the reference never rotates them; rope_dims = 0 marks
                    // that for the muse route.
                    rope_dims: if m.model_family == "muse_glimmer" {
                        0
                    } else if full_rope_freqs.is_some() {
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

#[cfg(test)]
mod tests {
    use super::{KvQuantSpec, ModelConfig, default_rms_norm_eps};

    fn manifest_with_kv_cache_quantization(
        kv_cache_quantization: Option<serde_json::Value>,
    ) -> ax_engine_core::NativeModelManifest {
        let mut value = serde_json::json!({
            "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
            "model_family": "qwen3",
            "tensor_format": "safetensors",
            "layer_count": 2,
            "hidden_size": 16,
            "attention_head_count": 2,
            "attention_head_dim": 8,
            "kv_head_count": 1,
            "vocab_size": 32,
            "tensors": [],
        });
        if let Some(table) = kv_cache_quantization {
            value["kv_cache_quantization"] = table;
        }
        serde_json::from_value(value).expect("manifest JSON should deserialize")
    }

    #[test]
    fn from_manifest_maps_kv_cache_quantization_bits_16_to_none() {
        let manifest = manifest_with_kv_cache_quantization(Some(serde_json::json!({
            "layer_bits": [8, 16],
            "layer_group_sizes": [64, 128],
            "basis": "measured",
        })));

        let config = ModelConfig::from_manifest(&manifest);

        assert_eq!(
            config.kv_cache_quant,
            vec![
                Some(KvQuantSpec {
                    bits: 8,
                    group_size: 64
                }),
                None,
            ]
        );
    }

    #[test]
    fn from_manifest_without_kv_cache_quantization_yields_all_none() {
        let manifest = manifest_with_kv_cache_quantization(None);

        let config = ModelConfig::from_manifest(&manifest);

        assert_eq!(config.kv_cache_quant, vec![None, None]);
    }

    #[test]
    fn gemma4_vl_uses_geglu_and_unit_query_scale() {
        // gemma4_vl packaging must not fall through to non-Gemma defaults
        // (SwiGLU + 1/sqrt(d) query scale), which desyncs the text tower.
        let value = serde_json::json!({
            "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
            "model_family": "gemma4_vl",
            "tensor_format": "safetensors",
            "layer_count": 2,
            "hidden_size": 64,
            "attention_head_count": 4,
            "attention_head_dim": 16,
            "kv_head_count": 1,
            "vocab_size": 32,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window_size": 128,
            "tensors": [],
        });
        let manifest: ax_engine_core::NativeModelManifest =
            serde_json::from_value(value).expect("gemma4_vl manifest");
        let cfg = ModelConfig::from_manifest(&manifest);
        assert!(cfg.uses_geglu, "gemma4_vl text tower is GeGLU");
        assert_eq!(
            cfg.query_scale, 1.0,
            "Gemma 4 Softcapping attention uses query_scale=1.0, not 1/sqrt(head_dim)"
        );
        assert_eq!(cfg.layer_configs.len(), 2);
        assert_eq!(cfg.layer_configs[0].sliding_window, Some(128));
        assert_eq!(cfg.layer_configs[1].sliding_window, None);
    }

    #[test]
    fn yarn_rope_honors_manifest_beta_fast_slow() {
        // Convert stores rope_beta_fast/slow from config.json rope_scaling.
        // Runtime must pass them into build_yarn_rope_freqs (not hardcode 32/1).
        // openai/gpt-oss uses 32/1; a non-default pair must change the divisors.
        let mut base = serde_json::json!({
            "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
            "model_family": "gpt_oss",
            "tensor_format": "safetensors",
            "layer_count": 2,
            "hidden_size": 64,
            "attention_head_count": 4,
            "attention_head_dim": 16,
            "kv_head_count": 1,
            "vocab_size": 32,
            "rope_theta": 150000,
            "rope_scaling_type": "yarn",
            "rope_scaling_factor": 32.0,
            "rope_original_context_len": 4096,
            "rope_beta_fast": 32.0,
            "rope_beta_slow": 1.0,
            "tensors": [],
        });
        let default_manifest: ax_engine_core::NativeModelManifest =
            serde_json::from_value(base.clone()).expect("default yarn manifest");
        let default_cfg = ModelConfig::from_manifest(&default_manifest);
        let default_freqs = default_cfg
            .rope_freqs
            .as_ref()
            .expect("yarn must precompute rope_freqs")
            .data_f32();

        base["rope_beta_fast"] = serde_json::json!(8.0);
        base["rope_beta_slow"] = serde_json::json!(2.0);
        let custom_manifest: ax_engine_core::NativeModelManifest =
            serde_json::from_value(base).expect("custom yarn manifest");
        assert_eq!(custom_manifest.rope_beta_fast, Some(8.0));
        assert_eq!(custom_manifest.rope_beta_slow, Some(2.0));
        let custom_cfg = ModelConfig::from_manifest(&custom_manifest);
        let custom_freqs = custom_cfg
            .rope_freqs
            .as_ref()
            .expect("yarn must precompute rope_freqs")
            .data_f32();

        assert_eq!(default_freqs.len(), custom_freqs.len());
        let max_abs = default_freqs
            .iter()
            .zip(custom_freqs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs > 1e-4,
            "non-default rope_beta_fast/slow must change YaRN freqs (max_abs={max_abs}); \
             runtime was likely hardcoding 32/1 and ignoring the manifest"
        );
        // mscale depends only on factor/mscale, not betas — still ~1.346 for factor=32.
        let expected_mscale = 0.1 * 32.0_f32.ln() + 1.0;
        assert!(
            (custom_cfg.rope_mscale - expected_mscale).abs() < 1e-4,
            "rope_mscale should follow yarn factor, got {}",
            custom_cfg.rope_mscale
        );
    }

    #[test]
    fn qwen3_next_moe_forces_norm_topk_true_for_stale_manifests() {
        // Qwen3.6-35B-A3B is family qwen3_next. Older converters wrote
        // moe_norm_topk_prob=false when config.json omitted the field; runtime
        // must still normalize top-k weights or expert mix desyncs from mlx_lm.
        let mut value = serde_json::json!({
            "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
            "model_family": "qwen3_next",
            "tensor_format": "safetensors",
            "layer_count": 2,
            "hidden_size": 16,
            "attention_head_count": 2,
            "attention_head_dim": 8,
            "kv_head_count": 1,
            "vocab_size": 32,
            "moe_norm_topk_prob": false,
            "moe": {
                "expert_count": 8,
                "experts_per_token": 2,
                "expert_intermediate_size": 8
            },
            "tensors": [],
        });
        let manifest: ax_engine_core::NativeModelManifest =
            serde_json::from_value(value.clone()).expect("manifest");
        assert!(!manifest.moe_norm_topk_prob);
        assert!(manifest.moe.is_enabled());
        let cfg = ModelConfig::from_manifest(&manifest);
        assert!(
            cfg.moe_norm_topk_prob,
            "qwen3_next MoE must force norm_topk_prob=true for stale manifests"
        );

        // Dense qwen3_next (no MoE) must keep the stored false — only MoE needs
        // the override.
        value["moe"] = serde_json::json!({});
        let dense: ax_engine_core::NativeModelManifest =
            serde_json::from_value(value).expect("dense manifest");
        assert!(!dense.moe.is_enabled());
        let dense_cfg = ModelConfig::from_manifest(&dense);
        assert!(!dense_cfg.moe_norm_topk_prob);
    }

    #[test]
    fn unlimited_ocr_and_deepseek_default_to_1e_6_rms_norm_eps() {
        // DeepSeek-V2 / Unlimited-OCR language towers use 1e-6; the generic
        // non-qwen/gemma fallback of 1e-5 would silently corrupt OCR quality.
        assert!((default_rms_norm_eps("unlimited_ocr") - 1e-6).abs() < f32::EPSILON);
        assert!((default_rms_norm_eps("deepseek_v3") - 1e-6).abs() < f32::EPSILON);
        assert!((default_rms_norm_eps("deepseek_v32") - 1e-6).abs() < f32::EPSILON);
        assert!((default_rms_norm_eps("qwen3") - 1e-6).abs() < f32::EPSILON);
        assert!((default_rms_norm_eps("llama3") - 1e-5).abs() < f32::EPSILON);
    }
}
