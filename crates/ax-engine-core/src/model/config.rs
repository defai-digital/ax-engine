//! Model configuration extracted from GGUF metadata.
//!
//! GGUF keys follow the pattern: `{arch}.{param}` where arch is e.g. "llama".
//! This module reads the architecture-specific keys to build a ModelConfig.

use crate::gguf::GgufHeader;

/// Gate activation function used in the FFN.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum GateActivation {
    /// SiLU / SwiGLU — used by LLaMA, Mistral, Qwen.
    #[default]
    SiLU,
    /// GELU / GeGLU — used by Gemma3.
    GELU,
}

/// RoPE scaling type.
#[derive(Debug, Default, Clone, PartialEq)]
pub enum RopeScaling {
    /// No scaling (standard RoPE).
    #[default]
    None,
    /// Linear scaling: freq_scale = 1.0 / factor.
    Linear(f32),
    /// YaRN (Yet another RoPE extensioN): per-dimension interpolation/extrapolation blend.
    Yarn {
        factor: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        orig_ctx_len: u32,
    },
}

impl RopeScaling {
    /// Effective position used by existing linear-style RoPE paths.
    ///
    /// Until the remaining YaRN-specific CPU/model paths are fully plumbed,
    /// YaRN follows the same position interpolation factor as linear RoPE.
    pub fn scaled_position(&self, position: usize) -> f32 {
        match self {
            Self::None => position as f32,
            Self::Linear(factor) => position as f32 / factor,
            Self::Yarn { factor, .. } => position as f32 / factor,
        }
    }

    /// Effective `(rope_start, rope_step)` pair for batched RoPE traversal.
    ///
    /// As above, YaRN currently follows the same positional interpolation
    /// factor as the existing linear path until the full parameter plumbing is
    /// completed across all forward variants.
    pub fn scaled_start_step(&self, base_seq_len: usize) -> (f32, f32) {
        match self {
            Self::None => (base_seq_len as f32, 1.0),
            Self::Linear(factor) => (base_seq_len as f32 / factor, 1.0 / factor),
            Self::Yarn { factor, .. } => (base_seq_len as f32 / factor, 1.0 / factor),
        }
    }
}

/// Pre-computed YaRN parameters passed to GPU kernels.
/// When `ext_factor == 0.0`, the kernel reduces to vanilla RoPE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeYarnParams {
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub orig_ctx_len: u32,
}

impl RopeYarnParams {
    /// Vanilla RoPE (no scaling). Kernel short-circuits to cos/sin.
    pub fn none() -> Self {
        Self {
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            orig_ctx_len: 0,
        }
    }

    /// Linear scaling (position interpolation only).
    pub fn linear(factor: f32) -> Self {
        Self {
            freq_scale: 1.0 / factor,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            orig_ctx_len: 0,
        }
    }

    /// Full YaRN with per-dimension blend.
    pub fn yarn(
        factor: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        orig_ctx_len: u32,
    ) -> Self {
        // Pre-compute effective attn_factor (mscale correction)
        let freq_scale = 1.0 / factor;
        let mscale = if ext_factor != 0.0 && factor > 1.0 {
            let base_mscale = 1.0 + 0.1 * (1.0 / freq_scale).ln();
            attn_factor * base_mscale
        } else {
            attn_factor
        };
        Self {
            freq_scale,
            ext_factor,
            attn_factor: mscale,
            beta_fast,
            beta_slow,
            orig_ctx_len,
        }
    }

    /// Build from RopeScaling enum.
    pub fn from_scaling(scaling: &RopeScaling) -> Self {
        match scaling {
            RopeScaling::None => Self::none(),
            RopeScaling::Linear(factor) => Self::linear(*factor),
            RopeScaling::Yarn {
                factor,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow,
                orig_ctx_len,
            } => Self::yarn(
                *factor,
                *ext_factor,
                *attn_factor,
                *beta_fast,
                *beta_slow,
                *orig_ctx_len,
            ),
        }
    }
}

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub embedding_dim: u32,
    pub head_dim: u32,
    pub intermediate_dim: u32,
    pub context_length: u32,
    pub vocab_size: u32,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    /// Whether Q/K/V projections have bias terms (Qwen3 uses this).
    pub has_qkv_bias: bool,
    /// Sliding window attention size (None = full attention).
    pub sliding_window_size: Option<u32>,
    /// Sliding window pattern: e.g. 2 = every other layer uses sliding window.
    pub sliding_window_pattern: Option<u32>,
    /// Gate activation function for FFN.
    pub gate_activation: GateActivation,
    /// Whether the LM head ties to embedding weights.
    pub tie_word_embeddings: bool,
    /// Logit scaling factor (e.g. 1/sqrt(head_dim) for Gemma3).
    pub logit_scale: Option<f32>,
    /// RoPE scaling type and factor.
    pub rope_scaling: RopeScaling,
    /// Whether to scale embeddings by sqrt(embedding_dim) (Gemma).
    pub embed_scale: bool,
    /// RoPE freq base for local (sliding window) attention layers.
    /// Gemma3 uses 10000.0 for local layers vs 1000000.0 for global.
    pub rope_freq_base_local: Option<f32>,
    /// Number of experts in MoE models. None for dense models.
    pub n_expert: Option<u32>,
    /// Number of experts used per token in MoE routing. None for dense models.
    pub n_expert_used: Option<u32>,
    /// Per-expert intermediate (FFN) dimension. May differ from `intermediate_dim`
    /// which is the shared expert / dense FFN dimension. None for dense models.
    pub expert_intermediate_dim: Option<u32>,
    /// Qwen3.5: every Nth layer is full attention, others are recurrent GDN.
    pub qwen35_full_attention_interval: Option<u32>,
    /// Qwen3.5 recurrent SSM conv kernel size.
    pub qwen35_ssm_conv_kernel: Option<u32>,
    /// Qwen3.5 recurrent inner/value dimension.
    pub qwen35_ssm_inner_size: Option<u32>,
    /// Qwen3.5 recurrent state dimension per head.
    pub qwen35_ssm_state_size: Option<u32>,
    /// Qwen3.5 recurrent value-head count / time-step rank.
    pub qwen35_ssm_time_step_rank: Option<u32>,
    /// Qwen3.5 recurrent key-head group count.
    pub qwen35_ssm_group_count: Option<u32>,
}

impl ModelConfig {
    /// Extract model config from GGUF header metadata.
    ///
    /// Reads architecture-prefixed keys (e.g. `llama.embedding_length`)
    /// with sensible defaults for optional fields.
    pub fn from_gguf(header: &GgufHeader) -> anyhow::Result<Self> {
        let arch = header.architecture().unwrap_or("llama").to_string();

        let get_u32 = |key: &str| -> anyhow::Result<u32> {
            header
                .get_u32(&format!("{arch}.{key}"))
                .ok_or_else(|| anyhow::anyhow!("missing GGUF key: {arch}.{key}"))
        };

        let get_f32_or = |key: &str, default: f32| -> f32 {
            header.get_f32(&format!("{arch}.{key}")).unwrap_or(default)
        };

        let n_layers = get_u32("block_count")?;
        let n_heads = get_u32("attention.head_count")?;
        let embedding_dim = get_u32("embedding_length")?;

        // Validate critical fields are non-zero to prevent division-by-zero
        anyhow::ensure!(n_layers > 0, "GGUF {arch}.block_count is 0");
        anyhow::ensure!(n_heads > 0, "GGUF {arch}.attention.head_count is 0");
        anyhow::ensure!(embedding_dim > 0, "GGUF {arch}.embedding_length is 0");

        // n_kv_heads defaults to n_heads (MHA) if not specified (GQA/MQA set this)
        let n_kv_heads = header
            .get_u32(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(n_heads);

        // Explicit head_dim from GGUF; fall back to embedding_dim / n_heads
        let head_dim = header
            .get_u32(&format!("{arch}.attention.key_length"))
            .unwrap_or(embedding_dim / n_heads);

        // feed_forward_length is the intermediate dim for MLP
        let intermediate_dim = get_u32("feed_forward_length")?;

        let context_length = header
            .get_u32(&format!("{arch}.context_length"))
            .unwrap_or(4096);

        // Vocab size can come from tokenizer metadata or model metadata
        let vocab_size = header
            .get_u32(&format!("{arch}.vocab_size"))
            .or_else(|| {
                header
                    .get_str_array("tokenizer.ggml.tokens")
                    .map(|t| t.len() as u32)
            })
            .unwrap_or(32000);

        let rms_norm_eps = get_f32_or("attention.layer_norm_rms_epsilon", 1e-5);
        let rope_freq_base = get_f32_or("rope.freq_base", 10000.0);

        // Architecture-specific: QKV bias
        let has_qkv_bias = header
            .get_bool(&format!("{arch}.attention.bias"))
            .unwrap_or(false);

        // Architecture-specific: sliding window
        let sliding_window_size = header.get_u32(&format!("{arch}.attention.sliding_window"));
        let sliding_window_pattern = header
            .get_u32(&format!("{arch}.attention.sliding_window_pattern"))
            .or_else(|| {
                // Gemma3 default: every 6th layer is global, rest use sliding window
                if matches!(arch.as_str(), "gemma" | "gemma2" | "gemma3")
                    && sliding_window_size.is_some()
                {
                    Some(6)
                } else {
                    None
                }
            });

        // Architecture-specific: gate activation + logit scale + tie embeddings
        let is_gemma = matches!(arch.as_str(), "gemma" | "gemma2" | "gemma3");
        let gate_activation = if is_gemma {
            GateActivation::GELU
        } else {
            GateActivation::SiLU
        };

        let tie_word_embeddings = header
            .get_bool(&format!("{arch}.tie_word_embeddings"))
            .or_else(|| header.get_bool("general.tie_word_embeddings"))
            .unwrap_or(false);

        // Gemma3: no final logit scaling — the 1/sqrt(head_dim) is already
        // in the attention score scaling (AttentionParams::scale). Only apply
        // if there's an explicit GGUF metadata key for it (e.g. Gemma2 had
        // final_logit_softcapping, but Gemma3 does not).
        let logit_scale = None;

        // RoPE scaling
        // NOTE: For Gemma3, the GGUF has rope.scaling.type="linear" with factor=8.0.
        // Per llama.cpp, this scaling is applied only to GLOBAL (dense) attention layers.
        // SWA (sliding window / local) layers use freq_scale=1.0 (no position scaling).
        // The per-layer application happens in gemma3.rs, not here.
        let rope_scaling = match header.get_str(&format!("{arch}.rope.scaling.type")) {
            Some("linear") => {
                let factor = header
                    .get_f32(&format!("{arch}.rope.scaling.factor"))
                    .unwrap_or(1.0);
                anyhow::ensure!(
                    factor.is_finite() && factor > 0.0,
                    "invalid {arch}.rope.scaling.factor: expected finite > 0, got {factor}"
                );
                RopeScaling::Linear(factor)
            }
            Some("yarn") => {
                let factor = header
                    .get_f32(&format!("{arch}.rope.scaling.factor"))
                    .unwrap_or(1.0);
                anyhow::ensure!(
                    factor.is_finite() && factor > 0.0,
                    "invalid {arch}.rope.scaling.factor: expected finite > 0, got {factor}"
                );
                let orig_ctx_len = header
                    .get_u32(&format!("{arch}.rope.scaling.original_context_length"))
                    .unwrap_or(context_length);
                let ext_factor = header
                    .get_f32(&format!("{arch}.rope.scaling.yarn_ext_factor"))
                    .unwrap_or(1.0);
                let attn_factor = header
                    .get_f32(&format!("{arch}.rope.scaling.attn_factor"))
                    .or_else(|| header.get_f32(&format!("{arch}.rope.scaling.yarn_attn_factor")))
                    .unwrap_or(1.0);
                let beta_fast = header
                    .get_f32(&format!("{arch}.rope.scaling.yarn_beta_fast"))
                    .unwrap_or(32.0);
                let beta_slow = header
                    .get_f32(&format!("{arch}.rope.scaling.yarn_beta_slow"))
                    .unwrap_or(1.0);
                tracing::info!(
                    factor,
                    ext_factor,
                    attn_factor,
                    beta_fast,
                    beta_slow,
                    orig_ctx_len,
                    "YaRN RoPE scaling detected"
                );
                RopeScaling::Yarn {
                    factor,
                    ext_factor,
                    attn_factor,
                    beta_fast,
                    beta_slow,
                    orig_ctx_len,
                }
            }
            Some(scaling_type) => {
                tracing::warn!(scaling_type, "unknown RoPE scaling type, ignoring");
                RopeScaling::None
            }
            None => RopeScaling::None,
        };

        let embed_scale = is_gemma;

        // Gemma3: local (sliding window) layers use a lower RoPE base (10000)
        // while global layers use the main rope_freq_base (typically 1000000).
        let rope_freq_base_local = if is_gemma && sliding_window_size.is_some() {
            Some(10000.0f32)
        } else {
            None
        };

        // MoE: expert count, experts-per-token, and per-expert FFN dimension
        let n_expert = header.get_u32(&format!("{arch}.expert_count"));
        let n_expert_used = header.get_u32(&format!("{arch}.expert_used_count"));
        let expert_intermediate_dim = header.get_u32(&format!("{arch}.expert_feed_forward_length"));
        let qwen35_full_attention_interval =
            header.get_u32(&format!("{arch}.full_attention_interval"));
        let qwen35_ssm_conv_kernel = header.get_u32(&format!("{arch}.ssm.conv_kernel"));
        let qwen35_ssm_inner_size = header.get_u32(&format!("{arch}.ssm.inner_size"));
        let qwen35_ssm_state_size = header.get_u32(&format!("{arch}.ssm.state_size"));
        let qwen35_ssm_time_step_rank = header.get_u32(&format!("{arch}.ssm.time_step_rank"));
        let qwen35_ssm_group_count = header.get_u32(&format!("{arch}.ssm.group_count"));

        tracing::info!(
            arch = %arch,
            n_layers,
            n_heads,
            n_kv_heads,
            embedding_dim,
            head_dim,
            intermediate_dim,
            context_length,
            vocab_size,
            rms_norm_eps,
            rope_freq_base,
            has_qkv_bias,
            ?sliding_window_size,
            ?gate_activation,
            tie_word_embeddings,
            ?logit_scale,
            ?rope_scaling,
            embed_scale,
            ?rope_freq_base_local,
            ?n_expert,
            ?n_expert_used,
            ?qwen35_full_attention_interval,
            ?qwen35_ssm_conv_kernel,
            ?qwen35_ssm_inner_size,
            ?qwen35_ssm_state_size,
            ?qwen35_ssm_time_step_rank,
            ?qwen35_ssm_group_count,
            "model config loaded"
        );

        Ok(Self {
            architecture: arch,
            n_layers,
            n_heads,
            n_kv_heads,
            embedding_dim,
            head_dim,
            intermediate_dim,
            context_length,
            vocab_size,
            rms_norm_eps,
            rope_freq_base,
            has_qkv_bias,
            sliding_window_size,
            sliding_window_pattern,
            gate_activation,
            tie_word_embeddings,
            logit_scale,
            rope_scaling,
            embed_scale,
            rope_freq_base_local,
            n_expert,
            n_expert_used,
            expert_intermediate_dim,
            qwen35_full_attention_interval,
            qwen35_ssm_conv_kernel,
            qwen35_ssm_inner_size,
            qwen35_ssm_state_size,
            qwen35_ssm_time_step_rank,
            qwen35_ssm_group_count,
        })
    }

    pub fn qwen35_is_recurrent_layer(&self, layer: usize) -> bool {
        let Some(interval) = self.qwen35_full_attention_interval else {
            return false;
        };
        let interval = interval as usize;
        interval > 0 && !(layer + 1).is_multiple_of(interval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::MetadataValue;
    use crate::gguf::header::GgufHeader;
    use std::collections::HashMap;

    fn make_header(kv: Vec<(&str, MetadataValue)>) -> GgufHeader {
        let mut metadata = HashMap::new();
        for (k, v) in kv {
            metadata.insert(k.to_string(), v);
        }
        GgufHeader {
            version: 3,
            tensor_count: 0,
            metadata,
        }
    }

    #[test]
    fn test_config_from_gguf() {
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count", MetadataValue::Uint32(32)),
            ("llama.attention.head_count", MetadataValue::Uint32(32)),
            ("llama.attention.head_count_kv", MetadataValue::Uint32(8)),
            ("llama.embedding_length", MetadataValue::Uint32(4096)),
            ("llama.feed_forward_length", MetadataValue::Uint32(11008)),
            ("llama.context_length", MetadataValue::Uint32(2048)),
            (
                "llama.attention.layer_norm_rms_epsilon",
                MetadataValue::Float32(1e-5),
            ),
            ("llama.rope.freq_base", MetadataValue::Float32(10000.0)),
        ]);

        let config = ModelConfig::from_gguf(&header).unwrap();
        assert_eq!(config.architecture, "llama");
        assert_eq!(config.n_layers, 32);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, 8);
        assert_eq!(config.embedding_dim, 4096);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_dim, 11008);
        assert_eq!(config.context_length, 2048);
        assert_eq!(config.rms_norm_eps, 1e-5);
        assert_eq!(config.rope_freq_base, 10000.0);
        // New fields default correctly for llama
        assert!(!config.has_qkv_bias);
        assert_eq!(config.sliding_window_size, None);
        assert_eq!(config.gate_activation, GateActivation::SiLU);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.logit_scale, None);
    }

    #[test]
    fn test_config_defaults() {
        // Minimal metadata — n_kv_heads defaults to n_heads, context to 4096
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count", MetadataValue::Uint32(12)),
            ("llama.attention.head_count", MetadataValue::Uint32(12)),
            ("llama.embedding_length", MetadataValue::Uint32(768)),
            ("llama.feed_forward_length", MetadataValue::Uint32(2048)),
        ]);

        let config = ModelConfig::from_gguf(&header).unwrap();
        assert_eq!(config.n_kv_heads, 12); // defaults to n_heads
        assert_eq!(config.head_dim, 64); // 768 / 12
        assert_eq!(config.context_length, 4096); // default
        assert_eq!(config.rope_freq_base, 10000.0); // default
    }

    #[test]
    fn test_config_zero_heads_rejected() {
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count", MetadataValue::Uint32(32)),
            ("llama.attention.head_count", MetadataValue::Uint32(0)),
            ("llama.embedding_length", MetadataValue::Uint32(4096)),
            ("llama.feed_forward_length", MetadataValue::Uint32(11008)),
        ]);

        let err = ModelConfig::from_gguf(&header).unwrap_err();
        assert!(
            err.to_string().contains("head_count is 0"),
            "expected head_count error, got: {err}"
        );
    }

    #[test]
    fn test_config_missing_required() {
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("llama".into()),
            ),
            // Missing block_count
            ("llama.attention.head_count", MetadataValue::Uint32(32)),
            ("llama.embedding_length", MetadataValue::Uint32(4096)),
            ("llama.feed_forward_length", MetadataValue::Uint32(11008)),
        ]);

        let err = ModelConfig::from_gguf(&header).unwrap_err();
        assert!(
            err.to_string().contains("block_count"),
            "expected block_count error, got: {err}"
        );
    }

    #[test]
    fn test_config_vocab_from_tokenizer() {
        // No vocab_size in model metadata — falls back to tokenizer token count
        let tokens = vec![
            MetadataValue::String("a".into()),
            MetadataValue::String("b".into()),
            MetadataValue::String("c".into()),
        ];
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count", MetadataValue::Uint32(1)),
            ("llama.attention.head_count", MetadataValue::Uint32(1)),
            ("llama.embedding_length", MetadataValue::Uint32(64)),
            ("llama.feed_forward_length", MetadataValue::Uint32(128)),
            ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
        ]);

        let config = ModelConfig::from_gguf(&header).unwrap();
        assert_eq!(config.vocab_size, 3);
    }

    #[test]
    fn test_config_explicit_head_dim() {
        // When key_length is set explicitly, it should override embedding_dim / n_heads
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("gemma3".into()),
            ),
            ("gemma3.block_count", MetadataValue::Uint32(26)),
            ("gemma3.attention.head_count", MetadataValue::Uint32(8)),
            ("gemma3.attention.head_count_kv", MetadataValue::Uint32(4)),
            ("gemma3.embedding_length", MetadataValue::Uint32(2560)),
            ("gemma3.feed_forward_length", MetadataValue::Uint32(10240)),
            ("gemma3.attention.key_length", MetadataValue::Uint32(320)),
        ]);

        let config = ModelConfig::from_gguf(&header).unwrap();
        assert_eq!(config.head_dim, 320); // explicit, not 2560/8=320
        assert_eq!(config.gate_activation, GateActivation::GELU);
        assert!(config.logit_scale.is_none()); // Gemma3 does NOT scale final logits
        assert!(config.embed_scale); // Gemma scales embeddings by sqrt(dim)
    }

    #[test]
    fn test_config_qwen3_qkv_bias() {
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("qwen3".into()),
            ),
            ("qwen3.block_count", MetadataValue::Uint32(32)),
            ("qwen3.attention.head_count", MetadataValue::Uint32(32)),
            ("qwen3.embedding_length", MetadataValue::Uint32(4096)),
            ("qwen3.feed_forward_length", MetadataValue::Uint32(11008)),
            ("qwen3.attention.bias", MetadataValue::Bool(true)),
        ]);

        let config = ModelConfig::from_gguf(&header).unwrap();
        assert!(config.has_qkv_bias);
        assert_eq!(config.gate_activation, GateActivation::SiLU);
    }

    #[test]
    fn test_config_gemma3_features() {
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("gemma3".into()),
            ),
            ("gemma3.block_count", MetadataValue::Uint32(26)),
            ("gemma3.attention.head_count", MetadataValue::Uint32(8)),
            ("gemma3.embedding_length", MetadataValue::Uint32(2560)),
            ("gemma3.feed_forward_length", MetadataValue::Uint32(10240)),
            (
                "gemma3.attention.sliding_window",
                MetadataValue::Uint32(4096),
            ),
            (
                "gemma3.attention.sliding_window_pattern",
                MetadataValue::Uint32(2),
            ),
            ("gemma3.tie_word_embeddings", MetadataValue::Bool(true)),
        ]);

        let config = ModelConfig::from_gguf(&header).unwrap();
        assert_eq!(config.sliding_window_size, Some(4096));
        assert_eq!(config.sliding_window_pattern, Some(2));
        assert_eq!(config.gate_activation, GateActivation::GELU);
        assert!(config.tie_word_embeddings);
        assert!(config.logit_scale.is_none()); // Gemma3 does NOT scale final logits
        assert!(config.embed_scale);
        assert_eq!(config.rope_freq_base_local, Some(10000.0));
    }

    #[test]
    fn test_config_gemma3_default_sliding_window_pattern() {
        // When GGUF has sliding_window but no pattern, Gemma3 defaults to 6
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("gemma3".into()),
            ),
            ("gemma3.block_count", MetadataValue::Uint32(34)),
            ("gemma3.attention.head_count", MetadataValue::Uint32(8)),
            ("gemma3.embedding_length", MetadataValue::Uint32(2560)),
            ("gemma3.feed_forward_length", MetadataValue::Uint32(10240)),
            (
                "gemma3.attention.sliding_window",
                MetadataValue::Uint32(1024),
            ),
            // No sliding_window_pattern key — should default to 6
        ]);

        let config = ModelConfig::from_gguf(&header).unwrap();
        assert_eq!(config.sliding_window_size, Some(1024));
        assert_eq!(config.sliding_window_pattern, Some(6));
        assert_eq!(config.rope_freq_base_local, Some(10000.0));
    }

    #[test]
    fn test_config_rejects_non_positive_rope_scaling_factor() {
        let header = make_header(vec![
            (
                "general.architecture",
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count", MetadataValue::Uint32(32)),
            ("llama.attention.head_count", MetadataValue::Uint32(32)),
            ("llama.embedding_length", MetadataValue::Uint32(4096)),
            ("llama.feed_forward_length", MetadataValue::Uint32(11008)),
            (
                "llama.rope.scaling.type",
                MetadataValue::String("linear".into()),
            ),
            ("llama.rope.scaling.factor", MetadataValue::Float32(0.0)),
        ]);

        let err = ModelConfig::from_gguf(&header).unwrap_err();
        assert!(err.to_string().contains("rope.scaling.factor"));
    }

    #[test]
    fn test_rope_scaling_yarn_uses_linear_style_scaled_position() {
        let scaling = RopeScaling::Yarn {
            factor: 8.0,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            orig_ctx_len: 8192,
        };

        assert!((scaling.scaled_position(96) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_scaling_yarn_uses_linear_style_scaled_start_step() {
        let scaling = RopeScaling::Yarn {
            factor: 4.0,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            orig_ctx_len: 16384,
        };

        let (rope_start, rope_step) = scaling.scaled_start_step(128);
        assert!((rope_start - 32.0).abs() < 1e-6);
        assert!((rope_step - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_rope_yarn_params_from_scaling_preserves_yarn_configuration() {
        let scaling = RopeScaling::Yarn {
            factor: 4.0,
            ext_factor: 0.5,
            attn_factor: 1.25,
            beta_fast: 16.0,
            beta_slow: 2.0,
            orig_ctx_len: 32768,
        };

        let params = RopeYarnParams::from_scaling(&scaling);
        assert!((params.freq_scale - 0.25).abs() < 1e-6);
        assert_eq!(params.ext_factor, 0.5);
        assert_eq!(params.beta_fast, 16.0);
        assert_eq!(params.beta_slow, 2.0);
        assert_eq!(params.orig_ctx_len, 32768);
        assert!(params.attn_factor > 1.25);
    }
}
