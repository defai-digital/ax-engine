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
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// No scaling (standard RoPE).
    #[default]
    None,
    /// Linear scaling: position = position / factor.
    Linear(f32),
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
    /// Number of experts in MoE models (Mixtral). None for dense models.
    pub n_expert: Option<u32>,
    /// Number of experts used per token in MoE routing. None for dense models.
    pub n_expert_used: Option<u32>,
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
        let is_falcon = matches!(arch.as_str(), "falcon");
        let is_starcoder2 = matches!(arch.as_str(), "starcoder2");
        let gate_activation = if is_gemma || is_falcon || is_starcoder2 {
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

        // MoE: expert count and experts-per-token (Mixtral)
        let n_expert = header.get_u32(&format!("{arch}.expert_count"));
        let n_expert_used = header.get_u32(&format!("{arch}.expert_used_count"));

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
        })
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
}
