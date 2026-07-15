use super::*;

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

pub(crate) fn load_hf_config(model_dir: &Path) -> Result<serde_json::Value, ConvertError> {
    let config_path = model_dir.join("config.json");
    let bytes = fs::read(&config_path).map_err(|source| ConvertError::ReadFile {
        path: config_path.clone(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| ConvertError::ParseJson {
        path: config_path,
        source,
    })
}

pub(crate) fn resolve_model_type(config: &serde_json::Value) -> Result<String, ConvertError> {
    config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or(ConvertError::MissingConfigField {
            field: "model_type",
        })
}

pub(crate) struct ArchitectureParams {
    pub(crate) layer_count: u32,
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) attention_head_count: u32,
    pub(crate) attention_head_dim: u32,
    pub(crate) kv_head_count: u32,
    pub(crate) vocab_size: u32,
}

/// Whether this model type nests architecture params under `text_config`.
pub(crate) fn uses_text_config(model_type: &str) -> bool {
    matches!(
        model_type,
        "gemma4"
            | "gemma4_unified"
            | "gemma4_unified_text"
            | "gemma4_assistant"
            | "diffusion_gemma"
            | "llama4"
            | "qwen3_5"
            | "qwen3_5_moe"
            | "qwen3_5_text"
            | "qwen3_next"
            | "qwen3_6"
            | "qwen3.5"
            | "qwen3.6"
            | "mistral3"
    )
}

pub(crate) fn is_qwen3_5_family(model_type: &str) -> bool {
    matches!(
        model_type,
        "qwen3_5" | "qwen3.5" | "qwen3_5_moe" | "qwen3_5_text"
    )
}

pub(crate) fn is_qwen_gated_delta_family(model_type: &str) -> bool {
    is_qwen3_5_family(model_type) || matches!(model_type, "qwen3_next" | "qwen3_6" | "qwen3.6")
}

pub(crate) fn is_gemma4_target_model_type(model_type: &str) -> bool {
    matches!(
        model_type,
        "gemma4" | "gemma4_unified" | "gemma4_unified_text" | "diffusion_gemma"
    )
}

pub(crate) fn is_gemma4_text_model_type(model_type: &str) -> bool {
    is_gemma4_target_model_type(model_type) || model_type == "gemma4_assistant"
}

pub(crate) fn is_gemma4_unified_model_type(model_type: &str) -> bool {
    matches!(model_type, "gemma4_unified" | "gemma4_unified_text")
}

/// EmbeddingGemma (Gemma3 text backbone served as a bidirectional + mean-pooled
/// embedding model). HF `model_type` is `gemma3_text`; the config is flat (no
/// `text_config` wrapper), uses `rope_local_base_freq` for sliding layers, and
/// ships a `layer_types` array.
pub(crate) fn is_embeddinggemma_model_type(model_type: &str) -> bool {
    matches!(model_type, "gemma3_text" | "embeddinggemma")
}

pub(crate) fn is_qwen_family_model_type(model_type: &str) -> bool {
    model_type.starts_with("qwen3")
}

pub(crate) fn is_glm4_moe_lite(model_type: &str) -> bool {
    model_type == "glm4_moe_lite"
}

/// Parse diffusion-specific config fields from config.json.
///
/// DiffusionGemma may expose these at the top level, nested under a
/// `diffusion_config` key, under `text_config`, or under `generation_config`
/// / `sampler_config`. This helper checks all relevant locations.
pub(crate) fn parse_diffusion_config(
    config: &serde_json::Value,
    model_type: &str,
) -> NativeDiffusionConfig {
    if model_type != "diffusion_gemma" {
        return NativeDiffusionConfig::default();
    }

    let diffusion = config
        .get("diffusion_config")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

    let generation_config = config.get("generation_config");
    let top_level_sampler_config = config.get("sampler_config");
    let generation_sampler_config = generation_config.and_then(|gc| gc.get("sampler_config"));

    let get_u32 = |key: &str| -> Option<u32> {
        diffusion
            .get(key)
            .and_then(|v| v.as_u64())
            .and_then(u64_to_u32)
            .or_else(|| arch_u64(config, model_type, key).and_then(u64_to_u32))
    };

    // Helper for u32 fields that may also be under generation_config.
    let get_u32_gen = |key: &str| -> Option<u32> {
        get_u32(key).or_else(|| {
            generation_config
                .and_then(|gc| gc.get(key))
                .and_then(|v| v.as_u64())
                .and_then(u64_to_u32)
        })
    };

    // Helper for f32 fields that may also be under sampler_config or generation_config.
    let get_f32_sampler = |key: &str| -> Option<f32> {
        diffusion
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .or_else(|| arch_f64(config, model_type, key).map(|v| v as f32))
            .or_else(|| {
                top_level_sampler_config
                    .and_then(|sc| sc.get(key))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
            })
            .or_else(|| {
                generation_sampler_config
                    .and_then(|sc| sc.get(key))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
            })
            .or_else(|| {
                generation_config
                    .and_then(|gc| gc.get(key))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
            })
    };

    let get_bool = |key: &str| -> Option<bool> {
        diffusion
            .get(key)
            .and_then(|v| v.as_bool())
            .or_else(|| arch_bool(config, model_type, key))
    };

    // canvas_size: check both "canvas_size" and "canvas_length" (real config uses canvas_length).
    // Reject canvas_size=0 (malformed manifest) — treat as not specified.
    let canvas_size = get_u32("canvas_size")
        .or_else(|| get_u32("canvas_length"))
        .filter(|&v| v > 0);

    // Temperature schedule: real config uses t_max/t_min inside generation_config.
    let temperature_start =
        get_f32_sampler("temperature_start").or_else(|| get_f32_sampler("t_max"));
    let temperature_end = get_f32_sampler("temperature_end").or_else(|| get_f32_sampler("t_min"));

    // Convergence steps: real config uses stability_threshold (integer) inside generation_config.
    // Reject 0 — would cause instant convergence trigger in the denoise loop.
    let convergence_steps = get_u32("convergence_steps")
        .or_else(|| get_u32_gen("stability_threshold"))
        .filter(|&v| v > 0);

    NativeDiffusionConfig {
        canvas_size,
        // Reject max_denoise_steps=0 — a denoise loop with zero iterations
        // produces degenerate output.
        max_denoise_steps: get_u32_gen("max_denoise_steps")
            .or_else(|| get_u32_gen("max_denoising_steps"))
            .filter(|&v| v > 0),
        self_conditioning: get_bool("self_conditioning"),
        entropy_bound: get_f32_sampler("entropy_bound"),
        entropy_threshold: get_f32_sampler("entropy_threshold")
            .or_else(|| get_f32_sampler("confidence_threshold")),
        convergence_steps,
        temperature_start,
        temperature_end,
        convergence_check_interval: None,
        acceptance_rate_threshold: None,
        sampler: None,
        confidence_threshold: None,
    }
}

pub(crate) fn is_mla_family(model_type: &str) -> bool {
    is_glm4_moe_lite(model_type) || matches!(model_type, "deepseek_v3" | "deepseek_v32")
}

pub(crate) fn defaults_attn_output_gate(model_type: &str) -> bool {
    is_qwen3_5_family(model_type) || matches!(model_type, "qwen3_next" | "qwen3_6" | "qwen3.6")
}

pub(crate) fn default_moe_norm_topk_prob(model_type: &str) -> bool {
    is_qwen3_5_family(model_type)
}

pub(crate) fn runtime_status_for_model_type(_model_type: &str) -> NativeRuntimeStatus {
    NativeRuntimeStatus::default()
}

/// Get a u64 field, checking both the top-level config and a nested `text_config`
/// (Gemma4 and Qwen3.5+ nest architecture params under `text_config`).
pub(crate) fn arch_u64(config: &serde_json::Value, model_type: &str, field: &str) -> Option<u64> {
    config.get(field).and_then(|v| v.as_u64()).or_else(|| {
        if uses_text_config(model_type) {
            config
                .get("text_config")
                .and_then(|tc| tc.get(field))
                .and_then(|v| v.as_u64())
        } else {
            None
        }
    })
}

pub(crate) fn arch_bool(config: &serde_json::Value, model_type: &str, field: &str) -> Option<bool> {
    config.get(field).and_then(|v| v.as_bool()).or_else(|| {
        if uses_text_config(model_type) {
            config
                .get("text_config")
                .and_then(|tc| tc.get(field))
                .and_then(|v| v.as_bool())
        } else {
            None
        }
    })
}

pub(crate) fn arch_f64(config: &serde_json::Value, model_type: &str, field: &str) -> Option<f64> {
    config.get(field).and_then(|v| v.as_f64()).or_else(|| {
        if uses_text_config(model_type) {
            config
                .get("text_config")
                .and_then(|tc| tc.get(field))
                .and_then(|v| v.as_f64())
        } else {
            None
        }
    })
}

pub(crate) fn parse_rms_norm_eps(config: &serde_json::Value, model_type: &str) -> Option<f32> {
    ["rms_norm_eps", "rms_norm_epsilon", "layer_norm_epsilon"]
        .into_iter()
        .find_map(|field| arch_f64(config, model_type, field))
        .map(|value| value as f32)
}

pub(crate) fn linear_attention_config(
    config: &serde_json::Value,
    model_type: &str,
) -> NativeLinearAttentionConfig {
    if !is_qwen_gated_delta_family(model_type) {
        return NativeLinearAttentionConfig::default();
    }

    NativeLinearAttentionConfig {
        full_attention_interval: arch_u64(config, model_type, "full_attention_interval")
            .and_then(u64_to_u32),
        num_value_heads: arch_u64(config, model_type, "linear_num_value_heads")
            .and_then(u64_to_u32),
        num_key_heads: arch_u64(config, model_type, "linear_num_key_heads").and_then(u64_to_u32),
        key_head_dim: arch_u64(config, model_type, "linear_key_head_dim").and_then(u64_to_u32),
        value_head_dim: arch_u64(config, model_type, "linear_value_head_dim").and_then(u64_to_u32),
        conv_kernel_dim: arch_u64(config, model_type, "linear_conv_kernel_dim")
            .and_then(u64_to_u32),
    }
}

pub(crate) fn mla_attention_config(
    config: &serde_json::Value,
    model_type: &str,
) -> NativeMlaAttentionConfig {
    if !is_mla_family(model_type) {
        return NativeMlaAttentionConfig::default();
    }

    NativeMlaAttentionConfig {
        q_lora_rank: arch_u64(config, model_type, "q_lora_rank").and_then(u64_to_u32),
        kv_lora_rank: arch_u64(config, model_type, "kv_lora_rank").and_then(u64_to_u32),
        qk_nope_head_dim: arch_u64(config, model_type, "qk_nope_head_dim").and_then(u64_to_u32),
        qk_rope_head_dim: arch_u64(config, model_type, "qk_rope_head_dim").and_then(u64_to_u32),
        value_head_dim: arch_u64(config, model_type, "v_head_dim").and_then(u64_to_u32),
    }
}

pub(crate) fn glm_router_config(
    config: &serde_json::Value,
    model_type: &str,
) -> NativeGlmRouterConfig {
    if !is_glm4_moe_lite(model_type) {
        return NativeGlmRouterConfig::default();
    }

    NativeGlmRouterConfig {
        first_dense_layer_count: arch_u64(config, model_type, "first_k_dense_replace")
            .and_then(u64_to_u32),
        routed_scaling_factor: arch_f64(config, model_type, "routed_scaling_factor")
            .map(|value| value as f32),
        n_group: arch_u64(config, model_type, "n_group")
            .and_then(u64_to_u32)
            .or(Some(1)),
        topk_group: arch_u64(config, model_type, "topk_group")
            .and_then(u64_to_u32)
            .or(Some(1)),
        has_shared_experts: arch_u64(config, model_type, "n_shared_experts").unwrap_or(0) > 0,
    }
}

pub(crate) fn moe_config(config: &serde_json::Value, model_type: &str) -> NativeMoeConfig {
    let is_gemma4_moe = arch_bool(config, model_type, "enable_moe_block").unwrap_or(false);
    let is_diffusion_gemma_moe =
        model_type == "diffusion_gemma" && config_has_moe_experts(config, model_type);
    let is_qwen3_moe = matches!(model_type, "qwen3_moe" | "qwen3_5_moe" | "qwen3_5_text")
        || (is_qwen3_5_family(model_type) && config_has_moe_experts(config, model_type));
    let is_qwen3_next_moe = matches!(model_type, "qwen3_next" | "qwen3_6" | "qwen3.6");
    let is_glm_moe = is_glm4_moe_lite(model_type);
    let is_mixtral = model_type == "mixtral";
    let is_deepseek_v3 = matches!(model_type, "deepseek_v3" | "deepseek_v32");
    let is_llama4 = model_type == "llama4";
    // GPT-OSS is always MoE (num_local_experts + num_experts_per_tok).
    let is_gpt_oss = model_type == "gpt_oss";
    if !is_gemma4_moe
        && !is_diffusion_gemma_moe
        && !is_qwen3_moe
        && !is_qwen3_next_moe
        && !is_glm_moe
        && !is_mixtral
        && !is_deepseek_v3
        && !is_llama4
        && !is_gpt_oss
    {
        return NativeMoeConfig::default();
    }

    let expert_count = arch_u64(config, model_type, "num_experts")
        .or_else(|| arch_u64(config, model_type, "num_local_experts"))
        .or_else(|| arch_u64(config, model_type, "n_routed_experts"))
        .and_then(u64_to_u32);
    let experts_per_token = arch_u64(config, model_type, "top_k_experts")
        .or_else(|| arch_u64(config, model_type, "num_experts_per_tok"))
        .or_else(|| arch_u64(config, model_type, "experts_per_token"))
        .and_then(u64_to_u32);

    let layer_freq = if is_deepseek_v3 {
        arch_u64(config, model_type, "moe_layer_freq").and_then(u64_to_u32)
    } else if is_llama4 {
        arch_u64(config, model_type, "interleave_moe_layer_step").and_then(u64_to_u32)
    } else {
        None
    };

    let first_dense_layers = if is_deepseek_v3 {
        arch_u64(config, model_type, "first_k_dense_replace").and_then(u64_to_u32)
    } else {
        None
    };

    let shared_expert_count = if is_deepseek_v3 {
        arch_u64(config, model_type, "n_shared_experts").and_then(u64_to_u32)
    } else if is_llama4 {
        // LLaMA 4 always has 1 shared expert when MoE is active
        Some(1)
    } else {
        None
    };

    // GPT-OSS and Llama 4 expose dense `intermediate_size` for the expert FFN
    // width rather than a separate `moe_intermediate_size`.
    let expert_intermediate_size = arch_u64(config, model_type, "moe_intermediate_size")
        .or_else(|| {
            if is_gpt_oss || is_llama4 {
                arch_u64(config, model_type, "intermediate_size")
            } else {
                None
            }
        })
        .and_then(u64_to_u32);

    NativeMoeConfig {
        expert_count,
        experts_per_token,
        expert_intermediate_size,
        layer_freq,
        first_dense_layers,
        shared_expert_count,
        sigmoid_routing: is_deepseek_v3,
        routed_scaling_factor: if is_deepseek_v3 {
            arch_f64(config, model_type, "routed_scaling_factor").map(|v| v as f32)
        } else {
            None
        },
        n_group: if is_deepseek_v3 {
            arch_u64(config, model_type, "n_group").and_then(u64_to_u32)
        } else {
            None
        },
        topk_group: if is_deepseek_v3 {
            arch_u64(config, model_type, "topk_group").and_then(u64_to_u32)
        } else {
            None
        },
    }
}

/// Parse per-model rope theta (main + SWA) and partial_rotary_factor.
///
/// Returns `(rope_theta, rope_theta_swa, partial_rotary_factor)`.
/// Gemma4 stores these nested under `text_config.rope_parameters.{full,sliding}_attention`.
/// Qwen3.5/Next stores them flat inside `text_config.rope_parameters`.
///
/// The gemma4 assistant drafter (`gemma4_assistant`) carries the identical
/// nested `rope_parameters` layout and must share the target's RoPE geometry,
/// so it takes the same parsing branch — otherwise `rope_theta` / `rope_theta_swa`
/// / `partial_rotary_factor` fall through to None and the drafter's Q rotation
/// stops matching the target's cached K.
pub(crate) fn parse_rope_params(
    config: &serde_json::Value,
    model_type: &str,
) -> (Option<u32>, Option<u32>, Option<f32>) {
    if is_gemma4_text_model_type(model_type) {
        let rp = config
            .get("text_config")
            .and_then(|tc| tc.get("rope_parameters"));
        let full_theta = rp
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| arch_f64(config, model_type, "rope_theta"))
            .and_then(f64_to_u32);
        let sliding_theta = rp
            .and_then(|rp| rp.get("sliding_attention"))
            .and_then(|sa| sa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| arch_f64(config, model_type, "rope_theta"))
            .and_then(f64_to_u32);
        let partial_rotary = rp
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .or_else(|| arch_f64(config, model_type, "partial_rotary_factor"))
            .map(|v| v as f32)
            .filter(|&v| v <= 1.0);
        return (full_theta, sliding_theta, partial_rotary);
    }

    if is_embeddinggemma_model_type(model_type) {
        // Gemma3 dual-RoPE: global (full-attention) layers use `rope_theta`;
        // sliding layers use `rope_local_base_freq`. Stored as rope_theta /
        // rope_theta_swa respectively (same convention as gemma4).
        let full_theta = arch_f64(config, model_type, "rope_theta").and_then(f64_to_u32);
        let sliding_theta = arch_f64(config, model_type, "rope_local_base_freq")
            .or_else(|| arch_f64(config, model_type, "rope_theta"))
            .and_then(f64_to_u32);
        return (full_theta, sliding_theta, None);
    }

    let theta = arch_f64(config, model_type, "rope_theta")
        .or_else(|| {
            let text_config = config.get("text_config")?;
            text_config
                .get("rope_parameters")
                .and_then(|rp| rp.get("rope_theta"))
                .and_then(|v| v.as_f64())
        })
        .and_then(f64_to_u32);

    let partial_rotary = arch_f64(config, model_type, "partial_rotary_factor")
        .or_else(|| {
            let text_config = config.get("text_config")?;
            text_config
                .get("rope_parameters")
                .and_then(|rp| rp.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
        })
        .map(|v| v as f32)
        .filter(|&v| v <= 1.0);

    (theta, None, partial_rotary)
}

/// Parse LLaMA 3 / LLaMA 4 rope_scaling dict from config.json.
#[allow(clippy::type_complexity)]
pub(crate) fn parse_rope_scaling(
    config: &serde_json::Value,
    model_type: &str,
) -> (
    Option<String>,
    Option<f32>,
    Option<f32>,
    Option<f32>,
    Option<u32>,
) {
    let rs = if uses_text_config(model_type) {
        config
            .get("text_config")
            .and_then(|tc| tc.get("rope_scaling"))
    } else {
        config.get("rope_scaling")
    };
    let Some(rs) = rs else {
        return (None, None, None, None, None);
    };
    let rope_type = rs
        .get("rope_type")
        .or_else(|| rs.get("type"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let factor = rs.get("factor").and_then(|v| v.as_f64()).map(|f| f as f32);
    let low_freq_factor = rs
        .get("low_freq_factor")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let high_freq_factor = rs
        .get("high_freq_factor")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let original_context_len = rs
        .get("original_max_position_embeddings")
        .and_then(|v| v.as_u64())
        .and_then(u64_to_u32);
    (
        rope_type,
        factor,
        low_freq_factor,
        high_freq_factor,
        original_context_len,
    )
}

/// Parse per-layer type list from config (e.g. Gemma4 / GPT-OSS `layer_types`).
///
/// - **Gemma4 / EmbeddingGemma:** use `layer_types` when present; otherwise
///   derive from `sliding_window_pattern` (default period 5).
/// - **GPT-OSS:** use `layer_types` when present (openai ships the full list);
///   otherwise alternate `sliding_attention` / `full_attention` matching
///   mlx-lm `gpt_oss.GptOssMoeModel` defaults.
pub(crate) fn parse_layer_types(
    config: &serde_json::Value,
    model_type: &str,
    layer_count: u32,
) -> Vec<String> {
    let is_gpt_oss = model_type == "gpt_oss";
    if !is_gemma4_text_model_type(model_type)
        && !is_embeddinggemma_model_type(model_type)
        && !is_gpt_oss
    {
        return Vec::new();
    }
    if let Some(layer_types) = config
        .get("layer_types")
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("layer_types"))
        })
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| {
                    v.as_str()
                        .map(String::from)
                        .unwrap_or_else(|| v.to_string())
                })
                .collect()
        })
    {
        return layer_types;
    }

    if is_gpt_oss {
        // mlx-lm: ["sliding_attention", "full_attention"] * (n_layers // 2)
        return (0..layer_count)
            .map(|i| {
                if i % 2 == 0 {
                    "sliding_attention".to_string()
                } else {
                    "full_attention".to_string()
                }
            })
            .collect();
    }

    let pattern = arch_u64(config, model_type, "sliding_window_pattern")
        .filter(|&pattern| pattern > 0)
        .unwrap_or(5);
    (0..layer_count)
        .map(|i| {
            if (u64::from(i) + 1).is_multiple_of(pattern) {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect()
}

/// Compute the KV-source mapping for KV-shared layers (Gemma4).
///
/// The last `num_kv_shared_layers` layers reuse K/V from the last non-shared
/// layer of the same type (sliding vs. full).
pub(crate) fn compute_kv_shared_sources(
    config: &serde_json::Value,
    model_type: &str,
    layer_types: &[String],
    layer_count: u32,
) -> BTreeMap<u32, u32> {
    if !is_gemma4_target_model_type(model_type) || layer_types.is_empty() {
        return BTreeMap::new();
    }
    let default_shared_layers = if model_type == "gemma4" { 20 } else { 0 };
    let num_shared = arch_u64(config, model_type, "num_kv_shared_layers")
        .unwrap_or(default_shared_layers)
        .min(u64::from(layer_count)) as usize;
    let non_shared_count = layer_count as usize - num_shared;

    let mut last_full: Option<u32> = None;
    let mut last_sliding: Option<u32> = None;
    let mut sources = BTreeMap::new();

    for (i, lt) in layer_types.iter().enumerate() {
        if i < non_shared_count {
            if lt == "full_attention" {
                last_full = Some(i as u32);
            } else {
                last_sliding = Some(i as u32);
            }
        } else if let Some(src) = if lt == "full_attention" {
            last_full
        } else {
            last_sliding
        } {
            sources.insert(i as u32, src);
        }
    }
    sources
}

pub(crate) fn compute_attention_value_from_key_layers(
    config: &serde_json::Value,
    model_type: &str,
    layer_types: &[String],
    kv_shared_source_layers: &BTreeMap<u32, u32>,
    layer_count: u32,
) -> Vec<u32> {
    // The reference config dataclasses default this field differently per model
    // type: standard gemma4 (27B) defaults False, while gemma4_unified (12B)
    // and diffusion_gemma default True (full attention layers share V from K).
    // AX reads config.json directly without a dataclass to supply
    // the default, so we mirror the per-type default when the field is absent.
    let default_k_eq_v =
        is_gemma4_unified_model_type(model_type) || model_type == "diffusion_gemma";
    if !is_gemma4_target_model_type(model_type)
        || !arch_bool(config, model_type, "attention_k_eq_v").unwrap_or(default_k_eq_v)
    {
        return Vec::new();
    }

    (0..layer_count)
        .filter(|&i| {
            layer_types
                .get(i as usize)
                .is_some_and(|layer_type| layer_type == "full_attention")
                && !kv_shared_source_layers.contains_key(&i)
        })
        .collect()
}

pub(crate) fn f64_to_u32(value: f64) -> Option<u32> {
    if value.is_finite() && value >= 0.0 && value <= f64::from(u32::MAX) {
        Some(value as u32)
    } else {
        None
    }
}

pub(crate) fn u64_to_u32(value: u64) -> Option<u32> {
    u32::try_from(value).ok()
}

pub(crate) fn require_arch_u64(
    config: &serde_json::Value,
    model_type: &str,
    field: &'static str,
) -> Result<u64, ConvertError> {
    arch_u64(config, model_type, field).ok_or(ConvertError::MissingConfigField { field })
}

pub(crate) fn resolve_architecture(
    config: &serde_json::Value,
    model_type: &str,
) -> Result<ArchitectureParams, ConvertError> {
    let hidden_size = require_arch_u64(config, model_type, "hidden_size")? as u32;
    let attention_head_count = require_arch_u64(config, model_type, "num_attention_heads")? as u32;
    let kv_head_count = arch_u64(config, model_type, "num_key_value_heads")
        .map(|v| v as u32)
        .unwrap_or(attention_head_count);
    let attention_head_dim = if is_mla_family(model_type) {
        let qk_nope = require_arch_u64(config, model_type, "qk_nope_head_dim")?;
        let qk_rope = require_arch_u64(config, model_type, "qk_rope_head_dim")?;
        (qk_nope + qk_rope) as u32
    } else {
        arch_u64(config, model_type, "head_dim")
            .map(|v| v as u32)
            .unwrap_or_else(|| hidden_size.checked_div(attention_head_count).unwrap_or(0))
    };
    let layer_count = require_arch_u64(config, model_type, "num_hidden_layers")? as u32;
    let vocab_size = require_arch_u64(config, model_type, "vocab_size")? as u32;
    let intermediate_size = arch_u64(config, model_type, "intermediate_size")
        .map(|v| v as u32)
        .unwrap_or(0);

    Ok(ArchitectureParams {
        layer_count,
        hidden_size,
        intermediate_size,
        attention_head_count,
        attention_head_dim,
        kv_head_count,
        vocab_size,
    })
}
