use super::*;

pub(crate) struct ModelFamily {
    pub(crate) family_name: &'static str,
    pub(crate) tensor_map: &'static [(&'static str, TensorMapping)],
    pub(crate) extra_tensor_map: Option<&'static [(&'static str, TensorMapping)]>,
    pub(crate) uses_language_model_prefix: bool,
    /// DiffusionGemma stores tensors under `model.decoder.*` instead of `model.*`.
    pub(crate) uses_decoder_prefix: bool,
}

pub(crate) fn config_has_moe_experts(config: &serde_json::Value, model_type: &str) -> bool {
    arch_u64(config, model_type, "num_experts")
        .or_else(|| arch_u64(config, model_type, "num_local_experts"))
        .or_else(|| arch_u64(config, model_type, "n_routed_experts"))
        .is_some_and(|n| n > 0)
}

pub(crate) fn model_family_for_type(
    model_type: &str,
    config: &serde_json::Value,
) -> Result<ModelFamily, ConvertError> {
    match model_type {
        "qwen3" => Ok(ModelFamily {
            family_name: "qwen3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: None,
            uses_language_model_prefix: false,
            uses_decoder_prefix: false,
        }),
        "qwen3_5" | "qwen3.5" | "qwen3_5_moe" | "qwen3_5_text" => Ok(ModelFamily {
            family_name: "qwen3_5",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: if matches!(model_type, "qwen3_5_moe" | "qwen3_5_text")
                || config_has_moe_experts(config, model_type)
            {
                Some(QWEN3_MOE_EXTRA_TENSOR_MAP)
            } else {
                None
            },
            uses_language_model_prefix: true,
            uses_decoder_prefix: false,
        }),
        "qwen3_next" | "qwen3.6" | "qwen3_6" => Ok(ModelFamily {
            family_name: "qwen3_next",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(QWEN3_MOE_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: true,
            uses_decoder_prefix: false,
        }),
        "qwen3_moe" => Ok(ModelFamily {
            family_name: "qwen3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(QWEN3_MOE_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
            uses_decoder_prefix: false,
        }),
        "gemma4" | "gemma4_unified" | "gemma4_unified_text" => Ok(ModelFamily {
            family_name: "gemma4",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: if matches!(model_type, "gemma4_unified" | "gemma4_unified_text") {
                Some(GEMMA4_UNIFIED_EXTRA_TENSOR_MAP)
            } else {
                None
            },
            uses_language_model_prefix: true,
            uses_decoder_prefix: false,
        }),
        "gemma4_assistant" => Ok(ModelFamily {
            family_name: "gemma4_assistant",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(GEMMA4_ASSISTANT_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
            uses_decoder_prefix: false,
        }),
        // EmbeddingGemma-300m: Gemma3 text backbone (model.layers.* standard map)
        // with a sentence-transformers Dense head (dense.0/dense.1). Served as a
        // bidirectional encoder + mean pooling embedding model.
        "gemma3_text" | "embeddinggemma" => Ok(ModelFamily {
            family_name: "embeddinggemma",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(EMBEDDINGGEMMA_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
            uses_decoder_prefix: false,
        }),
        "diffusion_gemma" => Ok(ModelFamily {
            family_name: "diffusion_gemma",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: None,
            uses_language_model_prefix: true,
            uses_decoder_prefix: true,
        }),
        "glm4_moe_lite" => Ok(ModelFamily {
            family_name: "glm4_moe_lite",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(GLM4_MOE_LITE_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
            uses_decoder_prefix: false,
        }),
        "llama" => Ok(ModelFamily {
            family_name: "llama3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: None,
            uses_language_model_prefix: false,
            uses_decoder_prefix: false,
        }),
        // Classic Mistral dense (Ministral, Devstral, Nemo) uses model.layers.*.
        // Mistral 3 multimodal text towers (Mistral Small 3.x) use language_model.model.*.
        // Enable both maps via uses_language_model_prefix; standard map is always tried first.
        "mistral" | "mistral3" | "ministral3" => Ok(ModelFamily {
            family_name: "mistral3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: None,
            uses_language_model_prefix: true,
            uses_decoder_prefix: false,
        }),
        "mixtral" => Ok(ModelFamily {
            family_name: "mixtral",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(MIXTRAL_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
            uses_decoder_prefix: false,
        }),
        "deepseek_v3" | "deepseek_v32" => Ok(ModelFamily {
            family_name: "deepseek_v3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(DEEPSEEK_V3_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
            uses_decoder_prefix: false,
        }),
        "llama4" => Ok(ModelFamily {
            family_name: "llama4",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(LLAMA4_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: true,
            uses_decoder_prefix: false,
        }),
        "gpt_oss" => Ok(ModelFamily {
            family_name: "gpt_oss",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(GPT_OSS_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
            uses_decoder_prefix: false,
        }),
        // Nemotron-H hybrid Mamba-2 + GQA + ReLU² MoE (Nemotron 3 Nano, …).
        // Weights live under `backbone.layers.*.mixer.*` (see NEMOTRON_H_TENSOR_MAP).
        "nemotron_h" => Ok(ModelFamily {
            family_name: "nemotron_h",
            tensor_map: NEMOTRON_H_TENSOR_MAP,
            extra_tensor_map: None,
            uses_language_model_prefix: false,
            uses_decoder_prefix: false,
        }),
        // Unlimited-OCR / DeepSeek-OCR: language_model.* MoE SWA tower + vision/sam/projector.
        "unlimited_ocr" | "unlimited-ocr" | "deepseekocr" => Ok(ModelFamily {
            family_name: "unlimited_ocr",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(UNLIMITED_OCR_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: true,
            uses_decoder_prefix: false,
        }),
        other => Err(ConvertError::UnsupportedModelType {
            model_type: other.to_string(),
        }),
    }
}
