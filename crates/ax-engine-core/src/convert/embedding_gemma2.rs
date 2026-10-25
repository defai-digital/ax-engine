//! Text-encoder contract for the published EmbeddingGemma 2 packs.

use super::*;

pub(super) fn prepare_config(
    config: &mut serde_json::Value,
    model_type: &str,
) -> Result<(), ConvertError> {
    for (field, expected) in [
        ("image_token_id", 258880),
        ("audio_token_id", 258881),
        ("video_token_id", 258884),
    ] {
        if config
            .get(field)
            .and_then(|v| v.as_u64())
            .is_some_and(|id| id != expected)
        {
            return invalid_model_contract(
                model_type,
                format!("unsupported {field}; text-only media-token rejection requires {expected}"),
            );
        }
    }
    let text = config.get("text_config").unwrap_or(config);
    if text
        .get("hidden_activation")
        .and_then(|v| v.as_str())
        .is_some_and(|name| name != "gelu_pytorch_tanh")
        || text
            .get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .is_some_and(|factor| factor != 1.0)
    {
        return invalid_model_contract(
            model_type,
            "encoder requires approximate GELU and full rotary dimensions",
        );
    }
    if let Some(rope) = text.get("rope_parameters").and_then(|v| v.as_object()) {
        for params in rope.values() {
            if params
                .get("rope_type")
                .and_then(|v| v.as_str())
                .is_some_and(|kind| kind != "default")
                || params
                    .get("partial_rotary_factor")
                    .and_then(|v| v.as_f64())
                    .is_some_and(|factor| factor != 1.0)
            {
                return invalid_model_contract(
                    model_type,
                    "encoder supports default full-dimensional rotary embeddings only",
                );
            }
        }
    }
    let layers = text
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let heads = text.get("num_attention_heads").and_then(|v| v.as_u64());
    let types = text.get("layer_types").and_then(|v| v.as_array());
    let Some(types) = types.filter(|types| types.len() == layers as usize && layers > 0) else {
        return invalid_model_contract(model_type, "layer_types must describe every encoder layer");
    };
    if types.last().and_then(|v| v.as_str()) != Some("full_attention") {
        return invalid_model_contract(
            model_type,
            "the final encoder layer must use full_attention",
        );
    }
    let overrides = text.get("per_layer_config").and_then(|v| v.as_object());
    let mut global = None;
    for (index, kind) in types.iter().enumerate() {
        let name = format!("{index:02}");
        let override_ = overrides.and_then(|o| o.get(&name).or_else(|| o.get(&index.to_string())));
        let dim = override_
            .and_then(|o| o.get("head_dim"))
            .or_else(|| text.get("head_dim"))
            .and_then(|v| v.as_u64());
        let kv = override_
            .and_then(|o| o.get("num_key_value_heads"))
            .or_else(|| text.get("num_key_value_heads"))
            .and_then(|v| v.as_u64());
        if override_
            .and_then(|o| o.get("num_attention_heads"))
            .and_then(|v| v.as_u64())
            .is_some_and(|n| Some(n) != heads)
        {
            return invalid_model_contract(
                model_type,
                "per-layer query head overrides are unsupported",
            );
        }
        match kind.as_str() {
            Some("full_attention") => {
                let pair = (dim, kv);
                if global.is_some_and(|previous| previous != pair) {
                    return invalid_model_contract(
                        model_type,
                        "full-attention layers must share head geometry",
                    );
                }
                global = Some(pair);
            }
            Some("sliding_attention")
                if dim == text.get("head_dim").and_then(|v| v.as_u64())
                    && kv == text.get("num_key_value_heads").and_then(|v| v.as_u64()) => {}
            _ => {
                return invalid_model_contract(
                    model_type,
                    "unsupported encoder layer type or sliding head override",
                );
            }
        }
    }
    let Some((Some(dim), Some(kv))) = global else {
        return invalid_model_contract(model_type, "missing full-attention head geometry");
    };
    if text
        .get("attention_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return invalid_model_contract(model_type, "attention biases are unsupported");
    }
    config["global_head_dim"] = dim.into();
    config["num_global_key_value_heads"] = kv.into();
    Ok(())
}

pub(super) fn match_tensor(name: &str) -> Option<(NativeTensorRole, Option<u32>)> {
    let name = name.strip_prefix("model.").unwrap_or(name);
    let text = name.strip_prefix("language_model.")?;
    let role = match text {
        "embed_tokens.weight" => NativeTensorRole::TokenEmbedding,
        "norm.weight" => NativeTensorRole::FinalNorm,
        "embedding_projection.weight" => NativeTensorRole::EmbeddingDense0,
        "ple.per_layer_model_projection.weight" => NativeTensorRole::PerLayerModelProjection,
        "ple.per_layer_projection_norm.weight" => NativeTensorRole::PerLayerProjectionNorm,
        _ => {
            let (index, suffix) = text.strip_prefix("layers.")?.split_once('.')?;
            let suffix = suffix.strip_prefix("ple_block.").unwrap_or(suffix);
            let (_, TensorMapping::PerLayer(role)) =
                HF_STANDARD_TENSOR_MAP.iter().find(|(n, _)| *n == suffix)?
            else {
                return None;
            };
            return Some((*role, Some(index.parse().ok()?)));
        }
    };
    Some((role, None))
}
