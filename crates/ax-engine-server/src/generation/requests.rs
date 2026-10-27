use ax_engine_sdk::{GenerateRequest, GenerateSampling, RequestMultimodalInputs};
use axum::Json;
use axum::http::StatusCode;
use serde::Deserialize;

use crate::app_state::LiveState;
use crate::errors::{ErrorResponse, error_response};
use crate::openai::requests::{GenerateRequestParts, build_generate_request_internal};

#[derive(Debug, Deserialize)]
pub(crate) struct GenerateHttpRequest {
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    input_tokens: Vec<u32>,
    #[serde(default)]
    input_text: Option<String>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_sampling")]
    sampling: Option<GenerateSampling>,
    #[serde(default)]
    multimodal_inputs: RequestMultimodalInputs,
    #[serde(default)]
    metadata: Option<String>,
}

/// A JSON `seed` key is an explicit seed, including `0`: SDKs send
/// `"seed": 0` and cannot set the internal `seed_explicit` flag.
fn deserialize_sampling<'de, D>(deserializer: D) -> Result<Option<GenerateSampling>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(value) = Option::<serde_json::Value>::deserialize(deserializer)? else {
        return Ok(None);
    };
    let seed_present = value.get("seed").is_some_and(|seed| !seed.is_null());
    let mut sampling: GenerateSampling =
        serde_json::from_value(value).map_err(serde::de::Error::custom)?;
    sampling.seed_explicit |= seed_present;
    Ok(Some(sampling))
}

impl GenerateHttpRequest {
    pub(crate) fn reject_video_inputs(
        &self,
        live: &LiveState,
    ) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
        let has_video = self
            .multimodal_inputs
            .gemma4_unified
            .as_ref()
            .is_some_and(|inputs| !inputs.videos.is_empty());
        if !has_video {
            return Ok(());
        }
        if crate::metadata::model_supports_video(live) {
            return Ok(());
        }
        Err(error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_modality",
            "video input is not supported for this model; requires gemma4_unified vision roles, \
no convert-time media drops, and AX_MLX_GEMMA4_VIDEO enabled (data URI only)"
                .to_string(),
        ))
    }
}

pub(crate) fn build_generate_request(
    live: &LiveState,
    request: GenerateHttpRequest,
) -> GenerateRequest {
    build_generate_request_internal(
        live,
        GenerateRequestParts {
            input_tokens: request.input_tokens,
            input_text: request.input_text,
            multimodal_inputs: request.multimodal_inputs,
            max_output_tokens: request.max_output_tokens.unwrap_or(256),
            sampling: request.sampling.unwrap_or_default(),
            stop_sequences: Vec::new(),
            metadata: request.metadata,
            // The raw generate endpoint takes caller-supplied token ids, so
            // there is no rendered prompt to derive semantic cut points from.
            prefill_hints: Default::default(),
        },
    )
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use ax_engine_sdk::{Gemma4UnifiedModality, Gemma4UnifiedTokenSpan};
    use serde_json::json;

    use super::*;

    #[test]
    fn json_seed_zero_is_an_explicit_seed() {
        let parse = |body: serde_json::Value| {
            serde_json::from_value::<GenerateHttpRequest>(body)
                .expect("request")
                .sampling
                .expect("sampling")
        };
        let zero = parse(serde_json::json!({"sampling": {"temperature": 0.8, "seed": 0}}));
        assert_eq!(zero.seed, 0);
        assert!(zero.seed_explicit, "seed 0 must be reproducible");
        let absent = parse(serde_json::json!({"sampling": {"temperature": 0.8}}));
        assert!(!absent.seed_explicit);
        let seven = parse(serde_json::json!({"sampling": {"seed": 7}}));
        assert_eq!(seven.seed, 7);
        assert!(
            serde_json::from_value::<GenerateHttpRequest>(
                serde_json::json!({"sampling": {"seed": -1}})
            )
            .is_err()
        );
    }

    #[test]
    fn generate_http_request_deserializes_gemma4_processed_multimodal_inputs() {
        let request: GenerateHttpRequest = serde_json::from_value(json!({
            "model": "gemma-4-12b-it",
            "input_tokens": [10, 258880, 11],
            "max_output_tokens": 1,
            "multimodal_inputs": {
                "gemma4_unified": {
                    "images": [{
                        "span": {
                            "modality": "image",
                            "placeholder_index": 1,
                            "replacement_start": 1,
                            "soft_token_count": 1,
                            "replacement_token_count": 3
                        },
                        "pixel_values": [0.0, 1.0, 2.0],
                        "pixel_position_ids": [[0, 0]]
                    }],
                    "audios": [],
                    "videos": []
                }
            }
        }))
        .expect("native generate request should accept processed Gemma4 inputs");

        let inputs = request
            .multimodal_inputs
            .gemma4_unified
            .expect("Gemma4 inputs should deserialize");
        assert_eq!(inputs.images.len(), 1);
        assert_eq!(
            inputs.images[0].span,
            Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Image,
                placeholder_index: 1,
                replacement_start: 1,
                soft_token_count: 1,
                replacement_token_count: 3,
            }
        );
        assert_eq!(inputs.images[0].pixel_position_ids, vec![[0, 0]]);
    }
}
