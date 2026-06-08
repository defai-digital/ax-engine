use ax_engine_sdk::{GenerateRequest, GenerateSampling, RequestMultimodalInputs};
use serde::Deserialize;

use crate::app_state::AppState;
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
    #[serde(default)]
    sampling: Option<GenerateSampling>,
    #[serde(default)]
    multimodal_inputs: RequestMultimodalInputs,
    #[serde(default)]
    metadata: Option<String>,
}

pub(crate) fn build_generate_request(
    state: &AppState,
    request: GenerateHttpRequest,
) -> GenerateRequest {
    build_generate_request_internal(
        state,
        GenerateRequestParts {
            input_tokens: request.input_tokens,
            input_text: request.input_text,
            multimodal_inputs: request.multimodal_inputs,
            max_output_tokens: request.max_output_tokens.unwrap_or(256),
            sampling: request.sampling.unwrap_or_default(),
            stop_sequences: Vec::new(),
            metadata: request.metadata,
        },
    )
}

#[cfg(test)]
mod tests {
    use ax_engine_sdk::{Gemma4UnifiedModality, Gemma4UnifiedTokenSpan};
    use serde_json::json;

    use super::*;

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
