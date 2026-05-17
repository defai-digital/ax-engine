use ax_engine_sdk::{GenerateRequest, GenerateSampling};
use serde::Deserialize;

use crate::app_state::AppState;
use crate::openai::requests::build_generate_request_internal;

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
    metadata: Option<String>,
}

pub(crate) fn build_generate_request(
    state: &AppState,
    request: GenerateHttpRequest,
) -> GenerateRequest {
    build_generate_request_internal(
        state,
        request.input_tokens,
        request.input_text,
        request.max_output_tokens.unwrap_or(256),
        request.sampling.unwrap_or_default(),
        Vec::new(),
        request.metadata,
    )
}
