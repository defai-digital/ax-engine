use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::backends::{llama_cpp, mlx_lm};
use crate::errors::ErrorResponse;
use crate::openai::generation::{
    run_openai_llama_cpp_chat_generation, run_openai_mlx_lm_chat_generation,
    run_openai_text_generation,
};
use crate::openai::requests::build_openai_chat_request_offloading_media;
use crate::openai::schema::{OpenAiChatCompletionHttpRequest, OpenAiStreamKind};
use crate::openai::validation::select_openai_model;

pub(crate) async fn openai_chat_completions(
    State(state): State<AppState>,
    payload: Result<Json<OpenAiChatCompletionHttpRequest>, JsonRejection>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    // Fold axum's JSON rejections into the documented 400 invalid_request
    // envelope; the default extractor answers with a plain-text 422 OpenAI
    // clients cannot parse (same pattern as openai::embeddings).
    let Json(request) = payload.map_err(|rejection| {
        crate::errors::error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("invalid request body: {}", rejection.body_text()),
        )
    })?;
    let live = select_openai_model(&state, request.model.as_deref())?;
    if request.skip_special_tokens.is_some() || request.vllm_xargs.is_some() {
        return Err(crate::errors::error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "CUDA runtime extensions moved to AX Serving and are not accepted by AX Engine"
                .to_string(),
        ));
    }
    if mlx_lm::is_selected(&live) {
        return run_openai_mlx_lm_chat_generation(state, live, request).await;
    }
    if llama_cpp::supports_server_chat(&live) {
        return run_openai_llama_cpp_chat_generation(state, live, request).await;
    }
    let request = build_openai_chat_request_offloading_media(&live, &state.media, request).await?;

    run_openai_text_generation(state, live, request, OpenAiStreamKind::ChatCompletion).await
}
