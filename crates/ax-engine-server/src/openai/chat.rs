use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::backends::{llama_cpp, mlx_lm};
use crate::errors::ErrorResponse;
use crate::openai::generation::{
    run_openai_llama_cpp_chat_generation, run_openai_mlx_lm_chat_generation,
    run_openai_text_generation,
};
use crate::openai::requests::build_openai_chat_request;
use crate::openai::schema::{OpenAiChatCompletionHttpRequest, OpenAiStreamKind};
use crate::openai::validation::validate_openai_request;

pub(crate) async fn openai_chat_completions(
    State(state): State<AppState>,
    Json(request): Json<OpenAiChatCompletionHttpRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    validate_openai_request(&state, request.model.as_deref())?;
    if mlx_lm::is_selected(&state) {
        return run_openai_mlx_lm_chat_generation(state, request).await;
    }
    if llama_cpp::supports_server_chat(&state) {
        return run_openai_llama_cpp_chat_generation(state, request).await;
    }
    let request = build_openai_chat_request(&state, request)?;

    run_openai_text_generation(state, request, OpenAiStreamKind::ChatCompletion).await
}
