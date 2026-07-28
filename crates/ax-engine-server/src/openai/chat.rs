use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::backends::{edge_llm, llama_cpp, mlx_lm, vllm};
use crate::errors::ErrorResponse;
use crate::openai::generation::{
    run_openai_edge_llm_chat_generation, run_openai_llama_cpp_chat_generation,
    run_openai_mlx_lm_chat_generation, run_openai_text_generation, run_openai_vllm_chat_generation,
};
use crate::openai::requests::build_openai_chat_request_offloading_media;
use crate::openai::schema::{OpenAiChatCompletionHttpRequest, OpenAiStreamKind};
use crate::openai::validation::select_openai_model;

pub(crate) async fn openai_chat_completions(
    State(state): State<AppState>,
    Json(request): Json<OpenAiChatCompletionHttpRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let live = select_openai_model(&state, request.model.as_deref())?;
    if vllm::is_selected(&live) {
        return run_openai_vllm_chat_generation(state, live, request).await;
    }
    if request.skip_special_tokens.is_some() || request.vllm_xargs.is_some() {
        return Err(crate::errors::error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "skip_special_tokens and vllm_xargs require the vLLM backend".to_string(),
        ));
    }
    if mlx_lm::is_selected(&live) {
        return run_openai_mlx_lm_chat_generation(state, live, request).await;
    }
    if edge_llm::is_selected(&live) {
        return run_openai_edge_llm_chat_generation(state, live, request).await;
    }
    if llama_cpp::supports_server_chat(&live) {
        return run_openai_llama_cpp_chat_generation(state, live, request).await;
    }
    let request = build_openai_chat_request_offloading_media(&live, request).await?;

    run_openai_text_generation(state, live, request, OpenAiStreamKind::ChatCompletion).await
}
