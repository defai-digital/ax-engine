use ax_engine_sdk::{
    EngineSessionError, run_blocking_chat_generate, run_blocking_llama_cpp_chat_generate,
};
use axum::Json;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::errors::{ErrorResponse, map_session_error};
use crate::generation::native::run_stateless_generate_request;
use crate::openai::requests::{
    OpenAiBuiltRequest, build_openai_llama_cpp_chat_request, build_openai_mlx_lm_chat_request,
};
use crate::openai::schema::{OpenAiChatCompletionHttpRequest, OpenAiStreamKind};
use crate::openai::streaming::{
    stream_openai_llama_cpp_chat_request, stream_openai_mlx_lm_chat_request, stream_openai_request,
};
use crate::tasks::run_blocking_session_task;

pub(crate) async fn run_openai_llama_cpp_chat_generation(
    state: AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let request = build_openai_llama_cpp_chat_request(&state, request)?;
    if request.stream {
        return stream_openai_llama_cpp_chat_request(state, request.chat_request).await;
    }

    let request_id = state.allocate_request_id();
    let runtime = state.runtime_report.clone();
    let llama_backend = state
        .session_config
        .llama_backend
        .clone()
        .ok_or(EngineSessionError::MissingLlamaCppConfig {
            selected_backend: state.runtime_report.selected_backend,
        })
        .map_err(map_session_error)?;
    let response = run_blocking_session_task(move || {
        run_blocking_llama_cpp_chat_generate(
            request_id,
            &runtime,
            &llama_backend,
            &request.chat_request,
        )
        .map_err(EngineSessionError::from)
    })
    .await?;

    Ok(OpenAiStreamKind::ChatCompletion.build_non_stream_response(&response, request_id))
}

pub(crate) async fn run_openai_mlx_lm_chat_generation(
    state: AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let request = build_openai_mlx_lm_chat_request(&state, request)?;
    if request.stream {
        return stream_openai_mlx_lm_chat_request(state, request.chat_request).await;
    }

    let request_id = state.allocate_request_id();
    let runtime = state.runtime_report.clone();
    let mlx_lm_backend = state
        .session_config
        .mlx_lm_backend
        .clone()
        .ok_or(EngineSessionError::MissingMlxLmConfig)
        .map_err(map_session_error)?;
    let response = run_blocking_session_task(move || {
        run_blocking_chat_generate(request_id, &runtime, &mlx_lm_backend, &request.chat_request)
            .map_err(EngineSessionError::from)
    })
    .await?;

    Ok(OpenAiStreamKind::ChatCompletion.build_non_stream_response(&response, request_id))
}

pub(crate) async fn run_openai_text_generation(
    state: AppState,
    request: OpenAiBuiltRequest,
    kind: OpenAiStreamKind,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    if request.stream {
        return stream_openai_request(state, request.generate_request, kind).await;
    }

    let (request_id, response) =
        run_stateless_generate_request(&state, request.generate_request).await?;

    Ok(kind.build_non_stream_response(&response, request_id))
}
