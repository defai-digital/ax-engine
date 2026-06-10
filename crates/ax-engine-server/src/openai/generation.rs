use ax_engine_sdk::{EngineTokenizer, GenerateResponse, SelectedBackend};
use axum::Json;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::backends::{llama_cpp, mlx_lm};
use crate::chat::decode_gemma4_chat_output;
use crate::errors::{ErrorResponse, error_response, map_session_error};
use crate::generation::native::run_stateless_generate_request;
use crate::openai::requests::{
    OpenAiBuiltLlamaCppChatRequest, OpenAiBuiltMlxLmChatRequest, OpenAiBuiltRequest,
    OpenAiOutputPostprocessing, build_openai_llama_cpp_chat_request,
    build_openai_mlx_lm_chat_request,
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
    let OpenAiBuiltLlamaCppChatRequest {
        chat_request,
        stream,
        output_postprocessing,
    } = build_openai_llama_cpp_chat_request(&state, request)?;
    if stream {
        return stream_openai_llama_cpp_chat_request(state, chat_request, output_postprocessing)
            .await;
    }

    let request_id = state.allocate_request_id();
    let runtime = state.runtime_report.clone();
    let llama_backend = llama_cpp::config(&state).map_err(map_session_error)?;
    let response = run_blocking_session_task(move || {
        llama_cpp::run_chat_generate(request_id, &runtime, &llama_backend, &chat_request)
    })
    .await?;
    let mut response = response;
    apply_openai_chat_output_postprocessing(&mut response, output_postprocessing);

    Ok(OpenAiStreamKind::ChatCompletion.build_non_stream_response(&response, request_id))
}

pub(crate) async fn run_openai_mlx_lm_chat_generation(
    state: AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let OpenAiBuiltMlxLmChatRequest {
        chat_request,
        stream,
        output_postprocessing,
    } = build_openai_mlx_lm_chat_request(&state, request)?;
    if stream {
        return stream_openai_mlx_lm_chat_request(state, chat_request, output_postprocessing).await;
    }

    let request_id = state.allocate_request_id();
    let runtime = state.runtime_report.clone();
    let mlx_lm_backend = mlx_lm::config(&state).map_err(map_session_error)?;
    let response = run_blocking_session_task(move || {
        mlx_lm::run_chat_generate(request_id, &runtime, &mlx_lm_backend, &chat_request)
    })
    .await?;
    let mut response = response;
    apply_openai_chat_output_postprocessing(&mut response, output_postprocessing);

    Ok(OpenAiStreamKind::ChatCompletion.build_non_stream_response(&response, request_id))
}

pub(crate) async fn run_openai_text_generation(
    state: AppState,
    request: OpenAiBuiltRequest,
    kind: OpenAiStreamKind,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let OpenAiBuiltRequest {
        generate_request,
        stream,
        output_postprocessing,
    } = request;
    if stream {
        return stream_openai_request(state, generate_request, kind, output_postprocessing).await;
    }

    let (request_id, mut response) =
        run_stateless_generate_request(&state, generate_request).await?;
    populate_native_mlx_output_text(&state, &mut response, kind)?;
    apply_openai_chat_output_postprocessing(&mut response, output_postprocessing);

    Ok(kind.build_non_stream_response(&response, request_id))
}

fn apply_openai_chat_output_postprocessing(
    response: &mut GenerateResponse,
    output_postprocessing: OpenAiOutputPostprocessing,
) {
    if output_postprocessing.is_noop() {
        return;
    }

    if let Some(output_text) = response.output_text.take() {
        response.output_text = Some(output_postprocessing.normalize_output_text(&output_text));
    }
    response.finish_reason = output_postprocessing.apply_finish_reason(response.finish_reason);
}

fn populate_native_mlx_output_text(
    state: &AppState,
    response: &mut GenerateResponse,
    kind: OpenAiStreamKind,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if state.runtime_report.selected_backend != SelectedBackend::Mlx
        || response.output_text.is_some()
        || response.output_tokens.is_empty()
    {
        return Ok(());
    }

    let Some(model_dir) = state.session_config.mlx_model_artifacts_dir() else {
        return Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "native MLX OpenAI response decode requires mlx_model_artifacts_dir with tokenizer.json"
                .to_string(),
        ));
    };
    let tokenizer = EngineTokenizer::from_model_dir(model_dir).map_err(|error| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("failed to load tokenizer for native MLX OpenAI response decode: {error}"),
        )
    })?;
    // Chat responses strip Gemma 4 thinking-channel framing (the markers are
    // model-specific control tokens that must not surface as content); raw
    // completions keep the verbatim decode.
    let output_text = match kind {
        OpenAiStreamKind::ChatCompletion => {
            decode_gemma4_chat_output(&tokenizer, &response.output_tokens)
        }
        OpenAiStreamKind::Completion => tokenizer.decode(&response.output_tokens, true),
    }
    .map_err(|error| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("failed to decode native MLX OpenAI response tokens: {error}"),
        )
    })?;
    response.output_text = Some(output_text);

    Ok(())
}
