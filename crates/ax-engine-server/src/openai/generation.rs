use ax_engine_sdk::{EngineTokenizer, GenerateResponse, SelectedBackend};
use axum::Json;
use axum::http::StatusCode;

use crate::app_state::{AppState, LiveState};
use crate::backends::{llama_cpp, mlx_lm};
use crate::chat::{decode_gemma4_chat_output, decode_gemma4_chat_output_with_reasoning};
use crate::errors::{ErrorResponse, error_response, map_session_error};
use crate::generation::native::run_stateless_generate_request;
use crate::openai::requests::{
    OpenAiBuiltLlamaCppChatRequest, OpenAiBuiltMlxLmChatRequest, OpenAiBuiltRequest,
    OpenAiResponseOptions, build_openai_llama_cpp_chat_request, build_openai_mlx_lm_chat_request,
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
    let live = state.snapshot();
    let OpenAiBuiltLlamaCppChatRequest {
        chat_request,
        stream,
        response_options,
    } = build_openai_llama_cpp_chat_request(&live, request)?;
    if stream {
        return stream_openai_llama_cpp_chat_request(state, chat_request).await;
    }

    let request_id = state.allocate_request_id();
    let runtime = live.runtime_report.clone();
    let llama_backend = llama_cpp::config(&live).map_err(map_session_error)?;
    let response = run_blocking_session_task(move || {
        llama_cpp::run_chat_generate(request_id, &runtime, &llama_backend, &chat_request)
    })
    .await?;
    validate_openai_json_object_response(&response, response_options)?;

    Ok(OpenAiStreamKind::ChatCompletion.build_non_stream_response(
        &response,
        request_id,
        response_options,
        None,
    ))
}

pub(crate) async fn run_openai_mlx_lm_chat_generation(
    state: AppState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    let OpenAiBuiltMlxLmChatRequest {
        chat_request,
        stream,
        response_options,
    } = build_openai_mlx_lm_chat_request(&live, request)?;
    if stream {
        return stream_openai_mlx_lm_chat_request(state, chat_request).await;
    }

    let request_id = state.allocate_request_id();
    let runtime = live.runtime_report.clone();
    let mlx_lm_backend = mlx_lm::config(&live).map_err(map_session_error)?;
    let response = run_blocking_session_task(move || {
        mlx_lm::run_chat_generate(request_id, &runtime, &mlx_lm_backend, &chat_request)
    })
    .await?;
    validate_openai_json_object_response(&response, response_options)?;

    Ok(OpenAiStreamKind::ChatCompletion.build_non_stream_response(
        &response,
        request_id,
        response_options,
        None,
    ))
}

pub(crate) async fn run_openai_text_generation(
    state: AppState,
    request: OpenAiBuiltRequest,
    kind: OpenAiStreamKind,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    let OpenAiBuiltRequest {
        generate_request,
        stream,
        response_options,
    } = request;
    if stream {
        return stream_openai_request(state, generate_request, kind).await;
    }

    let (request_id, mut response) =
        run_stateless_generate_request(&state, generate_request).await?;
    let native_reasoning = populate_native_mlx_output_text(
        &live,
        &mut response,
        kind,
        response_options.include_reasoning,
    )?;
    validate_openai_json_object_response(&response, response_options)?;

    Ok(kind.build_non_stream_response(&response, request_id, response_options, native_reasoning))
}

pub(crate) fn validate_openai_json_object_response(
    response: &GenerateResponse,
    options: OpenAiResponseOptions,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if !options.validate_json_object {
        return Ok(());
    }
    let text = response.output_text.as_deref().unwrap_or("").trim();
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(serde_json::Value::Object(_)) => Ok(()),
        Ok(_) => Err(error_response(
            StatusCode::BAD_GATEWAY,
            "invalid_output",
            "model output did not satisfy response_format json_object: expected a JSON object"
                .to_string(),
        )),
        Err(error) => Err(error_response(
            StatusCode::BAD_GATEWAY,
            "invalid_output",
            format!("model output did not satisfy response_format json_object: {error}"),
        )),
    }
}

/// Decode native MLX output tokens into `response.output_text`. For chat
/// responses with an explicit reasoning contract, returns the Gemma 4
/// thinking-channel text extracted during decode (the channel framing is
/// token-level and never survives into `output_text`, so the response layer
/// cannot recover it afterwards).
pub(crate) fn populate_native_mlx_output_text(
    live: &LiveState,
    response: &mut GenerateResponse,
    kind: OpenAiStreamKind,
    include_reasoning: bool,
) -> Result<Option<String>, (StatusCode, Json<ErrorResponse>)> {
    if live.runtime_report.selected_backend != SelectedBackend::Mlx
        || response.output_text.is_some()
        || response.output_tokens.is_empty()
    {
        return Ok(None);
    }

    let Some(model_dir) = live.session_config.mlx_model_artifacts_dir() else {
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
    let (output_text, reasoning) = match kind {
        OpenAiStreamKind::ChatCompletion if include_reasoning => {
            decode_gemma4_chat_output_with_reasoning(&tokenizer, &response.output_tokens)
        }
        OpenAiStreamKind::ChatCompletion => {
            decode_gemma4_chat_output(&tokenizer, &response.output_tokens)
                .map(|content| (content, None))
        }
        OpenAiStreamKind::Completion => tokenizer
            .decode(&response.output_tokens, true)
            .map(|content| (content, None)),
    }
    .map_err(|error| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("failed to decode native MLX OpenAI response tokens: {error}"),
        )
    })?;
    response.output_text = Some(output_text);

    Ok(reasoning)
}
