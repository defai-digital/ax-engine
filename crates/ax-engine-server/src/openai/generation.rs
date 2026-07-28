use std::convert::Infallible;

use ax_engine_sdk::{EngineTokenizer, GenerateResponse, SelectedBackend};
use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::Event;
use serde::Serialize;
use tokio::sync::mpsc;

use crate::app_state::{AppState, LiveState};
use crate::backends::{edge_llm, llama_cpp, mlx_lm, vllm};
use crate::chat::{
    ChatPromptTemplate, decode_gemma4_chat_output, decode_gemma4_chat_output_with_reasoning,
    decode_glm_chat_output,
};
use crate::errors::{ErrorResponse, admission_error_response, error_response, map_session_error};
use crate::generation::native::run_stateless_generate_request;
use crate::generation::streaming::{StreamEvent, build_keep_alive_stream};
use crate::openai::chunks::{
    chat_delta_chunk, chat_final_chunk, chat_tool_calls_delta_chunk, chat_tool_calls_final_chunk,
};
use crate::openai::requests::{
    OpenAiBuiltEdgeLlmChatRequest, OpenAiBuiltLlamaCppChatRequest, OpenAiBuiltMlxLmChatRequest,
    OpenAiBuiltRequest, OpenAiBuiltVllmChatRequest, OpenAiResponseOptions,
    build_openai_edge_llm_chat_request, build_openai_llama_cpp_chat_request,
    build_openai_mlx_lm_chat_request, build_openai_vllm_chat_request,
};
use crate::openai::responses::openai_chat_completion_response;
use crate::openai::schema::{OpenAiChatCompletionHttpRequest, OpenAiStreamKind};
use crate::openai::streaming::{
    StreamReasoningFamily, stream_openai_edge_llm_chat_request,
    stream_openai_llama_cpp_chat_request, stream_openai_mlx_lm_chat_request, stream_openai_request,
    stream_openai_vllm_chat_request,
};
use crate::tasks::run_blocking_session_task;

pub(crate) async fn run_openai_llama_cpp_chat_generation(
    state: AppState,
    live: LiveState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let OpenAiBuiltLlamaCppChatRequest {
        chat_request,
        stream,
        response_options,
    } = build_openai_llama_cpp_chat_request(&live, request)?;
    if stream {
        if response_options.parse_tool_calls {
            return Err(streaming_delegated_tool_calls_error());
        }
        return stream_openai_llama_cpp_chat_request(
            state,
            live,
            chat_request,
            response_options.include_stream_usage,
        )
        .await;
    }

    let request_id = state.allocate_request_id();
    let runtime = live.runtime_report.clone();
    let llama_backend = llama_cpp::config(&live).map_err(map_session_error)?;
    let permit = state.try_admit(&live).map_err(admission_error_response)?;
    let response = run_blocking_session_task(move || {
        let _permit = permit;
        llama_cpp::run_chat_generate(request_id, &runtime, &llama_backend, &chat_request)
    })
    .await?;
    validate_openai_response_format(&response, &response_options)?;

    Ok(OpenAiStreamKind::ChatCompletion.build_non_stream_response(
        &response,
        request_id,
        response_options,
        None,
    ))
}

pub(crate) async fn run_openai_mlx_lm_chat_generation(
    state: AppState,
    live: LiveState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let OpenAiBuiltMlxLmChatRequest {
        chat_request,
        stream,
        response_options,
    } = build_openai_mlx_lm_chat_request(&live, request)?;
    if stream {
        if response_options.parse_tool_calls {
            return Err(streaming_delegated_tool_calls_error());
        }
        return stream_openai_mlx_lm_chat_request(
            state,
            live,
            chat_request,
            response_options.include_stream_usage,
        )
        .await;
    }

    let request_id = state.allocate_request_id();
    let runtime = live.runtime_report.clone();
    let mlx_lm_backend = mlx_lm::config(&live).map_err(map_session_error)?;
    let permit = state.try_admit(&live).map_err(admission_error_response)?;
    let response = run_blocking_session_task(move || {
        let _permit = permit;
        mlx_lm::run_chat_generate(request_id, &runtime, &mlx_lm_backend, &chat_request)
    })
    .await?;
    validate_openai_response_format(&response, &response_options)?;

    Ok(OpenAiStreamKind::ChatCompletion.build_non_stream_response(
        &response,
        request_id,
        response_options,
        None,
    ))
}

pub(crate) async fn run_openai_edge_llm_chat_generation(
    state: AppState,
    live: LiveState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let OpenAiBuiltEdgeLlmChatRequest {
        chat_request,
        stream,
        response_options,
    } = build_openai_edge_llm_chat_request(&live, request)?;
    if stream {
        if response_options.parse_tool_calls {
            return Err(streaming_delegated_tool_calls_error());
        }
        return stream_openai_edge_llm_chat_request(
            state,
            live,
            chat_request,
            response_options.include_stream_usage,
        )
        .await;
    }

    let request_id = state.allocate_request_id();
    let runtime = live.runtime_report.clone();
    let edge_backend = edge_llm::config(&live).map_err(map_session_error)?;
    let permit = state.try_admit(&live).map_err(admission_error_response)?;
    let response = run_blocking_session_task(move || {
        let _permit = permit;
        edge_llm::run_chat_generate(request_id, &runtime, &edge_backend, &chat_request)
    })
    .await?;
    validate_openai_response_format(&response, &response_options)?;

    Ok(OpenAiStreamKind::ChatCompletion.build_non_stream_response(
        &response,
        request_id,
        response_options,
        None,
    ))
}

pub(crate) async fn run_openai_vllm_chat_generation(
    state: AppState,
    live: LiveState,
    request: OpenAiChatCompletionHttpRequest,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let OpenAiBuiltVllmChatRequest {
        chat_request,
        stream,
        response_options,
    } = build_openai_vllm_chat_request(&live, request)?;
    if stream {
        if response_options.parse_tool_calls {
            return Err(streaming_delegated_tool_calls_error());
        }
        return stream_openai_vllm_chat_request(
            state,
            live,
            chat_request,
            response_options.include_stream_usage,
        )
        .await;
    }

    let request_id = state.allocate_request_id();
    let runtime = live.runtime_report.clone();
    let vllm_backend = vllm::config(&live).map_err(map_session_error)?;
    let permit = state.try_admit(&live).map_err(admission_error_response)?;
    let response = run_blocking_session_task(move || {
        let _permit = permit;
        vllm::run_chat_generate(request_id, &runtime, &vllm_backend, &chat_request)
    })
    .await?;
    validate_openai_response_format(&response, &response_options)?;

    Ok(OpenAiStreamKind::ChatCompletion.build_non_stream_response(
        &response,
        request_id,
        response_options,
        None,
    ))
}

pub(crate) async fn run_openai_text_generation(
    state: AppState,
    live: LiveState,
    request: OpenAiBuiltRequest,
    kind: OpenAiStreamKind,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let OpenAiBuiltRequest {
        generate_request,
        stream,
        response_options,
    } = request;
    if stream {
        let tool_chat =
            response_options.parse_tool_calls && matches!(kind, OpenAiStreamKind::ChatCompletion);
        // Incremental tool-call streaming (ADR-040 D1) covers the product
        // scope's text-marker families. GLM 4.x encodes tool markers as
        // special tokens the plain incremental decode strips, and GPT-OSS
        // calls ride Harmony commentary channels — both keep the buffered
        // fallback until their stream decodes preserve the markers.
        let incremental_tool_chat = tool_chat
            && live.runtime_report.selected_backend == SelectedBackend::Mlx
            && matches!(
                ChatPromptTemplate::for_model_id(live.model_id.as_ref()),
                ChatPromptTemplate::QwenChatMl | ChatPromptTemplate::Gemma4
            );
        if tool_chat && !incremental_tool_chat {
            return stream_buffered_openai_tool_chat_response(
                state,
                live,
                generate_request,
                response_options,
            )
            .await;
        }
        // Streaming reasoning (M2): the request build already rejected
        // reasoning+stream for families without a mechanism.
        let reasoning_family = if response_options.include_reasoning
            && matches!(kind, OpenAiStreamKind::ChatCompletion)
            && live.runtime_report.selected_backend == SelectedBackend::Mlx
        {
            match ChatPromptTemplate::for_model_id(live.model_id.as_ref()) {
                ChatPromptTemplate::QwenChatMl => Some(StreamReasoningFamily::QwenThink),
                ChatPromptTemplate::Gemma4 => Some(StreamReasoningFamily::Gemma4Channel),
                _ => None,
            }
        } else {
            None
        };
        return stream_openai_request(
            state,
            live,
            generate_request,
            kind,
            &response_options,
            incremental_tool_chat,
            reasoning_family,
        )
        .await;
    }

    let (request_id, mut response) =
        run_stateless_generate_request(&state, &live, generate_request).await?;
    let native_reasoning = populate_native_mlx_output_text(
        &live,
        &mut response,
        kind,
        response_options.include_reasoning,
    )?;
    validate_openai_response_format(&response, &response_options)?;

    Ok(kind.build_non_stream_response(&response, request_id, response_options, native_reasoning))
}

async fn stream_buffered_openai_tool_chat_response(
    state: AppState,
    live: LiveState,
    generate_request: ax_engine_sdk::GenerateRequest,
    response_options: OpenAiResponseOptions,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let (request_id, mut response) =
        run_stateless_generate_request(&state, &live, generate_request).await?;
    let native_reasoning = populate_native_mlx_output_text(
        &live,
        &mut response,
        OpenAiStreamKind::ChatCompletion,
        response_options.include_reasoning,
    )?;
    validate_openai_response_format(&response, &response_options)?;
    let include_stream_usage = response_options.include_stream_usage;

    let chat_response = openai_chat_completion_response(
        &response,
        OpenAiStreamKind::ChatCompletion.response_id(request_id),
        response_options,
        native_reasoning,
    );
    let Some(choice) = chat_response.choices.into_iter().next() else {
        return Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "OpenAI chat response did not contain a choice".to_string(),
        ));
    };

    let (tx, rx) = mpsc::channel::<StreamEvent>(8);
    let mut role_emitted = false;
    if !choice.message.content.is_empty() {
        let chunk = chat_delta_chunk(
            request_id,
            chat_response.model.clone(),
            Some("assistant"),
            choice.message.content,
        );
        role_emitted = true;
        send_openai_chunk_async(&tx, &chunk).await;
    }
    if let Some(tool_calls) = choice.message.tool_calls.as_ref()
        && !tool_calls.is_empty()
    {
        let chunk = chat_tool_calls_delta_chunk(
            request_id,
            chat_response.model.clone(),
            if role_emitted {
                None
            } else {
                Some("assistant")
            },
            tool_calls,
        );
        role_emitted = true;
        send_openai_chunk_async(&tx, &chunk).await;
    }
    if !role_emitted {
        let chunk = chat_delta_chunk(
            request_id,
            chat_response.model.clone(),
            Some("assistant"),
            String::new(),
        );
        send_openai_chunk_async(&tx, &chunk).await;
    }

    let final_chunk = if choice.finish_reason == Some("tool_calls") {
        chat_tool_calls_final_chunk(request_id, chat_response.model)
    } else {
        chat_final_chunk(request_id, chat_response.model, response.finish_reason)
    };
    send_openai_chunk_async(&tx, &final_chunk).await;
    if include_stream_usage && let Some(usage) = crate::openai::responses::openai_usage(&response) {
        let usage_chunk = crate::openai::chunks::stream_usage_chunk(
            request_id,
            response.model_id.clone(),
            OpenAiStreamKind::ChatCompletion,
            usage,
        );
        send_openai_chunk_async(&tx, &usage_chunk).await;
    }
    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    drop(tx);

    Ok(build_keep_alive_stream(rx, state.limits.stream_deadlines).into_response())
}

async fn send_openai_chunk_async<T: Serialize>(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    payload: &T,
) -> bool {
    match serde_json::to_string(payload) {
        Ok(data) => tx.send(Ok(Event::default().data(data))).await.is_ok(),
        Err(error) => {
            let payload = ErrorResponse::server_error(format!(
                "failed to serialize OpenAI stream chunk: {error}"
            ));
            let data = serde_json::to_string(&payload).unwrap_or_else(|_| {
                r#"{"error":{"code":"server_error","message":"failed to serialize OpenAI stream chunk"}}"#
                    .to_string()
            });
            let _ = tx
                .send(Ok(Event::default().event("error").data(data)))
                .await;
            false
        }
    }
}

fn streaming_delegated_tool_calls_error() -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::BAD_REQUEST,
        "unsupported_parameter",
        "streaming tool calls require native AX-rendered model-family tool prompts; delegated text backends do not expose structured tool-call deltas yet".to_string(),
    )
}

/// Post-hoc `response_format` validation for `json_object` and Phase A
/// `json_schema` (ADR-040 D3: validated, not constrained — Phase B upgrades
/// this to guided decoding without changing the request shape).
pub(crate) fn validate_openai_response_format(
    response: &GenerateResponse,
    options: &OpenAiResponseOptions,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if !options.validate_json_object && options.json_schema.is_none() {
        return Ok(());
    }
    let text = response.output_text.as_deref().unwrap_or("").trim();
    let value = serde_json::from_str::<serde_json::Value>(text).map_err(|error| {
        error_response(
            StatusCode::BAD_GATEWAY,
            "invalid_output",
            format!("model output did not satisfy response_format: {error}"),
        )
    })?;
    if options.validate_json_object && !value.is_object() {
        return Err(error_response(
            StatusCode::BAD_GATEWAY,
            "invalid_output",
            "model output did not satisfy response_format json_object: expected a JSON object"
                .to_string(),
        ));
    }
    if let Some(contract) = options.json_schema.as_deref() {
        crate::openai::json_schema::validate_value(&contract.schema, &value).map_err(
            |message| {
                let name = contract.name.as_deref().unwrap_or("response");
                error_response(
                    StatusCode::BAD_GATEWAY,
                    "invalid_output",
                    format!(
                        "model output did not satisfy response_format json_schema {name}: {message}"
                    ),
                )
            },
        )?;
    }
    Ok(())
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
    let tokenizer = EngineTokenizer::from_model_dir_cached(model_dir).map_err(|error| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("failed to load tokenizer for native MLX OpenAI response decode: {error}"),
        )
    })?;
    // Chat responses strip Gemma 4 thinking-channel framing (the markers are
    // model-specific control tokens that must not surface as content); raw
    // completions keep the verbatim decode.
    // GLM 4.x encodes tool calls with special tokens that a plain decode strips,
    // so chat output is decoded with those markers preserved for the tool-call
    // parser. GLM does not use Gemma 4 reasoning channels.
    let chat_template = ChatPromptTemplate::for_model_id(live.model_id.as_ref());
    let is_glm_chat = matches!(kind, OpenAiStreamKind::ChatCompletion)
        && matches!(chat_template, ChatPromptTemplate::Glm47);
    let is_gpt_oss_chat = matches!(kind, OpenAiStreamKind::ChatCompletion)
        && matches!(chat_template, ChatPromptTemplate::GptOssHarmony);
    let (output_text, reasoning) = if is_glm_chat {
        decode_glm_chat_output(&tokenizer, &response.output_tokens).map(|content| (content, None))
    } else if is_gpt_oss_chat {
        crate::chat::decode_gpt_oss_chat_output(&tokenizer, &response.output_tokens)
            .map(|content| (content, None))
    } else {
        match kind {
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
