use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::embeddings::{
    parse_embedding_max_tokens, parse_embedding_pooling, parse_embedding_timeout_ms,
};
use crate::errors::map_generation_service_error;
use crate::errors::{ErrorResponse, admission_error_response, error_response, map_session_error};
use crate::openai::compat::tokenizer_for_live_op;
use crate::openai::schema::{
    EmbeddingInput, OpenAiEmbeddingObject, OpenAiEmbeddingRequest, OpenAiEmbeddingResponse,
    OpenAiEmbeddingUsage,
};
use crate::openai::validation::select_model;

const DEFAULT_EMBED_MAX_TOKENS: usize = 8192;
const DEFAULT_EMBED_TIMEOUT_MS: u64 = 30_000;

pub(crate) async fn openai_embeddings(
    State(state): State<AppState>,
    Json(request): Json<OpenAiEmbeddingRequest>,
) -> Result<Json<OpenAiEmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    let live = select_model(&state, request.model.as_deref())?;
    if request
        .encoding_format
        .as_deref()
        .is_some_and(|format| format != "float")
    {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "encoding_format currently supports only 'float'".to_string(),
        ));
    }
    if request.dimensions.is_some() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "dimensions is not supported; AX returns the model's native embedding dimension"
                .to_string(),
        ));
    }

    let pooling = parse_embedding_pooling(request.pooling.as_deref())
        .map_err(|message| error_response(StatusCode::BAD_REQUEST, "invalid_request", message))?;
    let normalize = request.normalize.unwrap_or(true);
    let model_id = live.model_id.as_ref().clone();
    let was_single = request.input.is_single();
    let batch = embedding_input_tokens(&live, request.input)?;

    // Treat empty single inputs, empty batches, and empty batch items as the
    // same user-facing error, with a per-index hint for batch items.
    if batch.is_empty() || (was_single && batch[0].is_empty()) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "input must not be empty".into(),
        ));
    }
    for (i, ids) in batch.iter().enumerate() {
        if ids.is_empty() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("input[{i}] must not be empty"),
            ));
        }
    }
    let max_tokens = parse_embedding_max_tokens(
        std::env::var("AX_ENGINE_EMBED_MAX_TOKENS").ok(),
        DEFAULT_EMBED_MAX_TOKENS,
    );
    let token_count: usize = batch.iter().map(Vec::len).sum();
    if token_count > max_tokens {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!(
                "input token count ({token_count}) exceeds maximum ({max_tokens}); \
                 set AX_ENGINE_EMBED_MAX_TOKENS to override"
            ),
        ));
    }
    let embed_timeout = parse_embedding_timeout_ms(
        std::env::var("AX_ENGINE_EMBED_TIMEOUT_MS").ok(),
        DEFAULT_EMBED_TIMEOUT_MS,
    );
    let timeout = Duration::from_millis(embed_timeout);
    let permit = state.try_admit(&live).map_err(admission_error_response)?;

    // Single input -> microbatcher (lets concurrent callers coalesce into
    // one batched runner call). Multi-input -> direct `embed_batch_flat`
    // since the caller already pre-batched; double-batching with the
    // microbatcher would just add a queueing delay.
    let embeddings: Vec<Vec<f32>> = if was_single {
        let Some(single) = batch.into_iter().next() else {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "input must not be empty".into(),
            ));
        };
        vec![
            tokio::time::timeout(
                timeout,
                live.embedding_batcher
                    .embed(single, pooling, normalize, permit),
            )
            .await
            .map_err(|_| {
                error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service_unavailable",
                    format!("embedding request timed out after {embed_timeout}ms"),
                )
            })?
            .map_err(map_session_error)?,
        ]
    } else {
        let generation_service = live.generation_service.clone();
        tokio::time::timeout(
            timeout,
            generation_service.execute(move |session| {
                let _permit = permit;
                session.embed_batch_flat(&batch, pooling, normalize)
            }),
        )
        .await
        .map_err(|_| {
            error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                format!("embedding request timed out after {embed_timeout}ms"),
            )
        })?
        .map_err(map_generation_service_error)
        .map(|m| {
            (0..m.batch_size)
                .map(|i| m.row(i).to_vec())
                .collect::<Vec<_>>()
        })?
    };

    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| OpenAiEmbeddingObject {
            object: "embedding",
            embedding,
            index: index as u32,
        })
        .collect();

    Ok(Json(OpenAiEmbeddingResponse {
        object: "list",
        data,
        model: model_id,
        usage: OpenAiEmbeddingUsage {
            prompt_tokens: token_count,
            total_tokens: token_count,
        },
    }))
}

pub(crate) fn embedding_input_tokens(
    live: &crate::app_state::LiveState,
    input: EmbeddingInput,
) -> Result<Vec<Vec<u32>>, (StatusCode, Json<ErrorResponse>)> {
    match input {
        EmbeddingInput::Text(text) => {
            if text.is_empty() {
                return Ok(vec![Vec::new()]);
            }
            let tokenizer = tokenizer_for_live_op(live, "text embedding input")?;
            tokenizer
                .encode(&text, true)
                .map(|tokens| vec![tokens])
                .map_err(|error| {
                    error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request",
                        format!("failed to tokenize embedding input: {error}"),
                    )
                })
        }
        EmbeddingInput::TextBatch(texts) => {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            let tokenizer = tokenizer_for_live_op(live, "text embedding input")?;
            let text_refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
            let mut batch = tokenizer.encode_batch(&text_refs, true).map_err(|error| {
                error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    format!("failed to tokenize embedding input: {error}"),
                )
            })?;
            for (text, tokens) in texts.iter().zip(&mut batch) {
                if text.is_empty() {
                    tokens.clear();
                }
            }
            Ok(batch)
        }
        EmbeddingInput::Tokens(tokens) => Ok(vec![tokens]),
        EmbeddingInput::TokenBatch(batch) => Ok(batch),
    }
}
