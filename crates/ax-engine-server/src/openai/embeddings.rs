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
use crate::openai::schema::{
    OpenAiEmbeddingObject, OpenAiEmbeddingRequest, OpenAiEmbeddingResponse, OpenAiEmbeddingUsage,
};
use crate::openai::validation::select_model;

const DEFAULT_EMBED_MAX_TOKENS: usize = 8192;
const DEFAULT_EMBED_TIMEOUT_MS: u64 = 30_000;

pub(crate) async fn openai_embeddings(
    State(state): State<AppState>,
    Json(request): Json<OpenAiEmbeddingRequest>,
) -> Result<Json<OpenAiEmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    let live = select_model(&state, request.model.as_deref())?;

    let pooling = parse_embedding_pooling(request.pooling.as_deref())
        .map_err(|message| error_response(StatusCode::BAD_REQUEST, "invalid_request", message))?;
    let normalize = request.normalize.unwrap_or(true);
    let model_id = live.model_id.as_ref().clone();
    let was_single = request.input.is_single();
    let batch = request.input.into_batch();

    // Treat both `input: []` (Single empty) and `input: [[]]` / `input: []`
    // (Batch empty / batch with empty inner) as the same user-facing
    // error, with a per-index hint when the position is non-trivial.
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
        let single = batch
            .into_iter()
            .next()
            .expect("batch.len() == 1 because input was Single");
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
