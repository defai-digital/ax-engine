use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::embeddings::parse_embedding_pooling;
use crate::errors::{ErrorResponse, error_response, map_session_error};
use crate::openai::schema::{
    OpenAiEmbeddingObject, OpenAiEmbeddingRequest, OpenAiEmbeddingResponse, OpenAiEmbeddingUsage,
};
use crate::openai::validation::validate_model;

pub(crate) async fn openai_embeddings(
    State(state): State<AppState>,
    Json(request): Json<OpenAiEmbeddingRequest>,
) -> Result<Json<OpenAiEmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    validate_model(&live, request.model.as_deref())?;

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
    let token_count: usize = batch.iter().map(Vec::len).sum();

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
            live.embedding_batcher
                .embed(single, pooling, normalize)
                .await
                .map_err(map_session_error)?,
        ]
    } else {
        let session = live.request_session.clone();
        tokio::task::spawn_blocking(move || {
            let s = session.blocking_lock();
            s.embed_batch_flat(&batch, pooling, normalize)
        })
        .await
        .map_err(|_| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "embedding worker join failed".into(),
            )
        })?
        .map_err(map_session_error)
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
