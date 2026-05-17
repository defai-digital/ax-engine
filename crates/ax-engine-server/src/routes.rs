use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};

use super::app_state::AppState;
use super::generation::lifecycle::{
    cancel_request, request_snapshot, step_request, submit_request,
};
use super::metadata::{health, models, runtime_info};
use super::openai::embeddings::openai_embeddings;
use super::{
    MAX_REQUEST_BODY_BYTES, generate, generate_stream, openai_chat_completions, openai_completions,
};

pub(crate) fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/v1/runtime", get(runtime_info))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(openai_embeddings))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/step", post(step_request))
        .route("/v1/requests", post(submit_request))
        .route("/v1/requests/:request_id", get(request_snapshot))
        .route("/v1/requests/:request_id/cancel", post(cancel_request))
        .route("/v1/generate/stream", post(generate_stream))
        .route("/v1/generate", post(generate))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state)
}
