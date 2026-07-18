use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::errors::ErrorResponse;
use crate::openai::generation::run_openai_text_generation;
use crate::openai::requests::build_openai_completion_request;
use crate::openai::schema::{OpenAiCompletionHttpRequest, OpenAiStreamKind};
use crate::openai::validation::select_openai_model;

pub(crate) async fn openai_completions(
    State(state): State<AppState>,
    Json(request): Json<OpenAiCompletionHttpRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let live = select_openai_model(&state, request.model.as_deref())?;
    let request = build_openai_completion_request(&live, request)?;

    run_openai_text_generation(state, live, request, OpenAiStreamKind::Completion).await
}
