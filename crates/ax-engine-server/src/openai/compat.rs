use ax_engine_sdk::EngineTokenizer;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::app_state::AppState;
use crate::errors::{ErrorResponse, error_response};
use crate::openai::chat_requests::render_openai_chat_prompt;
use crate::openai::schema::OpenAiChatCompletionHttpRequest;
use crate::openai::validation::validate_model;

type HttpErrorResponse = (StatusCode, Json<ErrorResponse>);

#[derive(Debug, Deserialize)]
pub(crate) struct TokenizeRequest {
    content: String,
    #[serde(default)]
    add_special: bool,
    #[serde(default, rename = "parse_special")]
    _parse_special: bool,
    #[serde(default)]
    with_pieces: bool,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Serialize)]
struct TokenizeIdsResponse {
    tokens: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct TokenizePiecesResponse {
    tokens: Vec<TokenPiece>,
}

#[derive(Debug, Serialize)]
struct TokenPiece {
    id: u32,
    piece: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ApplyTemplateResponse {
    prompt: String,
}

pub(crate) async fn tokenize(
    State(state): State<AppState>,
    Json(request): Json<TokenizeRequest>,
) -> Result<Json<serde_json::Value>, HttpErrorResponse> {
    validate_model(&state, request.model.as_deref())?;
    let tokenizer = tokenizer_for_state(&state)?;
    let tokens = tokenizer
        .encode_with_special_tokens(&request.content, request.add_special)
        .map_err(|error| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("tokenization failed: {error}"),
            )
        })?;

    if request.with_pieces {
        let tokens = tokens
            .into_iter()
            .map(|id| TokenPiece {
                id,
                piece: tokenizer.id_to_token(id).unwrap_or_default(),
            })
            .collect();
        return serde_json::to_value(TokenizePiecesResponse { tokens })
            .map(Json)
            .map_err(|error| {
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    format!("failed to serialize tokenize response: {error}"),
                )
            });
    }

    serde_json::to_value(TokenizeIdsResponse { tokens })
        .map(Json)
        .map_err(|error| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                format!("failed to serialize tokenize response: {error}"),
            )
        })
}

pub(crate) async fn apply_template(
    State(state): State<AppState>,
    Json(request): Json<OpenAiChatCompletionHttpRequest>,
) -> Result<Json<ApplyTemplateResponse>, HttpErrorResponse> {
    validate_model(&state, request.model.as_deref())?;
    let prompt = render_openai_chat_prompt(state.model_id.as_ref(), &request.messages)?;
    Ok(Json(ApplyTemplateResponse { prompt }))
}

fn tokenizer_for_state(state: &AppState) -> Result<EngineTokenizer, HttpErrorResponse> {
    let Some(model_dir) = state.session_config.mlx_model_artifacts_dir() else {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "tokenize requires mlx_model_artifacts_dir with tokenizer.json".to_string(),
        ));
    };
    EngineTokenizer::from_model_dir(model_dir).map_err(|error| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("failed to load tokenizer for tokenize: {error}"),
        )
    })
}
