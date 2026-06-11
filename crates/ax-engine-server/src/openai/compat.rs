use ax_engine_sdk::EngineTokenizer;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::app_state::{AppState, LiveState};
use crate::errors::{ErrorResponse, error_response};
use crate::metadata::context_length;
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
    let live = state.snapshot();
    validate_model(&live, request.model.as_deref())?;
    let tokenizer = tokenizer_for_live(&live)?;
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
    let live = state.snapshot();
    validate_model(&live, request.model.as_deref())?;
    let prompt = render_openai_chat_prompt(live.model_id.as_ref(), &request.messages)?;
    Ok(Json(ApplyTemplateResponse { prompt }))
}

fn tokenizer_for_live(live: &LiveState) -> Result<EngineTokenizer, HttpErrorResponse> {
    tokenizer_for_live_op(live, "this endpoint")
}

fn tokenizer_for_live_op(live: &LiveState, op: &str) -> Result<EngineTokenizer, HttpErrorResponse> {
    let Some(model_dir) = live.session_config.mlx_model_artifacts_dir() else {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("{op} requires mlx_model_artifacts_dir with tokenizer.json"),
        ));
    };
    EngineTokenizer::from_model_dir(model_dir).map_err(|error| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("failed to load tokenizer: {error}"),
        )
    })
}

// ── /detokenize ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(crate) struct DetokenizeRequest {
    tokens: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DetokenizeResponse {
    content: String,
}

pub(crate) async fn detokenize(
    State(state): State<AppState>,
    Json(request): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, HttpErrorResponse> {
    let live = state.snapshot();
    let tokenizer = tokenizer_for_live_op(&live, "/detokenize")?;
    let content = tokenizer.decode(&request.tokens, false).map_err(|error| {
        error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("detokenization failed: {error}"),
        )
    })?;
    Ok(Json(DetokenizeResponse { content }))
}

// ── /props ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
struct DefaultGenerationSettings {
    n_ctx: u32,
    n_predict: i32,
    model: String,
    seed: i64,
    temperature: f32,
    dynatemp_range: f32,
    dynatemp_exponent: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    tfs_z: f32,
    typical_p: f32,
    repeat_last_n: i32,
    repeat_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    mirostat: u32,
    mirostat_tau: f32,
    mirostat_eta: f32,
    penalize_nl: bool,
    stop: Vec<String>,
    n_keep: u32,
    n_discard: u32,
    ignore_eos: bool,
    stream: bool,
    n_probs: u32,
    min_keep: u32,
    grammar: String,
    samplers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BuildInfo {
    build_number: u32,
    commit: String,
    compiler: &'static str,
    build_mode: &'static str,
}

#[derive(Debug, Serialize)]
pub(crate) struct PropsResponse {
    system_prompt: String,
    default_generation_settings: DefaultGenerationSettings,
    total_slots: u32,
    chat_template: String,
    build_info: BuildInfo,
}

pub(crate) async fn props(State(state): State<AppState>) -> Json<PropsResponse> {
    let live = state.snapshot();
    let n_ctx = context_length(&state);
    let chat_template = read_chat_template_live(&live);
    let model = live.model_id.as_ref().clone();

    Json(PropsResponse {
        system_prompt: String::new(),
        default_generation_settings: default_generation_settings(n_ctx, model),
        total_slots: 1,
        chat_template,
        build_info: BuildInfo {
            build_number: 0,
            commit: option_env!("GIT_COMMIT_HASH")
                .unwrap_or("unknown")
                .to_string(),
            compiler: "Rust/MLX",
            build_mode: "Release",
        },
    })
}

fn default_generation_settings(n_ctx: u32, model: String) -> DefaultGenerationSettings {
    DefaultGenerationSettings {
        n_ctx,
        n_predict: -1,
        model,
        seed: -1,
        temperature: 0.8,
        dynatemp_range: 0.0,
        dynatemp_exponent: 1.0,
        top_k: 40,
        top_p: 0.95,
        min_p: 0.05,
        tfs_z: 1.0,
        typical_p: 1.0,
        repeat_last_n: 64,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        mirostat: 0,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        penalize_nl: false,
        stop: vec![],
        n_keep: 0,
        n_discard: 0,
        ignore_eos: false,
        stream: false,
        n_probs: 0,
        min_keep: 0,
        grammar: String::new(),
        samplers: vec![
            "top_k".into(),
            "tfs_z".into(),
            "typical_p".into(),
            "top_p".into(),
            "min_p".into(),
            "temperature".into(),
        ],
    }
}

// ── /slots ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct SlotTimings {
    prompt_n: u32,
    prompt_ms: f64,
    prompt_per_token_ms: f64,
    prompt_per_second: f64,
    predicted_n: u32,
    predicted_ms: f64,
    predicted_per_token_ms: f64,
    predicted_per_second: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct SlotEntry {
    id: u32,
    id_task: i32,
    state: u32,
    state_str: &'static str,
    prompt: String,
    n_ctx: u32,
    n_past: u32,
    n_predict: i32,
    n_prompt_tokens: u32,
    n_decoded: u32,
    infill: bool,
    cache_tokens: u32,
    tokens: Vec<u32>,
    generation_settings: DefaultGenerationSettings,
    timings: SlotTimings,
}

pub(crate) async fn slots(State(state): State<AppState>) -> Json<Vec<SlotEntry>> {
    let live = state.snapshot();
    let busy = live.request_session.try_lock().is_err();
    let (slot_state, state_str) = if busy {
        (1u32, "processing")
    } else {
        (0u32, "idle")
    };
    let n_ctx = context_length(&state);
    let model = live.model_id.as_ref().clone();

    Json(vec![SlotEntry {
        id: 0,
        id_task: -1,
        state: slot_state,
        state_str,
        prompt: String::new(),
        n_ctx,
        n_past: 0,
        n_predict: -1,
        n_prompt_tokens: 0,
        n_decoded: 0,
        infill: false,
        cache_tokens: 0,
        tokens: vec![],
        generation_settings: default_generation_settings(n_ctx, model),
        timings: SlotTimings {
            prompt_n: 0,
            prompt_ms: 0.0,
            prompt_per_token_ms: 0.0,
            prompt_per_second: 0.0,
            predicted_n: 0,
            predicted_ms: 0.0,
            predicted_per_token_ms: 0.0,
            predicted_per_second: 0.0,
        },
    }])
}

fn read_chat_template_live(live: &LiveState) -> String {
    let Some(model_dir) = live.session_config.mlx_model_artifacts_dir() else {
        return String::new();
    };
    // Prefer the explicit jinja file (used by Gemma4 and instruction-tuned models).
    let jinja_path = model_dir.join("chat_template.jinja");
    if let Ok(template) = std::fs::read_to_string(&jinja_path) {
        return template;
    }
    // Fall back to the standard HuggingFace tokenizer_config.json field.
    // The field can be either a plain string or an array of {name, template}
    // objects (newer HF models like Qwen3, Llama 3.1+). In the array form,
    // prefer the entry named "default", otherwise take the first entry.
    std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            let field = v.get("chat_template")?;
            if let Some(s) = field.as_str() {
                return Some(s.to_owned());
            }
            if let Some(arr) = field.as_array() {
                let pick = arr
                    .iter()
                    .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
                    .or_else(|| arr.first())?;
                return pick
                    .get("template")
                    .and_then(|t| t.as_str())
                    .map(str::to_owned);
            }
            None
        })
        .unwrap_or_default()
}
