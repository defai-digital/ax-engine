use ax_engine_sdk::{RuntimeReport, SelectedBackend};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Serialize;
use serde_json::json;

use crate::app_state::AppState;
use crate::errors::{ErrorResponse, error_response};

pub(crate) const MODEL_OWNER: &str = "ax-engine";

#[derive(Debug, Serialize)]
pub(crate) struct ServerInfoResponse {
    service: &'static str,
    model_id: String,
    deterministic: bool,
    max_batch_tokens: u32,
    block_size_tokens: u32,
    runtime: RuntimeResponse,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
    capabilities: ModelCapabilities,
    limit: ModelLimit,
    context_length: u32,
    max_output_tokens: u32,
    ax_engine: AxEngineModelMetadata,
    runtime: RuntimeResponse,
}

pub(crate) type RuntimeResponse = RuntimeReport;

#[derive(Debug, Serialize)]
struct ModelCapabilities {
    temperature: bool,
    reasoning: bool,
    attachment: bool,
    toolcall: bool,
    input: ModelModalities,
    output: ModelModalities,
    interleaved: bool,
}

#[derive(Debug, Serialize)]
struct ModelModalities {
    text: bool,
    audio: bool,
    image: bool,
    video: bool,
    pdf: bool,
}

#[derive(Debug, Serialize)]
struct ModelLimit {
    context: u32,
    output: u32,
}

#[derive(Debug, Serialize)]
struct AxEngineModelMetadata {
    native_generate_supported: bool,
    openai_completions_supported: bool,
    openai_chat_completions_supported: bool,
    openai_tool_calling_supported: bool,
    openai_text_input_supported: bool,
}

pub(crate) async fn health(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    // `/health` is the readiness probe most callers (bench harness,
    // k8s, load balancers) poll while a server starts. Returning 200
    // when the server has bound a port but the inference session is
    // wedged (deadlocked on another in-flight call, runtime panicked,
    // weights not loadable on this device, etc.) sends those callers
    // into the failure pattern below. A `try_lock` is a sub-us probe
    // that confirms the session mutex is grabbable, which is the
    // strongest "ready" signal we can give without doing real work.
    let session_lock = state.request_session.try_lock();
    if session_lock.is_err() {
        return Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "session_busy",
            "ax-engine-server has not finished initialising its inference session".into(),
        ));
    }
    drop(session_lock);
    Ok(Json(json!({
        "status": "ok",
        "service": "ax-engine-server",
        "model_id": state.model_id.as_ref(),
        "runtime": runtime_response(&state),
    })))
}

pub(crate) async fn runtime_info(State(state): State<AppState>) -> Json<ServerInfoResponse> {
    Json(server_info_response(&state))
}

pub(crate) async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let context_length = context_length(&state);
    let max_output_tokens = max_output_tokens(&state, context_length);
    let openai_text = openai_text_supported(&state);
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: state.model_id.to_string(),
            object: "model",
            owned_by: MODEL_OWNER,
            capabilities: model_capabilities(openai_text),
            limit: ModelLimit {
                context: context_length,
                output: max_output_tokens,
            },
            context_length,
            max_output_tokens,
            ax_engine: ax_engine_model_metadata(openai_text),
            runtime: runtime_response(&state),
        }],
    })
}

fn server_info_response(state: &AppState) -> ServerInfoResponse {
    ServerInfoResponse {
        service: "ax-engine-server",
        model_id: state.model_id.to_string(),
        deterministic: state.session_config.deterministic,
        max_batch_tokens: state.session_config.max_batch_tokens,
        block_size_tokens: state.session_config.kv_config.block_size_tokens,
        runtime: runtime_response(state),
    }
}

fn runtime_response(state: &AppState) -> RuntimeResponse {
    state.runtime_report.clone()
}

fn model_capabilities(openai_text: bool) -> ModelCapabilities {
    ModelCapabilities {
        temperature: openai_text,
        reasoning: false,
        attachment: false,
        toolcall: false,
        input: ModelModalities {
            text: openai_text,
            audio: false,
            image: false,
            video: false,
            pdf: false,
        },
        output: ModelModalities {
            text: openai_text,
            audio: false,
            image: false,
            video: false,
            pdf: false,
        },
        interleaved: false,
    }
}

fn ax_engine_model_metadata(openai_text: bool) -> AxEngineModelMetadata {
    AxEngineModelMetadata {
        native_generate_supported: true,
        openai_completions_supported: openai_text,
        openai_chat_completions_supported: openai_text,
        openai_tool_calling_supported: false,
        openai_text_input_supported: openai_text,
    }
}

fn openai_text_supported(state: &AppState) -> bool {
    // Keep this in sync with `validate_openai_text_backend` in `openai::validation`:
    // every backend that serves the OpenAI text endpoints must advertise them here.
    matches!(
        state.runtime_report.selected_backend,
        SelectedBackend::LlamaCpp | SelectedBackend::MlxLmDelegated | SelectedBackend::Mlx
    )
}

fn context_length(state: &AppState) -> u32 {
    state
        .session_config
        .kv_config
        .block_size_tokens
        .saturating_mul(state.session_config.kv_config.total_blocks)
}

fn max_output_tokens(state: &AppState, context_length: u32) -> u32 {
    state
        .session_config
        .max_batch_tokens
        .min(context_length)
        .max(1)
}
