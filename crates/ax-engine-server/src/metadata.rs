use ax_engine_sdk::{RuntimeReport, SelectedBackend};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Serialize;
use serde_json::{Value, json};

use crate::app_state::{AppState, LiveState};
use crate::chat::{self, ChatPromptTemplate};
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
    native_multimodal_input_supported: bool,
    gemma4_unified_multimodal_input_supported: bool,
    openai_tokenized_multimodal_input_supported: bool,
    primary_use: &'static str,
    chat_default: bool,
    coding_supported: bool,
    coding_only: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct NativeProcessedMultimodalSupport {
    image: bool,
    audio: bool,
    video: bool,
}

impl NativeProcessedMultimodalSupport {
    const fn any(self) -> bool {
        self.image || self.audio || self.video
    }
}

pub(crate) async fn health(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let live = state.snapshot();
    // `/health` is the readiness probe most callers (bench harness,
    // k8s, load balancers) poll while a server starts. Returning 200
    // when the server has bound a port but the inference session is
    // wedged (deadlocked on another in-flight call, runtime panicked,
    // weights not loadable on this device, etc.) sends those callers
    // into the failure pattern below. A `try_lock` is a sub-us probe
    // that confirms the session mutex is grabbable, which is the
    // strongest "ready" signal we can give without doing real work.
    let session_lock = live.request_session.try_lock();
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
        "model_id": live.model_id.as_ref(),
        "runtime": live.runtime_report.clone(),
    })))
}

pub(crate) async fn runtime_info(State(state): State<AppState>) -> Json<ServerInfoResponse> {
    let live = state.snapshot();
    Json(server_info_response(&live))
}

pub(crate) async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let live = state.snapshot();
    let context_length = context_length(&live);
    let max_output_tokens = max_output_tokens_live(&live, context_length);
    let openai_text = openai_text_supported_live(&live);
    let native_multimodal = native_processed_multimodal_support_live(&live);
    let openai_tool_calling = openai_tool_calling_supported_live(&live, openai_text);
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: live.model_id.to_string(),
            object: "model",
            owned_by: MODEL_OWNER,
            capabilities: model_capabilities(openai_text, native_multimodal, openai_tool_calling),
            limit: ModelLimit {
                context: context_length,
                output: max_output_tokens,
            },
            context_length,
            max_output_tokens,
            ax_engine: ax_engine_model_metadata(
                live.model_id.as_ref(),
                openai_text,
                native_multimodal,
                openai_tool_calling,
            ),
            runtime: live.runtime_report.clone(),
        }],
    })
}

fn server_info_response(live: &LiveState) -> ServerInfoResponse {
    ServerInfoResponse {
        service: "ax-engine-server",
        model_id: live.model_id.to_string(),
        deterministic: live.session_config.deterministic,
        max_batch_tokens: live.session_config.max_batch_tokens,
        block_size_tokens: live.session_config.kv_config.block_size_tokens,
        runtime: live.runtime_report.clone(),
    }
}

fn model_capabilities(
    openai_text: bool,
    native_multimodal: NativeProcessedMultimodalSupport,
    openai_tool_calling: bool,
) -> ModelCapabilities {
    ModelCapabilities {
        temperature: openai_text,
        reasoning: false,
        attachment: native_multimodal.any(),
        toolcall: openai_tool_calling,
        input: ModelModalities {
            text: openai_text,
            audio: native_multimodal.audio,
            image: native_multimodal.image,
            video: native_multimodal.video,
            pdf: false,
        },
        output: ModelModalities {
            text: openai_text,
            audio: false,
            image: false,
            video: false,
            pdf: false,
        },
        interleaved: native_multimodal.any(),
    }
}

fn ax_engine_model_metadata(
    model_id: &str,
    openai_text: bool,
    native_multimodal: NativeProcessedMultimodalSupport,
    openai_tool_calling: bool,
) -> AxEngineModelMetadata {
    let native_multimodal_input = native_multimodal.any();
    let coding_only = chat::is_qwen_coder_model(model_id);
    let coding_supported = openai_tool_calling
        && matches!(
            ChatPromptTemplate::for_model_id(model_id),
            ChatPromptTemplate::QwenChatMl
        );
    AxEngineModelMetadata {
        native_generate_supported: true,
        openai_completions_supported: openai_text,
        openai_chat_completions_supported: openai_text,
        openai_tool_calling_supported: openai_tool_calling,
        openai_text_input_supported: openai_text,
        native_multimodal_input_supported: native_multimodal_input,
        gemma4_unified_multimodal_input_supported: native_multimodal_input,
        openai_tokenized_multimodal_input_supported: native_multimodal_input,
        primary_use: if coding_only { "coding" } else { "general" },
        chat_default: openai_text && !coding_only,
        coding_supported,
        coding_only,
    }
}

fn openai_tool_calling_supported_live(live: &LiveState, openai_text: bool) -> bool {
    openai_text
        && live.runtime_report.selected_backend == SelectedBackend::Mlx
        && matches!(
            ChatPromptTemplate::for_model_id(live.model_id.as_ref()),
            ChatPromptTemplate::QwenChatMl | ChatPromptTemplate::Gemma4 | ChatPromptTemplate::Glm47
        )
}

fn native_processed_multimodal_support_live(live: &LiveState) -> NativeProcessedMultimodalSupport {
    if live.runtime_report.selected_backend != SelectedBackend::Mlx {
        return NativeProcessedMultimodalSupport::default();
    }

    let Some(artifacts_dir) = live.session_config.mlx_model_artifacts_dir() else {
        return NativeProcessedMultimodalSupport::default();
    };
    let manifest_path = artifacts_dir.join("model-manifest.json");
    let Ok(manifest_bytes) = std::fs::read(manifest_path) else {
        return NativeProcessedMultimodalSupport::default();
    };
    let Ok(manifest) = serde_json::from_slice::<Value>(&manifest_bytes) else {
        return NativeProcessedMultimodalSupport::default();
    };
    let Some(tensors) = manifest.get("tensors").and_then(Value::as_array) else {
        return NativeProcessedMultimodalSupport::default();
    };

    let image = GEMMA4_UNIFIED_VISION_ROLES
        .iter()
        .all(|role| has_global_tensor_role(tensors, role));
    let audio = has_global_tensor_role(tensors, GEMMA4_UNIFIED_AUDIO_ROLE);
    NativeProcessedMultimodalSupport {
        image,
        audio,
        video: image,
    }
}

fn has_global_tensor_role(tensors: &[Value], role: &str) -> bool {
    tensors.iter().any(|tensor| {
        tensor.get("role").and_then(Value::as_str) == Some(role)
            && tensor
                .get("layer_index")
                .is_none_or(|layer_index| layer_index.is_null())
    })
}

const GEMMA4_UNIFIED_VISION_ROLES: &[&str] = &[
    "gemma4_unified_vision_patch_dense",
    "gemma4_unified_vision_patch_dense_bias",
    "gemma4_unified_vision_patch_norm1",
    "gemma4_unified_vision_patch_norm1_bias",
    "gemma4_unified_vision_patch_norm2",
    "gemma4_unified_vision_patch_norm2_bias",
    "gemma4_unified_vision_position_embedding",
    "gemma4_unified_vision_position_norm",
    "gemma4_unified_vision_position_norm_bias",
    "gemma4_unified_vision_projection",
];

const GEMMA4_UNIFIED_AUDIO_ROLE: &str = "gemma4_unified_audio_projection";

fn openai_text_supported_live(live: &LiveState) -> bool {
    // Keep this in sync with `validate_openai_text_backend` in `openai::validation`:
    // every backend that serves the OpenAI text endpoints must advertise them here.
    matches!(
        live.runtime_report.selected_backend,
        SelectedBackend::LlamaCpp | SelectedBackend::MlxLmDelegated | SelectedBackend::Mlx
    )
}

/// Computes context length from the caller's `LiveState` snapshot — callers
/// must pass the snapshot they are already serving the request from, never a
/// fresh one, so all fields in a response come from the same model.
pub(crate) fn context_length(live: &LiveState) -> u32 {
    live.session_config
        .kv_config
        .block_size_tokens
        .saturating_mul(live.session_config.kv_config.total_blocks)
}

fn max_output_tokens_live(live: &LiveState, context_length: u32) -> u32 {
    // Advertise the per-request output budget bounded by the scheduler batch
    // width and the model context window. A previous fixed `512` ceiling
    // under-reported the real capacity (the model can generate up to its full
    // context), so it was removed.
    live.session_config
        .max_batch_tokens
        .min(context_length)
        .max(1)
}
