use ax_engine_sdk::{
    RuntimeReport, SelectedBackend, UNLIMITED_OCR_DEFAULT_CONTEXT_LENGTH,
    UNLIMITED_OCR_DEFAULT_MAX_OUTPUT_TOKENS, VllmModelProfile,
};
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
    let unavailable = state.unavailable_model_ids();
    if !unavailable.is_empty() {
        return Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "generation_worker_unavailable",
            format!(
                "native generation workers are unavailable for loaded models: {}",
                unavailable.join(", ")
            ),
        ));
    }
    let live = state.snapshot();
    Ok(Json(json!({
        "status": "ok",
        "service": "ax-engine-server",
        "model_id": live.model_id.as_ref(),
        "models": state.model_ids(),
        "runtime": live.runtime_report.clone(),
    })))
}

/// Unauthenticated discovery document for LAN browse verification.
/// Schema: `ax.engine.discovery.v1` (see docs/LAN-DISCOVERY.md).
///
/// Fail closed when the generation worker is down so agents do not register a
/// dead peer after mDNS browse (same readiness bar as `/health`).
pub(crate) async fn discovery_info(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let unavailable = state.unavailable_model_ids();
    if !unavailable.is_empty() {
        return Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "generation_worker_unavailable",
            format!(
                "native generation workers are unavailable for loaded models: {}",
                unavailable.join(", ")
            ),
        ));
    }
    let live = state.snapshot();
    let auth_required = state.api_key.is_some();
    let mut operations = vec![
        "chat_completions".to_string(),
        "completions".to_string(),
        "embeddings".to_string(),
    ];
    operations.sort();
    Ok(Json(json!({
        "schema": "ax.engine.discovery.v1",
        "service": "ax-engine-server",
        "version": state.discovery.version,
        "model_id": live.model_id.as_ref(),
        "models": state.model_ids(),
        "auth_required": auth_required,
        "openai_base_path": "/v1",
        "operations": operations,
        "cluster": state.discovery.cluster,
        "instance_id": state.discovery.instance_id,
        "runtime": live.runtime_report.clone(),
    })))
}

pub(crate) async fn runtime_info(State(state): State<AppState>) -> Json<ServerInfoResponse> {
    let live = state.snapshot();
    Json(server_info_response(&live))
}

pub(crate) async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let data = state.snapshots().iter().map(model_card).collect();
    Json(ModelsResponse {
        object: "list",
        data,
    })
}

fn model_card(live: &LiveState) -> ModelCard {
    let context_length = context_length(live);
    let max_output_tokens = max_output_tokens_live(live, context_length);
    let openai_text = openai_text_supported_live(live);
    let native_multimodal = native_processed_multimodal_support_live(live);
    let delegated_multimodal = delegated_multimodal_support_live(live);
    let advertised_multimodal = NativeProcessedMultimodalSupport {
        image: native_multimodal.image || delegated_multimodal.image,
        audio: native_multimodal.audio || delegated_multimodal.audio,
        video: native_multimodal.video || delegated_multimodal.video,
    };
    let openai_tool_calling = openai_tool_calling_supported_live(live, openai_text);
    ModelCard {
        id: live.model_id.to_string(),
        object: "model",
        owned_by: MODEL_OWNER,
        capabilities: model_capabilities(openai_text, advertised_multimodal, openai_tool_calling),
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
    }
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

/// Public capability probe used by request rejection paths (WS-M1).
pub(crate) fn model_supports_video(live: &LiveState) -> bool {
    native_processed_multimodal_support_live(live).video
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

    let gemma4_image = GEMMA4_UNIFIED_VISION_ROLES
        .iter()
        .all(|role| has_global_tensor_role(tensors, role));
    let qwen3_vl_image = has_global_tensor_role(tensors, QWEN3_VL_VISION_PATCH_EMBED_ROLE)
        || has_global_tensor_role(tensors, QWEN3_VL_VISION_MERGER_ROLE)
        || family_from_manifest(&manifest).is_some_and(|f| f == "qwen3_vl" || f == "qwen3_vl_moe");
    let image = gemma4_image || qwen3_vl_image;
    let audio = has_global_tensor_role(tensors, GEMMA4_UNIFIED_AUDIO_ROLE);
    // WS-M1: advertise video only for gemma4_unified manifests that already
    // have vision roles, have no convert-time media drops, and are not
    // disabled via AX_MLX_GEMMA4_VIDEO=off. Media is data-URI only (no remote
    // fetch). Frame caps keep expanded soft tokens under atomic max_batch_tokens.
    let media_drops = manifest
        .get("dropped_tensors")
        .and_then(|d| d.get("media_role_hits"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let family = family_from_manifest(&manifest).unwrap_or_default();
    let video_env_off = matches!(
        std::env::var("AX_MLX_GEMMA4_VIDEO")
            .unwrap_or_else(|_| "on".into())
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    );
    let video = gemma4_image
        && !video_env_off
        && media_drops == 0
        && (family == "gemma4" || family == "gemma4_unified" || family.starts_with("gemma4"));
    NativeProcessedMultimodalSupport {
        image,
        audio,
        video,
    }
}

fn delegated_multimodal_support_live(live: &LiveState) -> NativeProcessedMultimodalSupport {
    if live.runtime_report.selected_backend != SelectedBackend::Vllm {
        return NativeProcessedMultimodalSupport::default();
    }
    let unlimited_ocr = live
        .session_config
        .vllm_backend
        .as_ref()
        .is_some_and(|config| config.server().model_profile == VllmModelProfile::UnlimitedOcr);
    NativeProcessedMultimodalSupport {
        image: unlimited_ocr,
        ..NativeProcessedMultimodalSupport::default()
    }
}

fn family_from_manifest(manifest: &Value) -> Option<String> {
    manifest
        .get("model_family")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Read `model_family` from the live session's model-manifest.json when present.
pub(crate) fn model_family_from_artifacts(live: &LiveState) -> Option<String> {
    if live.runtime_report.selected_backend != SelectedBackend::Mlx {
        return None;
    }
    let artifacts_dir = live.session_config.mlx_model_artifacts_dir()?;
    let manifest_path = artifacts_dir.join("model-manifest.json");
    let bytes = std::fs::read(manifest_path).ok()?;
    let manifest: Value = serde_json::from_slice(&bytes).ok()?;
    family_from_manifest(&manifest)
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

const QWEN3_VL_VISION_PATCH_EMBED_ROLE: &str = "qwen3_vl_vision_patch_embed";
const QWEN3_VL_VISION_MERGER_ROLE: &str = "qwen3_vl_vision_merger";

fn openai_text_supported_live(live: &LiveState) -> bool {
    // Keep this in sync with `validate_openai_text_backend` in `openai::validation`:
    // every backend that serves the OpenAI text endpoints must advertise them here.
    matches!(
        live.runtime_report.selected_backend,
        SelectedBackend::LlamaCpp
            | SelectedBackend::MlxLmDelegated
            | SelectedBackend::TensorRtEdgeLlm
            | SelectedBackend::TensorRtLlm
            | SelectedBackend::Vllm
            | SelectedBackend::Mlx
    )
}

/// Computes context length from the caller's `LiveState` snapshot — callers
/// must pass the snapshot they are already serving the request from, never a
/// fresh one, so all fields in a response come from the same model.
pub(crate) fn context_length(live: &LiveState) -> u32 {
    if live.runtime_report.selected_backend == SelectedBackend::Vllm {
        if let Some(max_model_len) = live
            .session_config
            .vllm_readiness
            .as_ref()
            .and_then(|readiness| readiness.max_model_len)
        {
            return max_model_len;
        }
        if live
            .session_config
            .vllm_backend
            .as_ref()
            .is_some_and(|config| config.server().model_profile == VllmModelProfile::UnlimitedOcr)
        {
            return UNLIMITED_OCR_DEFAULT_CONTEXT_LENGTH;
        }
    }
    live.session_config
        .kv_config
        .block_size_tokens
        .saturating_mul(live.session_config.kv_config.total_blocks)
}

fn max_output_tokens_live(live: &LiveState, context_length: u32) -> u32 {
    if live.runtime_report.selected_backend == SelectedBackend::Vllm
        && live
            .session_config
            .vllm_backend
            .as_ref()
            .is_some_and(|config| config.server().model_profile == VllmModelProfile::UnlimitedOcr)
    {
        return UNLIMITED_OCR_DEFAULT_MAX_OUTPUT_TOKENS.min(context_length);
    }
    // Advertise the per-request output budget bounded by the scheduler batch
    // width and the model context window. A previous fixed `512` ceiling
    // under-reported the real capacity (the model can generate up to its full
    // context), so it was removed.
    live.session_config
        .max_batch_tokens
        .min(context_length)
        .max(1)
}
