use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;

use ax_engine_sdk::{RuntimeReport, SelectedBackend};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use parking_lot::Mutex;
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
    openai_audio_transcriptions_supported: bool,
    openai_audio_translations_supported: bool,
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
    let snapshots = state.snapshots();
    let mut operations = Vec::new();
    if snapshots.iter().any(openai_text_supported_live) {
        operations.extend([
            "chat_completions".to_string(),
            "completions".to_string(),
            "embeddings".to_string(),
        ]);
    }
    if snapshots.iter().any(whisper_supported_live) {
        operations.extend([
            "audio_transcriptions".to_string(),
            "audio_translations".to_string(),
        ]);
    }
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
    let whisper = whisper_supported_live(live);
    let native_multimodal = native_processed_multimodal_support_live(live);
    let openai_tool_calling = openai_tool_calling_supported_live(live, openai_text);
    let openai_reasoning = openai_reasoning_supported_live(live, openai_text);
    ModelCard {
        id: live.model_id.to_string(),
        object: "model",
        owned_by: MODEL_OWNER,
        capabilities: model_capabilities(
            openai_text,
            native_multimodal,
            openai_tool_calling,
            openai_reasoning,
            whisper,
        ),
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
            whisper,
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
    openai_reasoning: bool,
    whisper: bool,
) -> ModelCapabilities {
    ModelCapabilities {
        temperature: openai_text,
        reasoning: openai_reasoning,
        attachment: native_multimodal.any() || whisper,
        toolcall: openai_tool_calling,
        input: ModelModalities {
            text: openai_text,
            audio: native_multimodal.audio || whisper,
            image: native_multimodal.image,
            video: native_multimodal.video,
            pdf: false,
        },
        output: ModelModalities {
            text: openai_text || whisper,
            audio: false,
            image: false,
            video: false,
            pdf: false,
        },
        interleaved: native_multimodal.any() && !whisper,
    }
}

fn ax_engine_model_metadata(
    model_id: &str,
    openai_text: bool,
    native_multimodal: NativeProcessedMultimodalSupport,
    openai_tool_calling: bool,
    whisper: bool,
) -> AxEngineModelMetadata {
    let native_multimodal_input = native_multimodal.any();
    let coding_only = chat::is_qwen_coder_model(model_id);
    let coding_supported = openai_tool_calling
        && matches!(
            ChatPromptTemplate::for_model_id(model_id),
            ChatPromptTemplate::QwenChatMl
        );
    AxEngineModelMetadata {
        native_generate_supported: !whisper,
        openai_completions_supported: openai_text,
        openai_chat_completions_supported: openai_text,
        openai_audio_transcriptions_supported: whisper,
        openai_audio_translations_supported: whisper,
        openai_tool_calling_supported: openai_tool_calling,
        openai_text_input_supported: openai_text,
        native_multimodal_input_supported: native_multimodal_input,
        gemma4_unified_multimodal_input_supported: native_multimodal_input,
        openai_tokenized_multimodal_input_supported: native_multimodal_input,
        primary_use: if whisper {
            "speech_recognition"
        } else if coding_only {
            "coding"
        } else {
            "general"
        },
        chat_default: openai_text && !coding_only,
        coding_supported,
        coding_only,
    }
}

fn openai_reasoning_supported_live(live: &LiveState, openai_text: bool) -> bool {
    openai_text
        && live.runtime_report.selected_backend == SelectedBackend::Mlx
        && (chat::is_qwen_thinking_model(live.model_id.as_ref())
            || chat::is_deepseek_model(live.model_id.as_ref())
            || matches!(
                chat::resolve_chat_template(
                    live.model_id.as_ref(),
                    model_family_from_artifacts(live).as_deref()
                ),
                ChatPromptTemplate::Gemma4
            ))
}

pub(crate) fn model_supports_reasoning(live: &LiveState) -> bool {
    openai_reasoning_supported_live(live, openai_text_supported_live(live))
}

pub(crate) fn model_supports_image(live: &LiveState) -> bool {
    native_processed_multimodal_support_live(live).image
}

/// Single source of truth for tool-call capability, shared by `/v1/models`
/// and the Ollama discovery/gating surfaces so they can never disagree.
pub(crate) fn model_supports_tool_calling(live: &LiveState) -> bool {
    openai_tool_calling_supported_live(live, openai_text_supported_live(live))
}

fn openai_tool_calling_supported_live(live: &LiveState, openai_text: bool) -> bool {
    openai_text
        && live.runtime_report.selected_backend == SelectedBackend::Mlx
        && matches!(
            chat::resolve_chat_template(
                live.model_id.as_ref(),
                model_family_from_artifacts(live).as_deref()
            ),
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

    let gemma4_unified_image = GEMMA4_UNIFIED_VISION_ROLES
        .iter()
        .all(|role| has_global_tensor_role(tensors, role));
    let family = family_from_manifest(&manifest).unwrap_or_default();
    // Standard encoder-VL packaging (family `gemma4` or `gemma4_vl`) ships a
    // ViT under vision_tower.* plus embed_vision projection — same capability
    // surface; gemma4_vl is only a separate label for registry/gating.
    let gemma4_standard_image = matches!(family.as_str(), "gemma4" | "gemma4_vl")
        && (has_tensor_name_prefix(tensors, "vision_tower.")
            || has_tensor_name_prefix(tensors, "model.vision_tower."))
        && (has_tensor_name_prefix(tensors, "embed_vision.")
            || has_tensor_name_prefix(tensors, "model.embed_vision."));
    let qwen3_vl_image = (matches!(family.as_str(), "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5")
        && (has_tensor_name_prefix(tensors, "vision_tower.")
            || has_tensor_name_prefix(tensors, "visual.")
            || has_tensor_name_prefix(tensors, "model.visual.")))
        || has_global_tensor_role(tensors, QWEN3_VL_VISION_PATCH_EMBED_ROLE)
        || has_global_tensor_role(tensors, QWEN3_VL_VISION_MERGER_ROLE);
    let minicpm_v46_image = family == "minicpmv4_6"
        && (has_tensor_name_prefix(tensors, "vision_tower.")
            || has_tensor_name_prefix(tensors, "model.vision_tower.")
            || has_tensor_name_prefix(tensors, "model.vpm."))
        && (has_tensor_name_prefix(tensors, "vit_merger.")
            || has_tensor_name_prefix(tensors, "merger.")
            || has_tensor_name_prefix(tensors, "model.vit_merger.")
            || has_tensor_name_prefix(tensors, "model.merger."));
    let nemotron_omni_image = has_tensor_name_prefix(tensors, "vision_model.radio_model.")
        && has_tensor_name_prefix(tensors, "mlp1.");
    let nemotron_omni_audio = has_tensor_name_prefix(tensors, "sound_encoder.")
        && has_tensor_name_prefix(tensors, "sound_projection.");
    let image = gemma4_unified_image
        || gemma4_standard_image
        || qwen3_vl_image
        || minicpm_v46_image
        || nemotron_omni_image;
    let audio = has_global_tensor_role(tensors, GEMMA4_UNIFIED_AUDIO_ROLE) || nemotron_omni_audio;
    // Advertise video only when the loaded tower has the corresponding native
    // frame path: Gemma4 unified or Qwen3-VL/Qwen3.5. Media is data-URI only
    // (no remote fetch), and convert-time media drops fail the capability closed.
    let media_drops = manifest
        .get("dropped_tensors")
        .and_then(|d| d.get("media_role_hits"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let video_env_off = matches!(
        std::env::var("AX_MLX_GEMMA4_VIDEO")
            .unwrap_or_else(|_| "on".into())
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    );
    // Standard Gemma checkpoints may intentionally omit the optional
    // Conformer audio tower. That must not suppress their independent
    // per-frame ViT video path.
    let gemma4_video = !video_env_off
        && (gemma4_standard_image
            || (gemma4_unified_image
                && media_drops == 0
                && (family == "gemma4_unified" || family.starts_with("gemma4"))));
    let qwen3_vl_video = qwen3_vl_image && media_drops == 0;
    let video = gemma4_video || qwen3_vl_video;
    NativeProcessedMultimodalSupport {
        image,
        audio,
        video,
    }
}

fn family_from_manifest(manifest: &Value) -> Option<String> {
    manifest
        .get("model_family")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Read `model_family` from the live session's model-manifest.json when present.
///
/// Cached per manifest path with a `(len, mtime)` fingerprint (same pattern
/// as `validate_gemma4_instruct_eos_cached`), because per-request chat
/// resolution (ADR-025 D2) consults this on every chat request and the
/// manifest embeds the full tensor list.
pub(crate) fn model_family_from_artifacts(live: &LiveState) -> Option<String> {
    if live.runtime_report.selected_backend != SelectedBackend::Mlx {
        return None;
    }
    let artifacts_dir = live.session_config.mlx_model_artifacts_dir()?;
    let manifest_path = artifacts_dir.join("model-manifest.json");
    model_family_from_manifest_path_cached(&manifest_path)
}

type ManifestFingerprint = Option<(u64, SystemTime)>;

fn model_family_from_manifest_path_cached(manifest_path: &Path) -> Option<String> {
    type FamilyCache = HashMap<PathBuf, (ManifestFingerprint, Option<String>)>;
    static FAMILIES: OnceLock<Mutex<FamilyCache>> = OnceLock::new();
    let cache = FAMILIES.get_or_init(|| Mutex::new(HashMap::new()));
    let fingerprint = std::fs::metadata(manifest_path)
        .ok()
        .and_then(|meta| Some((meta.len(), meta.modified().ok()?)));
    if let Some((cached_fingerprint, family)) = cache.lock().get(manifest_path)
        && *cached_fingerprint == fingerprint
    {
        return family.clone();
    }
    let family = std::fs::read(manifest_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|manifest| family_from_manifest(&manifest));
    cache
        .lock()
        .insert(manifest_path.to_path_buf(), (fingerprint, family.clone()));
    family
}

fn has_global_tensor_role(tensors: &[Value], role: &str) -> bool {
    tensors.iter().any(|tensor| {
        tensor.get("role").and_then(Value::as_str) == Some(role)
            && tensor
                .get("layer_index")
                .is_none_or(|layer_index| layer_index.is_null())
    })
}

fn has_tensor_name_prefix(tensors: &[Value], prefix: &str) -> bool {
    tensors.iter().any(|tensor| {
        tensor
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| name.starts_with(prefix))
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
    !whisper_supported_live(live)
        && matches!(
            live.runtime_report.selected_backend,
            SelectedBackend::LlamaCpp | SelectedBackend::MlxLmDelegated | SelectedBackend::Mlx
        )
}

fn whisper_supported_live(live: &LiveState) -> bool {
    live.runtime_report.selected_backend == SelectedBackend::Mlx
        && model_family_from_artifacts(live).as_deref() == Some("whisper")
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

pub(crate) fn max_output_tokens_live(live: &LiveState, context_length: u32) -> u32 {
    // Advertise the per-request output budget bounded by the model context
    // window. An explicit `max_output_tokens` override wins so operators can
    // decouple the client-facing output budget from the scheduler batch width
    // (`max_batch_tokens`); otherwise fall back to the batch width. A previous
    // fixed `512` ceiling under-reported the real capacity (the model can
    // generate up to its full context), so it was removed.
    live.session_config
        .max_output_tokens
        .unwrap_or(live.session_config.max_batch_tokens)
        .min(context_length)
        .max(1)
}
