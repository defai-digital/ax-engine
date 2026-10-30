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
    /// The Gemma 4 unified (processed-tensor) vision contract specifically;
    /// encoder-VL / Qwen3-VL / MiniCPM-V packs take a different payload.
    gemma4_unified: bool,
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
        operations.extend(["chat_completions".to_string(), "completions".to_string()]);
    }
    // Only the native MLX backend embeds; delegated text backends cannot.
    if snapshots.iter().any(|live| {
        live.runtime_report.selected_backend == SelectedBackend::Mlx
            && (openai_text_supported_live(live) || encoder_embed_live(live))
    }) {
        operations.push("embeddings".to_string());
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
            encoder_embed_live(live),
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
    encoder_embed: bool,
) -> AxEngineModelMetadata {
    let native_multimodal_input = native_multimodal.any();
    let coding_only = chat::is_qwen_coder_model(model_id);
    let coding_supported = openai_tool_calling
        && matches!(
            ChatPromptTemplate::for_model_id(model_id),
            ChatPromptTemplate::QwenChatMl
        );
    AxEngineModelMetadata {
        native_generate_supported: !whisper && !encoder_embed,
        openai_completions_supported: openai_text,
        openai_chat_completions_supported: openai_text,
        openai_audio_transcriptions_supported: whisper,
        openai_audio_translations_supported: whisper,
        openai_tool_calling_supported: openai_tool_calling,
        openai_text_input_supported: openai_text,
        native_multimodal_input_supported: native_multimodal_input,
        // Only packs with the unified processed-tensor vision roles accept
        // the Gemma 4 unified payload; other native multimodal families
        // must not be advertised as such.
        gemma4_unified_multimodal_input_supported: native_multimodal.gemma4_unified,
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
    let family = model_family_from_artifacts(live);
    openai_text
        && live.runtime_report.selected_backend == SelectedBackend::Mlx
        // DiffusionGemma renders with the Gemma 4 template but never prefills
        // a thought channel, so it must not advertise reasoning.
        && !chat::is_diffusion_gemma(live.model_id.as_ref())
        && family.as_deref() != Some("diffusion_gemma")
        && (chat::is_qwen_thinking_model(live.model_id.as_ref())
            || chat::is_deepseek_model(live.model_id.as_ref())
            // Hybrid-thinking Qwen families keep the contract under any id.
            || matches!(family.as_deref(), Some("qwen3_5" | "qwen4_exp"))
            // MiniMax M3 and Muse ATEM render a thinking switch too, but no
            // response path splits `</mm:think>` or `to=self` reasoning yet.
            || matches!(
                chat::resolve_chat_template(live.model_id.as_ref(), family.as_deref()),
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
    // The loaded runtime identity wins, as in the Flash Next request gates:
    // an auto-converted snapshot need not persist a manifest, and the tower
    // is read from the safetensors index, not the manifest.
    let runtime_family = live
        .runtime_report
        .mlx_model
        .as_ref()
        .map(|model| model.model_family.as_str());
    if runtime_family == Some("qwen4_exp") {
        return NativeProcessedMultimodalSupport {
            image: qwen4_exp_vision_enabled() && qwen4_exp_vision_tower_loadable(artifacts_dir),
            ..NativeProcessedMultimodalSupport::default()
        };
    }
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
    // Flash Next ships its tower outside the manifest; the runtime loads it
    // only behind the preview flag and refuses DeepStack towers.
    let qwen4_exp_image = family == "qwen4_exp"
        && qwen4_exp_vision_enabled()
        && qwen4_exp_vision_tower_loadable(artifacts_dir);
    let image = gemma4_unified_image
        || gemma4_standard_image
        || qwen3_vl_image
        || minicpm_v46_image
        || nemotron_omni_image
        || qwen4_exp_image;
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
        gemma4_unified: gemma4_unified_image,
    }
}

/// Mirrors the runtime's `qwen4_exp::load_vision_tower` and
/// `qwen3_vl::load_qwen3_vl_vision_weights` at the name level: the tower is
/// listed in `model.safetensors.index.json` (not the manifest) under any of
/// the `vision_tower.` / `visual.` / `model.visual.` spellings, every index
/// entry must be a string, the tower shards must resolve inside the pack (or
/// its own HF blob store), `config.json` must carry a Qwen-visual
/// `model_type` and a complete numeric `vision_config`, every tensor implied
/// by `depth` plus the merger must be listed, and DeepStack fails closed. The
/// runtime degrades to text-only on any of these, so the probe must too.
/// Tensor shapes and shard contents are only checked at load time.
fn qwen4_exp_vision_tower_loadable(artifacts_dir: &std::path::Path) -> bool {
    const VISUAL_PREFIXES: [&str; 3] = ["vision_tower.", "visual.", "model.visual."];
    const VISION_CONFIG_KEYS: [&str; 10] = [
        "depth",
        "hidden_size",
        "intermediate_size",
        "out_hidden_size",
        "num_heads",
        "in_channels",
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
        "num_position_embeddings",
    ];
    const LAYER_TENSORS: [&str; 6] = [
        "attn.qkv.weight",
        "attn.proj.weight",
        "norm1.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        "norm2.weight",
    ];
    const FIXED_TENSORS: [&str; 5] = [
        "patch_embed.proj.weight",
        "pos_embed.weight",
        "merger.norm.weight",
        "merger.linear_fc1.weight",
        "merger.linear_fc2.weight",
    ];
    let read_json = |name: &str| {
        std::fs::read(artifacts_dir.join(name))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
    };
    let Some(config) = read_json("config.json") else {
        return false;
    };
    let qwen_visual_model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .is_some_and(|model_type| {
            matches!(
                model_type,
                "qwen3_vl"
                    | "qwen3-vl"
                    | "qwen3_vl_moe"
                    | "qwen3-vl-moe"
                    | "qwen3_5"
                    | "qwen3.5"
                    | "qwen3_5_moe"
                    | "qwen3_next"
                    | "qwen3.6"
                    | "qwen3_6"
                    | "qwen3_5_moe_text"
                    | "qwen4_exp"
            )
        });
    if !qwen_visual_model_type {
        return false;
    }
    let Some(vision_config) = config.get("vision_config").and_then(Value::as_object) else {
        return false;
    };
    let config_number = |key: &str| {
        vision_config
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
    };
    if !VISION_CONFIG_KEYS
        .iter()
        .all(|key| config_number(key).is_some())
    {
        return false;
    }
    let deepstack = vision_config
        .get("deepstack_visual_indexes")
        .and_then(Value::as_array)
        .is_some_and(|indexes| !indexes.is_empty());
    if deepstack {
        return false;
    }
    let Some(depth) = config_number("depth") else {
        return false;
    };
    let Some(index) = read_json("model.safetensors.index.json") else {
        return false;
    };
    let Some(weight_map) = index.get("weight_map").and_then(Value::as_object) else {
        return false;
    };
    // The runtime's index reader rejects the whole index on any non-string
    // entry, not only on tower entries.
    if !weight_map.values().all(Value::is_string) {
        return false;
    }
    let Ok(canonical_root) = artifacts_dir.canonicalize() else {
        return false;
    };
    let shard_resolves = |suffix: &str| {
        VISUAL_PREFIXES
            .iter()
            .filter_map(|prefix| weight_map.get(&format!("{prefix}{suffix}")))
            .filter_map(Value::as_str)
            .any(|file| shard_inside_pack(artifacts_dir, &canonical_root, file))
    };
    FIXED_TENSORS.iter().all(|suffix| shard_resolves(suffix))
        && (0..depth).all(|layer| {
            LAYER_TENSORS
                .iter()
                .all(|suffix| shard_resolves(&format!("blocks.{layer}.{suffix}")))
        })
}

/// Same containment rule as the runtime's `artifact_path::resolve_file`: a
/// shard resolves inside the pack, or, for a Hugging Face cache snapshot,
/// into that repository's own `blobs/` directory.
fn shard_inside_pack(
    artifacts_dir: &std::path::Path,
    canonical_root: &std::path::Path,
    file: &str,
) -> bool {
    let Ok(canonical) = artifacts_dir.join(file).canonicalize() else {
        return false;
    };
    if !canonical.is_file() {
        return false;
    }
    if canonical.starts_with(canonical_root) {
        return true;
    }
    let Some(snapshots) = canonical_root.parent() else {
        return false;
    };
    snapshots
        .file_name()
        .is_some_and(|name| name == "snapshots")
        && snapshots
            .parent()
            .is_some_and(|repo| canonical.parent() == Some(repo.join("blobs").as_path()))
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
        && !encoder_embed_live(live)
        && matches!(
            live.runtime_report.selected_backend,
            SelectedBackend::LlamaCpp | SelectedBackend::MlxLmDelegated | SelectedBackend::Mlx
        )
}

#[cfg(test)]
pub(crate) fn openai_text_supported_for_tests(live: &LiveState) -> bool {
    openai_text_supported_live(live)
}

/// Embedding encoders serve `/v1/embeddings` only; the OpenAI text validator
/// rejects them, so they must not advertise chat or completions.
fn encoder_embed_live(live: &LiveState) -> bool {
    model_family_from_artifacts(live)
        .as_deref()
        .and_then(ax_engine_core::lookup_architecture)
        .is_some_and(|entry| {
            entry.default_generation == ax_engine_core::GenerationKind::EncoderEmbed
        })
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

/// Mirrors the runtime's `AX_MLX_QWEN4_EXP_VISION` (default off while image
/// input is a preview; `1`, `true`, `yes` or `on` turn it on), which decides
/// whether the Flash Next vision tower was loaded. Read once per process,
/// like the runtime's own flag.
pub(crate) fn qwen4_exp_vision_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("AX_MLX_QWEN4_EXP_VISION").is_ok_and(|raw| {
            matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    })
}

/// `AX_ENGINE_JSON_CONSTRAINED_DECODING` (default on; `0`, `false`, `no` or
/// `off` turn it off): native MLX `response_format` requests constrain
/// decoding to their grammar instead of relying on post-hoc validation
/// alone. Read once per process.
pub(crate) fn json_constrained_decoding_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("AX_ENGINE_JSON_CONSTRAINED_DECODING").map_or(true, |raw| {
            !matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod qwen4_exp_vision_tower_tests {
    use super::qwen4_exp_vision_tower_loadable;
    use serde_json::{Value, json};

    const SHARD: &str = "model-00001.safetensors";

    fn tower_map(prefix: &str, depth: usize) -> Value {
        let mut names = vec![
            "patch_embed.proj.weight".to_string(),
            "pos_embed.weight".to_string(),
            "merger.norm.weight".to_string(),
            "merger.linear_fc1.weight".to_string(),
            "merger.linear_fc2.weight".to_string(),
        ];
        for layer in 0..depth {
            for suffix in [
                "attn.qkv.weight",
                "attn.proj.weight",
                "norm1.weight",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc2.weight",
                "norm2.weight",
            ] {
                names.push(format!("blocks.{layer}.{suffix}"));
            }
        }
        Value::Object(
            names
                .into_iter()
                .map(|name| (format!("{prefix}{name}"), json!(SHARD)))
                .collect(),
        )
    }

    fn config(depth: usize, deepstack: Value) -> Value {
        json!({
            "model_type": "qwen4_exp",
            "vision_config": {
                "depth": depth,
                "hidden_size": 64,
                "intermediate_size": 128,
                "out_hidden_size": 64,
                "num_heads": 4,
                "in_channels": 3,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "num_position_embeddings": 64,
                "deepstack_visual_indexes": deepstack,
            },
        })
    }

    fn loadable(label: &str, weight_map: Value, config: Value, write_shard: bool) -> bool {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ax-engine-server-qwen4-vision-{label}-{unique}"));
        std::fs::create_dir_all(&dir).expect("create pack dir");
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({ "weight_map": weight_map })).expect("index json"),
        )
        .expect("write index");
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config).expect("config json"),
        )
        .expect("write config");
        if write_shard {
            std::fs::write(dir.join(SHARD), b"").expect("write shard");
        }
        let result = qwen4_exp_vision_tower_loadable(&dir);
        let _ = std::fs::remove_dir_all(dir);
        result
    }

    #[test]
    fn complete_tower_without_deepstack_is_loadable() {
        for prefix in ["vision_tower.", "visual.", "model.visual."] {
            assert!(
                loadable("ok", tower_map(prefix, 2), config(2, json!([])), true),
                "{prefix}"
            );
        }
    }

    #[test]
    fn incomplete_or_unloadable_towers_are_not_loadable() {
        let complete = || tower_map("vision_tower.", 2);
        assert!(!loadable(
            "none",
            json!({ "model.layers.0.mlp.weight": SHARD }),
            config(2, json!([])),
            true,
        ));
        // depth 3 needs blocks.2.* which the map does not list.
        assert!(!loadable("short", complete(), config(3, json!([])), true));
        let mut no_merger = complete();
        no_merger
            .as_object_mut()
            .expect("map")
            .remove("vision_tower.merger.linear_fc2.weight");
        assert!(!loadable(
            "no-merger",
            no_merger,
            config(2, json!([])),
            true
        ));
        let mut non_string = complete();
        non_string
            .as_object_mut()
            .expect("map")
            .insert("model.norm.weight".into(), json!(1));
        assert!(!loadable(
            "non-string",
            non_string,
            config(2, json!([])),
            true
        ));
        let escaping = Value::Object(
            complete()
                .as_object()
                .expect("map")
                .keys()
                .map(|name| (name.clone(), json!("../outside.safetensors")))
                .collect(),
        );
        assert!(!loadable("escape", escaping, config(2, json!([])), true));
        assert!(!loadable(
            "no-shard",
            complete(),
            config(2, json!([])),
            false
        ));
        assert!(!loadable(
            "deepstack",
            complete(),
            config(2, json!([5, 11])),
            true
        ));
        let mut llama = config(2, json!([]));
        llama["model_type"] = json!("llama");
        assert!(!loadable("model-type", complete(), llama, true));
        let mut missing_field = config(2, json!([]));
        missing_field["vision_config"]
            .as_object_mut()
            .expect("vision config")
            .remove("num_position_embeddings");
        assert!(!loadable("no-field", complete(), missing_field, true));
    }
}
