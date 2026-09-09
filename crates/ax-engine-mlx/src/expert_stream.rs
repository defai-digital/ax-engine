//! SSD expert streaming (layer-stack paging), contract `axquant.expert-stream.v1`.
//!
//! Super-class MoE packs (e.g. Qwen 3.8 2.4T-A95B at 2-bit) keep their fused
//! expert stacks `[num_experts, out, in]` on SSD instead of in unified memory.
//! The initial load skips every tensor named in `ax_expert_stream.json`; the
//! [`ExpertStackPager`] then pages one layer's expert stack in on demand,
//! hands the MoE forward the same [`QuantizedWeight`] values the resident
//! path would have built, and evicts least-recently-used layer stacks once the
//! budget (`AX_STREAM_EXPERT_LAYERS`, default 1) is exceeded.
//!
//! v1 is layer-stack paging only: the existing `gather_qmm` kernel runs
//! unchanged on the paged packed tensors. No per-expert unfused kernels.
//!
//! Per-expert mode (`AX_STREAM_EXPERT_GRANULARITY=expert`) pages
//! individual experts instead: the [`ExpertRowPager`] caches `(layer, expert)`
//! rows under a byte budget, evicts by route hotness (router selection count,
//! decayed over time) rather than recency, preloads an optional offline
//! hotlist, and warms the next layer's predicted experts on a loader thread.
//! Decode assembles a compacted `[top_k, ...]` stack so the same `gather_qmm`
//! path runs unchanged; prefill pages the sorted union of the prompt's
//! selected experts (Qwen3-style routers only — per-expert-scale routers
//! stay on full stacks), and any row-mode failure falls back to the
//! layer-stack pager for that layer. An explicit env value always wins; when
//! the env is unset, qwen4_exp packs with a file-backed
//! `ax_expert_stream.json` default to `expert` and every other family (and
//! inferred manifests) stays on `layer` (see
//! [`stream_expert_granularity_for_family`]).

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use ax_engine_core::{NativeTensorRole, NativeTensorSpec};
use serde::{Deserialize, Serialize};

use crate::weights::QuantizedWeight;

pub const EXPERT_STREAM_MANIFEST_FILE: &str = "ax_expert_stream.json";
pub const EXPERT_STREAM_SCHEMA_V1: &str = "axquant.expert-stream.v1";
pub const EXPERT_STREAM_MODE_LAYER_STACK: &str = "layer-stack";
/// Serve/load admission: `AX_STREAM_EXPERTS=off|auto|on` (also `0`/`1`).
pub const STREAM_EXPERTS_ENV: &str = "AX_STREAM_EXPERTS";
/// Number of layer expert stacks kept resident concurrently (minimum 1).
pub const STREAM_EXPERT_LAYERS_ENV: &str = "AX_STREAM_EXPERT_LAYERS";
/// Extra unified-memory reserve (OS + KV + activations) when Auto decides
/// whether a pack can stay fully resident. 48 GiB matches a serve process
/// on a 192 GB Flash host without flipping the certified resident path.
pub const AUTO_RESIDENT_HEADROOM_BYTES: u64 = 48 * 1024 * 1024 * 1024;

/// How AX Engine admits SSD expert streaming.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StreamExpertsMode {
    /// Never page experts. Required packs fail closed.
    Off,
    /// Stream when the pack requires it, or when full residency plus
    /// [`AUTO_RESIDENT_HEADROOM_BYTES`] would exceed unified memory.
    /// This is the product default: capable, not always-on.
    #[default]
    Auto,
    /// Always page packed expert stacks (file manifest or inferred roles).
    On,
}

impl StreamExpertsMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" | "no" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "on" | "1" | "true" | "yes" => Ok(Self::On),
            other => Err(format!(
                "invalid stream-experts mode {other:?} (expected off, auto, or on)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::On => "on",
        }
    }
}

/// Packed expert projection slots understood by v1.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ExpertProj {
    /// Fused gate+up stack → `LayerWeights::gate_up_exps_packed`.
    GateUp,
    Gate,
    Up,
    Down,
}

impl ExpertProj {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "gate_up" => Some(Self::GateUp),
            "gate" => Some(Self::Gate),
            "up" => Some(Self::Up),
            "down" => Some(Self::Down),
            _ => None,
        }
    }
}

/// One streamed tensor entry from `ax_expert_stream.json`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExpertStreamTensor {
    /// Runtime / sanitized MLX module path (the name AX Engine already uses).
    pub name: String,
    /// Repo-relative safetensors shard holding the tensor.
    pub file: PathBuf,
    pub layer: u32,
    pub proj: String,
    pub expert_axis: u32,
    pub num_experts: u32,
    pub bits: u32,
    pub group_size: u32,
    #[serde(skip)]
    pub parsed_proj: Option<ExpertProj>,
}

/// Parsed and validated `ax_expert_stream.json`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExpertStreamManifest {
    pub schema_version: String,
    #[serde(default)]
    pub generated_by: String,
    #[serde(default)]
    pub required: bool,
    pub mode: String,
    pub num_experts: u32,
    #[serde(default)]
    pub experts_per_tok: u32,
    #[serde(default)]
    pub estimated_resident_bytes: u64,
    #[serde(default)]
    pub estimated_full_resident_bytes: u64,
    #[serde(default)]
    pub estimated_max_layer_expert_bytes: u64,
    #[serde(default)]
    pub resident_roles: Vec<String>,
    #[serde(default)]
    pub streamed_roles: Vec<String>,
    pub tensors: Vec<ExpertStreamTensor>,
    /// True when this plan was inferred from native tensor roles
    /// (`infer_layer_stack_manifest`) instead of parsed from the pack's
    /// `ax_expert_stream.json`. Inferred plans carry weight rows only (no
    /// sidecar rows), so the row pager must not default on for them.
    #[serde(skip)]
    pub inferred: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ExpertStreamError {
    #[error(
        "expert streaming is REQUIRED by this pack (ax_expert_stream.json: required=true); \
         re-run with --stream-experts or {env}=1. A full-resident load would need about \
         {estimated_full_resident_bytes} bytes and is refused to avoid OOM / swap thrash",
        env = STREAM_EXPERTS_ENV
    )]
    StreamRequired { estimated_full_resident_bytes: u64 },
    #[error(
        "expert streaming was requested but {file} is missing in the model directory; \
         refusing to guess which tensors to stream",
        file = EXPERT_STREAM_MANIFEST_FILE
    )]
    ManifestMissing,
    #[error("invalid expert stream manifest: {0}")]
    InvalidManifest(String),
    #[error("expert stream paging failed: {0}")]
    Paging(String),
}

impl ExpertStreamManifest {
    /// Parse and validate a manifest. Unknown `schema_version` or `mode` fail
    /// closed; v1 only supports `layer-stack` paging of packed expert stacks.
    pub fn parse(bytes: &[u8]) -> Result<Self, ExpertStreamError> {
        let mut manifest: Self = serde_json::from_slice(bytes)
            .map_err(|e| ExpertStreamError::InvalidManifest(format!("JSON parse: {e}")))?;
        if manifest.schema_version != EXPERT_STREAM_SCHEMA_V1 {
            return Err(ExpertStreamError::InvalidManifest(format!(
                "unsupported schema_version {:?} (expected {:?})",
                manifest.schema_version, EXPERT_STREAM_SCHEMA_V1
            )));
        }
        if manifest.mode != EXPERT_STREAM_MODE_LAYER_STACK {
            return Err(ExpertStreamError::InvalidManifest(format!(
                "unsupported mode {:?} (v1 only supports {:?})",
                manifest.mode, EXPERT_STREAM_MODE_LAYER_STACK
            )));
        }
        if manifest.tensors.is_empty() {
            return Err(ExpertStreamError::InvalidManifest(
                "tensors list is empty".to_string(),
            ));
        }
        for tensor in &mut manifest.tensors {
            let Some(proj) = ExpertProj::parse(&tensor.proj) else {
                return Err(ExpertStreamError::InvalidManifest(format!(
                    "tensor {}: unknown proj {:?} (expected one of gate_up, gate, up, down)",
                    tensor.name, tensor.proj
                )));
            };
            tensor.parsed_proj = Some(proj);
            if tensor.expert_axis != 0 {
                return Err(ExpertStreamError::InvalidManifest(format!(
                    "tensor {}: expert_axis must be 0 for packed [E, out, in] stacks",
                    tensor.name
                )));
            }
            if tensor.bits == 0 || tensor.group_size == 0 {
                return Err(ExpertStreamError::InvalidManifest(format!(
                    "tensor {}: bits and group_size must be positive",
                    tensor.name
                )));
            }
        }
        Ok(manifest)
    }

    /// Read the manifest from a model directory; `None` when absent.
    pub fn read_from_dir(dir: &Path) -> Result<Option<Self>, ExpertStreamError> {
        let path = dir.join(EXPERT_STREAM_MANIFEST_FILE);
        match std::fs::read(&path) {
            Ok(bytes) => Self::parse(&bytes).map(Some),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(ExpertStreamError::InvalidManifest(format!(
                "read {}: {error}",
                path.display()
            ))),
        }
    }

    /// Layer indices that have at least one streamed tensor.
    pub fn layer_indices(&self) -> Vec<u32> {
        let mut layers: Vec<u32> = self.tensors.iter().map(|t| t.layer).collect();
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    pub fn tensors_for_layer(&self, layer: u32) -> impl Iterator<Item = &ExpertStreamTensor> {
        self.tensors.iter().filter(move |t| t.layer == layer)
    }
}

/// AXQuant Super-class stream manifests list MLX affine sidecars
/// (`.scales` / `.biases` / `.bias`) as first-class tensors with the same
/// `proj` as the packed `.weight`. Those names must not be inserted as
/// expert slots; `load_layer` attaches them onto the weight by convention.
fn is_quantization_sidecar_name(name: &str) -> bool {
    name.ends_with(".scales") || name.ends_with(".biases") || name.ends_with(".bias")
}

/// All tensor names the initial load must skip for a manifest: each streamed
/// base name plus its MLX quantization sidecars (`.scales`, `.biases`) and any
/// dense switch `.bias`. These names never enter the resident name map and are
/// never `eval`ed at init.
pub fn streamed_skip_names(manifest: &ExpertStreamManifest) -> HashSet<String> {
    let mut skip = HashSet::new();
    for tensor in &manifest.tensors {
        let base = tensor
            .name
            .strip_suffix(".weight")
            .unwrap_or(tensor.name.as_str());
        skip.insert(tensor.name.clone());
        skip.insert(format!("{base}.scales"));
        skip.insert(format!("{base}.biases"));
        skip.insert(format!("{base}.bias"));
    }
    skip
}

/// Sidecar shard files declared by a manifest: `(layer, sidecar tensor name)`
/// → shard path. AXQuant lists every `.scales` / `.biases` / `.bias` as a
/// first-class row with its own `file`, and triplets may legally split across
/// shards, so the row pager resolves each sidecar's file from this map
/// instead of assuming co-location with the weight.
pub type SidecarFileMap = HashMap<(u32, String), PathBuf>;

/// Base name of a weight row (`{base}.weight` → `{base}`).
fn weight_base_name(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

/// Base name of a sidecar row (strips the matched sidecar suffix). Only
/// meaningful for names [`is_quantization_sidecar_name`] accepts.
fn sidecar_base_name(name: &str) -> &str {
    for suffix in [".scales", ".biases", ".bias"] {
        if let Some(base) = name.strip_suffix(suffix) {
            return base;
        }
    }
    name
}

/// Build the sidecar file map for a manifest (sidecar rows only; weight rows
/// resolve through their own `file`). Fails closed on duplicate `(layer,
/// name)` sidecar rows and on a sidecar row whose base weight row is not
/// streamed in the same layer: both are manifest structure errors, so a
/// name/layer drift can never land in the map and silently miss at read
/// time. The correspondence requirement is also what makes `.bias`
/// classification exact — a `switch.bias`-style row with no same-layer
/// `switch.weight` row is rejected, not mapped.
pub fn sidecar_file_map(
    manifest: &ExpertStreamManifest,
) -> Result<SidecarFileMap, ExpertStreamError> {
    let weight_bases: HashSet<(u32, &str)> = manifest
        .tensors
        .iter()
        .filter(|tensor| !is_quantization_sidecar_name(&tensor.name))
        .map(|tensor| (tensor.layer, weight_base_name(&tensor.name)))
        .collect();
    let mut map = SidecarFileMap::new();
    for tensor in manifest
        .tensors
        .iter()
        .filter(|tensor| is_quantization_sidecar_name(&tensor.name))
    {
        if !weight_bases.contains(&(tensor.layer, sidecar_base_name(&tensor.name))) {
            return Err(ExpertStreamError::InvalidManifest(format!(
                "sidecar tensor {} (layer {}) has no matching weight row in the same layer",
                tensor.name, tensor.layer
            )));
        }
        match map.entry((tensor.layer, tensor.name.clone())) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(ExpertStreamError::InvalidManifest(format!(
                    "duplicate sidecar tensor row {} (layer {})",
                    tensor.name, tensor.layer
                )));
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(tensor.file.clone());
            }
        }
    }
    Ok(map)
}

#[cfg(test)]
fn env_flag_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

pub fn stream_experts_mode_from_env(value: Option<&str>) -> Option<StreamExpertsMode> {
    let raw = value?.trim();
    if raw.is_empty() {
        return None;
    }
    StreamExpertsMode::parse(raw).ok()
}

/// Whether `AX_STREAM_EXPERTS` force-enables streaming (`1`/`true`/`on`).
pub fn stream_experts_env_enabled() -> bool {
    matches!(
        stream_experts_mode_from_env(std::env::var(STREAM_EXPERTS_ENV).ok().as_deref()),
        Some(StreamExpertsMode::On)
    )
}

/// Resident layer budget from `AX_STREAM_EXPERT_LAYERS`: `max(1, value)`,
/// default 1 layer stack.
pub fn expert_layer_budget_from_env(value: Option<&str>) -> usize {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(1)
}

pub fn expert_layer_budget() -> usize {
    expert_layer_budget_from_env(std::env::var(STREAM_EXPERT_LAYERS_ENV).ok().as_deref())
}

/// Per-expert paging granularity selector: `AX_STREAM_EXPERT_GRANULARITY`.
pub const STREAM_EXPERT_GRANULARITY_ENV: &str = "AX_STREAM_EXPERT_GRANULARITY";
/// Row-pager byte budget: `AX_STREAM_EXPERT_CACHE_BYTES`.
pub const STREAM_EXPERT_CACHE_BYTES_ENV: &str = "AX_STREAM_EXPERT_CACHE_BYTES";
/// Next-layer prefetch kill-switch: `AX_STREAM_EXPERT_PREFETCH=0`.
pub const STREAM_EXPERT_PREFETCH_ENV: &str = "AX_STREAM_EXPERT_PREFETCH";
/// Offline hotlist preload path: `AX_STREAM_EXPERT_HOTLIST`.
pub const STREAM_EXPERT_HOTLIST_ENV: &str = "AX_STREAM_EXPERT_HOTLIST";
/// Observed-hotness dump path: `AX_STREAM_EXPERT_HOTLIST_OUT`.
pub const STREAM_EXPERT_HOTLIST_OUT_ENV: &str = "AX_STREAM_EXPERT_HOTLIST_OUT";
/// Selections between hotness decay passes: `AX_STREAM_EXPERT_HOTNESS_DECAY`.
pub const STREAM_EXPERT_HOTNESS_DECAY_ENV: &str = "AX_STREAM_EXPERT_HOTNESS_DECAY";

/// Hotlist file schema (`ax.expert-hotlist.v1`): offline-measured expert
/// popularity used to seed the row-pager cache at startup.
pub const EXPERT_HOTLIST_SCHEMA_V1: &str = "ax.expert-hotlist.v1";

/// Pager granularity for streamed expert stacks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StreamExpertGranularity {
    /// Whole layer stacks (default for every family except qwen4_exp packs
    /// with a file-backed manifest).
    #[default]
    Layer,
    /// Individual experts with route-hotness eviction (file-backed qwen4_exp
    /// default; opt-in elsewhere via `AX_STREAM_EXPERT_GRANULARITY=expert`).
    Expert,
}

impl StreamExpertGranularity {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "layer" => Some(Self::Layer),
            "expert" => Some(Self::Expert),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Layer => "layer",
            Self::Expert => "expert",
        }
    }
}

/// Granularity from `AX_STREAM_EXPERT_GRANULARITY`; empty/unset is `layer`,
/// unknown values fail closed to `layer` with a warning.
pub fn stream_expert_granularity_from_env(value: Option<&str>) -> StreamExpertGranularity {
    let Some(raw) = value else {
        return StreamExpertGranularity::Layer;
    };
    if raw.trim().is_empty() {
        return StreamExpertGranularity::Layer;
    }
    match StreamExpertGranularity::parse(raw) {
        Some(granularity) => granularity,
        None => {
            tracing::warn!(
                target: "ax_engine_mlx",
                value = raw,
                "invalid {STREAM_EXPERT_GRANULARITY_ENV}; failing closed to layer-stack paging"
            );
            StreamExpertGranularity::Layer
        }
    }
}

pub fn stream_expert_granularity() -> StreamExpertGranularity {
    stream_expert_granularity_from_env(std::env::var(STREAM_EXPERT_GRANULARITY_ENV).ok().as_deref())
}

/// Family-aware granularity resolution at pager construction. An explicit
/// `AX_STREAM_EXPERT_GRANULARITY` (either value) always wins, and invalid
/// values keep failing closed to `layer`; only an unset/empty env falls back
/// to the family default: `expert` for qwen4_exp packs with a file-backed
/// `ax_expert_stream.json` (the row pager resolves this family's
/// split-sidecar triplets from the manifest), `layer` otherwise. An inferred
/// manifest carries weight rows only, so flipping there would re-open the
/// split-sidecar co-location class — `manifest_file_backed` locks the flip
/// to real pack manifests. The env-global [`stream_expert_granularity`]
/// stays family-blind so the family policy lives only here, where the caller
/// has family context.
pub fn stream_expert_granularity_for_family(
    value: Option<&str>,
    model_family: &str,
    manifest_file_backed: bool,
) -> StreamExpertGranularity {
    let family_default = if model_family == "qwen4_exp" && manifest_file_backed {
        StreamExpertGranularity::Expert
    } else {
        StreamExpertGranularity::Layer
    };
    match value {
        None => family_default,
        Some(raw) if raw.trim().is_empty() => family_default,
        some => stream_expert_granularity_from_env(some),
    }
}

/// Row-pager byte budget from `AX_STREAM_EXPERT_CACHE_BYTES`.
///
/// Default: `4 × estimated_max_layer_expert_bytes`. The floor keeps one
/// token's working set for the current layer plus the prefetched next layer
/// (`2 × experts_per_tok × per-expert bytes`) so decode cannot thrash inside
/// a single token.
pub fn expert_row_cache_bytes_from_env(
    value: Option<&str>,
    manifest: &ExpertStreamManifest,
) -> usize {
    let per_expert =
        (manifest.estimated_max_layer_expert_bytes / u64::from(manifest.num_experts.max(1))).max(1);
    let floor = 2u64
        .saturating_mul(u64::from(manifest.experts_per_tok.max(1)))
        .saturating_mul(per_expert);
    let default = 4u64
        .saturating_mul(manifest.estimated_max_layer_expert_bytes)
        .max(floor);
    let bytes = value
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(|v| v.max(floor))
        .unwrap_or(default);
    usize::try_from(bytes).unwrap_or(usize::MAX)
}

/// Whether next-layer prefetch runs in expert mode (default on; `0`/`off`/
/// `false`/`no` disables).
pub fn expert_prefetch_enabled_from_env(value: Option<&str>) -> bool {
    !matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0") | Some("false") | Some("no") | Some("off")
    )
}

/// Selections between hotness decay passes (default 4096, minimum 1).
pub fn expert_hotness_decay_from_env(value: Option<&str>) -> u64 {
    value
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(4096)
}

const STREAM_MODE_UNSET: u8 = 0;
const STREAM_MODE_OFF: u8 = 1;
const STREAM_MODE_AUTO: u8 = 2;
const STREAM_MODE_ON: u8 = 3;

static STREAM_EXPERTS_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(STREAM_MODE_UNSET);

fn mode_to_u8(mode: StreamExpertsMode) -> u8 {
    match mode {
        StreamExpertsMode::Off => STREAM_MODE_OFF,
        StreamExpertsMode::Auto => STREAM_MODE_AUTO,
        StreamExpertsMode::On => STREAM_MODE_ON,
    }
}

fn mode_from_u8(raw: u8) -> Option<StreamExpertsMode> {
    match raw {
        STREAM_MODE_OFF => Some(StreamExpertsMode::Off),
        STREAM_MODE_AUTO => Some(StreamExpertsMode::Auto),
        STREAM_MODE_ON => Some(StreamExpertsMode::On),
        _ => None,
    }
}

/// Install the CLI/SDK stream mode before weights load.
pub fn set_stream_experts_mode(mode: StreamExpertsMode) {
    STREAM_EXPERTS_OVERRIDE.store(mode_to_u8(mode), std::sync::atomic::Ordering::Relaxed);
}

/// Backward-compatible latch: `true` is On, `false` leaves Auto (env/default).
pub fn set_stream_experts_override(enabled: bool) {
    if enabled {
        set_stream_experts_mode(StreamExpertsMode::On);
    }
}

/// Effective mode: CLI/SDK override, else `AX_STREAM_EXPERTS`, else Auto.
pub fn stream_experts_mode() -> StreamExpertsMode {
    if let Some(mode) =
        mode_from_u8(STREAM_EXPERTS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed))
    {
        return mode;
    }
    stream_experts_mode_from_env(std::env::var(STREAM_EXPERTS_ENV).ok().as_deref())
        .unwrap_or(StreamExpertsMode::Auto)
}

/// Whether the current mode force-enables streaming.
pub fn stream_experts_requested() -> bool {
    matches!(stream_experts_mode(), StreamExpertsMode::On)
}

/// Host unified-memory size (`hw.memsize` on macOS). `None` when unknown.
pub fn unified_memory_bytes() -> Option<u64> {
    unified_memory_bytes_from_sysctl()
}

fn unified_memory_bytes_from_sysctl() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}

pub fn should_auto_stream(full_resident_bytes: u64, available_bytes: Option<u64>) -> bool {
    match available_bytes {
        None => false,
        Some(available) => {
            full_resident_bytes.saturating_add(AUTO_RESIDENT_HEADROOM_BYTES) > available
        }
    }
}

/// Decide whether to page experts.
///
/// - **Off:** resident load; `required=true` fails closed.
/// - **On:** use the file manifest, or infer packed expert roles.
/// - **Auto:** stream required packs, or optional/inferred packs that cannot
///   fit in unified memory plus [`AUTO_RESIDENT_HEADROOM_BYTES`].
pub fn resolve_expert_stream<F>(
    mode: StreamExpertsMode,
    file: Option<ExpertStreamManifest>,
    infer: F,
    available_bytes: Option<u64>,
) -> Result<Option<ExpertStreamManifest>, ExpertStreamError>
where
    F: FnOnce() -> Result<ExpertStreamManifest, ExpertStreamError>,
{
    match mode {
        StreamExpertsMode::Off => {
            if let Some(manifest) = &file
                && manifest.required
            {
                return Err(ExpertStreamError::StreamRequired {
                    estimated_full_resident_bytes: manifest.estimated_full_resident_bytes,
                });
            }
            Ok(None)
        }
        StreamExpertsMode::On => match file {
            Some(manifest) => Ok(Some(manifest)),
            None => infer().map(Some),
        },
        StreamExpertsMode::Auto => {
            if file.as_ref().is_some_and(|manifest| manifest.required) {
                return Ok(file);
            }
            let candidate = match file {
                Some(manifest) => Some(manifest),
                None => match infer() {
                    Ok(manifest) => Some(manifest),
                    Err(ExpertStreamError::ManifestMissing) => None,
                    Err(error) => return Err(error),
                },
            };
            match candidate {
                Some(manifest)
                    if manifest.required
                        || should_auto_stream(
                            manifest.estimated_full_resident_bytes,
                            available_bytes,
                        ) =>
                {
                    Ok(Some(manifest))
                }
                _ => Ok(None),
            }
        }
    }
}

/// File-backed admission used by tests and doctor.
pub fn admit_expert_stream(
    model_dir: &Path,
    requested: bool,
) -> Result<Option<ExpertStreamManifest>, ExpertStreamError> {
    let file = ExpertStreamManifest::read_from_dir(model_dir)?;
    let mode = if requested {
        StreamExpertsMode::On
    } else {
        StreamExpertsMode::Off
    };
    resolve_expert_stream(mode, file, || Err(ExpertStreamError::ManifestMissing), None)
}

fn expert_role_to_proj(role: NativeTensorRole) -> Option<ExpertProj> {
    match role {
        NativeTensorRole::FfnGateUpExpsPacked => Some(ExpertProj::GateUp),
        NativeTensorRole::FfnGateExps => Some(ExpertProj::Gate),
        NativeTensorRole::FfnUpExps => Some(ExpertProj::Up),
        NativeTensorRole::FfnDownExps => Some(ExpertProj::Down),
        _ => None,
    }
}

/// Build a non-required layer-stack plan from native expert roles.
///
/// Used when `--stream-experts` is set but the pack has no
/// `ax_expert_stream.json` — the published DeepSeek V4 Flash AXQ 2/3-bit
/// packs and any other fused-expert MoE that already maps onto
/// `FfnGateUpExpsPacked` / `Ffn{Gate,Up,Down}Exps`.
pub fn infer_layer_stack_manifest(
    specs: &[NativeTensorSpec],
    experts_per_tok: u32,
) -> Result<ExpertStreamManifest, ExpertStreamError> {
    if experts_per_tok < 1 {
        return Err(ExpertStreamError::InvalidManifest(
            "inferred stream plan requires experts_per_tok >= 1".into(),
        ));
    }
    let mut tensors = Vec::new();
    let mut expert_counts = HashSet::new();
    let mut expert_bytes = 0u64;
    let mut layer_bytes: HashMap<u32, u64> = HashMap::new();
    let mut full_bytes = 0u64;
    for spec in specs {
        full_bytes = full_bytes.saturating_add(spec.length_bytes);
        let Some(proj) = expert_role_to_proj(spec.role) else {
            continue;
        };
        if !spec.name.ends_with(".weight")
            && !spec.name.ends_with("_blocks")
            && !spec.name.ends_with(".gate")
            && !spec.name.ends_with(".up")
            && !spec.name.ends_with(".down")
        {
            continue;
        }
        let Some(layer) = spec.layer_index else {
            return Err(ExpertStreamError::InvalidManifest(format!(
                "expert tensor {} is missing a layer index",
                spec.name
            )));
        };
        if spec.shape.is_empty() || spec.shape[0] == 0 {
            return Err(ExpertStreamError::InvalidManifest(format!(
                "expert tensor {} has no expert axis",
                spec.name
            )));
        }
        let num_experts = spec.shape[0] as u32;
        expert_counts.insert(num_experts);
        let bits = spec.quantization.as_ref().map(|q| q.bits).unwrap_or(4);
        let group_size = spec
            .quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64);
        if bits == 0 || group_size == 0 {
            return Err(ExpertStreamError::InvalidManifest(format!(
                "expert tensor {} has invalid bits/group_size",
                spec.name
            )));
        }
        tensors.push(ExpertStreamTensor {
            name: spec.name.clone(),
            file: spec.file.clone(),
            layer,
            proj: match proj {
                ExpertProj::GateUp => "gate_up".into(),
                ExpertProj::Gate => "gate".into(),
                ExpertProj::Up => "up".into(),
                ExpertProj::Down => "down".into(),
            },
            expert_axis: 0,
            num_experts,
            bits,
            group_size,
            parsed_proj: Some(proj),
        });
        expert_bytes = expert_bytes.saturating_add(spec.length_bytes);
        *layer_bytes.entry(layer).or_insert(0) += spec.length_bytes;
    }
    if tensors.is_empty() {
        return Err(ExpertStreamError::ManifestMissing);
    }
    if expert_counts.len() != 1 {
        return Err(ExpertStreamError::InvalidManifest(format!(
            "packed expert tensors disagree on expert-axis size: {expert_counts:?}"
        )));
    }
    let num_experts = *expert_counts.iter().next().expect("count set is non-empty");
    let max_layer = layer_bytes.values().copied().max().unwrap_or(1).max(1);
    let resident = full_bytes.saturating_sub(expert_bytes);
    Ok(ExpertStreamManifest {
        schema_version: EXPERT_STREAM_SCHEMA_V1.to_string(),
        generated_by: "ax-engine-infer".into(),
        required: false,
        mode: EXPERT_STREAM_MODE_LAYER_STACK.to_string(),
        num_experts,
        experts_per_tok,
        estimated_resident_bytes: resident,
        estimated_full_resident_bytes: full_bytes.max(1),
        estimated_max_layer_expert_bytes: max_layer,
        resident_roles: vec![
            "embedding".into(),
            "attention".into(),
            "router".into(),
            "shared_expert".into(),
            "norm".into(),
            "lm_head".into(),
            "mtp".into(),
        ],
        streamed_roles: vec!["expert".into()],
        tensors,
        inferred: true,
    })
}

/// One layer's paged expert stack — the same slots the resident loader fills.
/// Clones are cheap refcount bumps on the underlying MLX arrays.
#[derive(Clone, Default)]
pub struct LayerExpertStack {
    pub gate_up_exps_packed: Option<QuantizedWeight>,
    pub gate_exps: Option<QuantizedWeight>,
    pub up_exps: Option<QuantizedWeight>,
    pub down_exps: Option<QuantizedWeight>,
}

impl LayerExpertStack {
    fn insert(&mut self, proj: ExpertProj, weight: QuantizedWeight) {
        match proj {
            ExpertProj::GateUp => self.gate_up_exps_packed = Some(weight),
            ExpertProj::Gate => self.gate_exps = Some(weight),
            ExpertProj::Up => self.up_exps = Some(weight),
            ExpertProj::Down => self.down_exps = Some(weight),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.gate_up_exps_packed.is_none()
            && self.gate_exps.is_none()
            && self.up_exps.is_none()
            && self.down_exps.is_none()
    }
}

struct PagerCache {
    /// Resident stacks keyed by layer index.
    entries: HashMap<u32, LayerExpertStack>,
    /// LRU order: front = least recently used.
    order: VecDeque<u32>,
}

/// Layer-stack pager: on MoE forward for layer L, `ensure_layer` loads L's
/// streamed tensors (only those tensors) from their shards, builds the same
/// `QuantizedWeight`s the resident path would have, caches them, and evicts
/// the LRU layer stack when the resident budget is exceeded.
pub struct ExpertStackPager {
    manifest: Arc<ExpertStreamManifest>,
    root: PathBuf,
    budget_layers: usize,
    /// When true, concatenate paged `gate`+`up` into one packed projection
    /// (DeepSeek V4 resident path). MiniMax-M3 mlxcel uses split SwitchLinear
    /// (`gate_proj` / `up_proj` / `down_proj`) and never fuses; Super-class
    /// MiniMax packs are stream-required, so this must stay false for them.
    fuse_split_experts: bool,
    cache: Mutex<PagerCache>,
}

impl ExpertStackPager {
    pub fn new(manifest: Arc<ExpertStreamManifest>, root: PathBuf, budget_layers: usize) -> Self {
        Self::new_with_fuse(manifest, root, budget_layers, false)
    }

    pub fn new_with_fuse(
        manifest: Arc<ExpertStreamManifest>,
        root: PathBuf,
        budget_layers: usize,
        fuse_split_experts: bool,
    ) -> Self {
        Self {
            manifest,
            root,
            budget_layers: budget_layers.max(1),
            fuse_split_experts,
            cache: Mutex::new(PagerCache {
                entries: HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }

    pub fn manifest(&self) -> &ExpertStreamManifest {
        &self.manifest
    }

    pub fn budget_layers(&self) -> usize {
        self.budget_layers
    }

    pub fn cached_layer_count(&self) -> usize {
        self.cache
            .lock()
            .expect("expert stream cache lock")
            .entries
            .len()
    }

    /// Cached layer indices in LRU order (front = least recently used).
    pub fn cached_layer_indices(&self) -> Vec<u32> {
        self.cache
            .lock()
            .expect("expert stream cache lock")
            .order
            .iter()
            .copied()
            .collect()
    }

    /// Make layer `layer`'s expert stack resident and return cheap clones of
    /// its `QuantizedWeight`s. Loads from disk on a cache miss, then evicts
    /// LRU layers beyond the budget.
    pub fn ensure_layer(&self, layer: u32) -> Result<LayerExpertStack, ExpertStreamError> {
        {
            let mut cache = self.cache.lock().expect("expert stream cache lock");
            if let Some(stack) = cache.entries.get(&layer).cloned() {
                if let Some(pos) = cache.order.iter().position(|l| *l == layer) {
                    cache.order.remove(pos);
                }
                cache.order.push_back(layer);
                return Ok(stack);
            }
        }

        let stack = self.load_layer(layer)?;

        let mut cache = self.cache.lock().expect("expert stream cache lock");
        // A racing thread may have filled the layer while we read from disk;
        // keep the existing entry rather than double-counting it.
        if let Some(existing) = cache.entries.get(&layer).cloned() {
            if let Some(pos) = cache.order.iter().position(|l| *l == layer) {
                cache.order.remove(pos);
            }
            cache.order.push_back(layer);
            return Ok(existing);
        }
        cache.entries.insert(layer, stack.clone());
        cache.order.push_back(layer);
        while cache.order.len() > self.budget_layers {
            let evict = cache
                .order
                .pop_front()
                .expect("LRU order must be non-empty when over budget");
            cache.entries.remove(&evict);
        }
        Ok(stack)
    }

    /// Read only this layer's streamed tensors from their shards and assemble
    /// the resident-path `QuantizedWeight` values.
    fn load_layer(&self, layer: u32) -> Result<LayerExpertStack, ExpertStreamError> {
        let tensors: Vec<&ExpertStreamTensor> = self.manifest.tensors_for_layer(layer).collect();
        if tensors.is_empty() {
            return Err(ExpertStreamError::Paging(format!(
                "manifest has no streamed tensors for layer {layer}"
            )));
        }

        // Shard → keep-set (base name + quantization sidecars).
        let mut keep_by_file: HashMap<&Path, HashSet<String>> = HashMap::new();
        for tensor in &tensors {
            let keep = keep_by_file.entry(tensor.file.as_path()).or_default();
            let base = tensor
                .name
                .strip_suffix(".weight")
                .unwrap_or(tensor.name.as_str());
            keep.insert(tensor.name.clone());
            keep.insert(format!("{base}.scales"));
            keep.insert(format!("{base}.biases"));
            keep.insert(format!("{base}.bias"));
        }

        let mut loaded: HashMap<String, mlx_sys::MlxArray> = HashMap::new();
        for (file, keep) in &keep_by_file {
            let path = self.root.join(file);
            let tensors = mlx_sys::load_safetensors_filtered(
                &path,
                mlx_sys::SafetensorsNameFilter::Keep(keep),
            )
            .map_err(ExpertStreamError::Paging)?;
            loaded.extend(tensors);
        }
        // Wire the freshly created arrays into MLX's working set, mirroring the
        // initial-load eval for both loader paths. Use try_eval so a paging
        // failure stays on the ExpertStreamError path instead of panicking.
        let refs: Vec<&mlx_sys::MlxArray> = loaded.values().collect();
        mlx_sys::try_eval(&refs).map_err(ExpertStreamError::Paging)?;

        let mut stack = LayerExpertStack::default();
        for tensor in &tensors {
            if is_quantization_sidecar_name(&tensor.name) {
                continue;
            }
            let proj = tensor.parsed_proj.ok_or_else(|| {
                ExpertStreamError::Paging(format!(
                    "tensor {} lost its parsed proj; manifest must be validated before paging",
                    tensor.name
                ))
            })?;
            let weight = loaded.remove(&tensor.name).ok_or_else(|| {
                ExpertStreamError::Paging(format!(
                    "tensor {} missing from shard {}",
                    tensor.name,
                    tensor.file.display()
                ))
            })?;
            let base = tensor
                .name
                .strip_suffix(".weight")
                .unwrap_or(tensor.name.as_str());
            let scales = loaded.remove(&format!("{base}.scales"));
            let biases = loaded.remove(&format!("{base}.biases"));
            let linear_bias = loaded.remove(&format!("{base}.bias"));
            let quantized = QuantizedWeight {
                weight,
                scales,
                biases,
                group_size: tensor.group_size as i32,
                bits: tensor.bits as i32,
                mode: "affine".to_string(),
                linear_bias,
                decode_weight_t: None,
                decode_q2_weight: None,
                decode_q2_scales: None,
                decode_q2_biases: None,
            };
            stack.insert(proj, quantized);
        }
        if stack.is_empty() {
            return Err(ExpertStreamError::Paging(format!(
                "no expert tensors were paged for layer {layer}"
            )));
        }
        if self.fuse_split_experts {
            let (packed, gate, up) = crate::weights::try_fuse_paged_split_moe_experts(
                stack.gate_up_exps_packed,
                stack.gate_exps,
                stack.up_exps,
            )
            .map_err(ExpertStreamError::Paging)?;
            stack.gate_up_exps_packed = packed;
            stack.gate_exps = gate;
            stack.up_exps = up;
        }
        Ok(stack)
    }
}

// ------------------------------------------------------------------
// Per-expert (row) paging: route-hotness cache over (layer, expert) rows.
// ------------------------------------------------------------------

/// Hotlist file (`ax.expert-hotlist.v1`): offline-measured expert popularity.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExpertHotlistFile {
    pub schema_version: String,
    #[serde(default)]
    pub generated_by: String,
    pub entries: Vec<ExpertHotlistEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExpertHotlistEntry {
    pub layer: u32,
    pub expert: u32,
    /// Selection count measured offline; higher is hotter.
    #[serde(default)]
    pub weight: u64,
}

/// Compacted per-token expert stacks assembled from cached rows: the same
/// slots the resident path fills, plus the flat per-position slot indices to
/// consume them with. Assembly follows request order, so `ensure_experts`
/// leaves `remap` as the identity `0..k` (decode compaction); the row-mode
/// prefill caller overwrites it with the per-position union-slot lookup.
#[derive(Clone)]
pub struct CompactedExperts {
    pub stack: LayerExpertStack,
    pub remap: Vec<u32>,
}

struct ExpertRowEntry {
    rows: LayerExpertStack,
    bytes: usize,
    tick: u64,
    pinned: bool,
}

#[derive(Default)]
struct RowPagerState {
    entries: HashMap<(u32, u32), ExpertRowEntry>,
    resident_bytes: usize,
    hotness: HashMap<(u32, u32), u64>,
    tick: u64,
    selections_since_decay: u64,
    /// Per-layer expert selection of the previous token (prefetch prediction).
    last_ids: HashMap<u32, Vec<u32>>,
}

struct RowPagerCore {
    manifest: Arc<ExpertStreamManifest>,
    root: PathBuf,
    headers: crate::expert_stream_slice::ShardHeaderCache,
    sidecar_files: SidecarFileMap,
    state: Mutex<RowPagerState>,
    budget_bytes: usize,
    load_delay: Option<std::time::Duration>,
}

enum PrefetchMsg {
    Warm { layer: u32, experts: Vec<u32> },
    Stop,
}

fn expert_stack_bytes(stack: &LayerExpertStack) -> usize {
    fn qw_bytes(qw: &QuantizedWeight) -> usize {
        qw.weight.nbytes()
            + qw.scales.as_ref().map_or(0, |a| a.nbytes())
            + qw.biases.as_ref().map_or(0, |a| a.nbytes())
            + qw.linear_bias.as_ref().map_or(0, |a| a.nbytes())
    }
    [
        &stack.gate_up_exps_packed,
        &stack.gate_exps,
        &stack.up_exps,
        &stack.down_exps,
    ]
    .into_iter()
    .flatten()
    .map(qw_bytes)
    .sum()
}

/// Read `(layer, experts)` rows for every non-sidecar manifest tensor of the
/// layer. Mirrors `ExpertStackPager::load_layer` slot semantics per expert.
fn load_rows(
    core: &RowPagerCore,
    layer: u32,
    experts: &[u32],
) -> Result<Vec<LayerExpertStack>, ExpertStreamError> {
    if experts.is_empty() {
        return Ok(Vec::new());
    }
    // Evidence/testing hook: simulate a slow SSD for overlap measurements.
    if let Some(delay) = core.load_delay {
        std::thread::sleep(delay);
    }
    let mut per_expert: Vec<LayerExpertStack> = (0..experts.len())
        .map(|_| LayerExpertStack::default())
        .collect();
    let mut any = false;
    for tensor in core.manifest.tensors_for_layer(layer) {
        if is_quantization_sidecar_name(&tensor.name) {
            continue;
        }
        let proj = tensor.parsed_proj.ok_or_else(|| {
            ExpertStreamError::Paging(format!(
                "tensor {} lost its parsed proj; manifest must be validated before paging",
                tensor.name
            ))
        })?;
        let rows = crate::expert_stream_slice::read_expert_proj_rows(
            &core.root,
            &core.headers,
            &core.sidecar_files,
            tensor,
            experts,
        )
        .map_err(ExpertStreamError::Paging)?;
        for (row, slot) in rows.into_iter().zip(per_expert.iter_mut()) {
            slot.insert(
                proj,
                QuantizedWeight {
                    weight: row.weight,
                    scales: row.scales,
                    biases: row.biases,
                    group_size: tensor.group_size as i32,
                    bits: tensor.bits as i32,
                    mode: "affine".to_string(),
                    linear_bias: row.linear_bias,
                    decode_weight_t: None,
                    decode_q2_weight: None,
                    decode_q2_scales: None,
                    decode_q2_biases: None,
                },
            );
        }
        any = true;
    }
    if !any {
        return Err(ExpertStreamError::Paging(format!(
            "manifest has no streamed tensors for layer {layer}"
        )));
    }
    Ok(per_expert)
}

/// Concatenate per-expert rows (`[1, ...]`) into one compacted `[k, ...]`
/// projection. Sidecars must be present on every row or none; mixed rows mean
/// the manifest/shards disagree, which fails closed.
fn concat_quantized_rows(
    rows: Vec<QuantizedWeight>,
    what: &str,
) -> Result<QuantizedWeight, ExpertStreamError> {
    let Some(first) = rows.first() else {
        return Err(ExpertStreamError::Paging(format!(
            "no expert rows to assemble for {what}"
        )));
    };
    let (group_size, bits, mode) = (first.group_size, first.bits, first.mode.clone());
    let concat_sidecar = |pick: fn(&QuantizedWeight) -> Option<&mlx_sys::MlxArray>|
     -> Result<Option<mlx_sys::MlxArray>, ExpertStreamError> {
        let refs: Vec<&mlx_sys::MlxArray> = rows.iter().filter_map(pick).collect();
        if refs.is_empty() {
            return Ok(None);
        }
        if refs.len() != rows.len() {
            return Err(ExpertStreamError::Paging(format!(
                "expert rows disagree on sidecar presence for {what}"
            )));
        }
        Ok(Some(mlx_sys::concatenate(&refs, 0, None)))
    };
    let weight_refs: Vec<&mlx_sys::MlxArray> = rows.iter().map(|r| &r.weight).collect();
    Ok(QuantizedWeight {
        weight: mlx_sys::concatenate(&weight_refs, 0, None),
        scales: concat_sidecar(|r| r.scales.as_ref())?,
        biases: concat_sidecar(|r| r.biases.as_ref())?,
        group_size,
        bits,
        mode,
        linear_bias: concat_sidecar(|r| r.linear_bias.as_ref())?,
        decode_weight_t: None,
        decode_q2_weight: None,
        decode_q2_scales: None,
        decode_q2_biases: None,
    })
}

/// Assemble compacted per-token stacks from per-expert rows in request order.
fn assemble_stack(
    rows: &[LayerExpertStack],
    fuse_split_experts: bool,
) -> Result<LayerExpertStack, ExpertStreamError> {
    fn collect(
        rows: &[LayerExpertStack],
        pick: impl Fn(&LayerExpertStack) -> &Option<QuantizedWeight>,
        what: &str,
    ) -> Result<Option<QuantizedWeight>, ExpertStreamError> {
        let present = rows.iter().filter(|r| pick(r).is_some()).count();
        if present == 0 {
            return Ok(None);
        }
        if present != rows.len() {
            return Err(ExpertStreamError::Paging(format!(
                "expert rows disagree on projection presence for {what}"
            )));
        }
        let picked: Vec<QuantizedWeight> = rows.iter().filter_map(|r| pick(r).clone()).collect();
        concat_quantized_rows(picked, what).map(Some)
    }
    let mut stack = LayerExpertStack {
        gate_up_exps_packed: collect(rows, |r| &r.gate_up_exps_packed, "gate_up")?,
        gate_exps: collect(rows, |r| &r.gate_exps, "gate")?,
        up_exps: collect(rows, |r| &r.up_exps, "up")?,
        down_exps: collect(rows, |r| &r.down_exps, "down")?,
    };
    if stack.is_empty() {
        return Err(ExpertStreamError::Paging(
            "no expert rows were assembled".to_string(),
        ));
    }
    if fuse_split_experts {
        let (packed, gate, up) = crate::weights::try_fuse_paged_split_moe_experts(
            stack.gate_up_exps_packed,
            stack.gate_exps,
            stack.up_exps,
        )
        .map_err(ExpertStreamError::Paging)?;
        stack.gate_up_exps_packed = packed;
        stack.gate_exps = gate;
        stack.up_exps = up;
    }
    Ok(stack)
}

/// Evict unpinned entries by route hotness (lowest selection count first,
/// oldest tick breaks ties) until the byte budget is met. The budget is a
/// soft cap: when everything is pinned mid-assembly, correctness wins.
fn evict_to_budget(state: &mut RowPagerState, budget_bytes: usize) {
    while state.resident_bytes > budget_bytes {
        let victim = state
            .entries
            .iter()
            .filter(|(_, entry)| !entry.pinned)
            .min_by_key(|(key, entry)| (state.hotness.get(key).copied().unwrap_or(0), entry.tick))
            .map(|(key, _)| *key);
        let Some(victim) = victim else {
            break;
        };
        if let Some(entry) = state.entries.remove(&victim) {
            state.resident_bytes = state.resident_bytes.saturating_sub(entry.bytes);
        }
    }
}

/// Synchronously warm `(layer, experts)` into the cache; returns how many
/// experts were newly loaded. Hotness is not touched: experts that were
/// warmed but never selected are the first eviction candidates.
fn warm_experts(
    core: &RowPagerCore,
    layer: u32,
    experts: &[u32],
) -> Result<usize, ExpertStreamError> {
    let num_experts = core.manifest.num_experts;
    let missing: Vec<u32> = {
        let state = core.state.lock().expect("expert row pager lock");
        experts
            .iter()
            .copied()
            .filter(|id| *id < num_experts && !state.entries.contains_key(&(layer, *id)))
            .collect()
    };
    if missing.is_empty() {
        return Ok(0);
    }
    let loaded = load_rows(core, layer, &missing)?;
    let mut state = core.state.lock().expect("expert row pager lock");
    let state = &mut *state;
    let mut warmed = 0usize;
    for (expert, rows) in missing.into_iter().zip(loaded) {
        if state.entries.contains_key(&(layer, expert)) {
            continue;
        }
        let bytes = expert_stack_bytes(&rows);
        state.tick += 1;
        let tick = state.tick;
        state.resident_bytes = state.resident_bytes.saturating_add(bytes);
        state.entries.insert(
            (layer, expert),
            ExpertRowEntry {
                rows,
                bytes,
                tick,
                pinned: false,
            },
        );
        warmed += 1;
    }
    evict_to_budget(state, core.budget_bytes);
    Ok(warmed)
}

/// In-flight state of a split ensure between
/// [`ExpertRowPager::ensure_experts_split_begin`] and
/// [`ExpertRowPager::ensure_experts_split_finish`]. Holds the selection's
/// pins; dropping without finishing releases them and sends the pending
/// next-layer prefetch, so an early error path can never strand a pin.
pub struct SplitExpertsBegin {
    core: Arc<RowPagerCore>,
    layer: u32,
    ids: Vec<u32>,
    prefetch_tx: Option<Sender<PrefetchMsg>>,
    /// Resident part, assembled in request order (`stack` is empty when the
    /// selection is entirely missing).
    pub resident: CompactedExperts,
    /// Missing expert ids in request order (empty when fully resident).
    pub missing_ids: Vec<u32>,
    /// Request position → row index in the `[resident | missing]` row
    /// concat (missing rows start after the resident part).
    pub order: Vec<u32>,
    done: bool,
}

impl SplitExpertsBegin {
    /// Whether the whole selection was already resident (the caller uses
    /// `resident` as the full compaction and skips the finish load).
    pub fn fully_resident(&self) -> bool {
        self.missing_ids.is_empty()
    }

    /// Whether nothing was resident (the finish load produces the full
    /// compaction).
    pub fn fully_missing(&self) -> bool {
        self.resident.remap.is_empty()
    }

    /// Take the resident part out of the guard (drop still releases pins).
    pub fn take_resident(&mut self) -> CompactedExperts {
        std::mem::replace(
            &mut self.resident,
            CompactedExperts {
                stack: LayerExpertStack::default(),
                remap: Vec::new(),
            },
        )
    }

    /// Unpin the selection and send the (single) next-layer prefetch —
    /// shared by the finish path and the drop path, exactly once.
    fn release(&mut self) {
        if self.done {
            return;
        }
        self.done = true;
        let mut state = self.core.state.lock().expect("expert row pager lock");
        for id in &self.ids {
            if let Some(entry) = state.entries.get_mut(&(self.layer, *id)) {
                entry.pinned = false;
            }
        }
        if let Some(tx) = self.prefetch_tx.as_ref() {
            let next = self.layer + 1;
            if self.core.manifest.tensors_for_layer(next).next().is_some()
                && let Some(predicted) = state.last_ids.get(&next).cloned()
            {
                let _ = tx.send(PrefetchMsg::Warm {
                    layer: next,
                    experts: predicted,
                });
            }
        }
    }
}

impl Drop for SplitExpertsBegin {
    fn drop(&mut self) {
        self.release();
    }
}

/// Construction knobs for [`ExpertRowPager`], usually derived from env.
#[derive(Clone, Debug)]
pub struct ExpertRowPagerConfig {
    pub budget_bytes: usize,
    pub fuse_split_experts: bool,
    pub prefetch: bool,
    pub decay_interval: u64,
    pub hotlist_out: Option<PathBuf>,
    /// Artificial delay injected once per SSD load batch. Evidence/testing
    /// hook for the split-submit overlap (never set by operator config).
    pub load_delay: Option<std::time::Duration>,
}

/// Per-expert pager: caches `(layer, expert)` rows under a byte budget and
/// evicts by route hotness — the number of times the router *selected* an
/// expert (counting selections, not cache hits, so a repeatedly selected
/// expert that was evicted before its second hit is not punished), halved
/// every `decay_interval` selections. Selected experts are pinned while
/// their token's stacks are assembled.
///
/// Decode calls [`Self::ensure_experts`] with the router's top-k ids and gets
/// compacted `[top_k, ...]` stacks back, so the existing `gather_qmm` path
/// runs unchanged. A loader thread warms the next layer's predicted experts
/// (its previous-token selection) while the GPU works on the current one.
pub struct ExpertRowPager {
    core: Arc<RowPagerCore>,
    fuse_split_experts: bool,
    decay_interval: u64,
    prefetch_tx: Mutex<Option<Sender<PrefetchMsg>>>,
    prefetch_worker: Mutex<Option<JoinHandle<()>>>,
    hotlist_out: Mutex<Option<PathBuf>>,
}

impl ExpertRowPager {
    pub fn new(
        manifest: Arc<ExpertStreamManifest>,
        root: PathBuf,
        config: ExpertRowPagerConfig,
    ) -> Result<Self, ExpertStreamError> {
        let core = Arc::new(RowPagerCore {
            sidecar_files: sidecar_file_map(&manifest)?,
            manifest,
            root,
            headers: crate::expert_stream_slice::ShardHeaderCache::new(),
            state: Mutex::new(RowPagerState::default()),
            budget_bytes: config.budget_bytes.max(1),
            load_delay: config.load_delay,
        });
        let (prefetch_tx, prefetch_worker) = if config.prefetch {
            let (tx, rx) = mpsc::channel::<PrefetchMsg>();
            let worker_core = core.clone();
            let handle = std::thread::Builder::new()
                .name("ax-expert-prefetch".to_string())
                .spawn(move || {
                    while let Ok(msg) = rx.recv() {
                        match msg {
                            PrefetchMsg::Warm { layer, experts } => {
                                let _ = warm_experts(&worker_core, layer, &experts);
                            }
                            PrefetchMsg::Stop => break,
                        }
                    }
                })
                .expect("spawn expert prefetch thread");
            (Some(tx), Some(handle))
        } else {
            (None, None)
        };
        Ok(Self {
            core,
            fuse_split_experts: config.fuse_split_experts,
            decay_interval: config.decay_interval.max(1),
            prefetch_tx: Mutex::new(prefetch_tx),
            prefetch_worker: Mutex::new(prefetch_worker),
            hotlist_out: Mutex::new(config.hotlist_out),
        })
    }

    pub fn budget_bytes(&self) -> usize {
        self.core.budget_bytes
    }

    pub fn cached_expert_count(&self) -> usize {
        self.core
            .state
            .lock()
            .expect("expert row pager lock")
            .entries
            .len()
    }

    /// Cached `(layer, expert)` keys in sorted order (diagnostics, tests).
    pub fn cached_expert_keys(&self) -> Vec<(u32, u32)> {
        let mut keys: Vec<(u32, u32)> = self
            .core
            .state
            .lock()
            .expect("expert row pager lock")
            .entries
            .keys()
            .copied()
            .collect();
        keys.sort_unstable();
        keys
    }

    pub fn resident_bytes(&self) -> usize {
        self.core
            .state
            .lock()
            .expect("expert row pager lock")
            .resident_bytes
    }

    /// Route-hotness score for `(layer, expert)` (0 when never selected).
    pub fn hotness_of(&self, layer: u32, expert: u32) -> u64 {
        self.core
            .state
            .lock()
            .expect("expert row pager lock")
            .hotness
            .get(&(layer, expert))
            .copied()
            .unwrap_or(0)
    }

    /// Synchronously warm `(layer, experts)` into the cache (prefetch path).
    pub fn warm_experts(&self, layer: u32, experts: &[u32]) -> Result<usize, ExpertStreamError> {
        warm_experts(&self.core, layer, experts)
    }

    #[cfg(test)]
    fn test_remove_expert(&self, layer: u32, expert: u32) {
        let mut state = self.core.state.lock().expect("expert row pager lock");
        if let Some(entry) = state.entries.remove(&(layer, expert)) {
            state.resident_bytes = state.resident_bytes.saturating_sub(entry.bytes);
        }
    }

    fn unpin(&self, layer: u32, ids: &[u32]) {
        let mut state = self.core.state.lock().expect("expert row pager lock");
        for id in ids {
            if let Some(entry) = state.entries.get_mut(&(layer, *id)) {
                entry.pinned = false;
            }
        }
    }

    /// Make `ids` of `layer` resident and assemble compacted `[top_k, ...]`
    /// stacks in request order. Missing experts are read from SSD; the cache
    /// is evicted back under the byte budget afterwards, keeping the
    /// currently selected experts pinned until assembly completes.
    pub fn ensure_experts(
        &self,
        layer: u32,
        ids: &[u32],
    ) -> Result<CompactedExperts, ExpertStreamError> {
        if ids.is_empty() {
            return Err(ExpertStreamError::Paging(
                "no expert ids to page".to_string(),
            ));
        }
        let num_experts = self.core.manifest.num_experts;
        if let Some(bad) = ids.iter().find(|id| **id >= num_experts) {
            return Err(ExpertStreamError::Paging(format!(
                "expert id {bad} out of range ({num_experts} experts)"
            )));
        }
        let prefetch_tx = self.prefetch_tx.lock().expect("prefetch lock").clone();
        let missing: Vec<u32> = {
            let mut state = self.core.state.lock().expect("expert row pager lock");
            for id in ids {
                *state.hotness.entry((layer, *id)).or_insert(0) += 1;
                if let Some(entry) = state.entries.get_mut(&(layer, *id)) {
                    entry.pinned = true;
                }
            }
            state.selections_since_decay += ids.len() as u64;
            if state.selections_since_decay >= self.decay_interval {
                for value in state.hotness.values_mut() {
                    *value /= 2;
                }
                state.selections_since_decay = 0;
            }
            state.last_ids.insert(layer, ids.to_vec());
            ids.iter()
                .copied()
                .filter(|id| !state.entries.contains_key(&(layer, *id)))
                .collect()
        };

        let loaded = match load_rows(&self.core, layer, &missing) {
            Ok(rows) => rows,
            Err(error) => {
                self.unpin(layer, ids);
                return Err(error);
            }
        };

        let snapshot: Vec<LayerExpertStack> = {
            let mut state = self.core.state.lock().expect("expert row pager lock");
            let state = &mut *state;
            for (expert, rows) in missing.iter().copied().zip(loaded) {
                let std::collections::hash_map::Entry::Vacant(slot) =
                    state.entries.entry((layer, expert))
                else {
                    continue; // a racing loader filled this expert already
                };
                let bytes = expert_stack_bytes(&rows);
                state.tick += 1;
                let tick = state.tick;
                state.resident_bytes = state.resident_bytes.saturating_add(bytes);
                slot.insert(ExpertRowEntry {
                    rows,
                    bytes,
                    tick,
                    pinned: true,
                });
            }
            evict_to_budget(state, self.core.budget_bytes);
            let snapshot: Result<Vec<_>, _> = ids
                .iter()
                .map(|id| {
                    state
                        .entries
                        .get(&(layer, *id))
                        .map(|entry| entry.rows.clone())
                        .ok_or_else(|| {
                            ExpertStreamError::Paging(format!(
                                "expert {id} of layer {layer} was evicted during assembly"
                            ))
                        })
                })
                .collect();
            for id in ids {
                if let Some(entry) = state.entries.get_mut(&(layer, *id)) {
                    entry.pinned = false;
                }
            }
            if let Some(tx) = prefetch_tx.as_ref() {
                let next = layer + 1;
                if self.core.manifest.tensors_for_layer(next).next().is_some()
                    && let Some(predicted) = state.last_ids.get(&next).cloned()
                {
                    let _ = tx.send(PrefetchMsg::Warm {
                        layer: next,
                        experts: predicted,
                    });
                }
            }
            snapshot?
        };

        let stack = assemble_stack(&snapshot, self.fuse_split_experts)?;
        Ok(CompactedExperts {
            stack,
            remap: (0..ids.len() as u32).collect(),
        })
    }

    /// Split variant of [`Self::ensure_experts`] (ADR-028 Phase 1
    /// split-submit overlap): identical hotness/pin/decay/last-ids
    /// accounting, but the selection is partitioned into an already-resident
    /// set — assembled and returned immediately so the caller can submit its
    /// GPU work — and a missing set that
    /// [`Self::ensure_experts_split_finish`] loads from SSD afterwards.
    ///
    /// Resident experts are pinned from this call until the guard releases
    /// (finish or drop), so an in-flight overlap can never lose a row to
    /// eviction; the missing experts land pinned at finish. When the
    /// selection is fully resident or fully missing the parts degenerate to
    /// exactly what `ensure_experts` would have returned.
    pub fn ensure_experts_split_begin(
        &self,
        layer: u32,
        ids: &[u32],
    ) -> Result<SplitExpertsBegin, ExpertStreamError> {
        if ids.is_empty() {
            return Err(ExpertStreamError::Paging(
                "no expert ids to page".to_string(),
            ));
        }
        let num_experts = self.core.manifest.num_experts;
        if let Some(bad) = ids.iter().find(|id| **id >= num_experts) {
            return Err(ExpertStreamError::Paging(format!(
                "expert id {bad} out of range ({num_experts} experts)"
            )));
        }
        let prefetch_tx = self.prefetch_tx.lock().expect("prefetch lock").clone();
        let (resident_rows, missing_ids, order) = {
            let mut state = self.core.state.lock().expect("expert row pager lock");
            for id in ids {
                *state.hotness.entry((layer, *id)).or_insert(0) += 1;
                if let Some(entry) = state.entries.get_mut(&(layer, *id)) {
                    entry.pinned = true;
                }
            }
            state.selections_since_decay += ids.len() as u64;
            if state.selections_since_decay >= self.decay_interval {
                for value in state.hotness.values_mut() {
                    *value /= 2;
                }
                state.selections_since_decay = 0;
            }
            state.last_ids.insert(layer, ids.to_vec());
            let mut resident_rows = Vec::new();
            let mut missing_ids = Vec::new();
            // Per request position: Ok(slot in the resident part) or
            // Err(slot in the missing part).
            let mut parts: Vec<Result<u32, u32>> = Vec::with_capacity(ids.len());
            for id in ids {
                match state.entries.get(&(layer, *id)) {
                    Some(entry) => {
                        parts.push(Ok(resident_rows.len() as u32));
                        resident_rows.push(entry.rows.clone());
                    }
                    None => {
                        parts.push(Err(missing_ids.len() as u32));
                        missing_ids.push(*id);
                    }
                }
            }
            let resident_total = resident_rows.len() as u32;
            let order = parts
                .iter()
                .map(|part| match part {
                    Ok(slot) => *slot,
                    Err(slot) => resident_total + *slot,
                })
                .collect();
            (resident_rows, missing_ids, order)
        };
        let resident = if resident_rows.is_empty() {
            CompactedExperts {
                stack: LayerExpertStack::default(),
                remap: Vec::new(),
            }
        } else {
            match assemble_stack(&resident_rows, self.fuse_split_experts) {
                Ok(stack) => CompactedExperts {
                    stack,
                    remap: (0..resident_rows.len() as u32).collect(),
                },
                Err(error) => {
                    self.unpin(layer, ids);
                    return Err(error);
                }
            }
        };
        Ok(SplitExpertsBegin {
            core: self.core.clone(),
            layer,
            ids: ids.to_vec(),
            prefetch_tx,
            resident,
            missing_ids,
            order,
            done: false,
        })
    }

    /// Load the missing experts of a split ensure (off the state lock),
    /// insert them pinned, evict to budget, and assemble the missing part's
    /// compacted stack in request order. Releases the selection's pins and
    /// sends the next-layer prefetch, completing the ensure. On error the
    /// begin guard still releases the pins on drop.
    pub fn ensure_experts_split_finish(
        &self,
        begin: &mut SplitExpertsBegin,
    ) -> Result<CompactedExperts, ExpertStreamError> {
        if begin.missing_ids.is_empty() {
            begin.release();
            return Ok(CompactedExperts {
                stack: LayerExpertStack::default(),
                remap: Vec::new(),
            });
        }
        let loaded = load_rows(&self.core, begin.layer, &begin.missing_ids)?;
        let missing_rows = {
            let mut state = self.core.state.lock().expect("expert row pager lock");
            let state = &mut *state;
            for (expert, rows) in begin.missing_ids.iter().copied().zip(loaded) {
                let std::collections::hash_map::Entry::Vacant(slot) =
                    state.entries.entry((begin.layer, expert))
                else {
                    continue; // a racing loader filled this expert already
                };
                let bytes = expert_stack_bytes(&rows);
                state.tick += 1;
                let tick = state.tick;
                state.resident_bytes = state.resident_bytes.saturating_add(bytes);
                slot.insert(ExpertRowEntry {
                    rows,
                    bytes,
                    tick,
                    pinned: true,
                });
            }
            evict_to_budget(state, self.core.budget_bytes);
            let snapshot: Result<Vec<_>, _> = begin
                .missing_ids
                .iter()
                .map(|id| {
                    state
                        .entries
                        .get(&(begin.layer, *id))
                        .map(|entry| entry.rows.clone())
                        .ok_or_else(|| {
                            ExpertStreamError::Paging(format!(
                                "expert {id} of layer {} was evicted during assembly",
                                begin.layer
                            ))
                        })
                })
                .collect();
            snapshot?
        };
        let stack = assemble_stack(&missing_rows, self.fuse_split_experts)?;
        let remap = (0..begin.missing_ids.len() as u32).collect();
        begin.release();
        Ok(CompactedExperts { stack, remap })
    }

    /// Preload an offline hotlist (`ax.expert-hotlist.v1`) in file order
    /// (hottest first), seeding hotness with the measured weight so valuable
    /// experts survive early evictions. Stops at the byte budget; returns the
    /// number of experts warmed.
    pub fn preload_hotlist(&self, path: &Path) -> Result<usize, ExpertStreamError> {
        let bytes = std::fs::read(path).map_err(|e| {
            ExpertStreamError::InvalidManifest(format!("read hotlist {}: {e}", path.display()))
        })?;
        let hotlist: ExpertHotlistFile = serde_json::from_slice(&bytes).map_err(|e| {
            ExpertStreamError::InvalidManifest(format!(
                "hotlist {}: JSON parse: {e}",
                path.display()
            ))
        })?;
        if hotlist.schema_version != EXPERT_HOTLIST_SCHEMA_V1 {
            return Err(ExpertStreamError::InvalidManifest(format!(
                "hotlist {}: unsupported schema_version {:?} (expected {:?})",
                path.display(),
                hotlist.schema_version,
                EXPERT_HOTLIST_SCHEMA_V1
            )));
        }
        let num_experts = self.core.manifest.num_experts;
        let mut warmed = 0usize;
        for entry in &hotlist.entries {
            if entry.expert >= num_experts
                || self
                    .core
                    .manifest
                    .tensors_for_layer(entry.layer)
                    .next()
                    .is_none()
            {
                continue;
            }
            enum HotlistAction {
                Load,
                Skip,
                Stop,
            }
            let action = {
                let state = self.core.state.lock().expect("expert row pager lock");
                if state.entries.contains_key(&(entry.layer, entry.expert)) {
                    HotlistAction::Skip
                } else if state.resident_bytes >= self.core.budget_bytes {
                    HotlistAction::Stop
                } else {
                    HotlistAction::Load
                }
            };
            match action {
                HotlistAction::Skip => continue,
                HotlistAction::Stop => break,
                HotlistAction::Load => {}
            }
            let loaded = load_rows(&self.core, entry.layer, &[entry.expert])?;
            let mut state = self.core.state.lock().expect("expert row pager lock");
            let state = &mut *state;
            if let Some(rows) = loaded.into_iter().next() {
                let bytes = expert_stack_bytes(&rows);
                state.tick += 1;
                let tick = state.tick;
                state.resident_bytes = state.resident_bytes.saturating_add(bytes);
                state.entries.insert(
                    (entry.layer, entry.expert),
                    ExpertRowEntry {
                        rows,
                        bytes,
                        tick,
                        pinned: false,
                    },
                );
                state
                    .hotness
                    .insert((entry.layer, entry.expert), entry.weight.max(1));
                warmed += 1;
            }
        }
        Ok(warmed)
    }

    /// Serialize the observed route-hotness histogram as an
    /// `ax.expert-hotlist.v1` file (hottest first).
    pub fn dump_hotlist(&self, path: &Path) -> Result<usize, ExpertStreamError> {
        let state = self.core.state.lock().expect("expert row pager lock");
        let mut entries: Vec<ExpertHotlistEntry> = state
            .hotness
            .iter()
            .filter(|(_, count)| **count > 0)
            .map(|((layer, expert), count)| ExpertHotlistEntry {
                layer: *layer,
                expert: *expert,
                weight: *count,
            })
            .collect();
        entries.sort_by(|a, b| {
            b.weight
                .cmp(&a.weight)
                .then(a.layer.cmp(&b.layer))
                .then(a.expert.cmp(&b.expert))
        });
        let count = entries.len();
        let file = ExpertHotlistFile {
            schema_version: EXPERT_HOTLIST_SCHEMA_V1.to_string(),
            generated_by: "ax-engine".to_string(),
            entries,
        };
        let bytes = serde_json::to_vec_pretty(&file)
            .map_err(|e| ExpertStreamError::InvalidManifest(format!("serialize hotlist: {e}")))?;
        std::fs::write(path, bytes).map_err(|e| {
            ExpertStreamError::InvalidManifest(format!("write hotlist {}: {e}", path.display()))
        })?;
        Ok(count)
    }
}

impl Drop for ExpertRowPager {
    fn drop(&mut self) {
        if let Some(tx) = self.prefetch_tx.lock().expect("prefetch lock").take() {
            let _ = tx.send(PrefetchMsg::Stop);
        }
        if let Some(handle) = self.prefetch_worker.lock().expect("prefetch lock").take() {
            let _ = handle.join();
        }
        if let Some(path) = self.hotlist_out.lock().expect("hotlist lock").take()
            && let Err(error) = self.dump_hotlist(&path)
        {
            tracing::warn!(
                target: "ax_engine_mlx",
                path = %path.display(),
                %error,
                "failed to dump expert hotlist"
            );
        }
    }
}

/// Per-layer handle stashed on `LayerWeights` when the layer's expert stack
/// is streamed instead of resident.
enum ExpertPagerBackend {
    LayerStack(Arc<ExpertStackPager>),
    Rows {
        stack: Arc<ExpertStackPager>,
        rows: Arc<ExpertRowPager>,
    },
}

pub struct ExpertLayerSource {
    backend: ExpertPagerBackend,
    layer: u32,
}

impl ExpertLayerSource {
    pub fn new(pager: Arc<ExpertStackPager>, layer: u32) -> Self {
        Self {
            backend: ExpertPagerBackend::LayerStack(pager),
            layer,
        }
    }

    /// Expert-granularity handle: the row pager serves decode, the
    /// layer-stack pager serves prefill and is the fail-closed fallback.
    pub fn new_with_rows(
        stack: Arc<ExpertStackPager>,
        rows: Arc<ExpertRowPager>,
        layer: u32,
    ) -> Self {
        Self {
            backend: ExpertPagerBackend::Rows { stack, rows },
            layer,
        }
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// Whether per-expert decode paging is available for this layer.
    pub fn is_per_expert(&self) -> bool {
        matches!(self.backend, ExpertPagerBackend::Rows { .. })
    }

    /// Resolve this layer's full expert stack, paging it in when needed.
    pub fn stack(&self) -> Result<LayerExpertStack, ExpertStreamError> {
        match &self.backend {
            ExpertPagerBackend::LayerStack(pager) => pager.ensure_layer(self.layer),
            ExpertPagerBackend::Rows { stack, .. } => stack.ensure_layer(self.layer),
        }
    }

    /// Page only the selected experts and return compacted `[top_k, ...]`
    /// stacks. `Err` when this layer has no row pager — callers must fall
    /// back to [`Self::stack`].
    pub fn experts_for_ids(&self, ids: &[u32]) -> Result<CompactedExperts, ExpertStreamError> {
        match &self.backend {
            ExpertPagerBackend::Rows { rows, .. } => rows.ensure_experts(self.layer, ids),
            ExpertPagerBackend::LayerStack(_) => Err(ExpertStreamError::Paging(
                "layer has no per-expert row pager".to_string(),
            )),
        }
    }

    /// Begin a split ensure on the row pager
    /// ([`ExpertRowPager::ensure_experts_split_begin`]); `Err` when this
    /// layer has no row pager.
    pub fn experts_split_begin(&self, ids: &[u32]) -> Result<SplitExpertsBegin, ExpertStreamError> {
        match &self.backend {
            ExpertPagerBackend::Rows { rows, .. } => {
                rows.ensure_experts_split_begin(self.layer, ids)
            }
            ExpertPagerBackend::LayerStack(_) => Err(ExpertStreamError::Paging(
                "layer has no per-expert row pager".to_string(),
            )),
        }
    }

    /// Finish a split ensure on the row pager
    /// ([`ExpertRowPager::ensure_experts_split_finish`]).
    pub fn experts_split_finish(
        &self,
        begin: &mut SplitExpertsBegin,
    ) -> Result<CompactedExperts, ExpertStreamError> {
        match &self.backend {
            ExpertPagerBackend::Rows { rows, .. } => rows.ensure_experts_split_finish(begin),
            ExpertPagerBackend::LayerStack(_) => Err(ExpertStreamError::Paging(
                "layer has no per-expert row pager".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fixture(dir: &Path, name: &str, value: &serde_json::Value) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
        path
    }

    fn manifest_json(required: bool) -> serde_json::Value {
        serde_json::json!({
            "schema_version": "axquant.expert-stream.v1",
            "generated_by": "axquant",
            "required": required,
            "mode": "layer-stack",
            "num_experts": 256,
            "experts_per_tok": 8,
            "estimated_resident_bytes": 40_000_000_000_u64,
            "estimated_full_resident_bytes": 800_000_000_000_u64,
            "estimated_max_layer_expert_bytes": 10_000_000_000_u64,
            "resident_roles": ["embedding", "attention", "router", "shared_expert", "norm", "lm_head", "mtp"],
            "streamed_roles": ["expert"],
            "tensors": [
                {
                    "name": "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                    "file": "model-00001-of-00080.safetensors",
                    "layer": 0,
                    "proj": "gate_up",
                    "expert_axis": 0,
                    "num_experts": 256,
                    "bits": 2,
                    "group_size": 64
                }
            ]
        })
    }

    #[test]
    fn parses_contract_manifest() {
        let manifest =
            ExpertStreamManifest::parse(&serde_json::to_vec(&manifest_json(true)).unwrap())
                .expect("contract-shaped manifest must parse");
        assert!(manifest.required);
        assert_eq!(manifest.mode, "layer-stack");
        assert_eq!(manifest.num_experts, 256);
        assert_eq!(manifest.estimated_full_resident_bytes, 800_000_000_000);
        assert_eq!(manifest.tensors.len(), 1);
        assert_eq!(manifest.tensors[0].parsed_proj, Some(ExpertProj::GateUp));
        assert_eq!(manifest.layer_indices(), vec![0]);
    }

    #[test]
    fn unknown_schema_version_fails_closed() {
        let mut value = manifest_json(true);
        value["schema_version"] = serde_json::json!("axquant.expert-stream.v9");
        let error = ExpertStreamManifest::parse(&serde_json::to_vec(&value).unwrap())
            .expect_err("unknown schema must fail closed");
        assert!(
            matches!(&error, ExpertStreamError::InvalidManifest(msg) if msg.contains("schema_version")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unknown_mode_fails_closed() {
        let mut value = manifest_json(true);
        value["mode"] = serde_json::json!("per-expert-unfused");
        let error = ExpertStreamManifest::parse(&serde_json::to_vec(&value).unwrap())
            .expect_err("unknown mode must fail closed");
        assert!(
            matches!(&error, ExpertStreamError::InvalidManifest(msg) if msg.contains("mode")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unknown_proj_fails_closed() {
        let mut value = manifest_json(true);
        value["tensors"][0]["proj"] = serde_json::json!("sideways");
        assert!(ExpertStreamManifest::parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn nonzero_expert_axis_fails_closed() {
        let mut value = manifest_json(true);
        value["tensors"][0]["expert_axis"] = serde_json::json!(1);
        assert!(ExpertStreamManifest::parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn skip_names_cover_base_and_sidecars() {
        let manifest =
            ExpertStreamManifest::parse(&serde_json::to_vec(&manifest_json(true)).unwrap())
                .unwrap();
        let skip = streamed_skip_names(&manifest);
        let base = "model.layers.0.mlp.switch_mlp.gate_proj";
        assert!(skip.contains("model.layers.0.mlp.switch_mlp.gate_proj.weight"));
        assert!(skip.contains(&format!("{base}.scales")));
        assert!(skip.contains(&format!("{base}.biases")));
        assert!(skip.contains(&format!("{base}.bias")));
        assert!(!skip.contains("model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn admission_required_without_flag_fails_closed() {
        let dir = std::env::temp_dir().join("ax_expert_stream_admission_required");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir, EXPERT_STREAM_MANIFEST_FILE, &manifest_json(true));

        let error = admit_expert_stream(&dir, false).expect_err("required pack must refuse load");
        match error {
            ExpertStreamError::StreamRequired {
                estimated_full_resident_bytes,
            } => {
                assert_eq!(estimated_full_resident_bytes, 800_000_000_000);
                assert!(error.to_string().contains("800000000000"));
            }
            other => panic!("unexpected error: {other}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn admission_required_with_flag_streams() {
        let dir = std::env::temp_dir().join("ax_expert_stream_admission_flag");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir, EXPERT_STREAM_MANIFEST_FILE, &manifest_json(true));
        assert!(admit_expert_stream(&dir, true).unwrap().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn admission_flag_without_manifest_defers_to_inference() {
        let dir = std::env::temp_dir().join("ax_expert_stream_admission_missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(
            resolve_expert_stream(
                StreamExpertsMode::Auto,
                None,
                || Err(ExpertStreamError::ManifestMissing),
                Some(512 * 1024 * 1024 * 1024),
            )
            .unwrap()
            .is_none()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn auto_streams_required_pack_without_explicit_on() {
        let file = ExpertStreamManifest::parse(&serde_json::to_vec(&manifest_json(true)).unwrap())
            .unwrap();
        let decided = resolve_expert_stream(
            StreamExpertsMode::Auto,
            Some(file),
            || Err(ExpertStreamError::ManifestMissing),
            Some(512 * 1024 * 1024 * 1024),
        )
        .unwrap();
        assert!(decided.is_some_and(|manifest| manifest.required));
    }

    #[test]
    fn auto_keeps_flash_sized_pack_resident_on_512gb() {
        let file = ExpertStreamManifest::parse(&serde_json::to_vec(&manifest_json(false)).unwrap())
            .unwrap();
        // ~115 GiB Flash-class optional pack on 512 GiB.
        let mut file = file;
        file.estimated_full_resident_bytes = 115 * 1024 * 1024 * 1024;
        let decided = resolve_expert_stream(
            StreamExpertsMode::Auto,
            Some(file),
            || Err(ExpertStreamError::ManifestMissing),
            Some(512 * 1024 * 1024 * 1024),
        )
        .unwrap();
        assert!(decided.is_none());
    }

    #[test]
    fn auto_streams_when_pack_exceeds_memory_plus_headroom() {
        let file = ExpertStreamManifest::parse(&serde_json::to_vec(&manifest_json(false)).unwrap())
            .unwrap();
        let mut file = file;
        file.estimated_full_resident_bytes = 800 * 1024 * 1024 * 1024;
        let decided = resolve_expert_stream(
            StreamExpertsMode::Auto,
            Some(file),
            || Err(ExpertStreamError::ManifestMissing),
            Some(512 * 1024 * 1024 * 1024),
        )
        .unwrap();
        assert!(decided.is_some());
    }

    #[test]
    fn should_auto_stream_uses_headroom() {
        let flash = 115 * 1024 * 1024 * 1024;
        let studio_192 = 192 * 1024 * 1024 * 1024;
        let studio_512 = 512 * 1024 * 1024 * 1024;
        assert!(!should_auto_stream(flash, Some(studio_192)));
        assert!(!should_auto_stream(flash, Some(studio_512)));
        assert!(should_auto_stream(
            800 * 1024 * 1024 * 1024,
            Some(studio_512)
        ));
        assert!(!should_auto_stream(flash, None));
    }

    #[test]
    fn admission_no_manifest_no_flag_is_default_resident() {
        let dir = std::env::temp_dir().join("ax_expert_stream_admission_none");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(admit_expert_stream(&dir, false).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn admission_optional_manifest_without_flag_stays_resident() {
        let dir = std::env::temp_dir().join("ax_expert_stream_admission_optional");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir, EXPERT_STREAM_MANIFEST_FILE, &manifest_json(false));
        assert!(admit_expert_stream(&dir, false).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn budget_env_clamps_to_at_least_one_layer() {
        assert_eq!(expert_layer_budget_from_env(None), 1);
        assert_eq!(expert_layer_budget_from_env(Some("")), 1);
        assert_eq!(expert_layer_budget_from_env(Some("0")), 1);
        assert_eq!(expert_layer_budget_from_env(Some("junk")), 1);
        assert_eq!(expert_layer_budget_from_env(Some("3")), 3);
    }

    #[test]
    fn env_flag_parsing() {
        assert!(!env_flag_enabled(None));
        assert!(!env_flag_enabled(Some("0")));
        assert!(env_flag_enabled(Some("1")));
        assert!(env_flag_enabled(Some("true")));
    }

    fn infer_spec(
        name: &str,
        role: NativeTensorRole,
        layer: u32,
        experts: u64,
        bytes: u64,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index: Some(layer),
            dtype: ax_engine_core::NativeTensorDataType::U32,
            source_tensor_type: None,
            source_quantized: true,
            quantization: Some(ax_engine_core::NativeTensorQuantization {
                mode: "affine".into(),
                group_size: 64,
                bits: 2,
            }),
            quantized_source: None,
            shape: vec![experts, 8, 4],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: bytes,
        }
    }

    #[test]
    fn infer_flash_switch_mlp_roles_without_manifest_file() {
        let specs = vec![
            infer_spec(
                "model.layers.0.ffn.switch_mlp.gate_proj.weight",
                NativeTensorRole::FfnGateUpExpsPacked,
                0,
                4,
                100,
            ),
            infer_spec(
                "model.layers.0.ffn.switch_mlp.down_proj.weight",
                NativeTensorRole::FfnDownExps,
                0,
                4,
                80,
            ),
            infer_spec(
                "model.layers.1.ffn.switch_mlp.gate_proj.weight",
                NativeTensorRole::FfnGateUpExpsPacked,
                1,
                4,
                100,
            ),
            infer_spec(
                "model.layers.1.ffn.switch_mlp.down_proj.weight",
                NativeTensorRole::FfnDownExps,
                1,
                4,
                80,
            ),
            NativeTensorSpec {
                name: "model.layers.0.ffn.shared_experts.gate_proj.weight".into(),
                role: NativeTensorRole::FfnSharedExpertGate,
                layer_index: Some(0),
                dtype: ax_engine_core::NativeTensorDataType::Bf16,
                source_tensor_type: None,
                source_quantized: false,
                quantization: None,
                quantized_source: None,
                shape: vec![8, 4],
                file: PathBuf::from("model.safetensors"),
                offset_bytes: 0,
                length_bytes: 64,
            },
        ];
        let manifest = infer_layer_stack_manifest(&specs, 8).expect("flash roles must infer");
        assert!(!manifest.required);
        assert_eq!(manifest.num_experts, 4);
        assert_eq!(manifest.experts_per_tok, 8);
        assert_eq!(manifest.layer_indices(), vec![0, 1]);
        assert_eq!(
            manifest
                .tensors
                .iter()
                .map(|t| t.proj.as_str())
                .collect::<HashSet<_>>(),
            HashSet::from(["gate_up", "down"])
        );
        assert!(
            manifest
                .tensors
                .iter()
                .all(|t| !t.name.contains("shared_experts"))
        );
    }

    #[test]
    fn infer_without_expert_roles_fails_closed() {
        let specs = vec![NativeTensorSpec {
            name: "model.embed_tokens.weight".into(),
            role: NativeTensorRole::TokenEmbedding,
            layer_index: None,
            dtype: ax_engine_core::NativeTensorDataType::Bf16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![8, 4],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 64,
        }];
        assert!(matches!(
            infer_layer_stack_manifest(&specs, 8),
            Err(ExpertStreamError::ManifestMissing)
        ));
    }

    // ------------------------------------------------------------------
    // Synthetic 2-layer x 4-expert paging fixture (no real checkpoint).
    // ------------------------------------------------------------------

    const SYN_HIDDEN: i32 = 4;
    const SYN_INTER: i32 = 2;
    const SYN_EXPERTS: i32 = 4;

    fn write_safetensors_f32(
        dir: &Path,
        file_name: &str,
        tensors: &[(&str, Vec<i32>, Vec<f32>)],
    ) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&data);
        let path = dir.join(file_name);
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    fn synth_expert_values(layer: u32, out: i32, inn: i32) -> Vec<f32> {
        // Expert e fills its [out, in] matrix with (layer + 1) * 10 + e.
        let mut values = Vec::with_capacity((SYN_EXPERTS * out * inn) as usize);
        for expert in 0..SYN_EXPERTS {
            let fill = (layer as f32 + 1.0) * 10.0 + expert as f32;
            values.resize(values.len() + (out * inn) as usize, fill);
        }
        values
    }

    fn synth_tensors(layer: u32) -> [(&'static str, Vec<i32>, Vec<f32>); 2] {
        [
            (
                if layer == 0 {
                    "model.layers.0.mlp.switch_mlp.gate_up_proj.weight"
                } else {
                    "model.layers.1.mlp.switch_mlp.gate_up_proj.weight"
                },
                vec![SYN_EXPERTS, 2 * SYN_INTER, SYN_HIDDEN],
                synth_expert_values(layer, 2 * SYN_INTER, SYN_HIDDEN),
            ),
            (
                if layer == 0 {
                    "model.layers.0.mlp.switch_mlp.down_proj.weight"
                } else {
                    "model.layers.1.mlp.switch_mlp.down_proj.weight"
                },
                vec![SYN_EXPERTS, SYN_HIDDEN, SYN_INTER],
                synth_expert_values(layer, SYN_HIDDEN, SYN_INTER),
            ),
        ]
    }

    fn synth_manifest() -> ExpertStreamManifest {
        let mut tensors = Vec::new();
        for layer in 0u32..2 {
            let prefix = format!("model.layers.{layer}.mlp.switch_mlp");
            tensors.push(serde_json::json!({
                "name": format!("{prefix}.gate_up_proj.weight"),
                "file": "experts.safetensors",
                "layer": layer,
                "proj": "gate_up",
                "expert_axis": 0,
                "num_experts": SYN_EXPERTS,
                "bits": 2,
                "group_size": 64
            }));
            tensors.push(serde_json::json!({
                "name": format!("{prefix}.down_proj.weight"),
                "file": "experts.safetensors",
                "layer": layer,
                "proj": "down",
                "expert_axis": 0,
                "num_experts": SYN_EXPERTS,
                "bits": 2,
                "group_size": 64
            }));
        }
        let json = serde_json::json!({
            "schema_version": "axquant.expert-stream.v1",
            "generated_by": "ax-engine-test",
            "required": true,
            "mode": "layer-stack",
            "num_experts": SYN_EXPERTS,
            "experts_per_tok": 2,
            "estimated_resident_bytes": 1000,
            "estimated_full_resident_bytes": 5000,
            "estimated_max_layer_expert_bytes": 2000,
            "resident_roles": ["embedding", "attention", "router", "norm", "lm_head"],
            "streamed_roles": ["expert"],
            "tensors": tensors,
        });
        ExpertStreamManifest::parse(&serde_json::to_vec(&json).unwrap()).unwrap()
    }

    fn synth_fixture(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ax_expert_stream_synth_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut tensors: Vec<(&str, Vec<i32>, Vec<f32>)> = Vec::new();
        let layer0 = synth_tensors(0);
        let layer1 = synth_tensors(1);
        tensors.extend(layer0);
        tensors.extend(layer1);
        write_safetensors_f32(&dir, "experts.safetensors", &tensors);
        dir
    }

    #[test]
    fn pager_pages_layer_zero_and_keeps_layer_one_absent() {
        let dir = synth_fixture("page_l0");
        let pager = ExpertStackPager::new(
            Arc::new(synth_manifest()),
            dir.clone(),
            expert_layer_budget_from_env(None),
        );
        assert_eq!(pager.budget_layers(), 1);
        assert_eq!(pager.cached_layer_count(), 0);

        let stack = pager.ensure_layer(0).expect("layer 0 must page in");
        let gate_up = stack
            .gate_up_exps_packed
            .as_ref()
            .expect("gate_up slot mapped from proj=gate_up");
        let down = stack.down_exps.as_ref().expect("down slot mapped");
        assert!(stack.gate_exps.is_none() && stack.up_exps.is_none());
        assert_eq!(
            gate_up.weight.shape(),
            vec![SYN_EXPERTS, 2 * SYN_INTER, SYN_HIDDEN]
        );
        assert_eq!(
            down.weight.shape(),
            vec![SYN_EXPERTS, SYN_HIDDEN, SYN_INTER]
        );
        assert_eq!(gate_up.bits, 2);
        assert_eq!(gate_up.group_size, 64);

        // Layer 1 must still be absent after paging layer 0.
        assert_eq!(pager.cached_layer_count(), 1);
        assert_eq!(pager.cached_layer_indices(), vec![0]);

        // The paged weight must flow through the existing gather kernel.
        // Dense fixture (no .scales sidecar) exercises the same lane
        // `qw_gather` uses for dense experts: transpose + gather_mm.
        let ones: Vec<f32> = vec![1.0; SYN_HIDDEN as usize];
        let mut x_data = Vec::new();
        for v in &ones {
            x_data.extend_from_slice(&v.to_le_bytes());
        }
        let x = mlx_sys::MlxArray::from_raw_data(
            x_data.as_ptr(),
            x_data.len(),
            &[1, 1, SYN_HIDDEN],
            mlx_sys::MlxDtype::Float32,
        );
        let mut idx_data = Vec::new();
        for idx in [0u32, 2] {
            idx_data.extend_from_slice(&idx.to_le_bytes());
        }
        let indices = mlx_sys::MlxArray::from_raw_data(
            idx_data.as_ptr(),
            idx_data.len(),
            &[1, 1, 2],
            mlx_sys::MlxDtype::Uint32,
        );
        let wt = mlx_sys::transpose(&gate_up.weight, &[0, 2, 1], None);
        let out = mlx_sys::gather_mm(&x, &wt, &indices, false, None);
        mlx_sys::eval(&[&out]);
        // gather_mm keeps the switch singleton (squeezed by the real MoE
        // path after the down projection).
        assert_eq!(out.shape(), vec![1, 1, 2, 1, 2 * SYN_INTER]);
        let values = out.data_f32();
        // Expert e output element = fill_e * SYN_HIDDEN.
        assert_eq!(values[0], 10.0 * SYN_HIDDEN as f32);
        assert_eq!(values[2 * SYN_INTER as usize], 12.0 * SYN_HIDDEN as f32);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pager_attaches_axquant_sidecar_entries_instead_of_overwriting_weight() {
        // MiniMax / Super-class ax_expert_stream.json lists
        // `.biases`, `.scales`, then `.weight` as three tensors with the
        // same proj. Paging must keep the packed weight and attach the
        // sidecars, not insert scales/biases as the expert slot.
        let dir = std::env::temp_dir().join("ax_expert_stream_sidecar_overwrite");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let weight = synth_expert_values(0, SYN_INTER, SYN_HIDDEN);
        let scales = vec![0.5_f32; (SYN_EXPERTS * SYN_INTER) as usize];
        let biases = vec![0.1_f32; (SYN_EXPERTS * SYN_INTER) as usize];
        write_safetensors_f32(
            &dir,
            "experts.safetensors",
            &[
                (
                    "language_model.model.layers.3.block_sparse_moe.switch_mlp.gate_proj.weight",
                    vec![SYN_EXPERTS, SYN_INTER, SYN_HIDDEN],
                    weight,
                ),
                (
                    "language_model.model.layers.3.block_sparse_moe.switch_mlp.gate_proj.scales",
                    vec![SYN_EXPERTS, SYN_INTER],
                    scales,
                ),
                (
                    "language_model.model.layers.3.block_sparse_moe.switch_mlp.gate_proj.biases",
                    vec![SYN_EXPERTS, SYN_INTER],
                    biases,
                ),
            ],
        );
        let prefix = "language_model.model.layers.3.block_sparse_moe.switch_mlp.gate_proj";
        let json = serde_json::json!({
            "schema_version": "axquant.expert-stream.v1",
            "generated_by": "axquant",
            "required": true,
            "mode": "layer-stack",
            "num_experts": SYN_EXPERTS,
            "experts_per_tok": 4,
            "estimated_resident_bytes": 1000,
            "estimated_full_resident_bytes": 5000,
            "estimated_max_layer_expert_bytes": 2000,
            "resident_roles": ["embedding", "attention", "router", "shared_expert", "norm", "lm_head"],
            "streamed_roles": ["expert"],
            "tensors": [
                {
                    "name": format!("{prefix}.biases"),
                    "file": "experts.safetensors",
                    "layer": 3,
                    "proj": "gate",
                    "expert_axis": 0,
                    "num_experts": SYN_EXPERTS,
                    "bits": 2,
                    "group_size": 32
                },
                {
                    "name": format!("{prefix}.scales"),
                    "file": "experts.safetensors",
                    "layer": 3,
                    "proj": "gate",
                    "expert_axis": 0,
                    "num_experts": SYN_EXPERTS,
                    "bits": 2,
                    "group_size": 32
                },
                {
                    "name": format!("{prefix}.weight"),
                    "file": "experts.safetensors",
                    "layer": 3,
                    "proj": "gate",
                    "expert_axis": 0,
                    "num_experts": SYN_EXPERTS,
                    "bits": 2,
                    "group_size": 32
                }
            ],
        });
        let manifest = ExpertStreamManifest::parse(&serde_json::to_vec(&json).unwrap()).unwrap();
        let pager = ExpertStackPager::new(
            Arc::new(manifest),
            dir.clone(),
            expert_layer_budget_from_env(None),
        );
        let stack = pager.ensure_layer(3).expect("layer 3 must page in");
        let gate = stack
            .gate_exps
            .as_ref()
            .expect("gate slot must keep the packed weight");
        assert_eq!(
            gate.weight.shape(),
            vec![SYN_EXPERTS, SYN_INTER, SYN_HIDDEN],
            "sidecar entries must not overwrite the packed weight slot"
        );
        assert!(
            gate.scales.is_some(),
            "MiniMax stream sidecars must attach as quantization scales"
        );
        assert!(
            gate.biases.is_some(),
            "MiniMax stream sidecars must attach as quantization biases"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pager_pages_deepseek_v4_flash_switch_mlp_names() {
        let dir = std::env::temp_dir().join("ax_expert_stream_flash_names");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_safetensors_f32(
            &dir,
            "experts.safetensors",
            &[
                (
                    "model.layers.0.ffn.switch_mlp.gate_proj.weight",
                    vec![SYN_EXPERTS, 2 * SYN_INTER, SYN_HIDDEN],
                    synth_expert_values(0, 2 * SYN_INTER, SYN_HIDDEN),
                ),
                (
                    "model.layers.0.ffn.switch_mlp.down_proj.weight",
                    vec![SYN_EXPERTS, SYN_HIDDEN, SYN_INTER],
                    synth_expert_values(0, SYN_HIDDEN, SYN_INTER),
                ),
            ],
        );
        let json = serde_json::json!({
            "schema_version": "axquant.expert-stream.v1",
            "required": false,
            "mode": "layer-stack",
            "num_experts": SYN_EXPERTS,
            "experts_per_tok": 2,
            "estimated_resident_bytes": 100,
            "estimated_full_resident_bytes": 1000,
            "estimated_max_layer_expert_bytes": 500,
            "resident_roles": ["embedding", "attention", "router", "shared_expert", "norm", "lm_head"],
            "streamed_roles": ["expert"],
            "tensors": [
                {
                    "name": "model.layers.0.ffn.switch_mlp.gate_proj.weight",
                    "file": "experts.safetensors",
                    "layer": 0,
                    "proj": "gate_up",
                    "expert_axis": 0,
                    "num_experts": SYN_EXPERTS,
                    "bits": 2,
                    "group_size": 64
                },
                {
                    "name": "model.layers.0.ffn.switch_mlp.down_proj.weight",
                    "file": "experts.safetensors",
                    "layer": 0,
                    "proj": "down",
                    "expert_axis": 0,
                    "num_experts": SYN_EXPERTS,
                    "bits": 2,
                    "group_size": 64
                }
            ]
        });
        let pager = ExpertStackPager::new(
            Arc::new(ExpertStreamManifest::parse(&serde_json::to_vec(&json).unwrap()).unwrap()),
            dir.clone(),
            1,
        );
        let stack = pager.ensure_layer(0).expect("flash layer 0 must page");
        assert!(stack.gate_up_exps_packed.is_some());
        assert!(stack.down_exps.is_some());
        assert!(stack.gate_exps.is_none() && stack.up_exps.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pager_evicts_lru_layer_when_budget_exceeded() {
        let dir = synth_fixture("evict");
        let pager = ExpertStackPager::new(Arc::new(synth_manifest()), dir.clone(), 1);

        pager.ensure_layer(0).unwrap();
        assert_eq!(pager.cached_layer_indices(), vec![0]);

        // Paging layer 1 with a 1-layer budget evicts layer 0.
        let stack1 = pager.ensure_layer(1).unwrap();
        assert_eq!(pager.cached_layer_count(), 1);
        assert_eq!(pager.cached_layer_indices(), vec![1]);
        let gate_up1 = stack1.gate_up_exps_packed.unwrap();
        mlx_sys::eval(&[&gate_up1.weight]);
        // Layer-1 expert 0 fills with 20.0, proving the right shard data was read.
        assert_eq!(gate_up1.weight.data_f32()[0], 20.0);

        // Re-paging layer 0 works after eviction (cache miss → disk load).
        pager.ensure_layer(0).unwrap();
        assert_eq!(pager.cached_layer_indices(), vec![0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pager_budget_two_keeps_both_layers() {
        let dir = synth_fixture("budget2");
        let pager = ExpertStackPager::new(Arc::new(synth_manifest()), dir.clone(), 2);
        pager.ensure_layer(0).unwrap();
        pager.ensure_layer(1).unwrap();
        assert_eq!(pager.cached_layer_count(), 2);
        assert_eq!(pager.cached_layer_indices(), vec![0, 1]);
        // Touch layer 0 → LRU order flips.
        pager.ensure_layer(0).unwrap();
        assert_eq!(pager.cached_layer_indices(), vec![1, 0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn layer_source_handle_pages_through_pager() {
        let dir = synth_fixture("handle");
        let pager = Arc::new(ExpertStackPager::new(
            Arc::new(synth_manifest()),
            dir.clone(),
            1,
        ));
        let source = ExpertLayerSource::new(pager.clone(), 1);
        assert_eq!(source.layer(), 1);
        let stack = source.stack().expect("handle must page layer 1");
        assert!(stack.down_exps.is_some());
        assert_eq!(pager.cached_layer_indices(), vec![1]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn initial_load_filtered_loader_excludes_streamed_names() {
        // Mixed shard: one resident tensor + one streamed expert tensor with
        // its quantization sidecars. The initial load (Exclude filter) must
        // materialize only the resident tensor — the streamed names never
        // enter the resident map and are never eval'd.
        let dir = std::env::temp_dir().join("ax_expert_stream_skip_list");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_safetensors_f32(
            &dir,
            "mixed.safetensors",
            &[
                ("model.embed.weight", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
                (
                    "model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
                    vec![1, 2],
                    vec![9.0, 9.0],
                ),
                (
                    "model.layers.0.mlp.switch_mlp.gate_up_proj.scales",
                    vec![1, 1],
                    vec![0.5],
                ),
                (
                    "model.layers.0.mlp.switch_mlp.gate_up_proj.biases",
                    vec![1, 1],
                    vec![0.1],
                ),
            ],
        );

        let manifest = synth_manifest();
        let skip = streamed_skip_names(&manifest);
        // The fixture's expert name matches the synthetic manifest.
        assert!(skip.contains("model.layers.0.mlp.switch_mlp.gate_up_proj.weight"));

        // Mirror load_weights' per-spec gate: a spec whose name is skipped
        // never triggers a file load for that tensor.
        let resident_spec_name = "model.embed.weight";
        let streamed_spec_name = "model.layers.0.mlp.switch_mlp.gate_up_proj.weight";
        assert!(!skip.contains(resident_spec_name));
        assert!(skip.contains(streamed_spec_name));

        let tensors = mlx_sys::load_safetensors_filtered(
            &dir.join("mixed.safetensors"),
            mlx_sys::SafetensorsNameFilter::Exclude(&skip),
        )
        .expect("filtered load must succeed");
        assert!(tensors.contains_key(resident_spec_name));
        assert!(!tensors.contains_key(streamed_spec_name));
        assert!(!tensors.contains_key("model.layers.0.mlp.switch_mlp.gate_up_proj.scales"));
        assert!(!tensors.contains_key("model.layers.0.mlp.switch_mlp.gate_up_proj.biases"));

        let embed = tensors.get(resident_spec_name).unwrap();
        mlx_sys::eval(&[embed]);
        assert_eq!(embed.data_f32(), &[1.0, 2.0, 3.0, 4.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pager_keep_filter_reads_only_the_requested_layer() {
        // Both layers share one shard file; paging layer 1 must only
        // materialize layer-1 tensors (single-tensor slice semantics).
        let dir = synth_fixture("keep_filter");
        let pager = ExpertStackPager::new(Arc::new(synth_manifest()), dir.clone(), 1);
        let stack = pager.ensure_layer(1).unwrap();
        let gate_up = stack.gate_up_exps_packed.unwrap();
        mlx_sys::eval(&[&gate_up.weight]);
        // Layer 1 expert 3 fills with 23.0; layer 0 would read 13.0.
        let values = gate_up.weight.data_f32();
        let expert3_offset = (3 * 2 * SYN_INTER * SYN_HIDDEN) as usize;
        assert_eq!(values[expert3_offset], 23.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // ExpertRowPager (per-expert granularity) tests.
    // ------------------------------------------------------------------

    /// gate_up row [1, 2*SYN_INTER, SYN_HIDDEN] + down row [1, SYN_HIDDEN,
    /// SYN_INTER] in F32: (16 + 8) * 4 bytes.
    const ROW_BYTES: usize = 96;

    fn row_pager(dir: &Path, budget_bytes: usize, prefetch: bool) -> ExpertRowPager {
        row_pager_with_decay(dir, budget_bytes, prefetch, 4096)
    }

    fn row_pager_with_decay(
        dir: &Path,
        budget_bytes: usize,
        prefetch: bool,
        decay: u64,
    ) -> ExpertRowPager {
        ExpertRowPager::new(
            Arc::new(synth_manifest()),
            dir.to_path_buf(),
            ExpertRowPagerConfig {
                budget_bytes,
                fuse_split_experts: false,
                prefetch,
                decay_interval: decay,
                hotlist_out: None,
                load_delay: None,
            },
        )
        .unwrap()
    }

    fn u32_index_array(ids: &[u32]) -> mlx_sys::MlxArray {
        let mut data = Vec::new();
        for id in ids {
            data.extend_from_slice(&id.to_le_bytes());
        }
        mlx_sys::MlxArray::from_raw_data(
            data.as_ptr(),
            data.len(),
            &[1, 1, ids.len() as i32],
            mlx_sys::MlxDtype::Uint32,
        )
    }

    fn ones_x() -> mlx_sys::MlxArray {
        let mut data = Vec::new();
        for value in [1.0_f32; SYN_HIDDEN as usize] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        mlx_sys::MlxArray::from_raw_data(
            data.as_ptr(),
            data.len(),
            &[1, 1, SYN_HIDDEN],
            mlx_sys::MlxDtype::Float32,
        )
    }

    #[test]
    fn row_pager_compacted_gather_matches_full_stack_gather() {
        let dir = synth_fixture("row_gather");
        let stack_pager = ExpertStackPager::new(Arc::new(synth_manifest()), dir.clone(), 4);
        let full = stack_pager.ensure_layer(0).unwrap();
        let rows_pager = row_pager(&dir, 1 << 20, false);
        let compacted = rows_pager.ensure_experts(0, &[3, 1]).unwrap();

        assert_eq!(compacted.remap, vec![0, 1]);
        let full_gu = full.gate_up_exps_packed.as_ref().unwrap();
        let row_gu = compacted.stack.gate_up_exps_packed.as_ref().unwrap();
        assert_eq!(row_gu.weight.shape(), vec![2, 2 * SYN_INTER, SYN_HIDDEN]);
        assert_eq!(
            compacted.stack.down_exps.as_ref().unwrap().weight.shape(),
            vec![2, SYN_HIDDEN, SYN_INTER]
        );
        // Request order is preserved: slot 0 holds expert 3, slot 1 expert 1.
        mlx_sys::eval(&[&row_gu.weight]);
        let row_values = row_gu.weight.data_f32();
        assert_eq!(row_values[0], 13.0);
        assert_eq!(row_values[(2 * SYN_INTER * SYN_HIDDEN) as usize], 11.0);

        // gather_mm on the compacted stack with remapped ids must produce
        // bit-identical output to the full stack with the original ids.
        let x = ones_x();
        let full_out = mlx_sys::gather_mm(
            &x,
            &mlx_sys::transpose(&full_gu.weight, &[0, 2, 1], None),
            &u32_index_array(&[3, 1]),
            false,
            None,
        );
        let row_out = mlx_sys::gather_mm(
            &x,
            &mlx_sys::transpose(&row_gu.weight, &[0, 2, 1], None),
            &u32_index_array(&[0, 1]),
            false,
            None,
        );
        mlx_sys::eval(&[&full_out, &row_out]);
        assert_eq!(full_out.shape(), row_out.shape());
        assert_eq!(full_out.data_f32(), row_out.data_f32());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_hotness_eviction_beats_recency() {
        let dir = synth_fixture("row_hot");
        let pager = row_pager(&dir, 2 * ROW_BYTES, false);
        pager.ensure_experts(0, &[0]).unwrap();
        pager.ensure_experts(0, &[0]).unwrap(); // (0,0) hot 2, but oldest
        pager.ensure_experts(0, &[1]).unwrap(); // (0,1) hot 1, more recent
        pager.ensure_experts(0, &[2]).unwrap(); // forces one eviction
        // LRU would evict (0,0); route hotness evicts the colder (0,1).
        assert_eq!(pager.cached_expert_keys(), vec![(0, 0), (0, 2)]);
        assert_eq!(pager.hotness_of(0, 0), 2);
        assert_eq!(pager.hotness_of(0, 1), 1); // history survives eviction
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_decay_halves_selection_counts() {
        let dir = synth_fixture("row_decay");
        let pager = row_pager_with_decay(&dir, 1 << 20, false, 4);
        pager.ensure_experts(0, &[0]).unwrap();
        pager.ensure_experts(0, &[0]).unwrap();
        pager.ensure_experts(0, &[1]).unwrap();
        assert_eq!(pager.hotness_of(0, 0), 2);
        assert_eq!(pager.hotness_of(0, 1), 1);
        pager.ensure_experts(0, &[0]).unwrap(); // 4th selection → decay pass
        assert_eq!(pager.hotness_of(0, 0), 1); // 3 / 2
        assert_eq!(pager.hotness_of(0, 1), 0); // 1 / 2
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_soft_budget_keeps_pinned_selection() {
        let dir = synth_fixture("row_pinned");
        let pager = row_pager(&dir, ROW_BYTES, false);
        // Budget fits one expert, but the pinned two-expert selection must
        // still assemble correctly (the budget is a soft cap).
        let compacted = pager.ensure_experts(0, &[1, 2]).unwrap();
        assert_eq!(
            compacted
                .stack
                .gate_up_exps_packed
                .as_ref()
                .unwrap()
                .weight
                .shape(),
            vec![2, 2 * SYN_INTER, SYN_HIDDEN]
        );
        // The next call evicts the now-unpinned previous selection.
        pager.ensure_experts(0, &[0]).unwrap();
        assert_eq!(pager.cached_expert_keys(), vec![(0, 0)]);
        assert!(pager.resident_bytes() <= ROW_BYTES);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_hotlist_preload_and_dump_round_trip() {
        let dir = synth_fixture("row_hotlist");
        let hotlist_path = dir.join("hotlist.json");
        std::fs::write(
            &hotlist_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": "ax.expert-hotlist.v1",
                "generated_by": "test",
                "entries": [
                    {"layer": 0, "expert": 0, "weight": 100},
                    {"layer": 0, "expert": 1, "weight": 50},
                    {"layer": 0, "expert": 2, "weight": 10},
                    {"layer": 9, "expert": 0, "weight": 5},
                    {"layer": 0, "expert": 7, "weight": 5}
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        let pager = row_pager(&dir, 2 * ROW_BYTES, false);
        let warmed = pager.preload_hotlist(&hotlist_path).unwrap();
        assert_eq!(warmed, 2); // the byte budget stops the third entry
        assert_eq!(pager.cached_expert_keys(), vec![(0, 0), (0, 1)]);
        assert_eq!(pager.hotness_of(0, 0), 100);
        assert_eq!(pager.hotness_of(0, 1), 50);

        pager.ensure_experts(0, &[3]).unwrap();
        let dump_path = dir.join("dump.json");
        let dumped = pager.dump_hotlist(&dump_path).unwrap();
        assert_eq!(dumped, 3);
        let parsed: ExpertHotlistFile =
            serde_json::from_slice(&std::fs::read(&dump_path).unwrap()).unwrap();
        assert_eq!(parsed.schema_version, EXPERT_HOTLIST_SCHEMA_V1);
        assert_eq!(parsed.entries.len(), 3);
        assert_eq!(parsed.entries[0].weight, 100);
        assert_eq!(parsed.entries[1].weight, 50);
        assert_eq!(parsed.entries[2].weight, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_bad_hotlist_schema_fails_closed() {
        let dir = synth_fixture("row_bad_hotlist");
        let hotlist_path = dir.join("hotlist.json");
        std::fs::write(
            &hotlist_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": "ax.expert-hotlist.v9",
                "entries": []
            }))
            .unwrap(),
        )
        .unwrap();
        let pager = row_pager(&dir, 1 << 20, false);
        assert!(matches!(
            pager.preload_hotlist(&hotlist_path),
            Err(ExpertStreamError::InvalidManifest(_))
        ));
        assert_eq!(pager.cached_expert_count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_warm_experts_is_idempotent_and_hotness_neutral() {
        let dir = synth_fixture("row_warm");
        let pager = row_pager(&dir, 1 << 20, false);
        assert_eq!(pager.warm_experts(1, &[2, 3]).unwrap(), 2);
        assert_eq!(pager.warm_experts(1, &[2, 3]).unwrap(), 0);
        assert_eq!(pager.warm_experts(1, &[9]).unwrap(), 0); // out of range filtered
        assert_eq!(pager.cached_expert_keys(), vec![(1, 2), (1, 3)]);
        assert_eq!(pager.hotness_of(1, 2), 0); // warming never touches hotness
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_prefetch_worker_warms_predicted_next_layer() {
        let dir = synth_fixture("row_prefetch");
        let pager = row_pager(&dir, 1 << 20, true);
        pager.ensure_experts(0, &[0]).unwrap();
        pager.ensure_experts(1, &[1]).unwrap(); // seeds layer-1 prediction
        pager.test_remove_expert(1, 1);
        pager.ensure_experts(0, &[0]).unwrap(); // warms layer 1 in background
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !pager.cached_expert_keys().contains(&(1, 1)) {
            assert!(
                std::time::Instant::now() < deadline,
                "prefetch worker did not warm (1, 1)"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        drop(pager); // Drop must stop and join the worker cleanly.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_layer_source_fallback_contract() {
        let dir = synth_fixture("row_source");
        let stack = Arc::new(ExpertStackPager::new(
            Arc::new(synth_manifest()),
            dir.clone(),
            1,
        ));
        let layer_only = ExpertLayerSource::new(stack.clone(), 0);
        assert!(!layer_only.is_per_expert());
        assert!(layer_only.experts_for_ids(&[0]).is_err()); // callers fall back
        assert!(layer_only.stack().is_ok());

        let rows = Arc::new(row_pager(&dir, 1 << 20, false));
        let both = ExpertLayerSource::new_with_rows(stack, rows, 0);
        assert!(both.is_per_expert());
        let compacted = both.experts_for_ids(&[1]).unwrap();
        assert_eq!(compacted.remap, vec![0]);
        assert!(both.stack().is_ok()); // layer-stack fallback stays available
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // Split-sidecar fixtures: the down_proj triplet is split across files
    // and the manifest declares each sidecar as a first-class row (the
    // published 4-bit qwen4_exp layout).
    // ------------------------------------------------------------------

    fn synth_split_sidecar_manifest() -> ExpertStreamManifest {
        let prefix = "model.layers.0.mlp.switch_mlp";
        let json = serde_json::json!({
            "schema_version": "axquant.expert-stream.v1",
            "generated_by": "ax-engine-test",
            "required": true,
            "mode": "layer-stack",
            "num_experts": SYN_EXPERTS,
            "experts_per_tok": 2,
            "estimated_resident_bytes": 1000,
            "estimated_full_resident_bytes": 5000,
            "estimated_max_layer_expert_bytes": 2000,
            "resident_roles": ["embedding", "attention", "router", "norm", "lm_head"],
            "streamed_roles": ["expert"],
            "tensors": [
                {"name": format!("{prefix}.gate_up_proj.weight"), "file": "experts.safetensors", "layer": 0, "proj": "gate_up", "expert_axis": 0, "num_experts": SYN_EXPERTS, "bits": 2, "group_size": 64},
                {"name": format!("{prefix}.down_proj.weight"), "file": "experts.safetensors", "layer": 0, "proj": "down", "expert_axis": 0, "num_experts": SYN_EXPERTS, "bits": 2, "group_size": 64},
                {"name": format!("{prefix}.down_proj.scales"), "file": "sidecars.safetensors", "layer": 0, "proj": "down", "expert_axis": 0, "num_experts": SYN_EXPERTS, "bits": 2, "group_size": 64},
                {"name": format!("{prefix}.down_proj.biases"), "file": "sidecars.safetensors", "layer": 0, "proj": "down", "expert_axis": 0, "num_experts": SYN_EXPERTS, "bits": 2, "group_size": 64}
            ],
        });
        ExpertStreamManifest::parse(&serde_json::to_vec(&json).unwrap()).unwrap()
    }

    fn sidecar_fill_values(base: f32, rows: i32) -> Vec<f32> {
        let mut values = Vec::with_capacity((SYN_EXPERTS * rows) as usize);
        for expert in 0..SYN_EXPERTS {
            values.resize(values.len() + rows as usize, base + expert as f32);
        }
        values
    }

    fn synth_split_sidecar_fixture(tag: &str, include_scales: bool) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ax_expert_stream_split_sidecar_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_safetensors_f32(
            &dir,
            "experts.safetensors",
            &[
                (
                    "model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
                    vec![SYN_EXPERTS, 2 * SYN_INTER, SYN_HIDDEN],
                    synth_expert_values(0, 2 * SYN_INTER, SYN_HIDDEN),
                ),
                (
                    "model.layers.0.mlp.switch_mlp.down_proj.weight",
                    vec![SYN_EXPERTS, SYN_HIDDEN, SYN_INTER],
                    synth_expert_values(0, SYN_HIDDEN, SYN_INTER),
                ),
            ],
        );
        let mut sidecar_tensors: Vec<(&str, Vec<i32>, Vec<f32>)> = Vec::new();
        if include_scales {
            sidecar_tensors.push((
                "model.layers.0.mlp.switch_mlp.down_proj.scales",
                vec![SYN_EXPERTS, SYN_HIDDEN],
                sidecar_fill_values(100.0, SYN_HIDDEN),
            ));
        }
        sidecar_tensors.push((
            "model.layers.0.mlp.switch_mlp.down_proj.biases",
            vec![SYN_EXPERTS, SYN_HIDDEN],
            sidecar_fill_values(200.0, SYN_HIDDEN),
        ));
        write_safetensors_f32(&dir, "sidecars.safetensors", &sidecar_tensors);
        dir
    }

    #[test]
    fn sidecar_file_map_indexes_declared_rows_only() {
        let map = sidecar_file_map(&synth_split_sidecar_manifest()).unwrap();
        assert_eq!(map.len(), 2);
        let prefix = "model.layers.0.mlp.switch_mlp.down_proj";
        assert_eq!(
            map.get(&(0, format!("{prefix}.scales"))),
            Some(&PathBuf::from("sidecars.safetensors"))
        );
        assert_eq!(
            map.get(&(0, format!("{prefix}.biases"))),
            Some(&PathBuf::from("sidecars.safetensors"))
        );
        assert!(!map.contains_key(&(0, format!("{prefix}.weight"))));
    }

    /// Manifest with caller-chosen tensor rows (name, file, layer, proj).
    fn manifest_with_rows(rows: &[(&str, &str, u32, &str)]) -> ExpertStreamManifest {
        let tensors: Vec<serde_json::Value> = rows
            .iter()
            .map(|(name, file, layer, proj)| {
                serde_json::json!({
                    "name": name,
                    "file": file,
                    "layer": layer,
                    "proj": proj,
                    "expert_axis": 0,
                    "num_experts": SYN_EXPERTS,
                    "bits": 2,
                    "group_size": 64
                })
            })
            .collect();
        let json = serde_json::json!({
            "schema_version": "axquant.expert-stream.v1",
            "generated_by": "ax-engine-test",
            "required": true,
            "mode": "layer-stack",
            "num_experts": SYN_EXPERTS,
            "experts_per_tok": 2,
            "estimated_resident_bytes": 1000,
            "estimated_full_resident_bytes": 5000,
            "estimated_max_layer_expert_bytes": 2000,
            "resident_roles": ["embedding", "attention", "router", "norm", "lm_head"],
            "streamed_roles": ["expert"],
            "tensors": tensors,
        });
        ExpertStreamManifest::parse(&serde_json::to_vec(&json).unwrap()).unwrap()
    }

    #[test]
    fn sidecar_file_map_rejects_duplicate_rows() {
        let prefix = "model.layers.0.mlp.switch_mlp.down_proj";
        let manifest = manifest_with_rows(&[
            (
                "model.layers.0.mlp.switch_mlp.down_proj.weight",
                "experts.safetensors",
                0,
                "down",
            ),
            (
                "model.layers.0.mlp.switch_mlp.down_proj.scales",
                "a.safetensors",
                0,
                "down",
            ),
            (
                "model.layers.0.mlp.switch_mlp.down_proj.scales",
                "a.safetensors",
                0,
                "down",
            ),
        ]);
        let error = sidecar_file_map(&manifest).expect_err("duplicate rows must fail closed");
        assert!(
            matches!(&error, ExpertStreamError::InvalidManifest(msg) if msg.contains(prefix)),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn sidecar_file_map_rejects_sidecar_without_matching_weight_row() {
        // A sidecar row whose (layer, base) matches no weight row is a
        // name/layer drift: fail closed at map build instead of letting the
        // lookup miss silently at read time. This is also what tightens the
        // `.bias` classifier — a `switch.bias`-style row with no same-layer
        // `switch.weight` never enters the map.
        let manifest = manifest_with_rows(&[
            (
                "model.layers.0.mlp.switch_mlp.down_proj.weight",
                "experts.safetensors",
                0,
                "down",
            ),
            (
                "model.layers.0.mlp.switch_mlp.down_proj.scales",
                "a.safetensors",
                1, // declared for the wrong layer
                "down",
            ),
        ]);
        let error = sidecar_file_map(&manifest)
            .expect_err("a sidecar row with no same-layer weight row must fail closed");
        assert!(
            matches!(&error, ExpertStreamError::InvalidManifest(msg) if msg.contains("down_proj.scales")),
            "unexpected error: {error}"
        );

        let dense_bias = manifest_with_rows(&[
            (
                "model.layers.0.mlp.switch_mlp.down_proj.weight",
                "experts.safetensors",
                0,
                "down",
            ),
            (
                "model.layers.0.mlp.switch.bias",
                "experts.safetensors",
                0,
                "down",
            ),
        ]);
        assert!(
            sidecar_file_map(&dense_bias).is_err(),
            "a dangling switch.bias row must fail closed"
        );
    }

    #[test]
    fn row_pager_partial_sidecar_declaration_fails_closed() {
        // scales declared, biases co-located but undeclared, affine tensor:
        // the manifest is authoritative for the whole quantized triplet, so
        // the row load fails with a Paging error (layer-stack fallback).
        let dir = synth_split_sidecar_fixture("partial", true);
        let manifest = manifest_with_rows(&[
            (
                "model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
                "experts.safetensors",
                0,
                "gate_up",
            ),
            (
                "model.layers.0.mlp.switch_mlp.down_proj.weight",
                "experts.safetensors",
                0,
                "down",
            ),
            (
                "model.layers.0.mlp.switch_mlp.down_proj.scales",
                "sidecars.safetensors",
                0,
                "down",
            ),
        ]);
        let pager = ExpertRowPager::new(
            Arc::new(manifest),
            dir.clone(),
            ExpertRowPagerConfig {
                budget_bytes: 1 << 20,
                fuse_split_experts: false,
                prefetch: false,
                decay_interval: 4096,
                hotlist_out: None,
                load_delay: None,
            },
        )
        .unwrap();
        let Err(error) = pager.ensure_experts(0, &[1]) else {
            panic!("partial sidecar declaration must fail closed");
        };
        assert!(
            matches!(&error, ExpertStreamError::Paging(msg) if msg.contains("down_proj.biases")),
            "unexpected error: {error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_reads_manifest_declared_split_sidecars() {
        let dir = synth_split_sidecar_fixture("load", true);
        let pager = ExpertRowPager::new(
            Arc::new(synth_split_sidecar_manifest()),
            dir.clone(),
            ExpertRowPagerConfig {
                budget_bytes: 1 << 20,
                fuse_split_experts: false,
                prefetch: false,
                decay_interval: 4096,
                hotlist_out: None,
                load_delay: None,
            },
        )
        .unwrap();
        let compacted = pager.ensure_experts(0, &[1]).unwrap();
        let down = compacted.stack.down_exps.as_ref().unwrap();
        let scales = down.scales.as_ref().expect("split scales must attach");
        let biases = down.biases.as_ref().expect("split biases must attach");
        mlx_sys::eval(&[&down.weight, scales, biases]);
        assert_eq!(scales.shape(), vec![1, SYN_HIDDEN]);
        assert!(down.weight.data_f32().iter().all(|v| *v == 11.0));
        assert!(scales.data_f32().iter().all(|v| *v == 101.0));
        assert!(biases.data_f32().iter().all(|v| *v == 201.0));
        let gate_up = compacted.stack.gate_up_exps_packed.as_ref().unwrap();
        assert!(gate_up.scales.is_none() && gate_up.biases.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_declared_missing_sidecar_fails_closed() {
        // The manifest declares down_proj.scales in sidecars.safetensors but
        // the shard does not hold it: the row load must fail with a Paging
        // error so the MoE forward falls back to layer-stack paging.
        let dir = synth_split_sidecar_fixture("missing", false);
        let pager = ExpertRowPager::new(
            Arc::new(synth_split_sidecar_manifest()),
            dir.clone(),
            ExpertRowPagerConfig {
                budget_bytes: 1 << 20,
                fuse_split_experts: false,
                prefetch: false,
                decay_interval: 4096,
                hotlist_out: None,
                load_delay: None,
            },
        )
        .unwrap();
        let Err(error) = pager.ensure_experts(0, &[1]) else {
            panic!("declared-but-absent sidecar must fail closed");
        };
        assert!(
            matches!(&error, ExpertStreamError::Paging(msg) if msg.contains("down_proj.scales")),
            "unexpected error: {error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn synth_split_fixture(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ax_expert_stream_split_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_safetensors_f32(
            &dir,
            "experts.safetensors",
            &[
                (
                    "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                    vec![SYN_EXPERTS, SYN_INTER, SYN_HIDDEN],
                    synth_expert_values(0, SYN_INTER, SYN_HIDDEN),
                ),
                (
                    "model.layers.0.mlp.switch_mlp.up_proj.weight",
                    vec![SYN_EXPERTS, SYN_INTER, SYN_HIDDEN],
                    synth_expert_values(0, SYN_INTER, SYN_HIDDEN),
                ),
                (
                    "model.layers.0.mlp.switch_mlp.down_proj.weight",
                    vec![SYN_EXPERTS, SYN_HIDDEN, SYN_INTER],
                    synth_expert_values(0, SYN_HIDDEN, SYN_INTER),
                ),
            ],
        );
        dir
    }

    fn synth_split_manifest() -> ExpertStreamManifest {
        let prefix = "model.layers.0.mlp.switch_mlp";
        let json = serde_json::json!({
            "schema_version": "axquant.expert-stream.v1",
            "generated_by": "ax-engine-test",
            "required": true,
            "mode": "layer-stack",
            "num_experts": SYN_EXPERTS,
            "experts_per_tok": 2,
            "estimated_resident_bytes": 1000,
            "estimated_full_resident_bytes": 5000,
            "estimated_max_layer_expert_bytes": 2000,
            "resident_roles": ["embedding", "attention", "router", "norm", "lm_head"],
            "streamed_roles": ["expert"],
            "tensors": [
                {"name": format!("{prefix}.gate_proj.weight"), "file": "experts.safetensors", "layer": 0, "proj": "gate", "expert_axis": 0, "num_experts": SYN_EXPERTS, "bits": 2, "group_size": 64},
                {"name": format!("{prefix}.up_proj.weight"), "file": "experts.safetensors", "layer": 0, "proj": "up", "expert_axis": 0, "num_experts": SYN_EXPERTS, "bits": 2, "group_size": 64},
                {"name": format!("{prefix}.down_proj.weight"), "file": "experts.safetensors", "layer": 0, "proj": "down", "expert_axis": 0, "num_experts": SYN_EXPERTS, "bits": 2, "group_size": 64}
            ],
        });
        ExpertStreamManifest::parse(&serde_json::to_vec(&json).unwrap()).unwrap()
    }

    #[test]
    fn row_pager_split_projections_assemble_and_fuse() {
        let dir = synth_split_fixture("assemble");
        let manifest = || Arc::new(synth_split_manifest());

        let plain = ExpertRowPager::new(
            manifest(),
            dir.clone(),
            ExpertRowPagerConfig {
                budget_bytes: 1 << 20,
                fuse_split_experts: false,
                prefetch: false,
                decay_interval: 4096,
                hotlist_out: None,
                load_delay: None,
            },
        )
        .unwrap();
        let compacted = plain.ensure_experts(0, &[0, 2]).unwrap();
        let gate = compacted.stack.gate_exps.as_ref().unwrap();
        assert_eq!(gate.weight.shape(), vec![2, SYN_INTER, SYN_HIDDEN]);
        assert!(compacted.stack.up_exps.is_some());
        assert!(compacted.stack.down_exps.is_some());
        assert!(compacted.stack.gate_up_exps_packed.is_none());

        let fused = ExpertRowPager::new(
            manifest(),
            dir.clone(),
            ExpertRowPagerConfig {
                budget_bytes: 1 << 20,
                fuse_split_experts: true,
                prefetch: false,
                decay_interval: 4096,
                hotlist_out: None,
                load_delay: None,
            },
        )
        .unwrap();
        let compacted = fused.ensure_experts(0, &[0, 2]).unwrap();
        let packed = compacted
            .stack
            .gate_up_exps_packed
            .as_ref()
            .expect("split rows must fuse into a packed gate_up stack");
        assert_eq!(packed.weight.shape(), vec![2, 2 * SYN_INTER, SYN_HIDDEN]);
        assert!(compacted.stack.gate_exps.is_none() && compacted.stack.up_exps.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn granularity_env_parsing() {
        assert_eq!(
            stream_expert_granularity_from_env(None),
            StreamExpertGranularity::Layer
        );
        assert_eq!(
            stream_expert_granularity_from_env(Some("")),
            StreamExpertGranularity::Layer
        );
        assert_eq!(
            stream_expert_granularity_from_env(Some("layer")),
            StreamExpertGranularity::Layer
        );
        assert_eq!(
            stream_expert_granularity_from_env(Some("expert")),
            StreamExpertGranularity::Expert
        );
        assert_eq!(
            stream_expert_granularity_from_env(Some("EXPERT")),
            StreamExpertGranularity::Expert
        );
        // Unknown values fail closed to the layer-stack default.
        assert_eq!(
            stream_expert_granularity_from_env(Some("bogus")),
            StreamExpertGranularity::Layer
        );
    }

    #[test]
    fn granularity_family_default_flips_only_qwen4_exp() {
        // Env unset/empty: qwen4_exp packs with a file-backed manifest decode
        // through the row pager; every other family keeps whole-layer paging.
        assert_eq!(
            stream_expert_granularity_for_family(None, "qwen4_exp", true),
            StreamExpertGranularity::Expert
        );
        assert_eq!(
            stream_expert_granularity_for_family(Some(""), "qwen4_exp", true),
            StreamExpertGranularity::Expert
        );
        assert_eq!(
            stream_expert_granularity_for_family(Some("  "), "qwen4_exp", true),
            StreamExpertGranularity::Expert
        );
        assert_eq!(
            stream_expert_granularity_for_family(None, "deepseek_v4", true),
            StreamExpertGranularity::Layer
        );
        assert_eq!(
            stream_expert_granularity_for_family(None, "minimax_m3", true),
            StreamExpertGranularity::Layer
        );
        // An explicit env value always wins over the family default.
        assert_eq!(
            stream_expert_granularity_for_family(Some("layer"), "qwen4_exp", true),
            StreamExpertGranularity::Layer
        );
        assert_eq!(
            stream_expert_granularity_for_family(Some("expert"), "deepseek_v4", true),
            StreamExpertGranularity::Expert
        );
        // Invalid values fail closed to layer, not to the family default.
        assert_eq!(
            stream_expert_granularity_for_family(Some("bogus"), "qwen4_exp", true),
            StreamExpertGranularity::Layer
        );
    }

    #[test]
    fn granularity_family_default_requires_file_backed_manifest() {
        // Inferred manifests (no ax_expert_stream.json in the pack) carry
        // weight rows only: the qwen4_exp flip must not engage for them.
        assert_eq!(
            stream_expert_granularity_for_family(None, "qwen4_exp", false),
            StreamExpertGranularity::Layer
        );
        assert_eq!(
            stream_expert_granularity_for_family(Some(""), "qwen4_exp", false),
            StreamExpertGranularity::Layer
        );
        // An explicit env value still wins over the inferred-manifest gate.
        assert_eq!(
            stream_expert_granularity_for_family(Some("expert"), "qwen4_exp", false),
            StreamExpertGranularity::Expert
        );
        assert_eq!(
            stream_expert_granularity_for_family(Some("expert"), "deepseek_v4", false),
            StreamExpertGranularity::Expert
        );
        // Invalid env + inferred manifest still fails closed to layer.
        assert_eq!(
            stream_expert_granularity_for_family(Some("bogus"), "qwen4_exp", false),
            StreamExpertGranularity::Layer
        );
    }

    #[test]
    fn inferred_manifest_marks_provenance_and_stays_on_layer() {
        // The review's regression test: a qwen4_exp pack whose stream plan is
        // inferred (no ax_expert_stream.json) must not flip to per-expert
        // paging — its manifest has no sidecar rows to resolve split
        // triplets from.
        let specs = vec![
            infer_spec(
                "model.layers.0.ffn.switch_mlp.gate_proj.weight",
                NativeTensorRole::FfnGateUpExpsPacked,
                0,
                4,
                100,
            ),
            infer_spec(
                "model.layers.0.ffn.switch_mlp.down_proj.weight",
                NativeTensorRole::FfnDownExps,
                0,
                4,
                80,
            ),
        ];
        let inferred = infer_layer_stack_manifest(&specs, 8).expect("roles must infer");
        assert!(inferred.inferred);
        assert_eq!(
            stream_expert_granularity_for_family(None, "qwen4_exp", !inferred.inferred),
            StreamExpertGranularity::Layer
        );
        let file_backed =
            ExpertStreamManifest::parse(&serde_json::to_vec(&manifest_json(true)).unwrap())
                .unwrap();
        assert!(!file_backed.inferred);
        assert_eq!(
            stream_expert_granularity_for_family(None, "qwen4_exp", !file_backed.inferred),
            StreamExpertGranularity::Expert
        );
    }

    #[test]
    fn row_cache_bytes_default_and_floor() {
        let manifest = synth_manifest(); // max layer 2000 B, 4 experts, top-2
        assert_eq!(expert_row_cache_bytes_from_env(None, &manifest), 8000);
        // Floor: 2 × experts_per_tok × per-expert bytes = 2000.
        assert_eq!(
            expert_row_cache_bytes_from_env(Some("100"), &manifest),
            2000
        );
        assert_eq!(
            expert_row_cache_bytes_from_env(Some("5000"), &manifest),
            5000
        );
        assert_eq!(
            expert_row_cache_bytes_from_env(Some("junk"), &manifest),
            8000
        );
    }

    #[test]
    fn prefetch_and_decay_env_parsing() {
        assert!(expert_prefetch_enabled_from_env(None));
        assert!(!expert_prefetch_enabled_from_env(Some("0")));
        assert!(!expert_prefetch_enabled_from_env(Some("off")));
        assert!(expert_prefetch_enabled_from_env(Some("1")));
        assert_eq!(expert_hotness_decay_from_env(None), 4096);
        assert_eq!(expert_hotness_decay_from_env(Some("0")), 4096);
        assert_eq!(expert_hotness_decay_from_env(Some("64")), 64);
    }

    // ------------------------------------------------------------------
    // Split-submit (resident/missing overlap) tests.
    // ------------------------------------------------------------------

    /// Bit-compare the gate_up and down weight arrays of two stacks.
    fn assert_stacks_bit_equal(a: &LayerExpertStack, b: &LayerExpertStack, what: &str) {
        for (pa, pb, name) in [
            (&a.gate_up_exps_packed, &b.gate_up_exps_packed, "gate_up"),
            (&a.down_exps, &b.down_exps, "down"),
        ] {
            match (pa, pb) {
                (Some(x), Some(y)) => {
                    mlx_sys::eval(&[&x.weight, &y.weight]);
                    assert_eq!(x.weight.shape(), y.weight.shape(), "{what}/{name} shape");
                    assert_eq!(
                        x.weight.data_f32(),
                        y.weight.data_f32(),
                        "{what}/{name} values"
                    );
                }
                (None, None) => {}
                _ => panic!("{what}/{name} presence mismatch"),
            }
        }
    }

    /// Recombine split parts into request order at the stack level (the same
    /// whole-row concat + take construction the mlp split path uses).
    fn recombine_parts(
        resident: &CompactedExperts,
        missing: &CompactedExperts,
        order: &[u32],
    ) -> LayerExpertStack {
        fn recombine_proj(
            r: &Option<QuantizedWeight>,
            m: &Option<QuantizedWeight>,
            order: &[u32],
        ) -> Option<QuantizedWeight> {
            let (Some(r), Some(m)) = (r, m) else {
                return None;
            };
            let mut order_data = Vec::new();
            for slot in order {
                order_data.extend_from_slice(&slot.to_le_bytes());
            }
            let order_arr = mlx_sys::MlxArray::from_raw_data(
                order_data.as_ptr(),
                order_data.len(),
                &[order.len() as i32],
                mlx_sys::MlxDtype::Uint32,
            );
            let weight = mlx_sys::take(
                &mlx_sys::concatenate(&[&r.weight, &m.weight], 0, None),
                &order_arr,
                0,
                None,
            );
            Some(QuantizedWeight {
                weight,
                scales: None,
                biases: None,
                group_size: r.group_size,
                bits: r.bits,
                mode: r.mode.clone(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q2_weight: None,
                decode_q2_scales: None,
                decode_q2_biases: None,
            })
        }
        LayerExpertStack {
            gate_up_exps_packed: recombine_proj(
                &resident.stack.gate_up_exps_packed,
                &missing.stack.gate_up_exps_packed,
                order,
            ),
            gate_exps: None,
            up_exps: None,
            down_exps: recombine_proj(&resident.stack.down_exps, &missing.stack.down_exps, order),
        }
    }

    /// Pager-state equality between a sync ensure and a split begin/finish.
    fn assert_pager_states_equal(sync: &ExpertRowPager, split: &ExpertRowPager, ids: &[u32]) {
        assert_eq!(sync.cached_expert_keys(), split.cached_expert_keys());
        for id in ids {
            assert_eq!(
                sync.hotness_of(0, *id),
                split.hotness_of(0, *id),
                "hotness mismatch for expert {id}"
            );
        }
        assert_eq!(sync.resident_bytes(), split.resident_bytes());
    }

    #[test]
    fn row_pager_split_matches_sync_all_resident() {
        let dir = synth_fixture("split_resident");
        let ids = [3u32, 1, 2, 0];
        let sync = row_pager(&dir, 1 << 20, false);
        sync.warm_experts(0, &ids).unwrap();
        let reference = sync.ensure_experts(0, &ids).unwrap();

        let split = row_pager(&dir, 1 << 20, false);
        split.warm_experts(0, &ids).unwrap();
        let mut begin = split.ensure_experts_split_begin(0, &ids).unwrap();
        assert!(begin.fully_resident());
        assert!(!begin.fully_missing());
        let whole = begin.take_resident();
        drop(begin); // releases pins without a finish load
        assert_stacks_bit_equal(&reference.stack, &whole.stack, "all-resident");
        assert_eq!(reference.remap, whole.remap);
        assert_pager_states_equal(&sync, &split, &ids);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_split_matches_sync_all_missing() {
        let dir = synth_fixture("split_missing");
        let ids = [3u32, 1];
        let sync = row_pager(&dir, 1 << 20, false);
        let reference = sync.ensure_experts(0, &ids).unwrap();

        let split = row_pager(&dir, 1 << 20, false);
        let mut begin = split.ensure_experts_split_begin(0, &ids).unwrap();
        assert!(begin.fully_missing());
        assert!(!begin.fully_resident());
        let missing = split.ensure_experts_split_finish(&mut begin).unwrap();
        assert_stacks_bit_equal(&reference.stack, &missing.stack, "all-missing");
        assert_eq!(reference.remap, missing.remap);
        assert_pager_states_equal(&sync, &split, &ids);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_split_matches_sync_interleaved() {
        let dir = synth_fixture("split_interleaved");
        let ids = [3u32, 1, 2, 0];
        let sync = row_pager(&dir, 1 << 20, false);
        sync.warm_experts(0, &[1, 0]).unwrap();
        let reference = sync.ensure_experts(0, &ids).unwrap();

        let split = row_pager(&dir, 1 << 20, false);
        split.warm_experts(0, &[1, 0]).unwrap();
        let mut begin = split.ensure_experts_split_begin(0, &ids).unwrap();
        assert!(!begin.fully_resident() && !begin.fully_missing());
        assert_eq!(begin.missing_ids, vec![3, 2]);
        assert_eq!(begin.order, vec![2, 0, 3, 1]);
        let missing = split.ensure_experts_split_finish(&mut begin).unwrap();
        let combined = recombine_parts(&begin.resident, &missing, &begin.order);
        assert_stacks_bit_equal(&reference.stack, &combined, "interleaved");
        assert_pager_states_equal(&sync, &split, &ids);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_split_hotness_and_decay_match_sync() {
        let dir = synth_fixture("split_hot");
        let ids = [0u32, 1];
        let sync = row_pager_with_decay(&dir, 1 << 20, false, 4);
        for round in [&ids[..], &ids[..], &[0][..], &[0][..]] {
            sync.ensure_experts(0, round).unwrap();
        }
        let split = row_pager_with_decay(&dir, 1 << 20, false, 4);
        for round in [&ids[..], &ids[..], &[0][..], &[0][..]] {
            let mut begin = split.ensure_experts_split_begin(0, round).unwrap();
            let _ = split.ensure_experts_split_finish(&mut begin).unwrap();
        }
        assert_pager_states_equal(&sync, &split, &ids);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_split_pins_protect_resident_during_overlap() {
        let dir = synth_fixture("split_pins");
        let pager = row_pager(&dir, ROW_BYTES, false); // budget: one expert
        pager.warm_experts(0, &[0]).unwrap();
        let mut begin = pager.ensure_experts_split_begin(0, &[0, 1]).unwrap();
        // Budget pressure mid-overlap must not evict the pinned resident
        // selection (the warm itself becomes the eviction candidate).
        pager.warm_experts(0, &[2]).unwrap();
        assert!(pager.cached_expert_keys().contains(&(0, 0)));
        let _ = pager.ensure_experts_split_finish(&mut begin).unwrap();
        // Both selected experts landed and stayed through the soft cap.
        assert_eq!(pager.cached_expert_keys(), vec![(0, 0), (0, 1)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_pager_split_finish_error_releases_pins() {
        let dir = synth_fixture("split_err");
        let pager = row_pager(&dir, ROW_BYTES, false); // budget: one expert
        pager.warm_experts(0, &[0]).unwrap();
        let mut begin = pager.ensure_experts_split_begin(0, &[0, 1]).unwrap();
        std::fs::remove_file(dir.join("experts.safetensors")).unwrap();
        assert!(
            pager.ensure_experts_split_finish(&mut begin).is_err(),
            "a missing shard must fail the finish load"
        );
        drop(begin);
        // Restore the shard: the failed selection's pin must not strand —
        // the next ensure evicts (0,0) under the one-expert budget.
        let tensors = synth_tensors(0);
        let layer1 = synth_tensors(1);
        let mut all: Vec<(&str, Vec<i32>, Vec<f32>)> = Vec::new();
        all.extend(tensors);
        all.extend(layer1);
        write_safetensors_f32(&dir, "experts.safetensors", &all);
        pager.ensure_experts(0, &[2]).unwrap();
        assert_eq!(pager.cached_expert_keys(), vec![(0, 2)]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
