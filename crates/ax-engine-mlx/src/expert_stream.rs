//! SSD expert streaming (layer-stack paging), contract `axquant.expert-stream.v1`.
//!
//! Super-class MoE packs (e.g. Qwen 3.8 2.4T-A95B at 2-bit) keep their fused
//! expert stacks `[num_experts, out, in]` on SSD instead of in unified memory.
//! The initial load skips every tensor named in `ax_expert_stream.json`; the
//! [`ExpertStackPager`] then pages one layer's expert stack in on demand,
//! hands the MoE forward the same [`QuantizedWeight`] values the resident
//! path would have built, and evicts least-recently-used layer stacks once the
//! budget (`AX_STREAM_EXPERT_LAYERS`, default 1, plus the
//! `AX_STREAM_EXPERT_CACHE_BYTES` byte cap the priced plan derives) is
//! exceeded.
//!
//! v1 is layer-stack paging only: the existing `gather_qmm` kernel runs
//! unchanged on the paged packed tensors. No per-expert unfused kernels.

mod quantization;
mod selected;
pub(crate) use quantization::{ExpertQuantizationMode, ExpertQuantizationModes};
pub(crate) use selected::take_selected_expert_read_stats;

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ax_engine_core::{NativeTensorRole, NativeTensorSpec};
use serde::{Deserialize, Serialize};

use crate::weights::QuantizedWeight;

pub const EXPERT_STREAM_MANIFEST_FILE: &str = "ax_expert_stream.json";
pub(crate) const EXPERT_STREAM_SCHEMA_V1: &str = "axquant.expert-stream.v1";
pub(crate) const EXPERT_STREAM_MODE_LAYER_STACK: &str = "layer-stack";
/// Serve/load admission: `AX_STREAM_EXPERTS=off|auto|on` (also `0`/`1`).
pub(crate) const STREAM_EXPERTS_ENV: &str = "AX_STREAM_EXPERTS";
/// Number of layer expert stacks kept resident concurrently (minimum 1).
pub(crate) const STREAM_EXPERT_LAYERS_ENV: &str = "AX_STREAM_EXPERT_LAYERS";
/// Legacy extra unified-memory reserve (OS + KV + activations) when Auto
/// decides whether a pack can stay fully resident. 48 GiB matches a serve
/// process on a 192 GB Flash host without flipping the certified resident
/// path.
///
/// Retained for `AX_MLX_PRICED_MEMORY=0` and for callers with no priced
/// inputs (doctor, raw constructors). With the priced plan on, Auto charges
/// the real components instead of this flat headroom.
pub const AUTO_RESIDENT_HEADROOM_BYTES: u64 = 48 * 1024 * 1024 * 1024;

// The `AX_MLX_PRICED_MEMORY`, `AX_STREAM_EXPERT_CACHE_BYTES`, and
// `AX_MLX_MEMORY_CEILING_BYTES` environment reads live in `fastpath` (the
// cached-accessor home), so a new knob here cannot become an unnoticed
// per-call read on a dispatch path.

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
    #[error("invalid {env} value: {0}", env = STREAM_EXPERTS_ENV)]
    InvalidMode(String),
    #[error(
        "insufficient unified memory for this pack: the resident part needs about {fixed_bytes} \
         bytes (non-expert weights, KV cache, hybrid state, prefill scratch, and runtime \
         reserves) but only about {ceiling_bytes} bytes fit after the OS reserve. Reduce the \
         context/pool, or use a smaller pack; paging experts cannot free the resident part"
    )]
    InsufficientMemory {
        fixed_bytes: u64,
        ceiling_bytes: u64,
    },
    #[error("expert stream paging failed: {0}")]
    Paging(String),
    #[error(
        "the priced plan can only keep {expert_cache_bytes} bytes of streamed expert cache \
         resident, but one layer stack of this pack needs about {layer_bytes} bytes. Every \
         decode step would re-read a whole layer from disk, so the load is refused instead of \
         thrashing. Raise AX_MLX_MEMORY_CEILING_BYTES, shrink the context/pool, or select a \
         smaller pack; an explicit --stream-experts load skips this byte cap and uses the \
         layer-count budget (AX_STREAM_EXPERT_LAYERS) instead"
    )]
    ExpertCacheBelowLayerStack {
        expert_cache_bytes: u64,
        layer_bytes: u64,
    },
}

/// `n` from the first `layers.<n>.` segment of a tensor name, when present.
fn layer_ordinal_in_name(name: &str) -> Option<u32> {
    name.split('.')
        .collect::<Vec<_>>()
        .windows(2)
        .find(|pair| pair[0] == "layers")
        .and_then(|pair| pair[1].parse().ok())
}

impl ExpertStreamManifest {
    /// Parse and validate a manifest. Unknown `schema_version` or `mode` fail
    /// closed; v1 only supports `layer-stack` paging of packed expert stacks.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ExpertStreamError> {
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
        let mut seen_slots: std::collections::HashSet<(u32, ExpertProj)> =
            std::collections::HashSet::new();
        let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut packed_layers: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut split_layers: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for tensor in &mut manifest.tensors {
            // Shards are resolved as `root.join(file)`: an absolute path would
            // replace the root and `..` would escape it, so a manifest must
            // never point outside its own pack.
            if tensor.file.is_absolute()
                || tensor
                    .file
                    .components()
                    .any(|component| !matches!(component, std::path::Component::Normal(_)))
            {
                return Err(ExpertStreamError::InvalidManifest(format!(
                    "tensor {}: file {} must be a relative path inside the pack",
                    tensor.name,
                    tensor.file.display()
                )));
            }
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
            if i32::try_from(tensor.bits).is_err() || i32::try_from(tensor.group_size).is_err() {
                return Err(ExpertStreamError::InvalidManifest(format!(
                    "tensor {}: bits {} / group_size {} exceed the supported range",
                    tensor.name, tensor.bits, tensor.group_size
                )));
            }
            // A second entry for the same slot would silently replace the
            // first when the layer is paged in.
            if !is_quantization_sidecar_name(&tensor.name)
                && !seen_slots.insert((tensor.layer, proj))
            {
                return Err(ExpertStreamError::InvalidManifest(format!(
                    "tensor {}: duplicate {:?} projection for layer {}",
                    tensor.name, tensor.proj, tensor.layer
                )));
            }
            // The same tensor name under two slots would satisfy only the
            // first slot at page-in and fail every forward through the layer.
            if !seen_names.insert(tensor.name.clone()) {
                return Err(ExpertStreamError::InvalidManifest(format!(
                    "tensor {}: listed more than once",
                    tensor.name
                )));
            }
            // The `layer` field binds the pager to a decoder layer; a name
            // that carries a different `layers.<n>.` ordinal would page one
            // layer's experts into another silently (shapes match).
            if let Some(ordinal) = layer_ordinal_in_name(&tensor.name)
                && ordinal != tensor.layer
            {
                return Err(ExpertStreamError::InvalidManifest(format!(
                    "tensor {}: layer {} does not match the layer ordinal in its name",
                    tensor.name, tensor.layer
                )));
            }
            if !is_quantization_sidecar_name(&tensor.name) {
                match proj {
                    ExpertProj::GateUp => {
                        packed_layers.insert(tensor.layer);
                    }
                    ExpertProj::Gate | ExpertProj::Up => {
                        split_layers.insert(tensor.layer);
                    }
                    ExpertProj::Down => {}
                }
            }
        }
        if let Some(layer) = packed_layers.intersection(&split_layers).min() {
            return Err(ExpertStreamError::InvalidManifest(format!(
                "layer {layer}: declares both a packed gate_up stack and split gate/up stacks"
            )));
        }
        // Auto pages an optional pack only when this estimate plus headroom
        // exceeds unified memory; a manifest that lost the field (serde
        // default 0) would admit any optional pack as fully resident. Required
        // packs page regardless of the estimate.
        if !manifest.required && manifest.estimated_full_resident_bytes == 0 {
            return Err(ExpertStreamError::InvalidManifest(
                "estimated_full_resident_bytes must be positive for a pack that is not stream-required"
                    .to_string(),
            ));
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
    pub(crate) fn layer_indices(&self) -> Vec<u32> {
        let mut layers: Vec<u32> = self.tensors.iter().map(|t| t.layer).collect();
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    pub(crate) fn tensors_for_layer(
        &self,
        layer: u32,
    ) -> impl Iterator<Item = &ExpertStreamTensor> {
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
pub(crate) fn streamed_skip_names(manifest: &ExpertStreamManifest) -> HashSet<String> {
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

#[cfg(test)]
fn env_flag_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// Parse `AX_STREAM_EXPERTS`. `Ok(None)` means the variable is unset or empty
/// (no operator intent); `Err` means it is set to a value this build does not
/// understand. An invalid value is never silently equivalent to unset: a typo
/// such as `offf` must not quietly replace an operator's explicit `off` with
/// Auto paging. Admission paths propagate the error and fail closed, matching
/// the Python binding's `ValueError` contract for the same variable.
pub(crate) fn stream_experts_mode_from_env(
    value: Option<&str>,
) -> Result<Option<StreamExpertsMode>, ExpertStreamError> {
    let Some(raw) = value.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(None);
    };
    StreamExpertsMode::parse(raw)
        .map(Some)
        .map_err(ExpertStreamError::InvalidMode)
}

/// Resident layer budget from `AX_STREAM_EXPERT_LAYERS`: `max(1, value)`,
/// default 1 layer stack.
pub(crate) fn expert_layer_budget_from_env(value: Option<&str>) -> usize {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(1)
}

pub(crate) fn expert_layer_budget() -> usize {
    expert_layer_budget_from_env(std::env::var(STREAM_EXPERT_LAYERS_ENV).ok().as_deref())
}

/// Parse `AX_MLX_PRICED_MEMORY`. Unset means on: the priced plan is the
/// product default, and only an explicit false-y value restores the legacy
/// 48 GiB fixed headroom.
pub(crate) fn priced_memory_enabled_from_env(value: Option<&str>) -> bool {
    !value.is_some_and(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

pub(crate) fn priced_memory_enabled() -> bool {
    priced_memory_enabled_from_env(crate::fastpath::priced_memory_env().as_deref())
}

/// Parse `AX_STREAM_EXPERT_CACHE_BYTES`: an explicit positive byte budget for
/// resident expert layer stacks. Blank, zero, and unparsable values mean "no
/// byte cap", leaving `AX_STREAM_EXPERT_LAYERS` as the only limit.
pub(crate) fn expert_cache_bytes_from_env(value: Option<&str>) -> Option<u64> {
    value
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|bytes| *bytes > 0)
}

/// Explicit `AX_MLX_MEMORY_CEILING_BYTES` ceiling for this process, if one is
/// set. The plan still bounds it at `RAM - 4 GiB` when RAM is known.
pub(crate) fn memory_ceiling_override_bytes() -> Option<u64> {
    ax_engine_core::memory_budget::parse_memory_plan_ceiling_override(
        crate::fastpath::memory_ceiling_override_env().as_deref(),
    )
}

/// Per-session costs the priced plan charges besides the pack's weights.
///
/// The runner installs this before the weights load on the same thread that
/// builds the runner, mirroring [`set_stream_experts_mode`]. Every field is
/// bytes; `None` capacity means the corresponding probe failed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AutoResidencyContext {
    /// KV cache bytes for the configured context at the admitted pool.
    pub kv_bytes: u64,
    /// GDN/QSA/PLE per-request state for the served concurrency.
    pub state_bytes: u64,
    /// Bound on prefill activations for one chunk.
    pub prefill_scratch_bytes: u64,
    /// Whole n-gram lexical table (weights plus scales/biases). Priced as
    /// disk/virtual: reported, never charged to the resident working set.
    pub ngram_table_bytes: u64,
    /// Sum of the tensors the loader actually keeps resident: every tensor
    /// the artifacts directory ships except the streamable expert stack and
    /// the disk-served n-gram table. A publisher's
    /// `estimated_*_resident_bytes` can understate what the loader
    /// materializes (MTP/vision sidecars), so the plan floors its non-expert
    /// share with this inventory instead of trusting the file. It never raises
    /// the expert term: experts are priced from the manifest and are never
    /// double counted.
    /// Zero when the caller has no artifacts (raw constructors, doctor).
    pub resident_tensor_inventory_bytes: u64,
    /// Physical unified memory (`hw.memsize`).
    pub host_ram_bytes: Option<u64>,
    /// Metal `recommendedMaxWorkingSetSize`, when the platform reports it.
    ///
    /// Advisory: the plan warns when the priced working set goes above it and
    /// keeps the RAM-derived ceiling either way.
    pub metal_recommended_max_working_set_bytes: Option<u64>,
    /// Bytes the runner passed to `mlx_set_wired_limit` for this session.
    ///
    /// Advisory, exactly like the Metal recommendation: a wired limit below
    /// the priced working set is what makes macOS page part of the weights
    /// out of GPU residency, so the plan reports the gap instead of changing
    /// the placement decision. `None` when the caller never wired.
    pub wired_limit_bytes: Option<u64>,
    /// Explicit `AX_MLX_MEMORY_CEILING_BYTES` override, bounded by
    /// `RAM - 4 GiB` once the plan knows RAM.
    pub ceiling_override_bytes: Option<u64>,
    /// Live decode sequences [`Self::state_bytes`] is priced for: the
    /// session's configured bound (the server's admitted concurrency, else the
    /// runner's cohort capacity). `None` keeps the legacy behaviour of not
    /// pricing live sequences at all.
    pub live_sequences: Option<u32>,
    /// Bytes of [`Self::state_bytes`] that scale with one live decode
    /// sequence. Lets the plan re-price itself for a smaller admitted count
    /// without approximating.
    pub state_bytes_per_live_sequence: u64,
    /// Whether [`Self::live_sequences`] came from an explicit operator cap
    /// (`--max-concurrent-requests`) rather than the cohort default.
    pub live_sequences_operator_capped: bool,
}

thread_local! {
    /// Priced-plan inputs for the session loading on this thread. Plain data
    /// that every session build replaces (including with `None`), so a later
    /// load cannot inherit an earlier session's KV pool.
    static AUTO_RESIDENCY_CONTEXT: std::cell::RefCell<Option<AutoResidencyContext>> =
        const { std::cell::RefCell::new(None) };

    /// Expert cache byte budget the priced plan derived for this session.
    /// Zero means "the plan did not bound bytes".
    static PRICED_EXPERT_CACHE_BYTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Install (or clear) the priced-plan inputs for the session about to load on
/// this thread.
pub fn set_auto_residency_context(context: Option<AutoResidencyContext>) {
    AUTO_RESIDENCY_CONTEXT.with(|slot| *slot.borrow_mut() = context);
}

/// Priced-plan inputs installed for the current thread, if any.
pub fn auto_residency_context() -> Option<AutoResidencyContext> {
    AUTO_RESIDENCY_CONTEXT.with(|slot| *slot.borrow())
}

fn set_priced_expert_cache_bytes(bytes: u64) {
    PRICED_EXPERT_CACHE_BYTES.with(|cell| cell.set(bytes));
}

fn priced_expert_cache_bytes() -> Option<u64> {
    let bytes = PRICED_EXPERT_CACHE_BYTES.with(|cell| cell.get());
    (bytes > 0).then_some(bytes)
}

/// Byte budget for resident expert layer stacks: `AX_STREAM_EXPERT_CACHE_BYTES`
/// wins, then the priced plan's remainder, then no byte cap.
fn resolved_expert_cache_budget() -> Option<u64> {
    expert_cache_bytes_from_env(crate::fastpath::stream_expert_cache_bytes_env().as_deref())
        .or_else(priced_expert_cache_bytes)
}

/// Advisory working-set report: MLX's recommendation, raised to the operator's
/// `iogpu.wired_limit_mb` when one is set.
///
/// The authoritative value is MLX's own device query. It is safe to call on
/// the loading thread before the weights load: the runner already queries it
/// at startup for `set_wired_limit`, on this same thread, so this read cannot
/// pull new device work into the load.
///
/// An operator's `iogpu.wired_limit_mb` is a *floor* here, not a cap: it is
/// the wired memory the GPU is actually allowed to use, and the documented
/// remedy for a pack that prices above Metal's recommendation is to raise it
/// (`sudo sysctl iogpu.wired_limit_mb=<MiB>`). MLX's
/// `recommendedMaxWorkingSetSize` is dynamic and underreports under memory
/// pressure, so the larger of the two probes is the honest stable bound. This
/// matches `combined_metal_budget` in the server's load admission, which has
/// always preferred the sysctl over a pressured device value. The kernel
/// default of `0` means "no override", so it contributes nothing until an
/// operator sets it.
///
/// The result is advisory: the priced plan warns when the working set exceeds
/// it, but never clamps the ceiling to it (a measured 6-bit Flash Next session
/// ran resident at a higher peak RSS with no swap growth).
pub fn metal_recommended_working_set_bytes() -> Option<u64> {
    let recommended = mlx_sys::device_recommended_working_set_bytes();
    let operator_floor = crate::hardware::sysctl_string(&["-n", "iogpu.wired_limit_mb"])
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|mebibytes| *mebibytes > 0)
        .map(|mebibytes| mebibytes.saturating_mul(1024 * 1024));
    select_metal_working_set(recommended, operator_floor)
}

/// Working-set selection: the larger of MLX's recommendation and the wired
/// limit the GPU is actually allowed to use, whichever is known when the other
/// is not, and `None` when neither probe answers (in which case the plan
/// cannot prove residency and pages instead).
///
/// Taking the maximum — rather than the tighter of the two — is the rule both
/// admission paths use: `recommendedMaxWorkingSetSize` dips under memory
/// pressure, so a lower device value must never turn transient pressure into a
/// spurious "does not fit", and an operator who raised the sysctl widened what
/// the GPU may wire, which is exactly the headroom the plan should report.
fn select_metal_working_set(recommended: Option<u64>, operator_floor: Option<u64>) -> Option<u64> {
    match (recommended, operator_floor) {
        (Some(recommended), Some(floor)) => Some(recommended.max(floor)),
        (Some(recommended), None) => Some(recommended),
        (None, Some(floor)) => Some(floor),
        (None, None) => None,
    }
}

/// Price one candidate stream manifest with this session's non-weight costs.
///
/// The plan is returned with its live-sequence cap resolved: when the
/// session's configured live-sequence count does not fit under the ceiling but
/// a smaller count does, the plan re-prices the per-request state term and
/// publishes that count in
/// [`MemoryPlan::live_sequence_cap`](ax_engine_core::memory_budget::MemoryPlan::live_sequence_cap)
/// so a pack with enough weights resident stays resident at reduced
/// concurrency instead of paging every expert. `None` when the manifest does
/// not price its resident share: charging zero non-expert bytes would make any
/// pack look resident, so such a manifest keeps the legacy formula rather than
/// being priced from a missing number.
pub(crate) fn memory_plan_for(
    manifest: &ExpertStreamManifest,
    context: &AutoResidencyContext,
) -> Option<ax_engine_core::memory_budget::MemoryPlan> {
    if manifest.estimated_resident_bytes == 0 || manifest.estimated_full_resident_bytes == 0 {
        return None;
    }
    // Never trust a manifest estimate that is smaller than the tensors the
    // artifacts directory actually ships: MTP and vision sidecars are routinely
    // missing from the publisher's resident estimate, and charging them as
    // expert (streamable) bytes would admit a resident load that pages later.
    // The delta lands on the resident share, which is the conservative side.
    let (resident_non_expert_bytes, expert_bytes) = priced_weight_split(manifest, context);
    Some(
        ax_engine_core::memory_budget::MemoryPlan {
            resident_non_expert_bytes,
            expert_bytes,
            ngram_table_bytes: context.ngram_table_bytes,
            kv_bytes: context.kv_bytes,
            state_bytes: context.state_bytes,
            prefill_scratch_bytes: context.prefill_scratch_bytes,
            host_ram_bytes: context.host_ram_bytes,
            metal_recommended_max_working_set_bytes: context
                .metal_recommended_max_working_set_bytes,
            wired_limit_bytes: context.wired_limit_bytes,
            ceiling_override_bytes: context.ceiling_override_bytes,
            charged_live_sequences: context.live_sequences,
            state_bytes_per_live_sequence: context.state_bytes_per_live_sequence,
            live_sequences_operator_capped: context.live_sequences_operator_capped,
            ..ax_engine_core::memory_budget::MemoryPlan::default()
        }
        .with_fitting_live_sequences(),
    )
}

/// Split the plan's weight bytes into resident and streamable shares.
///
/// The resident share is floored by
/// [`AutoResidencyContext::resident_tensor_inventory_bytes`], which the caller
/// prices from the artifacts' own tensor list minus the streamable experts and
/// the disk-served n-gram table. That is what lets the floor raise only
/// genuinely resident bytes: the n-gram table is already discounted out of the
/// manifest estimate (see [`discount_never_loaded_bytes`]), and the expert
/// share is priced once from the manifest, so neither is double counted.
fn priced_weight_split(
    manifest: &ExpertStreamManifest,
    context: &AutoResidencyContext,
) -> (u64, u64) {
    let resident_estimate = manifest.estimated_resident_bytes;
    let expert_estimate = manifest
        .estimated_full_resident_bytes
        .saturating_sub(resident_estimate);
    let resident = resident_estimate.max(context.resident_tensor_inventory_bytes);
    // Do not let a stale/understated full estimate shrink the total below the
    // resident floor plus the experts the manifest can page.
    let full = manifest
        .estimated_full_resident_bytes
        .max(resident.saturating_add(expert_estimate));
    (resident, full.saturating_sub(resident))
}

/// Activation element width used by the prefill scratch bound (f16/bf16).
const ACTIVATION_BYTES: u64 = 2;
/// Copies of one chunk's hidden state a prefill step can hold at once: the
/// residual stream, the attention output, and the gate/up pair.
const PREFILL_SCRATCH_MULTIPLIER: u64 = 4;
/// Bytes one QSA score graph can occupy. The indexer tiles its f32 score GEMM
/// so every issued tile stays inside this ceiling
/// (`qwen4_exp_qsa::QSA_SCORE_WORKSPACE_LIMIT`); a monolithic score tensor that
/// would exceed it is split, not materialized.
const QSA_SCORE_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;
/// KV bytes charged when the manifest geometry cannot be priced (diffusion, or
/// a manifest that predates the KV estimator). Matches the flat floor the
/// unpriced admission path already used. MLA packs no longer land here: their
/// latent cache is priced by `estimated_kv_pool_bytes`.
pub(crate) const UNKNOWN_KV_BYTES: u64 = 768 * 1024 * 1024;

/// Number of live decode sequences to charge per-request state for.
///
/// `configured` is the session's admitted concurrency bound (the server's
/// `--max-concurrent-requests` when it has one); `batched_cap` is the runner's
/// hard batched-cohort capacity. The configured bound is capped by the cohort
/// capacity, and an absent bound falls back to the capacity. The result is
/// never zero because one request always exists.
pub(crate) fn resolved_live_sequences(configured: Option<u32>, batched_cap: usize) -> u32 {
    let capacity = u32::try_from(batched_cap).unwrap_or(u32::MAX).max(1);
    configured.map_or(capacity, |live| live.max(1).min(capacity))
}

/// Bytes one live decode sequence's gated-delta-net state costs: the causal
/// conv ring plus the recurrent state for every linear-attention layer.
///
/// This is the only term of [`estimate_auto_residency_costs`] that scales with
/// concurrency, so it is what lets the priced plan re-price itself for a
/// smaller admitted count ([`resolved_live_sequences`] gives the configured
/// count, [`ax_engine_core::memory_budget::MemoryPlan::with_fitting_live_sequences`]
/// finds the count that fits). Zero for packs without the linear-attention
/// contract, which therefore never need a live-sequence cap.
pub(crate) fn per_live_sequence_state_bytes(manifest: &ax_engine_core::NativeModelManifest) -> u64 {
    hybrid_state_bytes(manifest)
}

/// KV-cache, hybrid-state, and prefill-scratch bytes this session charges.
///
/// Conservative bounds, not allocator limits. The KV term is the pool-wide
/// worst case from the manifest geometry at the admitted pool. The state term
/// splits by ownership: gated-delta-net conv/recurrent state is per live
/// sequence (each live row owns one, and
/// [`per_live_sequence_state_bytes`] is that share), while per-token state (the
/// QSA index keys and their pooled blocks) lives inside the shared KV pool and
/// can never exceed it, so it is charged once at the pool token bound. Charging
/// the index per sequence at full context double-counts the pool by the
/// concurrency and priced a resident pack out of the default eight-row cohort.
/// The scratch term bounds one prefill chunk's activations.
///
/// `max_live_sequences` is the served concurrency (`MemoryPlan::state_bytes`
/// means exactly that). It is the runner's live-row bound, never zero: one
/// request always exists. The runner resolves it with
/// [`resolved_live_sequences`].
pub(crate) fn estimate_auto_residency_costs(
    manifest: &ax_engine_core::NativeModelManifest,
    budget: SessionResidencyBudget,
    max_live_sequences: u32,
) -> (u64, u64, u64) {
    let kv_bytes = manifest_kv_geometry(manifest)
        .and_then(|geometry| {
            ax_engine_core::memory_budget::estimated_kv_pool_bytes(&geometry, budget.kv_pool_tokens)
        })
        .unwrap_or(UNKNOWN_KV_BYTES);
    let sequence_state = per_live_sequence_state_bytes(manifest)
        .saturating_mul(u64::from(max_live_sequences.max(1)));
    let pool_state = qsa_index_state_bytes(manifest, budget.kv_pool_tokens);
    let state_bytes = sequence_state.saturating_add(pool_state);
    let prefill_scratch_bytes = estimate_prefill_scratch_bytes(manifest, budget.prefill_chunk);
    (kv_bytes, state_bytes, prefill_scratch_bytes)
}

/// Bound on the prefill activations one chunk keeps live at once.
///
/// The flat `chunk × hidden × 2 × 4` estimate under-priced every large sparse
/// pack: it charged the residual stream and the gate/up pair but not the
/// selection and MoE staging a sparse prefill actually allocates. A profiled
/// 8192-token Flash Next prefill peaked about 3.5 GiB above that plan. Each
/// term below is derived from the manifest geometry the runtime uses, so a
/// pack pays for the shapes it really runs and a dense one pays almost nothing:
///
/// - residual/attention/activation copies (`chunk × hidden × 2 × 4`);
/// - gated-delta-net chunk intermediates for hybrid packs: the projected
///   q/k/v/g/beta rows and the conv output at f32;
/// - MoE selected-expert staging: `chunk × top_k` routed rows through the
///   gathered input, the gate/up pair, and the down projection back at hidden
///   width;
/// - QSA selection: one score workspace plus the batched gather of the selected
///   K/V rows, capped by the attention kernel's own per-sub-batch bound;
/// - hyper-connection read/write buffers.
///
/// This is a bound on simultaneously live intermediates, not an allocator
/// limit, and it stays a per-chunk figure: the chunk is what the runner
/// submits, so it is what the plan can prove.
pub(crate) fn estimate_prefill_scratch_bytes(
    manifest: &ax_engine_core::NativeModelManifest,
    prefill_chunk: usize,
) -> u64 {
    let chunk = prefill_chunk as u64;
    let hidden = u64::from(manifest.hidden_size);
    chunk
        .saturating_mul(hidden)
        .saturating_mul(ACTIVATION_BYTES)
        .saturating_mul(PREFILL_SCRATCH_MULTIPLIER)
        .saturating_add(gated_delta_chunk_bytes(manifest, chunk))
        .saturating_add(moe_staging_bytes(manifest, chunk))
        .saturating_add(qsa_prefill_bytes(manifest, chunk))
        .saturating_add(hyper_connection_chunk_bytes(manifest, chunk))
}

/// Gated-delta-net chunk intermediates: one projected row set
/// (`key_heads × key_dim × 2 + value_heads × value_dim` wide) plus its conv
/// output, both f32 for the Metal kernels. Zero for packs without the
/// linear-attention contract.
fn gated_delta_chunk_bytes(manifest: &ax_engine_core::NativeModelManifest, chunk: u64) -> u64 {
    let config = &manifest.linear_attention;
    if !config.is_enabled() {
        return 0;
    }
    let (Some(key_heads), Some(value_heads), Some(key_dim), Some(value_dim)) = (
        config.num_key_heads,
        config.num_value_heads,
        config.key_head_dim,
        config.value_head_dim,
    ) else {
        return 0;
    };
    let conv_width = u64::from(key_heads)
        .saturating_mul(u64::from(key_dim))
        .saturating_mul(2)
        .saturating_add(u64::from(value_heads).saturating_mul(u64::from(value_dim)));
    // Two f32 copies: the q/k/v/g/beta projections and the conv output.
    chunk
        .saturating_mul(conv_width)
        .saturating_mul(4)
        .saturating_mul(2)
}

/// MoE selected-expert staging for one chunk: the gathered input rows at
/// hidden width, the gate/up pair at the expert width, and the down projection
/// written back at hidden width. Zero for dense packs.
fn moe_staging_bytes(manifest: &ax_engine_core::NativeModelManifest, chunk: u64) -> u64 {
    let moe = &manifest.moe;
    let (Some(top_k), Some(intermediate)) = (moe.experts_per_token, moe.expert_intermediate_size)
    else {
        return 0;
    };
    if moe.expert_count.unwrap_or(0) == 0 || top_k == 0 || intermediate == 0 {
        return 0;
    }
    let hidden = u64::from(manifest.hidden_size);
    let routed_rows = chunk.saturating_mul(u64::from(top_k));
    let per_row = u64::from(intermediate)
        .saturating_mul(2)
        .saturating_add(hidden.saturating_mul(2));
    routed_rows
        .saturating_mul(per_row)
        .saturating_mul(ACTIVATION_BYTES)
}

/// QSA prefill workspace: the indexer score graph plus the attention
/// transients of one chunk. Zero for packs without the indexer geometry.
///
/// The attention term depends on which arm runs:
/// `AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL=1` attends the selected rows in place
/// and prices only the packed index rows and the query/output tiles; the
/// gathered arm (default) prices the padded `chunk × selected` K/V tile,
/// capped at [`crate::model::shared::qwen4_exp_attention::BATCHED_ATTENTION_BYTES`].
fn qsa_prefill_bytes(manifest: &ax_engine_core::NativeModelManifest, chunk: u64) -> u64 {
    qsa_prefill_bytes_with(
        manifest,
        chunk,
        crate::fastpath::flash_next_qsa_indexed_prefill_enabled(),
    )
}

fn qsa_prefill_bytes_with(
    manifest: &ax_engine_core::NativeModelManifest,
    chunk: u64,
    indexed: bool,
) -> u64 {
    let config = &manifest.qwen4_exp;
    let (Some(head_dim), Some(ratio), Some(budget)) = (
        config.indexer_head_dim,
        config.indexer_compress_ratio,
        config.indexer_budget,
    ) else {
        return 0;
    };
    if head_dim == 0 || ratio == 0 || budget == 0 || manifest.kv_head_count == 0 {
        return 0;
    }
    // Selection width: `token_budget / ratio` complete blocks plus the visible
    // tail of `ratio - 1` tokens.
    let kept = u64::from(budget)
        .saturating_add(u64::from(ratio))
        .saturating_sub(1);
    let gathered_for = |queries: u64| {
        queries
            .saturating_mul(kept)
            .saturating_mul(u64::from(manifest.kv_head_count))
            .saturating_mul(u64::from(manifest.attention_head_dim))
            .saturating_mul(ACTIVATION_BYTES)
            .min(crate::model::shared::qwen4_exp_attention::BATCHED_ATTENTION_BYTES as u64)
    };
    let attention = if indexed {
        // Chunks shorter than the batched-attention floor, and unsupported
        // geometries, keep the gathered arm even with the switch on, so the
        // plan still reserves room for one short gathered sub-batch.
        let short_chunk = gathered_for(
            u64::try_from(crate::model::shared::qwen4_exp_attention::BATCHED_ATTENTION_MIN_QUERIES)
                .unwrap_or(0),
        );
        indexed_attention_bytes(manifest, chunk, kept).max(short_chunk)
    } else {
        gathered_for(chunk)
    };
    QSA_SCORE_WORKSPACE_BYTES.saturating_add(attention)
}

/// Indexed-attention transients for one chunk: the packed per-query index rows
/// (int32), the per-query selected counts, one token-major query copy and the
/// output tile. No `chunk × selected` K/V tile exists on this arm, and the
/// kernel holds one K/V row plus one float accumulator per query head, so this
/// stays orders of magnitude below the gathered bound.
fn indexed_attention_bytes(
    manifest: &ax_engine_core::NativeModelManifest,
    chunk: u64,
    kept: u64,
) -> u64 {
    let query_width = u64::from(manifest.attention_head_count)
        .saturating_mul(u64::from(manifest.attention_head_dim));
    let indices = chunk.saturating_mul(kept).saturating_mul(4);
    let lengths = chunk.saturating_mul(4);
    // Query copy plus output tile, both at activation width.
    let tiles = chunk
        .saturating_mul(query_width)
        .saturating_mul(ACTIVATION_BYTES)
        .saturating_mul(2);
    indices.saturating_add(lengths).saturating_add(tiles)
}

/// Hyper-connection read/write buffers for one chunk.
fn hyper_connection_chunk_bytes(manifest: &ax_engine_core::NativeModelManifest, chunk: u64) -> u64 {
    let config = &manifest.qwen4_exp;
    let (Some(count), Some(lowrank)) = (config.hc_count, config.hc_lowrank) else {
        return 0;
    };
    chunk
        .saturating_mul(u64::from(count))
        .saturating_mul(u64::from(lowrank))
        .saturating_mul(ACTIVATION_BYTES)
}

/// What the pre-load pricing pass tells the runner about a resident pack.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PricedSessionResidency {
    /// Live decode sequences the engine may admit, when the plan resolved one.
    ///
    /// `Some(n)` means the pack stays resident while at most `n` requests
    /// decode or prefill at once, strictly below the configured count; `None`
    /// means the configured count fits (or the pack cannot be priced pre-load,
    /// or it does not stay resident), so residency imposes no cap and the
    /// operator's own admission limit stands.
    pub live_sequence_cap: Option<u32>,
    /// Priced resident working set to size the wired limit against. `None`
    /// when the pack streams (there is no full working set to wire).
    pub resident_working_set_bytes: Option<u64>,
}

/// Price the pack about to load, returning both the wired-limit input and the
/// live-sequence cap this session must enforce, estimated before a single
/// tensor is read.
///
/// The runner needs this ahead of `set_wired_limit` to size the wired limit
/// against the working set instead of always taking 0.9 of Metal's
/// recommendation, and ahead of request admission so the concurrency the plan
/// was priced for is the concurrency the engine actually admits. It
/// deliberately reuses the same inputs the load-time plan uses (session
/// KV/state/scratch, the pack's expert-stream sidecar, the resident tensor
/// inventory) so the two numbers cannot diverge, and it reports no working set
/// rather than guessing when the pack cannot be priced, requires streaming, or
/// the plan decides to stream: a streamed pack has no full working set to
/// wire and its expert cache, not its concurrency, is what bounds it.
///
/// `AX_MLX_PRICED_MEMORY=0` returns an empty report without pricing anything:
/// the runner then wires the historical 0.9 share and publishes no cap, which
/// is the admission behaviour that predates the priced plan.
pub fn price_session_residency(
    artifacts: &ax_engine_core::NativeModelArtifacts,
    budget: SessionResidencyBudget,
    max_live_sequences: u32,
) -> PricedSessionResidency {
    apply_priced_memory_switch(
        priced_memory_enabled() && residency_may_be_priced(stream_experts_mode_checked()),
        || price_session_residency_priced(artifacts, budget, max_live_sequences),
    )
}

/// Forced streaming never loads the full working set, so pricing it as
/// resident would wire memory and cap concurrency for a plan that cannot run.
/// An invalid mode fails load admission; it gets no priced plan either.
fn residency_may_be_priced(mode: Result<StreamExpertsMode, ExpertStreamError>) -> bool {
    matches!(mode, Ok(StreamExpertsMode::Auto | StreamExpertsMode::Off))
}

/// Gate a priced residency behind the `AX_MLX_PRICED_MEMORY` kill switch.
///
/// With the switch off the closure is not run at all, so no sidecar is read and
/// the caller sees exactly what an unpriced build reported: no cap and no
/// working-set input.
fn apply_priced_memory_switch(
    priced_memory: bool,
    price: impl FnOnce() -> PricedSessionResidency,
) -> PricedSessionResidency {
    if priced_memory {
        price()
    } else {
        PricedSessionResidency::default()
    }
}

fn price_session_residency_priced(
    artifacts: &ax_engine_core::NativeModelArtifacts,
    budget: SessionResidencyBudget,
    max_live_sequences: u32,
) -> PricedSessionResidency {
    let (kv_bytes, state_bytes, prefill_scratch_bytes) =
        estimate_auto_residency_costs(artifacts.manifest(), budget, max_live_sequences);
    let discount_ngram = crate::fastpath::expert_stream_discount_ngram_enabled();
    let never_loaded_bytes = if discount_ngram {
        crate::weights::qwen4_exp::never_loaded_ngram_bytes(artifacts.tensor_specs())
    } else {
        0
    };
    let manifest = ExpertStreamManifest::read_from_dir(artifacts.root_dir())
        .ok()
        .flatten()
        .or_else(|| {
            infer_layer_stack_manifest(
                artifacts.tensor_specs(),
                artifacts
                    .manifest()
                    .moe
                    .experts_per_token
                    .unwrap_or(1)
                    .max(1),
            )
            .ok()
        });
    let Some(manifest) = manifest else {
        return PricedSessionResidency::default();
    };
    if manifest.required {
        return PricedSessionResidency::default();
    }
    let manifest = discount_never_loaded_bytes(manifest, never_loaded_bytes);
    // Everything below is derived from this pack's own artifacts, including the
    // resident tensor inventory. Inheriting it from the thread-local context
    // would let a previous load on a reused worker thread floor this plan with
    // another model's tensors, which is enough to move the residency decision
    // and therefore the live-sequence cap.
    let context = AutoResidencyContext {
        kv_bytes,
        state_bytes,
        prefill_scratch_bytes,
        resident_tensor_inventory_bytes: resident_tensor_inventory_bytes(
            artifacts.tensor_specs(),
            discount_ngram,
        ),
        live_sequences: Some(max_live_sequences.max(1)),
        state_bytes_per_live_sequence: per_live_sequence_state_bytes(artifacts.manifest()),
        live_sequences_operator_capped: budget.max_live_sequences.is_some(),
        ..session_auto_residency_context(auto_residency_context(), never_loaded_bytes)
    };
    let Some(plan) = memory_plan_for(&manifest, &context) else {
        return PricedSessionResidency::default();
    };
    if !plan.decide().is_resident() {
        return PricedSessionResidency::default();
    }
    PricedSessionResidency {
        live_sequence_cap: plan.live_sequence_cap,
        resident_working_set_bytes: Some(plan.resident_working_set_bytes()),
    }
}

/// Project the typed manifest into the tolerant KV-geometry view.
fn manifest_kv_geometry(
    manifest: &ax_engine_core::NativeModelManifest,
) -> Option<ax_engine_core::memory_budget::ManifestKvGeometry> {
    serde_json::to_value(manifest)
        .ok()
        .and_then(|value| serde_json::from_value(value).ok())
}

/// One request's gated-delta-net state: the causal-conv ring plus the
/// recurrent state for every linear-attention layer.
fn hybrid_state_bytes(manifest: &ax_engine_core::NativeModelManifest) -> u64 {
    let config = &manifest.linear_attention;
    let (Some(key_heads), Some(value_heads), Some(key_dim), Some(value_dim)) = (
        config.num_key_heads,
        config.num_value_heads,
        config.key_head_dim,
        config.value_head_dim,
    ) else {
        return 0;
    };
    let conv_width = u64::from(key_heads)
        .saturating_mul(u64::from(key_dim))
        .saturating_mul(2)
        .saturating_add(u64::from(value_heads).saturating_mul(u64::from(value_dim)));
    let conv_taps = u64::from(config.conv_kernel_dim.unwrap_or(1)).saturating_sub(1);
    let recurrent = u64::from(value_heads)
        .saturating_mul(u64::from(key_dim))
        .saturating_mul(u64::from(value_dim));
    let per_layer = conv_width
        .saturating_mul(conv_taps)
        .saturating_add(recurrent);
    // f32 conv ring and recurrent state.
    u64::from(manifest.layer_count)
        .saturating_mul(per_layer)
        .saturating_mul(4)
}

/// QSA per-request index state for the Flash Next (`qwen4_exp`) trunk: the raw
/// index-key history plus the mean-pooled block keys per layer.
///
/// The index describes prompt positions, so it is bounded by the shared KV
/// pool, not by one request's context: the pool admits at most
/// `kv_pool_tokens` positions across every live sequence. It is therefore
/// charged once at the pool size in
/// [`estimate_auto_residency_costs`], never multiplied by the concurrency.
/// Every layer is charged: the indexer is built per layer, and a test-only
/// subset would under-price the decode working set that the plan exists to
/// prove.
fn qsa_index_state_bytes(manifest: &ax_engine_core::NativeModelManifest, pool_tokens: u64) -> u64 {
    let config = &manifest.qwen4_exp;
    let (Some(head_dim), Some(compress_ratio)) =
        (config.indexer_head_dim, config.indexer_compress_ratio)
    else {
        return 0;
    };
    if head_dim == 0 || compress_ratio == 0 {
        return 0;
    }
    let head_dim = u64::from(head_dim);
    let compress_ratio = u64::from(compress_ratio);
    let raw_keys = pool_tokens
        .saturating_mul(head_dim)
        .saturating_mul(ACTIVATION_BYTES);
    // One pooled block per complete `compress_ratio` window.
    let pooled = pool_tokens
        .saturating_div(compress_ratio)
        .saturating_mul(head_dim)
        .saturating_mul(ACTIVATION_BYTES);
    u64::from(manifest.layer_count).saturating_mul(raw_keys.saturating_add(pooled))
}

/// Merge the session's installed context with this pack's n-gram table, the
/// host probes, and the explicit ceiling override, giving the priced plan
/// everything it needs.
pub(crate) fn session_auto_residency_context(
    installed: Option<AutoResidencyContext>,
    ngram_table_bytes: u64,
) -> AutoResidencyContext {
    let mut context = installed.unwrap_or_default();
    context.ngram_table_bytes = ngram_table_bytes;
    if context.host_ram_bytes.is_none() {
        context.host_ram_bytes = unified_memory_bytes();
    }
    if context.metal_recommended_max_working_set_bytes.is_none() {
        context.metal_recommended_max_working_set_bytes = metal_recommended_working_set_bytes();
    }
    if context.ceiling_override_bytes.is_none() {
        context.ceiling_override_bytes = memory_ceiling_override_bytes();
    }
    context
}

const STREAM_MODE_UNSET: u8 = 0;
const STREAM_MODE_OFF: u8 = 1;
const STREAM_MODE_AUTO: u8 = 2;
const STREAM_MODE_ON: u8 = 3;

thread_local! {
    /// Per-session stream-mode latch. A session sets it on the thread that
    /// then builds the runner and loads the weights (the same thread), so
    /// concurrent multi-model builds cannot overwrite each other's mode the
    /// way a process-global latch did.
    static STREAM_EXPERTS_OVERRIDE: std::cell::Cell<u8> = const { std::cell::Cell::new(STREAM_MODE_UNSET) };

    /// First expert paging failure recorded on this thread during a forward
    /// that could not propagate a `Result` (the generic MoE path returns a
    /// bare array). The runner drains it at the end of the step and fails
    /// the affected requests instead of aborting the process.
    static PAGING_FAILURE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Record a recoverable expert paging failure for the current step.
pub(crate) fn record_paging_failure(message: String) {
    PAGING_FAILURE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(message);
        }
    });
}

/// Take the paging failure recorded on this thread, if any.
pub(crate) fn take_paging_failure() -> Option<String> {
    PAGING_FAILURE.with(|slot| slot.borrow_mut().take())
}

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
    STREAM_EXPERTS_OVERRIDE.with(|latch| latch.set(mode_to_u8(mode)));
}

/// Effective mode: CLI/SDK override, else `AX_STREAM_EXPERTS`, else Auto. An
/// explicit override wins and the environment is not parsed at all, so an
/// operator's explicit `--stream-experts auto` is never rejected because of an
/// unrelated invalid environment value.
///
/// Load admission must use this checked form: an invalid `AX_STREAM_EXPERTS`
/// is an error, not a silent Auto.
pub fn stream_experts_mode_checked() -> Result<StreamExpertsMode, ExpertStreamError> {
    if let Some(mode) = mode_from_u8(STREAM_EXPERTS_OVERRIDE.with(|latch| latch.get())) {
        return Ok(mode);
    }
    Ok(
        stream_experts_mode_from_env(std::env::var(STREAM_EXPERTS_ENV).ok().as_deref())?
            .unwrap_or(StreamExpertsMode::Auto),
    )
}

/// Host unified-memory size (`hw.memsize` on macOS). `None` when unknown.
pub fn unified_memory_bytes() -> Option<u64> {
    crate::hardware::sysctl_string(&["-n", "hw.memsize"])?
        .parse()
        .ok()
}

/// Session bounds known before loading weights. Raw weight loaders have no
/// admitted KV pool and retain the general Auto policy.
#[derive(Clone, Copy, Debug)]
pub struct SessionResidencyBudget {
    pub kv_pool_tokens: u64,
    pub prefill_chunk: usize,
    /// Configured session bound on concurrently live decode sequences (the
    /// server's admitted concurrency). `None` falls back to the runner's
    /// batched cohort capacity, which is the hard live-row bound. Either way
    /// the effective value is never zero: one request always exists.
    pub max_live_sequences: Option<u32>,
}

/// Whether the bounded Tiel session exception permits residency. This probes
/// metadata and current hardware/allocator state, but never loads weights.
/// Explicit modes and required packs must still pass the normal resolver.
pub fn session_auto_resident_fits(
    artifacts: &ax_engine_core::NativeModelArtifacts,
    budget: SessionResidencyBudget,
    manifest: &ExpertStreamManifest,
) -> bool {
    crate::tiel_memory_policy::session_auto_resident_fits(artifacts, budget, manifest)
}

/// Remove bytes the loader never makes resident (Flash Next's disk-served
/// n-gram table: about 53.6 GiB of a 132 GB pack) from a manifest's resident
/// estimates, so Auto admission prices what is actually loaded. Without this
/// the pinned Flash Next pack (about 69.5 GiB truly resident) is paged on a
/// 128 GiB host even though it fits with the full headroom.
pub(crate) fn discount_never_loaded_bytes(
    mut manifest: ExpertStreamManifest,
    never_loaded_bytes: u64,
) -> ExpertStreamManifest {
    // An estimate that does not even cover the never-loaded bytes cannot have
    // counted them: leave it untouched rather than pricing the pack at zero.
    if never_loaded_bytes >= manifest.estimated_full_resident_bytes {
        return manifest;
    }
    manifest.estimated_full_resident_bytes -= never_loaded_bytes;
    manifest.estimated_resident_bytes = manifest
        .estimated_resident_bytes
        .saturating_sub(never_loaded_bytes);
    manifest
}

/// Sum of the tensors the loader keeps resident.
///
/// Excludes the streamable expert stack, which is priced once from the manifest
/// ([`priced_weight_split`]) and must not be counted twice. When
/// `discount_ngram` mirrors `AX_MLX_EXPERT_STREAM_DISCOUNT_NGRAM`, the
/// disk-served n-gram table is excluded too, exactly as
/// [`discount_never_loaded_bytes`] removes it from the manifest estimate. With
/// the switch off the table stays in the inventory, matching an undiscounted
/// manifest.
pub(crate) fn resident_tensor_inventory_bytes(
    tensors: &[NativeTensorSpec],
    discount_ngram: bool,
) -> u64 {
    tensors
        .iter()
        .filter(|spec| expert_role_to_proj(spec.role).is_none())
        .filter(|spec| !(discount_ngram && crate::weights::qwen4_exp::is_ngram_table_tensor(spec)))
        .map(|spec| spec.length_bytes)
        .fold(0u64, u64::saturating_add)
}

pub(crate) fn should_auto_stream(full_resident_bytes: u64, available_bytes: Option<u64>) -> bool {
    match available_bytes {
        // Unknown capacity (both sysctl probes failed, e.g. a sandboxed
        // service) cannot prove residency: page rather than risk an OOM.
        None => full_resident_bytes > 0,
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
///   fit in unified memory. With the priced plan on (default) Auto compares
///   the real components against the ceiling; `AX_MLX_PRICED_MEMORY=0`
///   restores the legacy [`AUTO_RESIDENT_HEADROOM_BYTES`] formula.
///
/// The priced inputs are the ones [`set_auto_residency_context`] installed for
/// this session (the runner installs them before loading). Callers that need
/// to price a pack-specific table use [`resolve_expert_stream_with_plan`].
pub fn resolve_expert_stream<F>(
    mode: StreamExpertsMode,
    file: Option<ExpertStreamManifest>,
    infer: F,
    available_bytes: Option<u64>,
) -> Result<Option<ExpertStreamManifest>, ExpertStreamError>
where
    F: FnOnce() -> Result<ExpertStreamManifest, ExpertStreamError>,
{
    let installed = auto_residency_context();
    resolve_expert_stream_inner(
        mode,
        file,
        infer,
        available_bytes,
        installed.as_ref(),
        priced_memory_enabled(),
    )
}

/// Priced Auto resolution: the same contract as [`resolve_expert_stream`],
/// with this session's [`AutoResidencyContext`] charged against the ceiling.
///
/// Explicit modes and required packs keep precedence, and the plan never
/// overrides an operator's `off`/`on`. When the plan is disabled at runtime
/// or cannot price the pack, the legacy formula decides.
pub fn resolve_expert_stream_with_plan<F>(
    mode: StreamExpertsMode,
    file: Option<ExpertStreamManifest>,
    infer: F,
    available_bytes: Option<u64>,
    plan_context: Option<&AutoResidencyContext>,
) -> Result<Option<ExpertStreamManifest>, ExpertStreamError>
where
    F: FnOnce() -> Result<ExpertStreamManifest, ExpertStreamError>,
{
    resolve_expert_stream_inner(
        mode,
        file,
        infer,
        available_bytes,
        plan_context,
        priced_memory_enabled(),
    )
}

fn resolve_expert_stream_inner<F>(
    mode: StreamExpertsMode,
    file: Option<ExpertStreamManifest>,
    infer: F,
    available_bytes: Option<u64>,
    plan_context: Option<&AutoResidencyContext>,
    priced_memory: bool,
) -> Result<Option<ExpertStreamManifest>, ExpertStreamError>
where
    F: FnOnce() -> Result<ExpertStreamManifest, ExpertStreamError>,
{
    // A new admission clears the previous session's remainder so a resident
    // decision can never inherit a stale expert cache budget.
    set_priced_expert_cache_bytes(0);
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
            let Some(manifest) = candidate else {
                return Ok(None);
            };
            // Required packs always stream, whatever the plan prices.
            if manifest.required {
                return Ok(Some(manifest));
            }
            if priced_memory
                && let Some(context) = plan_context
                && let Some(plan) = memory_plan_for(&manifest, context)
            {
                let decision = plan.decide();
                log_memory_plan(&plan);
                return match decision {
                    ax_engine_core::memory_budget::MemoryPlanDecision::Resident { .. } => Ok(None),
                    ax_engine_core::memory_budget::MemoryPlanDecision::Stream {
                        expert_cache_bytes,
                        ceiling_bytes,
                        ..
                    } => {
                        let expert_cache_bytes = enforce_layer_stack_floor(
                            &manifest,
                            expert_cache_bytes,
                            ceiling_bytes,
                        )?;
                        set_priced_expert_cache_bytes(expert_cache_bytes);
                        Ok(Some(manifest))
                    }
                    ax_engine_core::memory_budget::MemoryPlanDecision::Refuse {
                        fixed_bytes,
                        ceiling_bytes,
                    } => Err(ExpertStreamError::InsufficientMemory {
                        fixed_bytes,
                        ceiling_bytes,
                    }),
                };
            }
            if priced_memory && plan_context.is_none() {
                note_priced_memory_skipped();
            }
            Ok(
                should_auto_stream(manifest.estimated_full_resident_bytes, available_bytes)
                    .then_some(manifest),
            )
        }
    }
}

/// Emit one priced plan, publish its wired-residency numbers, and warn when a
/// resident plan cannot fit the limit the runner wired.
///
/// Two separate warnings can come out of one admission and they are not the
/// same finding: the Metal recommendation is a vendor hint that the plan
/// deliberately ignores for placement, while the wired limit is the cap the
/// process actually set. Only the second one makes macOS evict weights, so it
/// names the gap and the exact `sysctl` that closes it.
fn log_memory_plan(plan: &ax_engine_core::memory_budget::MemoryPlan) {
    let line = plan.log_line();
    if plan.exceeds_recommended_working_set() {
        // Keep the RAM-derived decision, but surface both numbers: the priced
        // working set is above the platform's recommendation, which is
        // advisory rather than a limit.
        tracing::warn!(
            target: "ax_engine_mlx::expert_stream",
            working_set_bytes = plan.resident_working_set_bytes(),
            recommended_working_set_bytes = plan.metal_recommended_max_working_set_bytes,
            "{line}"
        );
    } else {
        tracing::info!(target: "ax_engine_mlx::expert_stream", "{line}");
    }
    // `/v1/memory` reports the same numbers an operator sees here, including
    // the ones this admission decided not to act on.
    ax_engine_core::memory_budget::publish_wired_memory_report(
        ax_engine_core::memory_budget::WiredMemoryReport::from_plan(plan),
    );
    if let (Some(wired_limit_bytes), Some(wired_shortfall_bytes), Some(remedy_mebibytes)) = (
        plan.wired_limit_bytes,
        plan.wired_shortfall_bytes(),
        plan.suggested_wired_limit_mebibytes(),
    ) && plan.wired_shortfall_needs_attention()
    {
        tracing::warn!(
            target: "ax_engine_mlx::expert_stream",
            working_set_bytes = plan.resident_working_set_bytes(),
            wired_limit_bytes,
            wired_shortfall_bytes,
            recommended_wired_limit_mebibytes = remedy_mebibytes,
            "memory plan keeps the pack resident but the wired limit cannot hold it: macOS can \
             page part of the weights out of GPU residency and long-context prefill/decode \
             collapse. Raise the limit and restart the process: \
             `sudo sysctl iogpu.wired_limit_mb={remedy_mebibytes}` (needs root, resets on \
             reboot). Placement is unchanged."
        );
    }
    // The configured live-sequence count did not fit but a smaller one does:
    // state exactly which count is admitted, why, and what the operator can do
    // about it. This is the difference between a resident pack at L rows and
    // the 0.17 tok/s disk-paging mode the same pack falls into at N rows.
    if let (Some(configured), Some(admitted)) =
        (plan.charged_live_sequences, plan.live_sequence_cap)
        && plan.live_sequences_reduced()
    {
        let reason = if plan.live_sequences_operator_capped {
            format!(
                "the operator's cap of {configured} live sequences is honoured, and the priced \
                 working set needs {admitted} of them to stay under the ceiling"
            )
        } else {
            format!(
                "the default live-sequence bound of {configured} does not fit the priced working \
                 set under the ceiling; {admitted} does"
            )
        };
        tracing::warn!(
            target: "ax_engine_mlx::expert_stream",
            configured_live_sequences = configured,
            live_sequence_cap = admitted,
            charged_working_set_bytes =
                plan.working_set_bytes_for_live_sequences(configured),
            admitted_working_set_bytes = plan.resident_working_set_bytes(),
            ceiling_bytes = plan.ceiling_bytes(),
            operator_capped = plan.live_sequences_operator_capped,
            "priced memory plan admits fewer live sequences than configured: {reason}. The \
             engine queues further decoding/prefilling requests instead of paging the experts; \
             raise --max-concurrent-requests only after reducing context or the KV pool."
        );
    }
}

/// Refuse a streamed byte cap that cannot hold one layer stack.
///
/// `expert_cache_bytes` is the plan's remainder (`ceiling - fixed`). When it is
/// below the largest streamed layer, the pager must evict the stack that the
/// next decode step immediately needs, so every step re-reads a whole layer
/// from disk: a hard stall, not a graceful degradation. The remainder cannot be
/// raised without crossing the ceiling, so the load fails with an actionable
/// error instead. Both numbers are named in the warning and in the error.
fn enforce_layer_stack_floor(
    manifest: &ExpertStreamManifest,
    expert_cache_bytes: u64,
    ceiling_bytes: Option<u64>,
) -> Result<u64, ExpertStreamError> {
    // No known ceiling means `decide()` returned the "unknown capacity" page
    // path with a zero cap: the layer-count budget alone decides, so there is
    // nothing to floor.
    if ceiling_bytes.is_none() {
        return Ok(expert_cache_bytes);
    }
    let layer_bytes = manifest.estimated_max_layer_expert_bytes;
    if layer_bytes == 0 || expert_cache_bytes >= layer_bytes {
        return Ok(expert_cache_bytes);
    }
    tracing::warn!(
        target: "ax_engine_mlx::expert_stream",
        expert_cache_bytes,
        largest_layer_expert_bytes = layer_bytes,
        "priced expert cache is smaller than one layer stack; refusing a load that would re-read the layer on every decode step"
    );
    Err(ExpertStreamError::ExpertCacheBelowLayerStack {
        expert_cache_bytes,
        layer_bytes,
    })
}

/// Tell the operator once per process that Auto admission had no priced inputs
/// and fell back to the legacy headroom formula.
fn note_priced_memory_skipped() {
    static FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if note_priced_memory_skipped_once(&FIRED) {
        PRICED_MEMORY_SKIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(
            target: "ax_engine_mlx::expert_stream",
            "priced memory inputs are not installed for this session; Auto expert-stream admission used the legacy headroom formula"
        );
    }
}

/// Claim the once-per-process note. Returns `true` only for the caller that
/// won the latch, so the note's idempotence is testable on its own state
/// instead of on the process-wide counter another test may already have
/// advanced.
fn note_priced_memory_skipped_once(fired: &std::sync::atomic::AtomicBool) -> bool {
    fired
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_ok()
}

/// Process-wide count of the once-per-process skipped-priced-memory note.
/// The latch bounds it at one; exposed so the note's idempotence is testable
/// without capturing tracing output.
static PRICED_MEMORY_SKIPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// How many times the skipped-priced-memory note fired in this process.
#[cfg(test)]
pub(crate) fn priced_memory_skip_count() -> u32 {
    PRICED_MEMORY_SKIPS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
/// File-backed admission used by tests and doctor.
pub(crate) fn admit_expert_stream(
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
        let entry = layer_bytes.entry(layer).or_insert(0);
        *entry = entry.saturating_add(spec.length_bytes);
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

    pub(crate) fn is_empty(&self) -> bool {
        self.gate_up_exps_packed.is_none()
            && self.gate_exps.is_none()
            && self.up_exps.is_none()
            && self.down_exps.is_none()
    }

    /// Materialized payload bytes of this stack: the packed weights plus their
    /// quantization sidecars and dense bias. This is what the pager's byte
    /// budget charges for a resident layer stack.
    pub(crate) fn resident_bytes(&self) -> u64 {
        [
            self.gate_up_exps_packed.as_ref(),
            self.gate_exps.as_ref(),
            self.up_exps.as_ref(),
            self.down_exps.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(quantized_weight_bytes)
        .fold(0u64, u64::saturating_add)
    }
}

fn quantized_weight_bytes(weight: &QuantizedWeight) -> u64 {
    [
        Some(&weight.weight),
        weight.scales.as_ref(),
        weight.biases.as_ref(),
        weight.linear_bias.as_ref(),
    ]
    .into_iter()
    .flatten()
    .fold(0u64, |total, array| {
        total.saturating_add(array.nbytes() as u64)
    })
}

struct PagerCache {
    /// Resident stacks keyed by layer index.
    entries: HashMap<u32, LayerExpertStack>,
    /// LRU order: front = least recently used.
    order: VecDeque<u32>,
    /// Materialized payload bytes per resident layer stack.
    entry_bytes: HashMap<u32, u64>,
    /// Sum of `entry_bytes` over `entries`.
    resident_bytes: u64,
}

impl PagerCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            entry_bytes: HashMap::new(),
            resident_bytes: 0,
        }
    }

    /// Move `layer` to the most-recently-used end.
    fn touch(&mut self, layer: u32) {
        if let Some(pos) = self.order.iter().position(|l| *l == layer) {
            self.order.remove(pos);
        }
        self.order.push_back(layer);
    }

    fn insert(&mut self, layer: u32, stack: LayerExpertStack, bytes: u64) {
        self.touch(layer);
        if let Some(previous) = self.entry_bytes.insert(layer, bytes) {
            self.resident_bytes = self.resident_bytes.saturating_sub(previous);
        }
        self.resident_bytes = self.resident_bytes.saturating_add(bytes);
        self.entries.insert(layer, stack);
    }

    /// Evict least-recently-used stacks until both the layer count and the
    /// byte budget hold. `keep` (the stack this load just made resident) is
    /// never evicted: it is the one the current forward is about to read.
    ///
    /// Eviction is an accounting decision only. Handed-out stacks are cheap
    /// clones of MLX arrays, so an in-flight forward keeps the arrays alive
    /// even if the cache drops its slot.
    fn enforce_budget(&mut self, budget_layers: usize, budget_bytes: Option<u64>, keep: u32) {
        loop {
            let over_layers = self.order.len() > budget_layers;
            let over_bytes = budget_bytes.is_some_and(|budget| self.resident_bytes > budget);
            if !over_layers && !over_bytes {
                return;
            }
            if self.order.len() <= 1 {
                // Only the protected stack is left; a budget smaller than one
                // layer stack cannot be honoured without evicting it.
                return;
            }
            let Some(evict) = self.order.front().copied() else {
                return;
            };
            if evict == keep {
                return;
            }
            self.order.pop_front();
            self.entries.remove(&evict);
            if let Some(bytes) = self.entry_bytes.remove(&evict) {
                self.resident_bytes = self.resident_bytes.saturating_sub(bytes);
            }
        }
    }
}

/// Layer-stack pager: on MoE forward for layer L, `ensure_layer` loads L's
/// streamed tensors (only those tensors) from their shards, builds the same
/// `QuantizedWeight`s the resident path would have, caches them, and evicts
/// the LRU layer stack when the resident budget is exceeded.
pub struct ExpertStackPager {
    manifest: Arc<ExpertStreamManifest>,
    root: PathBuf,
    budget_layers: usize,
    /// Byte budget for resident layer stacks (`AX_STREAM_EXPERT_CACHE_BYTES`,
    /// else the priced plan's remainder). `None` leaves the layer count as the
    /// only cap.
    cache_budget_bytes: Option<u64>,
    /// When true, concatenate paged `gate`+`up` into one packed projection
    /// (DeepSeek V4 resident path). MiniMax-M3 mlxcel uses split SwitchLinear
    /// (`gate_proj` / `up_proj` / `down_proj`) and never fuses; Super-class
    /// MiniMax packs are stream-required, so this must stay false for them.
    fuse_split_experts: bool,
    cache: Mutex<PagerCache>,
    selected_readers: Mutex<HashMap<u32, Arc<selected::SelectedExpertRows>>>,
    quantization_modes: Option<ExpertQuantizationModes>,
    /// Layers that missed the selected-prefill payload cap and fell back to
    /// whole-layer paging. Distinct from `cached_layer_count`, which also
    /// includes resident stacks loaded for any other reason.
    selected_prefill_capacity_fallbacks: Mutex<HashSet<u32>>,
}

impl ExpertStackPager {
    pub(crate) fn new(
        manifest: Arc<ExpertStreamManifest>,
        root: PathBuf,
        budget_layers: usize,
    ) -> Self {
        Self::new_with_fuse(manifest, root, budget_layers, false)
    }

    pub(crate) fn new_with_fuse(
        manifest: Arc<ExpertStreamManifest>,
        root: PathBuf,
        budget_layers: usize,
        fuse_split_experts: bool,
    ) -> Self {
        let cache_budget_bytes = resolved_expert_cache_budget();
        if let Some(bytes) = cache_budget_bytes {
            tracing::info!(
                target: "ax_engine_mlx::expert_stream",
                expert_cache_bytes = bytes,
                "expert layer cache byte budget (AX_STREAM_EXPERT_CACHE_BYTES overrides)"
            );
        }
        Self {
            manifest,
            root,
            budget_layers: budget_layers.max(1),
            cache_budget_bytes,
            fuse_split_experts,
            selected_readers: Mutex::new(HashMap::new()),
            quantization_modes: None,
            cache: Mutex::new(PagerCache::new()),
            selected_prefill_capacity_fallbacks: Mutex::new(HashSet::new()),
        }
    }

    #[cfg(test)]
    /// Pager with an explicit byte budget, bypassing the env/plan resolution.
    pub(crate) fn new_with_cache_budget(
        manifest: Arc<ExpertStreamManifest>,
        root: PathBuf,
        budget_layers: usize,
        cache_budget_bytes: Option<u64>,
    ) -> Self {
        let mut pager = Self::new_with_fuse(manifest, root, budget_layers, false);
        pager.cache_budget_bytes = cache_budget_bytes;
        pager
    }

    /// Bind every projection to the native mode. The map and caches belong
    /// to one model instance and never persist across model loads.
    pub(crate) fn new_with_quantization_modes(
        manifest: Arc<ExpertStreamManifest>,
        root: PathBuf,
        budget_layers: usize,
        modes: ExpertQuantizationModes,
    ) -> Result<Self, ExpertStreamError> {
        let mut names = HashSet::new();
        for tensor in manifest
            .tensors
            .iter()
            .filter(|t| !is_quantization_sidecar_name(&t.name))
        {
            let mode = quantization::mode_for(Some(&modes), &tensor.name)
                .map_err(ExpertStreamError::InvalidManifest)?;
            if !names.insert(tensor.name.as_str()) {
                return Err(ExpertStreamError::InvalidManifest(
                    "duplicate expert projection".into(),
                ));
            }
            if mode == ExpertQuantizationMode::Mxfp4 {
                if (tensor.bits, tensor.group_size) != (4, 32) {
                    return Err(ExpertStreamError::InvalidManifest(
                        "MXFP4 requires bits=4/group32".into(),
                    ));
                }
                let base = tensor.name.strip_suffix(".weight").unwrap_or(&tensor.name);
                if manifest
                    .tensors
                    .iter()
                    .any(|t| t.name == format!("{base}.biases") || t.name == format!("{base}.bias"))
                {
                    return Err(ExpertStreamError::InvalidManifest(
                        "MXFP4 expert bias is unsupported".into(),
                    ));
                }
            }
        }
        if names.len() != modes.len() {
            return Err(ExpertStreamError::InvalidManifest(
                "quantization map has unknown projections".into(),
            ));
        }
        let mut pager = Self::new(manifest, root, budget_layers);
        pager.quantization_modes = Some(modes);
        Ok(pager)
    }

    #[cfg(test)]
    pub(crate) fn manifest(&self) -> &ExpertStreamManifest {
        &self.manifest
    }

    pub(crate) fn budget_layers(&self) -> usize {
        self.budget_layers
    }

    /// Byte budget for resident layer stacks, if one applies.
    #[cfg(test)]
    pub(crate) fn cache_budget_bytes(&self) -> Option<u64> {
        self.cache_budget_bytes
    }

    #[cfg(test)]
    /// Materialized bytes currently held by the page cache.
    pub(crate) fn cached_resident_bytes(&self) -> u64 {
        self.cache
            .lock()
            .expect("expert stream cache lock")
            .resident_bytes
    }

    #[cfg(test)]
    pub(crate) fn cached_layer_count(&self) -> usize {
        self.cache
            .lock()
            .expect("expert stream cache lock")
            .entries
            .len()
    }

    #[cfg(test)]
    /// Cached layer indices in LRU order (front = least recently used).
    pub(crate) fn cached_layer_indices(&self) -> Vec<u32> {
        self.cache
            .lock()
            .expect("expert stream cache lock")
            .order
            .iter()
            .copied()
            .collect()
    }

    fn selected_reader(
        &self,
        layer: u32,
    ) -> Result<Arc<selected::SelectedExpertRows>, ExpertStreamError> {
        let reader = {
            let mut readers = self
                .selected_readers
                .lock()
                .map_err(|_| ExpertStreamError::Paging("selected reader lock poisoned".into()))?;
            if let Some(reader) = readers.get(&layer) {
                Arc::clone(reader)
            } else {
                let reader = Arc::new(
                    selected::SelectedExpertRows::open(
                        &self.manifest,
                        &self.root,
                        layer,
                        mlx_sys::DEFAULT_MAX_GATHER_BYTES,
                        self.quantization_modes.as_ref(),
                    )
                    .map_err(ExpertStreamError::Paging)?,
                );
                readers.insert(layer, Arc::clone(&reader));
                reader
            }
        };
        Ok(reader)
    }

    fn selected_stack(
        &self,
        layer: u32,
        ids: &[u64],
    ) -> Result<LayerExpertStack, ExpertStreamError> {
        self.selected_reader(layer)?
            .gather(ids)
            .map_err(ExpertStreamError::Paging)
    }

    fn selected_stack_if_fits(
        &self,
        layer: u32,
        ids: &[u64],
    ) -> Result<Option<LayerExpertStack>, ExpertStreamError> {
        let stack = self
            .selected_reader(layer)?
            .gather_if_fits(ids)
            .map_err(ExpertStreamError::Paging)?;
        if stack.is_none() {
            self.selected_prefill_capacity_fallbacks
                .lock()
                .expect("expert stream fallback lock")
                .insert(layer);
        }
        Ok(stack)
    }

    #[cfg(test)]
    /// Successful selected-row payload reads; excludes headers and full-layer reads.
    pub(crate) fn selected_payload_bytes_read(&self) -> Result<u64, ExpertStreamError> {
        let readers = self
            .selected_readers
            .lock()
            .map_err(|_| ExpertStreamError::Paging("selected reader lock poisoned".into()))?;
        Ok(readers
            .values()
            .map(|reader| reader.payload_bytes_read())
            .sum())
    }

    #[cfg(test)]
    /// Distinct layers that missed the selected-prefill payload cap.
    pub(crate) fn selected_prefill_capacity_fallback_layers(&self) -> usize {
        self.selected_prefill_capacity_fallbacks
            .lock()
            .expect("expert stream fallback lock")
            .len()
    }

    /// Make layer `layer`'s expert stack resident and return cheap clones of
    /// its `QuantizedWeight`s. Loads from disk on a cache miss, then evicts
    /// LRU layers beyond the layer count and byte budgets.
    pub(crate) fn ensure_layer(&self, layer: u32) -> Result<LayerExpertStack, ExpertStreamError> {
        {
            let mut cache = self.cache.lock().expect("expert stream cache lock");
            if let Some(stack) = cache.entries.get(&layer).cloned() {
                cache.touch(layer);
                #[cfg(test)]
                crate::model::qwen4_exp::profiling::layer_cache_hit(layer);
                return Ok(stack);
            }
        }

        let stack = self.load_layer(layer)?;
        let bytes = stack.resident_bytes();

        let mut cache = self.cache.lock().expect("expert stream cache lock");
        // A racing thread may have filled the layer while we read from disk;
        // keep the existing entry rather than double-counting it.
        if let Some(existing) = cache.entries.get(&layer).cloned() {
            cache.touch(layer);
            return Ok(existing);
        }
        cache.insert(layer, stack.clone(), bytes);
        cache.enforce_budget(self.budget_layers, self.cache_budget_bytes, layer);
        Ok(stack)
    }

    /// Read only this layer's streamed tensors from their shards and assemble
    /// the resident-path `QuantizedWeight` values.
    fn load_layer(&self, layer: u32) -> Result<LayerExpertStack, ExpertStreamError> {
        #[cfg(test)]
        let started = std::time::Instant::now();
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
                .or_else(|| tensor.name.strip_suffix(".scales"))
                .or_else(|| tensor.name.strip_suffix(".biases"))
                .or_else(|| tensor.name.strip_suffix(".bias"))
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
            for (name, array) in tensors {
                if loaded.insert(name.clone(), array).is_some() {
                    return Err(ExpertStreamError::Paging(format!(
                        "tensor {name} appears in multiple files for layer {layer}"
                    )));
                }
            }
        }
        for tensor in tensors
            .iter()
            .filter(|t| !is_quantization_sidecar_name(&t.name))
        {
            if quantization::mode_for(self.quantization_modes.as_ref(), &tensor.name)
                .map_err(ExpertStreamError::Paging)?
                == ExpertQuantizationMode::Mxfp4
            {
                let base = tensor.name.strip_suffix(".weight").unwrap_or(&tensor.name);
                let weight = loaded.get(&tensor.name).ok_or_else(|| {
                    ExpertStreamError::Paging(format!("missing MXFP4 weight {}", tensor.name))
                })?;
                let scales = loaded.get(&format!("{base}.scales")).ok_or_else(|| {
                    ExpertStreamError::Paging(format!("missing MXFP4 scales for {}", tensor.name))
                })?;
                if loaded.contains_key(&format!("{base}.biases"))
                    || loaded.contains_key(&format!("{base}.bias"))
                {
                    return Err(ExpertStreamError::Paging(
                        "MXFP4 expert bias is unsupported".into(),
                    ));
                }
                quantization::validate_mxfp4_pair(
                    &weight.shape(),
                    weight.dtype(),
                    &scales.shape(),
                    scales.dtype(),
                    tensor.num_experts,
                )
                .map_err(ExpertStreamError::Paging)?;
            }
        }
        #[cfg(test)]
        let materialized_payload_bytes = loaded.values().map(mlx_sys::MlxArray::nbytes).sum();
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
            let mode = quantization::mode_for(self.quantization_modes.as_ref(), &tensor.name)
                .map_err(ExpertStreamError::Paging)?;
            // Fail closed on the generic pager path (no bound quantization
            // modes, so every tensor above binds Affine): MXFP4 scales are
            // u8 (e4m3) while affine scales are float, so u8 scales here mean
            // an MXFP4 pack reached a pager that would silently dequantize
            // it as affine.
            if mode == ExpertQuantizationMode::Affine
                && let Some(scales) = scales.as_ref()
                && scales.dtype() == mlx_sys::MlxDtype::Uint8
            {
                return Err(ExpertStreamError::Paging(format!(
                    "tensor {}: u8 scales on an affine-bound pager look like MXFP4; \
                     bind quantization modes explicitly for this pack",
                    tensor.name
                )));
            }
            let quantized = QuantizedWeight {
                weight,
                scales,
                biases,
                group_size: i32::try_from(tensor.group_size).map_err(|_| {
                    ExpertStreamError::Paging(format!(
                        "tensor {}: group_size {} out of range",
                        tensor.name, tensor.group_size
                    ))
                })?,
                bits: i32::try_from(tensor.bits).map_err(|_| {
                    ExpertStreamError::Paging(format!(
                        "tensor {}: bits {} out of range",
                        tensor.name, tensor.bits
                    ))
                })?,
                mode: mode.as_str().to_string(),
                linear_bias,
                decode_weight_t: None,
                decode_q2_weight: None,
                decode_q2_scales: None,
                decode_q2_biases: None,
                weight_t_view: Default::default(),
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
        #[cfg(test)]
        crate::model::qwen4_exp::profiling::layer_loaded(
            layer,
            materialized_payload_bytes,
            started.elapsed().as_secs_f64(),
        );
        Ok(stack)
    }
}

/// Per-layer handle stashed on `LayerWeights` when the layer's expert stack
/// is streamed instead of resident.
pub struct ExpertLayerSource {
    pager: Arc<ExpertStackPager>,
    layer: u32,
}

impl ExpertLayerSource {
    pub(crate) fn new(pager: Arc<ExpertStackPager>, layer: u32) -> Self {
        Self { pager, layer }
    }

    #[cfg(test)]
    pub(crate) fn layer(&self) -> u32 {
        self.layer
    }

    /// Gather a compact stack in exactly the requested expert order.
    pub(crate) fn selected_stack(
        &self,
        ids: &[u64],
    ) -> Result<LayerExpertStack, ExpertStreamError> {
        self.pager.selected_stack(self.layer, ids)
    }

    /// Return None only when the validated union exceeds the selected payload cap.
    pub(crate) fn selected_stack_if_fits(
        &self,
        ids: &[u64],
    ) -> Result<Option<LayerExpertStack>, ExpertStreamError> {
        self.pager.selected_stack_if_fits(self.layer, ids)
    }

    /// Resolve this layer's expert stack, paging it in when needed.
    pub(crate) fn stack(&self) -> Result<LayerExpertStack, ExpertStreamError> {
        self.pager.ensure_layer(self.layer)
    }
}

#[cfg(test)]
mod tests {

    fn stream_manifest_json(required: bool, estimate: u64, tensors: serde_json::Value) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema_version": "axquant.expert-stream.v1",
            "mode": "layer-stack",
            "required": required,
            "num_experts": 8,
            "experts_per_tok": 2,
            "estimated_full_resident_bytes": estimate,
            "tensors": tensors,
        }))
        .expect("manifest json")
    }

    fn stream_tensor_json(name: &str, layer: u32, proj: &str) -> serde_json::Value {
        serde_json::json!({
            "name": name, "file": "shard.safetensors", "layer": layer, "proj": proj,
            "expert_axis": 0, "num_experts": 8, "bits": 4, "group_size": 64
        })
    }

    #[test]
    fn parse_rejects_crossed_layer_ordinals_duplicate_names_and_mixed_layouts() {
        // A `layer` that disagrees with the `layers.<n>.` ordinal in the name
        // would page one layer's experts into another.
        let crossed = stream_manifest_json(
            true,
            1,
            serde_json::json!([
                stream_tensor_json(
                    "model.layers.31.mlp.switch_mlp.gate_up_proj.weight",
                    30,
                    "gate_up"
                ),
                stream_tensor_json(
                    "model.layers.31.mlp.switch_mlp.down_proj.weight",
                    30,
                    "down"
                ),
            ]),
        );
        let error = ExpertStreamManifest::parse(&crossed).expect_err("crossed layer ordinal");
        assert!(
            error
                .to_string()
                .contains("does not match the layer ordinal"),
            "{error}"
        );

        // The same name under two slots satisfies only the first at page-in.
        let duplicate = stream_manifest_json(
            true,
            1,
            serde_json::json!([
                stream_tensor_json(
                    "model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
                    0,
                    "gate"
                ),
                stream_tensor_json("model.layers.0.mlp.switch_mlp.gate_up_proj.weight", 0, "up"),
            ]),
        );
        let error = ExpertStreamManifest::parse(&duplicate).expect_err("duplicate name");
        assert!(
            error.to_string().contains("listed more than once"),
            "{error}"
        );

        // Packed and split stacks for one layer: the forward would silently
        // prefer the packed copy.
        let mixed = stream_manifest_json(
            true,
            1,
            serde_json::json!([
                stream_tensor_json(
                    "model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
                    0,
                    "gate_up"
                ),
                stream_tensor_json("model.layers.0.mlp.switch_mlp.gate_proj.weight", 0, "gate"),
                stream_tensor_json("model.layers.0.mlp.switch_mlp.up_proj.weight", 0, "up"),
            ]),
        );
        let error = ExpertStreamManifest::parse(&mixed).expect_err("mixed layout");
        assert!(
            error.to_string().contains("both a packed gate_up stack"),
            "{error}"
        );

        // A well-formed manifest still parses, with or without an ordinal in
        // the name.
        let ok = stream_manifest_json(
            true,
            1,
            serde_json::json!([
                stream_tensor_json(
                    "model.layers.3.mlp.switch_mlp.gate_up_proj.weight",
                    3,
                    "gate_up"
                ),
                stream_tensor_json("experts.down.3", 3, "down"),
            ]),
        );
        assert!(ExpertStreamManifest::parse(&ok).is_ok());
    }

    #[test]
    fn parse_requires_a_resident_estimate_for_optional_packs() {
        let tensors = serde_json::json!([
            stream_tensor_json(
                "model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
                0,
                "gate_up"
            ),
            stream_tensor_json("model.layers.0.mlp.switch_mlp.down_proj.weight", 0, "down"),
        ]);
        // Optional pack without the estimate: Auto would treat it as free.
        let error = ExpertStreamManifest::parse(&stream_manifest_json(false, 0, tensors.clone()))
            .expect_err("optional pack needs an estimate");
        assert!(
            error.to_string().contains("estimated_full_resident_bytes"),
            "{error}"
        );
        // Required packs page regardless, so the estimate may be absent.
        assert!(
            ExpertStreamManifest::parse(&stream_manifest_json(true, 0, tensors.clone())).is_ok()
        );
        assert!(ExpertStreamManifest::parse(&stream_manifest_json(false, 1, tensors)).is_ok());
    }
    use super::*;

    #[test]
    fn stream_experts_env_unset_and_blank_carry_no_operator_intent() {
        for raw in [None, Some(""), Some("   "), Some("\t\n")] {
            assert!(
                matches!(stream_experts_mode_from_env(raw), Ok(None)),
                "{raw:?} must read as unset, not as a mode"
            );
        }
    }

    #[test]
    fn stream_experts_env_accepts_documented_aliases_with_trim_and_case() {
        for raw in ["off", "0", "false", "no", " OFF ", "Off"] {
            assert!(
                matches!(
                    stream_experts_mode_from_env(Some(raw)),
                    Ok(Some(StreamExpertsMode::Off))
                ),
                "{raw:?}"
            );
        }
        for raw in ["auto", " AUTO "] {
            assert!(
                matches!(
                    stream_experts_mode_from_env(Some(raw)),
                    Ok(Some(StreamExpertsMode::Auto))
                ),
                "{raw:?}"
            );
        }
        for raw in ["on", "1", "true", "yes", " On "] {
            assert!(
                matches!(
                    stream_experts_mode_from_env(Some(raw)),
                    Ok(Some(StreamExpertsMode::On))
                ),
                "{raw:?}"
            );
        }
    }

    /// Regression: an invalid value used to parse to `None`, which the mode
    /// resolver mapped to Auto. A typo for `off` therefore silently became a
    /// paging policy instead of failing closed, contradicting the Python
    /// binding's `ValueError` contract for the same variable.
    #[test]
    fn stream_experts_env_rejects_invalid_values_instead_of_degrading_to_auto() {
        for raw in [
            "offf",
            "onn",
            "2",
            "automatic",
            "disabled",
            "maybe",
            "-1",
            "o n",
        ] {
            match stream_experts_mode_from_env(Some(raw)) {
                Err(ExpertStreamError::InvalidMode(reported)) => {
                    assert_eq!(reported, StreamExpertsMode::parse(raw).unwrap_err());
                    // The rendered message must name the variable and the bad
                    // value so an operator can find and correct it.
                    let message = ExpertStreamError::InvalidMode(reported).to_string();
                    assert!(
                        message.contains(STREAM_EXPERTS_ENV) && message.contains(raw),
                        "error {message:?} must name {STREAM_EXPERTS_ENV} and {raw:?}"
                    );
                }
                other => panic!("{raw:?} must be InvalidMode, got {other:?}"),
            }
        }
    }

    #[test]
    fn stream_experts_env_enabled_is_false_for_invalid_values() {
        // `on`-only query: an invalid value is never "force streaming".
        assert!(!matches!(
            stream_experts_mode_from_env(Some("onn")),
            Ok(Some(StreamExpertsMode::On))
        ));
        assert!(matches!(
            stream_experts_mode_from_env(Some("on")),
            Ok(Some(StreamExpertsMode::On))
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn unified_memory_probe_without_sbin() {
        use std::os::unix::fs::PermissionsExt;

        const CHILD: &str = "AX_TEST_MEMORY_PROBE_CHILD";
        if std::env::var_os(CHILD).is_some() {
            assert!(
                unified_memory_bytes().is_some_and(|bytes| bytes > 0),
                "macOS memory detection must work without /usr/sbin in PATH"
            );
            assert!(crate::hardware::sysctl_string(&["-n", "machdep.cpu.brand_string"]).is_some());
            assert!(
                crate::hardware::sysctl_string(&["-n", "ax_engine.nonexistent_test_key"]).is_none()
            );
            return;
        }
        // Isolate PATH in a child: mutating the parent environment would race
        // other tests and requires unsafe in Rust 2024.
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ax-memory-probe-{}-{unique}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let shim = dir.join("sysctl");
        for script in [
            None,
            Some("#!/bin/sh\nexit 1\n"),
            Some("#!/bin/sh\nexit 0\n"),
        ] {
            if let Some(script) = script {
                std::fs::write(&shim, script).unwrap();
                std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "expert_stream::tests::unified_memory_probe_without_sbin",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("PATH", &dir)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "memory probe subprocess failed ({script:?}): {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

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
    fn shard_paths_outside_the_pack_fail_closed() {
        for file in [
            "/tmp/evil.safetensors",
            "../evil.safetensors",
            "sub/../../evil.safetensors",
        ] {
            let mut value = manifest_json(true);
            value["tensors"][0]["file"] = serde_json::json!(file);
            let error =
                ExpertStreamManifest::parse(&serde_json::to_vec(&value).unwrap()).expect_err(file);
            assert!(
                matches!(&error, ExpertStreamError::InvalidManifest(msg) if msg.contains("inside the pack")),
                "{file}: {error}"
            );
        }
        let mut value = manifest_json(true);
        value["tensors"][0]["file"] = serde_json::json!("shards/model-00001.safetensors");
        assert!(ExpertStreamManifest::parse(&serde_json::to_vec(&value).unwrap()).is_ok());
    }

    #[test]
    fn quantization_params_outside_i32_fail_closed() {
        let mut value = manifest_json(true);
        value["tensors"][0]["bits"] = serde_json::json!(2_147_483_649_u64);
        assert!(ExpertStreamManifest::parse(&serde_json::to_vec(&value).unwrap()).is_err());
        let mut value = manifest_json(true);
        value["tensors"][0]["group_size"] = serde_json::json!(4_294_967_295_u64);
        assert!(ExpertStreamManifest::parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn duplicate_layer_projection_fails_closed() {
        let mut value = manifest_json(true);
        let mut duplicate = value["tensors"][0].clone();
        duplicate["name"] = serde_json::json!("model.layers.0.mlp.switch_mlp.other.weight");
        duplicate["file"] = serde_json::json!("model-00002-of-00080.safetensors");
        value["tensors"].as_array_mut().unwrap().push(duplicate);
        let error = ExpertStreamManifest::parse(&serde_json::to_vec(&value).unwrap())
            .expect_err("duplicate slot must fail closed");
        assert!(
            matches!(&error, ExpertStreamError::InvalidManifest(msg) if msg.contains("duplicate")),
            "{error}"
        );

        // The same projection on a different layer is a distinct slot.
        let mut value = manifest_json(true);
        let mut other_layer = value["tensors"][0].clone();
        other_layer["name"] = serde_json::json!("model.layers.1.mlp.switch_mlp.gate_proj.weight");
        other_layer["layer"] = serde_json::json!(1);
        value["tensors"].as_array_mut().unwrap().push(other_layer);
        assert!(ExpertStreamManifest::parse(&serde_json::to_vec(&value).unwrap()).is_ok());
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
    fn admission_on_without_manifest_fails_closed() {
        // requested=true is the On admission path through admit_expert_stream;
        // with no manifest it must fail closed instead of streaming nothing
        // or guessing roles.
        let dir = std::env::temp_dir().join("ax_expert_stream_admission_missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(matches!(
            admit_expert_stream(&dir, true),
            Err(ExpertStreamError::ManifestMissing)
        ));
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

    /// Pinned Flash Next MXFP4 manifest (axquant): 132,229,016,568 bytes
    /// "full resident", of which the never-loaded n-gram table is 53.64 GiB.
    fn flash_next_manifest() -> ExpertStreamManifest {
        let mut file =
            ExpertStreamManifest::parse(&serde_json::to_vec(&manifest_json(false)).unwrap())
                .unwrap();
        file.estimated_full_resident_bytes = 132_229_016_568;
        file.estimated_resident_bytes = 68_056_165_368;
        file
    }
    const FLASH_NEXT_NGRAM_BYTES: u64 = 57_595_000_000;
    const M5_MAX_128_GIB: u64 = 128 * 1024 * 1024 * 1024;

    #[test]
    fn flash_next_pages_on_128gb_when_ngram_is_counted() {
        let decided = resolve_expert_stream(
            StreamExpertsMode::Auto,
            Some(flash_next_manifest()),
            || Err(ExpertStreamError::ManifestMissing),
            Some(M5_MAX_128_GIB),
        )
        .unwrap();
        assert!(decided.is_some(), "legacy accounting pages the pack");
    }

    #[test]
    fn flash_next_stays_resident_on_128gb_once_ngram_is_discounted() {
        let file = discount_never_loaded_bytes(flash_next_manifest(), FLASH_NEXT_NGRAM_BYTES);
        assert_eq!(file.estimated_full_resident_bytes, 74_634_016_568);
        assert_eq!(file.estimated_resident_bytes, 10_461_165_368);
        let decided = resolve_expert_stream(
            StreamExpertsMode::Auto,
            Some(file.clone()),
            || Err(ExpertStreamError::ManifestMissing),
            Some(M5_MAX_128_GIB),
        )
        .unwrap();
        assert!(decided.is_none(), "about 69.5 GiB + 48 GiB fits in 128 GiB");
        // A 96 GiB host still pages, and explicit On still pages.
        assert!(
            resolve_expert_stream(
                StreamExpertsMode::Auto,
                Some(file.clone()),
                || Err(ExpertStreamError::ManifestMissing),
                Some(96 * 1024 * 1024 * 1024),
            )
            .unwrap()
            .is_some()
        );
        assert!(
            resolve_expert_stream(
                StreamExpertsMode::On,
                Some(file),
                || Err(ExpertStreamError::ManifestMissing),
                Some(M5_MAX_128_GIB),
            )
            .unwrap()
            .is_some()
        );
    }

    #[test]
    fn discount_never_drops_a_required_pack_or_reaches_zero() {
        let mut required = flash_next_manifest();
        required.required = true;
        // An estimate smaller than the never-loaded bytes is left untouched.
        let discounted = discount_never_loaded_bytes(required, u64::MAX);
        assert_eq!(discounted.estimated_full_resident_bytes, 132_229_016_568);
        assert_eq!(discounted.estimated_resident_bytes, 68_056_165_368);
        assert!(
            resolve_expert_stream(
                StreamExpertsMode::Auto,
                Some(discounted),
                || Err(ExpertStreamError::ManifestMissing),
                Some(M5_MAX_128_GIB),
            )
            .unwrap()
            .is_some_and(|manifest| manifest.required)
        );
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
        // Unknown capacity pages: it cannot prove the pack fits.
        assert!(should_auto_stream(flash, None));
        assert!(!should_auto_stream(0, None));
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
    fn pager_rejects_duplicate_sidecars_before_caching() {
        let dir = synth_fixture("duplicate_sidecar");
        let name = "model.layers.0.mlp.switch_mlp.gate_up_proj.scales";
        let mut tensors = synth_tensors(0).to_vec();
        tensors.push((name, vec![SYN_EXPERTS, 2 * SYN_INTER, 1], vec![1.0; 16]));
        write_safetensors_f32(&dir, "experts.safetensors", &tensors);
        write_safetensors_f32(
            &dir,
            "sidecars.safetensors",
            &[(name, vec![SYN_EXPERTS, 2 * SYN_INTER, 1], vec![2.0; 16])],
        );
        let mut manifest = synth_manifest();
        let mut sidecar = manifest.tensors[0].clone();
        sidecar.name = name.into();
        sidecar.file = "sidecars.safetensors".into();
        manifest.tensors.push(sidecar);
        let pager = ExpertStackPager::new(Arc::new(manifest), dir.clone(), 1);
        let error = pager.ensure_layer(0).err().expect("duplicate must fail");
        assert!(error.to_string().contains("appears in multiple files"));
        assert_eq!(pager.cached_layer_count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
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
        assert_eq!(pager.selected_prefill_capacity_fallback_layers(), 0);

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
        assert_eq!(pager.selected_prefill_capacity_fallback_layers(), 0);

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

    // Priced memory plan -------------------------------------------------

    const GIB: u64 = 1024 * 1024 * 1024;

    /// Optional Auto candidate that prices both its resident share and its
    /// experts, so the priced plan can judge it.
    fn priced_manifest(expert_bytes: u64, resident_bytes: u64) -> ExpertStreamManifest {
        let json = serde_json::json!({
            "schema_version": "axquant.expert-stream.v1",
            "mode": "layer-stack",
            "required": false,
            "num_experts": 8,
            "experts_per_tok": 2,
            "estimated_resident_bytes": resident_bytes,
            "estimated_full_resident_bytes": resident_bytes + expert_bytes,
            "estimated_max_layer_expert_bytes": expert_bytes,
            "resident_roles": ["embedding", "attention", "norm", "lm_head"],
            "streamed_roles": ["expert"],
            "tensors": [stream_tensor_json(
                "model.layers.0.mlp.switch_mlp.down_proj.weight",
                0,
                "down"
            )],
        });
        ExpertStreamManifest::parse(&serde_json::to_vec(&json).expect("manifest json"))
            .expect("priced manifest")
    }

    #[test]
    fn expert_cache_byte_budget_ignores_blank_zero_and_unparsable_values() {
        // Only a positive integer is an operator byte cap; anything else must
        // leave the layer-count budget as the sole limit.
        assert_eq!(expert_cache_bytes_from_env(None), None);
        for raw in ["", "   ", "0", "-1", "abc", "1.5", "8GiB"] {
            assert_eq!(
                expert_cache_bytes_from_env(Some(raw)),
                None,
                "{raw:?} must not become a byte cap"
            );
        }
        assert_eq!(
            expert_cache_bytes_from_env(Some("8589934592")),
            Some(8 * 1024 * 1024 * 1024)
        );
        // Surrounding whitespace is trimmed before the parse.
        assert_eq!(
            expert_cache_bytes_from_env(Some(" 1073741824\n")),
            Some(1024 * 1024 * 1024)
        );
    }

    #[test]
    fn metal_working_set_prefers_the_wired_limit_the_gpu_is_actually_allowed() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // What MLX's device query reports on a 128 GiB M5 Max (measured
        // 2026-10-24). With the kernel default `iogpu.wired_limit_mb = 0`
        // (no operator override) this is the advisory value the plan reports.
        const M5_MAX_RECOMMENDED: u64 = 115_448_725_504;
        assert_eq!(
            select_metal_working_set(Some(M5_MAX_RECOMMENDED), None),
            Some(M5_MAX_RECOMMENDED)
        );
        // A lowered wired limit is what the GPU is allowed to wire, so it wins
        // over a recommendation that underreported under memory pressure. This
        // is the same rule `combined_metal_budget` uses for load admission:
        // taking the minimum here would turn transient device pressure into a
        // spurious "the pack does not fit".
        assert_eq!(
            select_metal_working_set(Some(64 * GIB), Some(96 * GIB)),
            Some(96 * GIB)
        );
        // An operator who raised the sysctl widened residency (the documented
        // remedy for a pack above the recommendation), so the raised value is
        // the headroom the plan reports rather than the stale device hint.
        assert_eq!(
            select_metal_working_set(Some(M5_MAX_RECOMMENDED), Some(120_000 * 1024 * 1024)),
            Some(120_000 * 1024 * 1024)
        );
        // Without a device answer (non-Metal), an explicit wired limit stands.
        assert_eq!(
            select_metal_working_set(None, Some(64 * GIB)),
            Some(64 * GIB)
        );
        // Neither probe answered: capacity is unknown, so the plan pages.
        assert_eq!(select_metal_working_set(None, None), None);
    }

    #[test]
    fn priced_plan_ceiling_ignores_the_metal_recommendation() {
        // 10 GiB resident non-expert weights plus 80 GiB of experts on a
        // 128 GiB host: 93 GiB of working set against a 115.20 GiB ceiling.
        // The recommendation sits far below both, and it still only warns.
        let manifest = priced_manifest(80 * GIB, 10 * GIB);
        let context = AutoResidencyContext {
            host_ram_bytes: Some(128 * GIB),
            metal_recommended_max_working_set_bytes: Some(64 * GIB),
            ..AutoResidencyContext::default()
        };
        let plan = memory_plan_for(&manifest, &context).expect("priced plan");
        assert_eq!(plan.ceiling_bytes(), Some(128 * GIB - 128 * GIB / 10));
        assert!(plan.exceeds_recommended_working_set());
        let line = plan.log_line();
        for component in [
            "metal-recommended=64.00GiB",
            "metal-recommended-exceeded=true",
            "ceiling-source=ram",
            "ceiling=115.20GiB",
            "warning=working-set-exceeds-metal-recommended",
            "decision=resident",
        ] {
            assert!(line.contains(component), "missing {component:?} in {line}");
        }
    }

    #[test]
    fn explicit_ceiling_override_flows_through_the_priced_plan() {
        let manifest = priced_manifest(80 * GIB, 10 * GIB);
        let tightened = AutoResidencyContext {
            host_ram_bytes: Some(128 * GIB),
            ceiling_override_bytes: Some(64 * GIB),
            ..AutoResidencyContext::default()
        };
        let plan = memory_plan_for(&manifest, &tightened).expect("priced plan");
        assert_eq!(plan.ceiling_bytes(), Some(64 * GIB));
        assert!(plan.log_line().contains("ceiling-source=override"));
        // 93 GiB of working set no longer fits the operator's 64 GiB ceiling,
        // so the experts page with the remainder instead of the load being
        // refused: the fixed part still fits.
        let decision = plan.decide();
        assert_eq!(
            decision.expert_cache_bytes(),
            Some(64 * GIB - plan.fixed_bytes())
        );
        assert!(plan.log_line().contains("decision=stream"));

        // The override is bounded by `RAM - 4 GiB` once RAM is known, so an
        // over-large request cannot starve the kernel.
        let raised = AutoResidencyContext {
            ceiling_override_bytes: Some(200 * GIB),
            ..tightened
        };
        let plan = memory_plan_for(&manifest, &raised).expect("priced plan");
        assert_eq!(plan.ceiling_bytes(), Some(128 * GIB - 4 * GIB));
    }

    #[test]
    fn priced_plan_kill_switch_restores_legacy_formula() {
        // The switch is on unless explicitly disabled.
        assert!(priced_memory_enabled_from_env(None));
        assert!(priced_memory_enabled_from_env(Some("1")));
        assert!(priced_memory_enabled_from_env(Some(" auto ")));
        assert!(priced_memory_enabled_from_env(Some("")));
        for disabled in ["0", "false", "no", "off", " OFF "] {
            assert!(
                !priced_memory_enabled_from_env(Some(disabled)),
                "{disabled:?} must disable the priced plan"
            );
        }

        // 10 GiB resident + 40 GiB of experts on a 64 GiB host. The priced
        // plan fits it (53 GiB against the 56 GiB ceiling); the legacy
        // formula pages it (50 GiB + 48 GiB headroom > 64 GiB).
        let manifest = priced_manifest(40 * GIB, 10 * GIB);
        let context = AutoResidencyContext {
            host_ram_bytes: Some(64 * GIB),
            ..AutoResidencyContext::default()
        };
        let available = Some(64 * GIB);
        let priced = resolve_expert_stream_inner(
            StreamExpertsMode::Auto,
            Some(manifest.clone()),
            || Err(ExpertStreamError::ManifestMissing),
            available,
            Some(&context),
            true,
        )
        .expect("priced admission");
        assert!(
            priced.is_none(),
            "priced plan must keep a pack whose whole working set fits resident"
        );

        let legacy = resolve_expert_stream_inner(
            StreamExpertsMode::Auto,
            Some(manifest.clone()),
            || Err(ExpertStreamError::ManifestMissing),
            available,
            Some(&context),
            false,
        )
        .expect("legacy admission");
        assert!(
            legacy.is_some(),
            "AX_MLX_PRICED_MEMORY=0 must restore the 48 GiB headroom formula"
        );

        // An explicit mode is never overridden by the plan.
        assert!(
            resolve_expert_stream_inner(
                StreamExpertsMode::Off,
                Some(manifest.clone()),
                || Err(ExpertStreamError::ManifestMissing),
                available,
                Some(&context),
                true,
            )
            .expect("off")
            .is_none(),
            "explicit Off must not be overridden"
        );
        assert!(
            resolve_expert_stream_inner(
                StreamExpertsMode::On,
                Some(manifest.clone()),
                || Err(ExpertStreamError::ManifestMissing),
                available,
                Some(&context),
                true,
            )
            .expect("on")
            .is_some(),
            "explicit On must not be overridden"
        );

        // A required pack streams even when the plan says it fits.
        let mut required = manifest.clone();
        required.required = true;
        assert!(
            resolve_expert_stream_inner(
                StreamExpertsMode::Auto,
                Some(required),
                || Err(ExpertStreamError::ManifestMissing),
                available,
                Some(&context),
                true,
            )
            .expect("required")
            .is_some(),
            "required packs always stream"
        );
    }

    /// With the priced plan off the runner must see exactly the pre-wave
    /// residency: no live-sequence cap and no working-set input, so the wired
    /// limit falls back to the historical 0.9 share and admission stays
    /// unbounded.
    #[test]
    fn priced_memory_kill_switch_clears_the_cap_and_the_working_set() {
        let priced = PricedSessionResidency {
            live_sequence_cap: Some(1),
            resident_working_set_bytes: Some(113 * GIB),
        };
        let mut priced_ran = false;
        let switched_off = apply_priced_memory_switch(false, || {
            priced_ran = true;
            priced
        });
        assert!(!priced_ran, "the kill switch must skip the pricing pass");
        assert_eq!(switched_off, PricedSessionResidency::default());
        assert_eq!(switched_off.live_sequence_cap, None);
        assert_eq!(switched_off.resident_working_set_bytes, None);

        // No working set keeps the historical wire; no cap keeps the
        // historical admit-everything admission.
        let recommended = 115_448_725_504usize;
        let wired = crate::tiel_memory_policy::resolve_wired_limit(
            recommended,
            switched_off.resident_working_set_bytes,
            None,
        );
        assert_eq!(
            wired.scale,
            crate::tiel_memory_policy::DEFAULT_WIRED_LIMIT_SCALE
        );
        assert_eq!(wired.limit_bytes, (recommended as f64 * 0.9) as u64);
        assert_eq!(
            wired.source,
            crate::tiel_memory_policy::WiredLimitSource::Default
        );

        // With the switch on the priced numbers pass through unchanged.
        assert_eq!(apply_priced_memory_switch(true, || priced), priced);
    }

    #[test]
    fn forced_streaming_is_never_priced_as_resident() {
        assert!(residency_may_be_priced(Ok(StreamExpertsMode::Auto)));
        assert!(residency_may_be_priced(Ok(StreamExpertsMode::Off)));
        assert!(!residency_may_be_priced(Ok(StreamExpertsMode::On)));
        assert!(!residency_may_be_priced(
            stream_experts_mode_from_env(Some("sometimes"))
                .map(|mode| mode.unwrap_or(StreamExpertsMode::Auto))
        ));
    }

    #[test]
    fn pager_cache_respects_byte_budget() {
        let dir = synth_fixture("cache_bytes");
        let manifest = Arc::new(synth_manifest());
        // One synthetic layer stack is 4*4*4 f32 (gate_up) plus 4*4*2 f32
        // (down) = 384 bytes.
        let pager =
            ExpertStackPager::new_with_cache_budget(manifest.clone(), dir.clone(), 8, Some(500));
        assert_eq!(pager.cache_budget_bytes(), Some(500));
        let per_layer = pager
            .ensure_layer(0)
            .expect("layer 0 must page in")
            .resident_bytes();
        assert_eq!(per_layer, 384);
        assert_eq!(pager.cached_layer_count(), 1);
        assert_eq!(pager.cached_resident_bytes(), per_layer);

        // A second stack would exceed 500 bytes, so the LRU stack goes first.
        pager.ensure_layer(1).expect("layer 1 must page in");
        assert_eq!(pager.cached_layer_count(), 1);
        assert_eq!(pager.cached_layer_indices(), vec![1]);
        assert_eq!(pager.cached_resident_bytes(), per_layer);

        // The byte budget is an extra cap; the layer budget still applies on
        // its own when no byte budget is set.
        let layers_only =
            ExpertStackPager::new_with_cache_budget(manifest.clone(), dir.clone(), 8, None);
        layers_only.ensure_layer(0).expect("layer 0");
        layers_only.ensure_layer(1).expect("layer 1");
        assert_eq!(layers_only.cached_layer_count(), 2);

        // A budget smaller than one stack must never evict the stack in use.
        let tight = ExpertStackPager::new_with_cache_budget(manifest, dir.clone(), 8, Some(1));
        tight.ensure_layer(0).expect("layer 0 under a tiny budget");
        assert_eq!(tight.cached_layer_count(), 1);
        assert_eq!(tight.cached_resident_bytes(), per_layer);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // Priced state and KV components -------------------------------------

    /// Minimal Flash Next manifest: three linear-attention layers plus the
    /// QSA indexer contract, so both per-request state terms are priceable.
    fn flash_next_state_manifest() -> ax_engine_core::NativeModelManifest {
        let json = serde_json::json!({
            "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
            "model_family": "qwen4_exp",
            "tensor_format": "safetensors",
            "layer_count": 4,
            "hidden_size": 128,
            "attention_head_count": 4,
            "attention_head_dim": 64,
            "kv_head_count": 1,
            "vocab_size": 1024,
            "tensors": [],
            "linear_attention": {
                "num_key_heads": 4,
                "num_value_heads": 4,
                "key_head_dim": 16,
                "value_head_dim": 16,
                "conv_kernel_dim": 4
            },
            "qwen4_exp": {
                "indexer_head_dim": 32,
                "indexer_compress_ratio": 8,
                "indexer_n_heads": 2,
                "indexer_kv_heads": 1,
                "indexer_budget": 64
            }
        });
        serde_json::from_value(json).expect("flash next state manifest")
    }

    fn session_budget(kv_pool_tokens: u64) -> SessionResidencyBudget {
        SessionResidencyBudget {
            kv_pool_tokens,
            prefill_chunk: 16,
            max_live_sequences: None,
        }
    }

    #[test]
    fn resolved_live_sequences_uses_the_configured_bound_and_caps_at_the_cohort() {
        // The server's admitted concurrency wins over the generic batched cap.
        assert_eq!(resolved_live_sequences(Some(1), 8), 1);
        assert_eq!(resolved_live_sequences(Some(4), 8), 4);
        // Never price more live rows than the runner can actually hold.
        assert_eq!(resolved_live_sequences(Some(32), 8), 8);
        // An absent bound falls back to the cohort capacity...
        assert_eq!(resolved_live_sequences(None, 8), 8);
        // ...and neither path can price zero live sequences.
        assert_eq!(resolved_live_sequences(Some(0), 8), 1);
        assert_eq!(resolved_live_sequences(None, 0), 1);
    }

    #[test]
    fn state_bytes_scale_with_live_sequences_and_price_the_qsa_index() {
        let manifest = flash_next_state_manifest();
        // One request: conv ring 4*16*2 + 4*16 = 192 wide, 3 taps; recurrent
        // 4*16*16 = 1024; 1600 per layer * 4 layers * 4 bytes (f32).
        let gdn = 4 * 1_600 * 4;
        assert_eq!(hybrid_state_bytes(&manifest), gdn);
        // QSA index: 4 layers * (1000 raw keys + 125 pooled blocks) * 32 dims
        // * 2 bytes.
        let qsa = 4 * (1_000 * 32 * 2 + 125 * 32 * 2);
        assert_eq!(qsa_index_state_bytes(&manifest, 1_000), qsa);

        let budget = session_budget(1_000);
        let (_, one, _) = estimate_auto_residency_costs(&manifest, budget, 1);
        let (_, eight, _) = estimate_auto_residency_costs(&manifest, budget, 8);
        // The gated-delta-net state is per live row; the QSA index is priced
        // once at the pool bound because the pool already caps every position
        // it can describe.
        assert_eq!(one, gdn + qsa);
        assert_eq!(eight, gdn * 8 + qsa);

        // A zero concurrency still charges the one request that exists.
        let (_, zero, _) = estimate_auto_residency_costs(&manifest, budget, 0);
        assert_eq!(zero, one);

        // A manifest without the QSA contract prices only the GDN state.
        let mut plain = flash_next_state_manifest();
        plain.qwen4_exp.indexer_head_dim = None;
        plain.qwen4_exp.indexer_compress_ratio = None;
        assert_eq!(qsa_index_state_bytes(&plain, 1_000), 0);
        let (_, plain_state, _) = estimate_auto_residency_costs(&plain, budget, 2);
        assert_eq!(plain_state, gdn * 2);
    }

    /// Real Flash Next 6-bit geometry, copied from the pinned pack's
    /// `model-manifest.json`, so the scratch bound is tested on the numbers the
    /// qualification host actually runs.
    fn flash_next_6bit_scratch_manifest() -> ax_engine_core::NativeModelManifest {
        let json = serde_json::json!({
            "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
            "model_family": "qwen4_exp",
            "tensor_format": "safetensors",
            "layer_count": 48,
            "hidden_size": 2560,
            "attention_head_count": 24,
            "attention_head_dim": 256,
            "kv_head_count": 2,
            "vocab_size": 248_320,
            "tensors": [],
            "moe": {
                "expert_count": 512,
                "experts_per_token": 10,
                "expert_intermediate_size": 640,
                "shared_expert_count": 1
            },
            "linear_attention": {
                "full_attention_interval": 4,
                "num_key_heads": 16,
                "num_value_heads": 48,
                "key_head_dim": 128,
                "value_head_dim": 128,
                "conv_kernel_dim": 4
            },
            "qwen4_exp": {
                "indexer_head_dim": 128,
                "indexer_compress_ratio": 4,
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_budget": 2048,
                "hc_count": 4,
                "hc_lowrank": 320
            }
        });
        serde_json::from_value(json).expect("flash next 6-bit manifest")
    }

    #[test]
    fn prefill_scratch_prices_the_selection_and_moe_staging_of_a_sparse_chunk() {
        let manifest = flash_next_6bit_scratch_manifest();
        // Residual/attention/activation copies: chunk * hidden * 2 bytes * 4.
        let base = 2048u64 * 2560 * 2 * 4;
        // GDN: (16*128*2 + 48*128) wide, f32, two copies.
        let gdn = 2048u64 * 10_240 * 4 * 2;
        // MoE: 2048*10 routed rows, (640*2 + 2560*2) wide, 2 bytes.
        let moe = 20_480u64 * 6_400 * 2;
        // QSA: the 256 MiB score workspace plus one sub-batch of gathered
        // selected K/V. The shape product (2048*2051*2*256*2 = 4.0 GiB) stays
        // capped by the attention kernel's own 1 GiB per-sub-batch bound.
        let qsa = 256 * 1024 * 1024 + (1u64 << 30);
        // Hyper-connections: 2048 * 4 * 320 * 2 bytes.
        let hc = 2048u64 * 4 * 320 * 2;

        assert_eq!(gated_delta_chunk_bytes(&manifest, 2048), gdn);
        assert_eq!(moe_staging_bytes(&manifest, 2048), moe);
        assert_eq!(qsa_prefill_bytes(&manifest, 2048), qsa);
        assert_eq!(hyper_connection_chunk_bytes(&manifest, 2048), hc);
        assert_eq!(
            estimate_prefill_scratch_bytes(&manifest, 2048),
            base + gdn + moe + qsa + hc
        );
        // The bound is ~37x the flat 40.00 MiB estimate that under-priced the
        // measured 8192-token prefill by about 3.5 GiB.
        assert_eq!(base, 41_943_040);
        assert!(
            estimate_prefill_scratch_bytes(&manifest, 2048) > 30 * base,
            "the scratch bound must reflect the packed prefill transients"
        );
        // Doubling the chunk doubles every term, so the bound stays
        // shape-derived rather than pinned to one chunk size.
        assert_eq!(
            estimate_prefill_scratch_bytes(&manifest, 4096),
            (base + gdn + moe + hc) * 2 + qsa
        );
    }

    #[test]
    fn indexed_attention_prices_no_selected_kv_tile() {
        let manifest = flash_next_6bit_scratch_manifest();
        let gathered = qsa_prefill_bytes_with(&manifest, 2048, false);
        let indexed = qsa_prefill_bytes_with(&manifest, 2048, true);
        // The gathered arm carries one 1 GiB sub-batch on top of the score
        // workspace; the indexed arm must not, and stays at index rows plus
        // query/output tiles.
        assert_eq!(gathered, 256 * 1024 * 1024 + (1u64 << 30));
        assert!(
            indexed < gathered,
            "indexed pricing {indexed} must undercut the gathered bound {gathered}"
        );
        assert!(
            indexed - QSA_SCORE_WORKSPACE_BYTES < (1u64 << 30) / 4,
            "indexed attention must not price a selected K/V tile"
        );
        // It still covers the packed index rows at the full selected width.
        let kept = 2048u64 + 4 - 1;
        let indices = 2048 * kept * 4;
        let tiles = 2048
            * u64::from(manifest.attention_head_count)
            * u64::from(manifest.attention_head_dim)
            * ACTIVATION_BYTES
            * 2;
        assert!(indexed - QSA_SCORE_WORKSPACE_BYTES >= indices + tiles);
    }

    #[test]
    fn prefill_scratch_of_a_dense_pack_keeps_the_legacy_activation_copy() {
        // A 27B-class dense pack: no MoE, no linear attention, no QSA indexer,
        // no hyper-connections. Nothing may be added to the activation copy, so
        // dense packs keep exactly the pricing they had.
        let json = serde_json::json!({
            "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
            "model_family": "qwen3_5",
            "tensor_format": "safetensors",
            "layer_count": 64,
            "hidden_size": 5120,
            "attention_head_count": 40,
            "attention_head_dim": 128,
            "kv_head_count": 8,
            "vocab_size": 151_936,
            "tensors": []
        });
        let manifest: ax_engine_core::NativeModelManifest =
            serde_json::from_value(json).expect("dense manifest");
        assert_eq!(
            estimate_prefill_scratch_bytes(&manifest, 2048),
            2048u64 * 5120 * 2 * 4
        );
    }

    #[test]
    fn qsa_gather_term_is_a_cap_not_a_constant() {
        // The small state fixture keeps far fewer than a gibibyte of selected
        // K/V, so the bound must fall back to the shape product.
        let manifest = flash_next_state_manifest();
        let chunk = 16u64;
        let kept = 64u64 + 8 - 1;
        // The gather runs over the attention K/V width (`attention_head_dim`,
        // not the indexer width) with the fixture's single KV head.
        let kv_head_count = 1u64;
        let gathered = chunk * kept * kv_head_count * 64 * 2;
        assert!(gathered < 1u64 << 30);
        assert_eq!(
            qsa_prefill_bytes(&manifest, chunk),
            256 * 1024 * 1024 + gathered
        );
        // A pack without the indexer geometry pays no selection workspace.
        let mut dense = flash_next_state_manifest();
        dense.qwen4_exp.indexer_head_dim = None;
        dense.qwen4_exp.indexer_compress_ratio = None;
        dense.qwen4_exp.indexer_budget = None;
        assert_eq!(qsa_prefill_bytes(&dense, chunk), 0);
    }

    #[test]
    fn mla_pack_prices_the_latent_cache_not_the_unknown_floor() {
        let json = serde_json::json!({
            "schema_version": ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
            "model_family": "glm4_moe",
            "tensor_format": "safetensors",
            "layer_count": 2,
            "hidden_size": 128,
            "attention_head_count": 4,
            "attention_head_dim": 64,
            "kv_head_count": 1,
            "vocab_size": 1024,
            "tensors": [],
            "mla_attention": {"kv_lora_rank": 512, "qk_rope_head_dim": 64}
        });
        let manifest: ax_engine_core::NativeModelManifest =
            serde_json::from_value(json).expect("mla manifest");
        let (kv_bytes, _, _) = estimate_auto_residency_costs(&manifest, session_budget(1_000), 1);
        // 2 layers * 1000 tokens * (512 + 64) latent elements * 2 bytes.
        assert_eq!(kv_bytes, 2 * 1_000 * 576 * 2);
        assert_ne!(
            kv_bytes, UNKNOWN_KV_BYTES,
            "MLA must be priced, not collapsed onto the flat 768 MiB floor"
        );
    }

    #[test]
    fn resident_inventory_floors_the_resident_share_without_double_counting_experts() {
        // The manifest claims 10 GiB resident and 30 GiB streamable experts.
        let manifest = priced_manifest(30 * GIB, 10 * GIB);
        let context = AutoResidencyContext {
            host_ram_bytes: Some(128 * GIB),
            ..AutoResidencyContext::default()
        };
        let plan = memory_plan_for(&manifest, &context).expect("priced plan");
        assert_eq!(plan.resident_non_expert_bytes, 10 * GIB);
        assert_eq!(plan.expert_bytes, 30 * GIB);

        // The artifacts directory ships 3 GiB more *resident* tensors than the
        // manifest admits (MTP/vision sidecars). That delta is resident, never
        // "streamable".
        let floored = AutoResidencyContext {
            resident_tensor_inventory_bytes: 13 * GIB,
            ..context
        };
        let plan = memory_plan_for(&manifest, &floored).expect("priced plan");
        assert_eq!(plan.resident_non_expert_bytes, 13 * GIB);
        // Experts are priced once from the manifest: the resident floor never
        // leaks into the expert term.
        assert_eq!(plan.expert_bytes, 30 * GIB);
        // Weights are exactly resident floor + experts; the plan's fixed
        // runtime reserves sit on top of them.
        assert_eq!(
            plan.resident_working_set_bytes(),
            plan.fixed_bytes() + 30 * GIB
        );
        assert!(plan.fixed_bytes() >= 13 * GIB);

        // A smaller resident inventory cannot shrink the manifest's estimate.
        let smaller = AutoResidencyContext {
            resident_tensor_inventory_bytes: 5 * GIB,
            ..context
        };
        let plan = memory_plan_for(&manifest, &smaller).expect("priced plan");
        assert_eq!(plan.resident_non_expert_bytes, 10 * GIB);
        assert_eq!(plan.expert_bytes, 30 * GIB);

        // The floor applies only to the resident side even when the manifest's
        // own full estimate is stale and smaller than the real resident bytes.
        let stale = ExpertStreamManifest {
            estimated_resident_bytes: GIB,
            estimated_full_resident_bytes: 2 * GIB,
            ..priced_manifest(30 * GIB, 10 * GIB)
        };
        let plan = memory_plan_for(&stale, &floored).expect("priced plan");
        assert_eq!(plan.resident_non_expert_bytes, 13 * GIB);
        assert_eq!(plan.expert_bytes, GIB);
        assert_eq!(plan.resident_working_set_bytes(), plan.fixed_bytes() + GIB);
    }

    #[test]
    fn resident_tensor_inventory_excludes_experts_and_the_ngram_table() {
        let specs = vec![
            infer_spec(
                "language_model.model.embed_tokens.weight",
                NativeTensorRole::TokenEmbedding,
                0,
                1,
                4 * GIB,
            ),
            infer_spec(
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                NativeTensorRole::FfnGateExps,
                0,
                4,
                10 * GIB,
            ),
            infer_spec(
                "language_model.model.layers.0.ngram_embedding.shards.0.weight",
                NativeTensorRole::NgramEmbedding,
                0,
                1,
                20 * GIB,
            ),
            // The quantized table's scales/biases do not carry the role; the
            // name segment is what prices the whole table.
            infer_spec(
                "language_model.model.layers.0.ngram_embedding.shards.0.scales",
                NativeTensorRole::LayerScalar,
                0,
                1,
                3 * GIB,
            ),
        ];
        // With the production default the table is disk-served: only the
        // embedding is genuinely resident.
        assert_eq!(resident_tensor_inventory_bytes(&specs, true), 4 * GIB);
        // With the discount disabled the table is priced resident, matching an
        // undiscounted manifest estimate.
        assert_eq!(
            resident_tensor_inventory_bytes(&specs, false),
            4 * GIB + 20 * GIB + 3 * GIB
        );
    }

    #[test]
    fn real_flash_next_packs_stay_resident_at_the_default_server_cohort() {
        // The two pinned Flash Next packs on the qualification host. The test
        // skips (not fails) on machines without them, and never writes into
        // the model cache.
        const SIX_BIT: &str = "/Volumes/Ext16TR0/models/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP";
        const MXFP4: &str = "/Volumes/Ext16TR0/models/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP";
        const HOST_RAM: u64 = 128 * GIB;
        // 4096 blocks x 16 tokens: the configured pool the log line reports.
        const POOL_TOKENS: u64 = 4_096 * 16;
        // The default server passes no `--max-concurrent-requests`, so the
        // charged count is the runner's cohort capacity
        // (`resolved_live_sequences(None, 8)`).
        const COHORT_LIVE_SEQUENCES: u32 = 8;
        let ceiling =
            HOST_RAM - ax_engine_core::memory_budget::memory_plan_os_reserve_bytes(HOST_RAM);

        // 6-bit at the default cohort: the per-request state term does not fit
        // 8 rows, but the pack's experts still do at fewer. The plan must stay
        // resident and publish the count that fits, because the alternative --
        // paging 91 GiB of experts from disk -- decodes at 0.17 tok/s.
        let Some(six_bit) =
            real_pack_plan(std::path::Path::new(SIX_BIT), HOST_RAM, POOL_TOKENS, None)
        else {
            eprintln!("skipping {SIX_BIT}: local pack metadata not readable");
            return;
        };
        println!("{SIX_BIT} (default cohort): {}", six_bit.log_line());
        assert_eq!(six_bit.charged_live_sequences, Some(COHORT_LIVE_SEQUENCES));
        assert!(
            six_bit.working_set_bytes_for_live_sequences(COHORT_LIVE_SEQUENCES) > ceiling,
            "the regression case must be a configured count that does not fit: {}",
            six_bit.log_line()
        );
        assert!(
            six_bit.decide().is_resident(),
            "the 6-bit pack must stay resident at a reduced live-sequence count: {}",
            six_bit.log_line()
        );
        let cap = six_bit
            .live_sequence_cap
            .expect("a reduced live-sequence count is published");
        assert!(
            (1..COHORT_LIVE_SEQUENCES).contains(&cap),
            "the cap must admit at least one row and fewer than the cohort: {cap} in {}",
            six_bit.log_line()
        );
        assert!(six_bit.live_sequences_reduced());
        assert!(
            six_bit.resident_working_set_bytes() <= ceiling,
            "the admitted working set must fit the 115.20 GiB ceiling: {}",
            six_bit.log_line()
        );
        // One row below the cap could not fit either, so the cap is the largest
        // count the ceiling allows rather than an arbitrary reduction.
        assert!(
            six_bit.working_set_bytes_for_live_sequences(cap + 1) > ceiling,
            "the cap must be the largest fitting count: {}",
            six_bit.log_line()
        );
        // The line names both counts and the reduction, which is what an
        // operator sees when the server silently serves fewer rows.
        let line = six_bit.log_line();
        assert!(
            line.contains(&format!("live-sequences={COHORT_LIVE_SEQUENCES}")),
            "{line}"
        );
        assert!(line.contains(&format!("live-sequence-cap={cap}")), "{line}");
        assert!(line.contains("warning=live-sequences-capped"), "{line}");

        // The same pack at one explicitly configured row stays resident and
        // publishes no cap: one row already fits, and a cap of one would only
        // make the engine queue requests the configured bound already admits.
        let Some(single) = real_pack_plan(
            std::path::Path::new(SIX_BIT),
            HOST_RAM,
            POOL_TOKENS,
            Some(1),
        ) else {
            return;
        };
        println!("{SIX_BIT} (1 live): {}", single.log_line());
        assert!(single.decide().is_resident(), "{}", single.log_line());
        assert_eq!(single.live_sequence_cap, None);
        assert!(!single.live_sequences_reduced());
        assert!(
            single.log_line().contains("live-sequence-cap=uncapped"),
            "{}",
            single.log_line()
        );

        // MXFP4 has ~33 GiB of headroom: the full cohort must survive. This is
        // the case a blanket concurrency reduction would have broken.
        let Some(mxfp4) = real_pack_plan(std::path::Path::new(MXFP4), HOST_RAM, POOL_TOKENS, None)
        else {
            eprintln!("skipping {MXFP4}: local pack metadata not readable");
            return;
        };
        println!("{MXFP4} (default cohort): {}", mxfp4.log_line());
        assert!(mxfp4.decide().is_resident(), "{}", mxfp4.log_line());
        assert_eq!(
            mxfp4.live_sequence_cap,
            None,
            "MXFP4 fits its full cohort, so no cap is published: {}",
            mxfp4.log_line()
        );
        assert!(!mxfp4.live_sequences_reduced());
        assert!(!mxfp4.log_line().contains("warning=live-sequences-capped"));
        assert!(
            mxfp4.log_line().contains("live-sequence-cap=uncapped"),
            "{}",
            mxfp4.log_line()
        );
        // Every case reports both sides of the wired limit, so an operator can
        // see why a resident pack still runs below the 0.9 share (or at the
        // full recommendation for the 6-bit single-stream case).
        for plan in [&six_bit, &mxfp4] {
            assert!(plan.wired_limit_bytes.is_some(), "{}", plan.log_line());
            assert!(
                plan.wired_shortfall_bytes().is_some(),
                "{}",
                plan.log_line()
            );
        }
    }

    #[test]
    fn real_flash_next_6bit_streams_when_the_host_cannot_hold_the_experts() {
        const PACK: &str = "/Volumes/Ext16TR0/models/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP";
        // 96 GiB cannot hold the 6-bit pack's ~91 GiB expert set next to its
        // ~23 GiB fixed part, so the plan must page experts. The 128 GiB case
        // at the same pool and cohort stays resident at a reduced live count
        // (see the test above): the live-sequence cap never turns a host that
        // cannot hold the experts into a resident one.
        const HOST_RAM: u64 = 96 * GIB;
        const POOL_TOKENS: u64 = 4_096 * 16;
        let Some(plan) = real_pack_plan(std::path::Path::new(PACK), HOST_RAM, POOL_TOKENS, None)
        else {
            eprintln!("skipping {PACK}: local pack metadata not readable");
            return;
        };
        println!("{PACK} (default cohort, 96 GiB): {}", plan.log_line());
        assert!(
            matches!(
                plan.decide(),
                ax_engine_core::memory_budget::MemoryPlanDecision::Stream { .. }
            ),
            "a host that cannot hold the expert set must stream: {}",
            plan.log_line()
        );
        // Not even one live sequence fits, so no cap is published: residency is
        // not what bounds this session.
        assert_eq!(plan.live_sequence_cap, None);
    }

    /// Price one real pack exactly as the runner does at load admission.
    ///
    /// `configured_live_sequences` mirrors `--max-concurrent-requests`: `None`
    /// is the default server, which charges the runner's cohort capacity. It
    /// only reads the native manifest, `ax_expert_stream.json`, `config.json`
    /// and safetensors headers; no tensor payload is loaded. Returns `None`
    /// when the directory (or its metadata) is absent, so the tests skip on a
    /// host without the pinned packs.
    fn real_pack_plan(
        dir: &std::path::Path,
        host_ram_bytes: u64,
        pool_tokens: u64,
        configured_live_sequences: Option<u32>,
    ) -> Option<ax_engine_core::memory_budget::MemoryPlan> {
        if !dir.is_dir() {
            return None;
        }
        let artifacts = real_pack_artifacts(dir)?;
        let file_manifest = ExpertStreamManifest::read_from_dir(dir).ok()??;
        let discount_ngram = crate::fastpath::expert_stream_discount_ngram_enabled();
        let never_loaded_bytes = if discount_ngram {
            crate::weights::qwen4_exp::never_loaded_ngram_bytes(artifacts.tensor_specs())
        } else {
            0
        };
        let manifest = discount_never_loaded_bytes(file_manifest, never_loaded_bytes);
        // The runner's cohort capacity (`AX_MLX_BATCHED_DECODE_MAX` default).
        // `resolved_live_sequences` caps a configured bound by it and falls back
        // to it when the server set no bound, which is the default server.
        const COHORT_CAP: usize = 8;
        let live_sequences = resolved_live_sequences(configured_live_sequences, COHORT_CAP);
        let budget = SessionResidencyBudget {
            kv_pool_tokens: pool_tokens,
            prefill_chunk: crate::generate::DEFAULT_PREFILL_CHUNK,
            max_live_sequences: configured_live_sequences,
        };
        let (kv_bytes, state_bytes, prefill_scratch_bytes) =
            estimate_auto_residency_costs(artifacts.manifest(), budget, live_sequences);
        let mut context = AutoResidencyContext {
            kv_bytes,
            state_bytes,
            prefill_scratch_bytes,
            ngram_table_bytes: never_loaded_bytes,
            resident_tensor_inventory_bytes: resident_tensor_inventory_bytes(
                artifacts.tensor_specs(),
                discount_ngram,
            ),
            host_ram_bytes: Some(host_ram_bytes),
            metal_recommended_max_working_set_bytes: Some(115_448_725_504),
            wired_limit_bytes: None,
            ceiling_override_bytes: None,
            live_sequences: Some(live_sequences.max(1)),
            state_bytes_per_live_sequence: per_live_sequence_state_bytes(artifacts.manifest()),
            live_sequences_operator_capped: configured_live_sequences.is_some(),
        };
        // Mirror the runner's two passes: it prices the pack to size the wired
        // limit, then builds the admission plan with that limit. The working set
        // is identical between passes (nothing in the plan depends on the
        // limit), so the second pass is the one production sees.
        let first = memory_plan_for(&manifest, &context)?;
        let wired = crate::tiel_memory_policy::resolve_wired_limit(
            115_448_725_504,
            first
                .decide()
                .is_resident()
                .then(|| first.resident_working_set_bytes()),
            None,
        );
        context.wired_limit_bytes = Some(wired.limit_bytes);
        memory_plan_for(&manifest, &context)
    }

    /// Read a real pack's native manifest without writing into the pack.
    ///
    /// Some pinned packs ship `config.json` plus safetensors headers without a
    /// generated `model-manifest.json`; those are converted in memory so the
    /// test never mutates the model cache.
    fn real_pack_artifacts(dir: &std::path::Path) -> Option<ax_engine_core::NativeModelArtifacts> {
        if let Ok(artifacts) = ax_engine_core::NativeModelArtifacts::from_dir(dir) {
            return Some(artifacts);
        }
        let manifest = ax_engine_core::convert::convert_hf_model_dir(dir).ok()?;
        ax_engine_core::NativeModelArtifacts::from_manifest_and_root(dir.to_path_buf(), manifest)
            .ok()
    }

    #[test]
    fn auto_refuses_an_expert_cache_smaller_than_one_layer_stack() {
        // 10 GiB resident + 80 GiB of experts on a 64 GiB host: the plan's
        // remainder (57.6 GiB ceiling - 13 GiB fixed) cannot hold one layer.
        let manifest = priced_manifest(80 * GIB, 10 * GIB);
        let context = AutoResidencyContext {
            host_ram_bytes: Some(64 * GIB),
            ..AutoResidencyContext::default()
        };
        let error = resolve_expert_stream_inner(
            StreamExpertsMode::Auto,
            Some(manifest.clone()),
            || Err(ExpertStreamError::ManifestMissing),
            Some(64 * GIB),
            Some(&context),
            true,
        )
        .expect_err("a cache below one layer stack must refuse");
        assert!(matches!(
            error,
            ExpertStreamError::ExpertCacheBelowLayerStack {
                expert_cache_bytes,
                layer_bytes
            } if expert_cache_bytes < layer_bytes
        ));

        // Unknown capacity pages with no byte cap at all, so the pager's
        // layer-count budget decides and there is nothing to floor.
        let unknown = AutoResidencyContext::default();
        assert!(
            resolve_expert_stream_inner(
                StreamExpertsMode::Auto,
                Some(manifest),
                || Err(ExpertStreamError::ManifestMissing),
                None,
                Some(&unknown),
                true,
            )
            .expect("unknown capacity must still admit")
            .is_some()
        );
    }

    #[test]
    fn auto_without_priced_inputs_uses_the_legacy_formula_and_notes_it_once() {
        // The latch is exercised on its own state: an earlier test in this
        // process may already have fired the process-wide note, which would
        // make a bare counter assertion pass without testing anything.
        let fired = std::sync::atomic::AtomicBool::new(false);
        assert!(
            note_priced_memory_skipped_once(&fired),
            "the first caller must win the latch"
        );
        assert!(
            !note_priced_memory_skipped_once(&fired),
            "the latch must refuse every later caller"
        );
        assert!(!note_priced_memory_skipped_once(&fired));

        // Re-calling the process-wide note never adds more than the one claim
        // it is allowed, whatever the rest of the test binary did.
        let before = priced_memory_skip_count();
        note_priced_memory_skipped();
        note_priced_memory_skipped();
        let delta = priced_memory_skip_count().saturating_sub(before);
        assert!(delta <= 1, "the note fired {delta} times in two calls");

        set_auto_residency_context(None);
        // 10 GiB resident + 40 GiB experts: the legacy formula pages this
        // (50 GiB + 48 GiB headroom > 64 GiB).
        let manifest = priced_manifest(40 * GIB, 10 * GIB);
        assert!(
            resolve_expert_stream(
                StreamExpertsMode::Auto,
                Some(manifest),
                || Err(ExpertStreamError::ManifestMissing),
                Some(64 * GIB),
            )
            .expect("legacy admission")
            .is_some()
        );
    }
}
