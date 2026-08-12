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

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ax_engine_core::{NativeTensorQuantization, NativeTensorRole, NativeTensorSpec};
use serde::{Deserialize, Serialize};

use crate::weights::QuantizedWeight;

pub const EXPERT_STREAM_MANIFEST_FILE: &str = "ax_expert_stream.json";
pub const EXPERT_STREAM_SCHEMA_V1: &str = "axquant.expert-stream.v1";
pub const EXPERT_STREAM_MODE_LAYER_STACK: &str = "layer-stack";
/// Serve/load admission flag: `AX_STREAM_EXPERTS=1` opts into streaming.
pub const STREAM_EXPERTS_ENV: &str = "AX_STREAM_EXPERTS";
/// Number of layer expert stacks kept resident concurrently (minimum 1).
pub const STREAM_EXPERT_LAYERS_ENV: &str = "AX_STREAM_EXPERT_LAYERS";

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

fn env_flag_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Whether `AX_STREAM_EXPERTS` opts this process into expert streaming.
pub fn stream_experts_env_enabled() -> bool {
    env_flag_enabled(std::env::var(STREAM_EXPERTS_ENV).ok().as_deref())
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

static STREAM_EXPERTS_OVERRIDE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Install the CLI admission flag (`--stream-experts`) before weights load.
/// Follows the `set_speculation_profile_override` pattern: the server/SDK
/// crates cannot mutate the process environment, so the flag latches here.
pub fn set_stream_experts_override(enabled: bool) {
    STREAM_EXPERTS_OVERRIDE.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Admission state: CLI override OR `AX_STREAM_EXPERTS=1`.
pub fn stream_experts_requested() -> bool {
    STREAM_EXPERTS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed)
        || stream_experts_env_enabled()
}

/// Admission gate for a file-backed `ax_expert_stream.json`:
/// - no file + streaming off → `Ok(None)` (default resident load)
/// - no file + streaming on → `Ok(None)` so the caller can infer a layer-stack
///   plan from native expert roles (DeepSeek V4 Flash published packs, Qwen 3.8)
/// - file `required=true` + streaming off → fail closed with the estimated
///   full-resident byte count (never fall through to a full load)
/// - file present + streaming on → `Ok(Some(manifest))`
pub fn admit_expert_stream(
    model_dir: &Path,
    requested: bool,
) -> Result<Option<ExpertStreamManifest>, ExpertStreamError> {
    let manifest = match ExpertStreamManifest::read_from_dir(model_dir)? {
        Some(manifest) => manifest,
        None => {
            return Ok(None);
        }
    };
    if manifest.required && !requested {
        return Err(ExpertStreamError::StreamRequired {
            estimated_full_resident_bytes: manifest.estimated_full_resident_bytes,
        });
    }
    if !requested {
        // Optional manifest without the admission flag: keep the default
        // fully-resident load.
        return Ok(None);
    }
    Ok(Some(manifest))
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
        let group_size = spec.quantization.as_ref().map(|q| q.group_size).unwrap_or(64);
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
    cache: Mutex<PagerCache>,
}

impl ExpertStackPager {
    pub fn new(manifest: Arc<ExpertStreamManifest>, root: PathBuf, budget_layers: usize) -> Self {
        Self {
            manifest,
            root,
            budget_layers: budget_layers.max(1),
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
        // initial-load eval for both loader paths.
        let refs: Vec<&mlx_sys::MlxArray> = loaded.values().collect();
        mlx_sys::eval(&refs);

        let mut stack = LayerExpertStack::default();
        for tensor in &tensors {
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
            };
            stack.insert(proj, quantized);
        }
        if stack.is_empty() {
            return Err(ExpertStreamError::Paging(format!(
                "no expert tensors were paged for layer {layer}"
            )));
        }
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
    pub fn new(pager: Arc<ExpertStackPager>, layer: u32) -> Self {
        Self { pager, layer }
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// Resolve this layer's expert stack, paging it in when needed.
    pub fn stack(&self) -> Result<LayerExpertStack, ExpertStreamError> {
        self.pager.ensure_layer(self.layer)
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
        assert!(admit_expert_stream(&dir, true).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
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
            quantization: Some(NativeTensorQuantization {
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
}
