//! Shared working-set estimation for model-load admission.
//!
//! Pure arithmetic over a parsed `model-manifest.json` attention-geometry
//! projection and on-disk weight bytes. Filesystem probes (summing
//! `*.safetensors`, reading the manifest) stay with the callers; this module
//! only turns a geometry plus a pool size into byte estimates.
//!
//! Every number here is a conservative load-admission heuristic. The
//! unknown-geometry fallback (a `None` KV estimate) is NOT proof that a
//! request or the whole working set fits, and none of these estimates is a
//! hard allocator limit — the engine can still fail an allocation at runtime.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::model::{
    NativeDeepseekV4Config, NativeDiffusionConfig, NativeLinearAttentionConfig,
    NativeMlaAttentionConfig,
};

/// Flat floor over on-disk weight bytes when the KV pool is not charged
/// explicitly: quantized weights land in memory at ~disk size; the extra
/// 1/8 runtime factor (applied by [`estimated_footprint_bytes`]) covers
/// compiled graphs, speculative-decoding state, and allocator slack; this
/// floor covers the KV baseline and runtime buffers. Conservative by
/// design — the estimator's job is to fail a load that cannot fit, loudly
/// and early.
const LOAD_FOOTPRINT_FIXED_FLOOR_BYTES: u64 = 768 * 1024 * 1024;

/// Floor applied when the KV pool is charged explicitly: the remainder
/// covers runtime buffers, linear-attention per-request state, MTP state,
/// and allocator slack — everything the flat floor above bundles together
/// with the "KV baseline".
const LOAD_FOOTPRINT_RUNTIME_FLOOR_BYTES: u64 = 512 * 1024 * 1024;

/// K + V, at fp16/bf16 element width. Quantized-KV serving would shrink
/// this; the admission bound stays at the unquantized worst case.
const KV_CACHE_BYTES_PER_HEAD_ELEMENT: u64 = 2 * 2;

/// One compressed MLA latent element (fp16/bf16). MLA stores a single
/// `kv_lora_rank` latent plus the decoupled RoPE slice, not K and V per
/// head, so it is charged at half [`KV_CACHE_BYTES_PER_HEAD_ELEMENT`].
const MLA_CACHE_BYTES_PER_ELEMENT: u64 = 2;

/// Estimated resident footprint for one model: on-disk weight bytes scaled
/// by a runtime factor (`1 + 1/8`, since quantized weights land in memory at
/// roughly disk size and the extra eighth covers compiled graphs,
/// speculative-decoding state, and allocator slack) plus a floor. With the
/// KV pool charged explicitly the floor covers only runtime buffers; without
/// it, a larger flat floor also bundles a KV baseline.
///
/// This is a conservative load-admission heuristic: it is not a hard
/// allocator limit, and the flat-floor fallback taken when the KV geometry
/// is unknown is NOT proof that a request or the whole working set fits.
pub fn estimated_footprint_bytes(weight_bytes: u64, kv_pool_bytes: Option<u64>) -> u64 {
    let base = weight_bytes.saturating_add(weight_bytes / 8);
    match kv_pool_bytes {
        Some(kv_bytes) => base
            .saturating_add(kv_bytes)
            .saturating_add(LOAD_FOOTPRINT_RUNTIME_FLOOR_BYTES),
        None => base.saturating_add(LOAD_FOOTPRINT_FIXED_FLOOR_BYTES),
    }
}

/// Tolerant projection of `model-manifest.json` for KV-geometry estimation.
/// Only the fields the estimator needs; identity checks live elsewhere and
/// stay independent. Unknown manifest fields never fail this parse.
#[derive(Debug, Deserialize)]
pub struct ManifestKvGeometry {
    #[serde(default)]
    model_family: String,
    layer_count: u32,
    attention_head_dim: u32,
    kv_head_count: u32,
    /// ISWA full-attention layers use this head dim; sliding layers use
    /// `attention_head_dim`.
    #[serde(default)]
    global_head_dim: Option<u32>,
    /// Explicit KV head count for ISWA full-attention layers. Older manifests
    /// omit this and retain the constant-total-KV-width rule.
    #[serde(default)]
    global_kv_head_count: Option<u32>,
    /// Minimal tensor projection used to recover per-layer KV geometry from
    /// manifests generated before `global_kv_head_count` was introduced.
    #[serde(default)]
    tensors: Vec<ManifestKvTensorGeometry>,
    /// Per-layer annotations ("sliding_attention" / "full_attention");
    /// empty for homogeneous models.
    #[serde(default)]
    layer_types: Vec<String>,
    /// Layers that read another layer's K/V and allocate none of their own.
    #[serde(default)]
    kv_shared_source_layers: BTreeMap<u32, u32>,
    #[serde(default)]
    linear_attention: NativeLinearAttentionConfig,
    #[serde(default)]
    mla_attention: NativeMlaAttentionConfig,
    /// DeepSeek V4's fused latent KV geometry (`head_dim` doubles as K and V).
    /// Its `NativeMlaAttentionConfig` stays disabled, so the MLA branch needs
    /// this config to price the latent cache instead of falling back.
    #[serde(default)]
    deepseek_v4: NativeDeepseekV4Config,
    #[serde(default)]
    diffusion: NativeDiffusionConfig,
}

#[derive(Debug, Deserialize)]
struct ManifestKvTensorGeometry {
    #[serde(default)]
    role: String,
    #[serde(default)]
    layer_index: Option<u32>,
    #[serde(default)]
    shape: Vec<u64>,
}

fn full_kv_head_count_for_layer(
    geometry: &ManifestKvGeometry,
    layer_idx: u32,
    full_head_dim: u64,
    legacy_full_kv_heads: Option<u64>,
) -> Option<u64> {
    match geometry.global_kv_head_count {
        Some(count) if count > 0 => return Some(u64::from(count)),
        Some(_) => return None,
        None => {}
    }

    let inferred = geometry
        .tensors
        .iter()
        .find(|tensor| tensor.role == "attention_k" && tensor.layer_index == Some(layer_idx))
        .and_then(|tensor| {
            let rows = *tensor.shape.first()?;
            (tensor.shape.len() == 2
                && rows > 0
                && full_head_dim > 0
                && rows.is_multiple_of(full_head_dim))
            .then_some(rows / full_head_dim)
        });
    inferred.or(legacy_full_kv_heads)
}

/// Worst-case KV-cache bytes for one model at its configured pool, from the
/// manifest's attention geometry. `None` when the geometry is unknowable —
/// diffusion keeps no autoregressive cache, and a zero dim or an unresolvable
/// hybrid interval means the manifest predates this estimator — in which case
/// the caller falls back to the flat floor baked into
/// [`estimated_footprint_bytes`] rather than guessing.
///
/// MLA (GLM/DeepSeek) packs are priced from their latent width
/// (`kv_lora_rank + qk_rope_head_dim`, or DeepSeek V4's fused `head_dim`), and
/// fall back to the equivalent full-attention width when that geometry is
/// missing: a latent cache is much smaller than K+V but is never free, and the
/// old `None` here collapsed to a flat 768 MiB floor that under-priced every
/// long-context MLA pool.
///
/// The result is a pool-wide worst case for load admission. Returning `None`
/// (the unknown-geometry fallback) is NOT proof that a request or the whole
/// working set fits, and the returned bound is not a hard allocator limit.
pub fn estimated_kv_pool_bytes(geometry: &ManifestKvGeometry, pool_tokens: u64) -> Option<u64> {
    if geometry.diffusion.is_enabled() {
        // Diffusion keeps no autoregressive KV cache at all.
        return None;
    }
    if geometry.mla_attention.is_enabled() || geometry.deepseek_v4.attention.head_dim.is_some() {
        return estimated_mla_kv_pool_bytes(geometry, pool_tokens);
    }
    if geometry.layer_count == 0 || geometry.kv_head_count == 0 || geometry.attention_head_dim == 0
    {
        return None;
    }
    let linear_interval = if geometry.linear_attention.is_enabled() {
        match geometry
            .linear_attention
            .resolved_full_attention_interval(&geometry.model_family)
        {
            Some(interval) if interval > 0 => Some(u64::from(interval)),
            _ => return None,
        }
    } else {
        None
    };
    let kv_heads = u64::from(geometry.kv_head_count);
    let sliding_bytes_per_token = kv_heads
        .saturating_mul(u64::from(geometry.attention_head_dim))
        .saturating_mul(KV_CACHE_BYTES_PER_HEAD_ELEMENT);
    let full_head_dim = geometry
        .global_head_dim
        .unwrap_or(geometry.attention_head_dim);
    let full_head_dim = u64::from(full_head_dim);
    // A tolerant metadata projection can contain an invalid zero global
    // dimension. Treat it as unknown instead of dividing by zero or
    // returning a zero-byte full-attention estimate.
    if full_head_dim == 0 {
        return None;
    }
    let base_kv_width = kv_heads.saturating_mul(u64::from(geometry.attention_head_dim));
    let legacy_full_kv_heads = (full_head_dim > 0 && base_kv_width.is_multiple_of(full_head_dim))
        .then_some(base_kv_width / full_head_dim);

    let mut total = 0u64;
    for layer_idx in 0..geometry.layer_count {
        if geometry.kv_shared_source_layers.contains_key(&layer_idx) {
            continue;
        }
        if let Some(interval) = linear_interval {
            // `layer_types` is the authoritative per-layer kind when the
            // manifest carries it: Nemotron-H writes full_attention_interval=1
            // as a sentinel that disables the Qwen gated-delta path, so the
            // interval rule would charge every mamba/moe/mlp layer for a
            // per-token KV cache the runtime never allocates.
            match geometry.layer_types.get(layer_idx as usize) {
                Some(kind) => match kind.as_str() {
                    "linear_attention" | "mamba" | "M" | "m" | "moe" | "E" | "e" | "mlp" | "-" => {
                        continue;
                    }
                    "sliding_attention" => {
                        total = total
                            .saturating_add(pool_tokens.saturating_mul(sliding_bytes_per_token));
                        continue;
                    }
                    // full_attention / attention / "*" / qwen_sparse_attention,
                    // and unknown kinds, fail closed at the full-attention
                    // charge (sparse stays conservative: its budget-bounded
                    // cache is smaller, but admission takes the worst case).
                    _ => {}
                },
                // Manifests that predate layer_types: hybrid linear-attention
                // layers keep per-request state, not a per-token cache; the
                // runtime floor covers them.
                None if u64::from(layer_idx) % interval != interval - 1 => continue,
                None => {}
            }
            let full_kv_heads = full_kv_head_count_for_layer(
                geometry,
                layer_idx,
                full_head_dim,
                legacy_full_kv_heads,
            )?;
            let full_bytes_per_token = full_kv_heads
                .saturating_mul(full_head_dim)
                .saturating_mul(KV_CACHE_BYTES_PER_HEAD_ELEMENT);
            total = total.saturating_add(pool_tokens.saturating_mul(full_bytes_per_token));
            continue;
        }
        let layer_type = geometry.layer_types.get(layer_idx as usize);
        if layer_type.is_some_and(|kind| kind == "sliding_attention") {
            // Sliding rings bound KV per REQUEST, not per pool: many
            // concurrent ≤window-length requests each own window-sized
            // rings, and their sum legally reaches the pool. Admission
            // charges the pool-wide worst case, so sliding differs from
            // full attention only in head dim.
            total = total.saturating_add(pool_tokens.saturating_mul(sliding_bytes_per_token));
        } else {
            let full_kv_heads = full_kv_head_count_for_layer(
                geometry,
                layer_idx,
                full_head_dim,
                legacy_full_kv_heads,
            )?;
            let full_bytes_per_token = full_kv_heads
                .saturating_mul(full_head_dim)
                .saturating_mul(KV_CACHE_BYTES_PER_HEAD_ELEMENT);
            total = total.saturating_add(pool_tokens.saturating_mul(full_bytes_per_token));
        }
    }
    Some(total)
}

/// Latent KV-cache bytes for an MLA (GLM/DeepSeek) pack.
///
/// MLA keeps one compressed latent per token per layer instead of K and V per
/// head, so the flat 768 MiB unknown-geometry floor massively under-prices a
/// long-context pool and silently flipped these packs to a resident decision.
/// The latent width is `kv_lora_rank + qk_rope_head_dim`; DeepSeek V4 fuses
/// both into a single `head_dim` row. When neither config carries usable
/// geometry, the pool is charged at the equivalent full-attention width rather
/// than guessed at zero.
fn estimated_mla_kv_pool_bytes(geometry: &ManifestKvGeometry, pool_tokens: u64) -> Option<u64> {
    if geometry.layer_count == 0 {
        return None;
    }
    let bytes_per_token_per_layer = mla_bytes_per_token_per_layer(geometry)
        .or_else(|| full_attention_equivalent_bytes_per_token_per_layer(geometry))?;
    let mut total = 0u64;
    for layer_idx in 0..geometry.layer_count {
        if geometry.kv_shared_source_layers.contains_key(&layer_idx) {
            continue;
        }
        total = total.saturating_add(pool_tokens.saturating_mul(bytes_per_token_per_layer));
    }
    Some(total)
}

/// One layer's MLA latent width in bytes, from the narrowest available
/// description of the compressed cache.
fn mla_bytes_per_token_per_layer(geometry: &ManifestKvGeometry) -> Option<u64> {
    // DeepSeek V4: one fused latent K/V row per token (`head_dim` covers both
    // the nope and rope slices), single head.
    if let Some(head_dim) = geometry
        .deepseek_v4
        .attention
        .head_dim
        .filter(|dim| *dim > 0)
    {
        return Some(u64::from(head_dim).saturating_mul(MLA_CACHE_BYTES_PER_ELEMENT));
    }
    // DeepSeek V2/V3-style MLA and GLM: compressed latent plus decoupled RoPE.
    let latent = geometry.mla_attention.kv_lora_rank;
    let rope = geometry.mla_attention.qk_rope_head_dim;
    match (latent, rope) {
        (Some(latent), Some(rope)) if latent > 0 && rope > 0 => Some(
            u64::from(latent)
                .saturating_add(u64::from(rope))
                .saturating_mul(MLA_CACHE_BYTES_PER_ELEMENT),
        ),
        _ => None,
    }
}

/// Conservative stand-in for an MLA pack whose latent geometry is missing:
/// charge the widest conventional full-attention K+V row the manifest still
/// describes. Never zero — an unpriced KV pool is what this branch exists to
/// avoid.
fn full_attention_equivalent_bytes_per_token_per_layer(
    geometry: &ManifestKvGeometry,
) -> Option<u64> {
    let sliding =
        u64::from(geometry.kv_head_count).saturating_mul(u64::from(geometry.attention_head_dim));
    let full = u64::from(
        geometry
            .global_kv_head_count
            .unwrap_or(geometry.kv_head_count),
    )
    .saturating_mul(u64::from(
        geometry
            .global_head_dim
            .unwrap_or(geometry.attention_head_dim),
    ));
    let width = sliding.max(full);
    (width > 0).then_some(width.saturating_mul(KV_CACHE_BYTES_PER_HEAD_ELEMENT))
}

/// MLX's own computed-graph buffer cache, charged to a priced load plan.
pub const MEMORY_PLAN_MLX_BUFFER_CACHE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Runtime reserve on top of the MLX buffer cache: compiled graphs,
/// tokenizer/request state, and allocator slack.
pub const MEMORY_PLAN_RUNTIME_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;

/// Floor of the OS reserve. The reserve is `max(this, RAM/10)` so a small
/// host still keeps room for the kernel, the display server, and the page
/// cache that serves disk-mapped tensors.
pub const MEMORY_PLAN_MIN_OS_RESERVE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// OS reserve for a host of `host_ram_bytes`: a tenth of RAM, never below
/// [`MEMORY_PLAN_MIN_OS_RESERVE_BYTES`]. 12.8 GiB on a 128 GiB host, which
/// leaves a 115.2 GiB ceiling.
pub fn memory_plan_os_reserve_bytes(host_ram_bytes: u64) -> u64 {
    MEMORY_PLAN_MIN_OS_RESERVE_BYTES.max(host_ram_bytes / 10)
}

/// Reserve the explicit `AX_MLX_MEMORY_CEILING_BYTES` override may never
/// cross: the override can move the ceiling in either direction, but never
/// into the last 4 GiB of unified memory, where the kernel and the window
/// server live.
pub const MEMORY_PLAN_OVERRIDE_RAM_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Environment variable holding an explicit ceiling override in bytes.
pub const MEMORY_PLAN_CEILING_OVERRIDE_ENV: &str = "AX_MLX_MEMORY_CEILING_BYTES";

/// Parse an explicit ceiling override. Blank, zero, negative, and unparsable
/// values mean "no override" so the RAM-derived ceiling stands.
pub fn parse_memory_plan_ceiling_override(value: Option<&str>) -> Option<u64> {
    value
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|bytes| *bytes > 0)
}

/// Priced decomposition of one model's resident and streamed costs.
///
/// Every field is bytes. The n-gram table is priced separately because the
/// Flash Next table is served by disk row gathers and is never resident; it
/// is reported but not charged to the working set. Expert weight bytes are
/// charged only when the plan decides to keep them resident.
///
/// This is a load-admission estimate, not an allocator limit. `ceiling_bytes`
/// is `None` when host RAM is unknown and no override is set, in which case
/// residency cannot be proven and the plan pages experts rather than risking
/// an OOM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryPlan {
    /// Non-expert weights that always stay resident (embeddings, attention,
    /// shared experts, norms, MTP head).
    pub resident_non_expert_bytes: u64,
    /// Packed expert weight bytes that may stay resident or page from disk.
    pub expert_bytes: u64,
    /// Whole n-gram lexical table including its quantization scales/biases.
    /// Disk/virtual: reported, never charged to the working set.
    pub ngram_table_bytes: u64,
    /// KV cache bytes for the configured context at the admitted pool.
    pub kv_bytes: u64,
    /// GDN/QSA/PLE state: conv rings and recurrent state per live sequence
    /// plus the index and pooled buffers once at the shared pool bound.
    pub state_bytes: u64,
    /// Bound on prefill activations for one chunk.
    pub prefill_scratch_bytes: u64,
    /// MLX computed-graph buffer cache allowance.
    pub mlx_buffer_cache_bytes: u64,
    /// Runtime reserve (compiled graphs, allocator slack).
    pub runtime_reserve_bytes: u64,
    /// Physical unified memory (`hw.memsize`).
    pub host_ram_bytes: Option<u64>,
    /// Metal `recommendedMaxWorkingSetSize`, when the platform reports it.
    ///
    /// Advisory only: it never bounds [`MemoryPlan::ceiling_bytes`]. Measured
    /// Flash Next 6-bit sessions ran resident at a peak RSS above this value,
    /// so the plan keeps its RAM-derived decision and only flags the gap in
    /// [`MemoryPlan::log_line`].
    pub metal_recommended_max_working_set_bytes: Option<u64>,
    /// Explicit ceiling override (`AX_MLX_MEMORY_CEILING_BYTES`). Bounded by
    /// `RAM - MEMORY_PLAN_OVERRIDE_RAM_RESERVE_BYTES` when RAM is known, and
    /// used as-is when it is not.
    pub ceiling_override_bytes: Option<u64>,
}

impl Default for MemoryPlan {
    fn default() -> Self {
        Self {
            resident_non_expert_bytes: 0,
            expert_bytes: 0,
            ngram_table_bytes: 0,
            kv_bytes: 0,
            state_bytes: 0,
            prefill_scratch_bytes: 0,
            mlx_buffer_cache_bytes: MEMORY_PLAN_MLX_BUFFER_CACHE_BYTES,
            runtime_reserve_bytes: MEMORY_PLAN_RUNTIME_RESERVE_BYTES,
            host_ram_bytes: None,
            metal_recommended_max_working_set_bytes: None,
            ceiling_override_bytes: None,
        }
    }
}

/// Outcome of [`MemoryPlan::decide`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryPlanDecision {
    /// Everything, experts included, fits under the ceiling.
    Resident {
        working_set_bytes: u64,
        ceiling_bytes: u64,
    },
    /// The fixed part fits but the experts do not; page them with at most
    /// `expert_cache_bytes` of resident layer stacks (`0` means "no byte
    /// cap": the layer-count budget alone decides).
    Stream {
        working_set_bytes: u64,
        expert_cache_bytes: u64,
        ceiling_bytes: Option<u64>,
    },
    /// Even the fixed part does not fit: the load cannot be admitted.
    Refuse {
        fixed_bytes: u64,
        ceiling_bytes: u64,
    },
}

impl MemoryPlanDecision {
    pub fn is_resident(self) -> bool {
        matches!(self, Self::Resident { .. })
    }

    /// Byte budget for resident expert layer stacks, when the plan streamed.
    pub fn expert_cache_bytes(self) -> Option<u64> {
        match self {
            Self::Stream {
                expert_cache_bytes, ..
            } => Some(expert_cache_bytes),
            _ => None,
        }
    }
}

impl MemoryPlan {
    /// Usable ceiling: `RAM - memory_plan_os_reserve_bytes(RAM)`, replaced by
    /// an explicit `AX_MLX_MEMORY_CEILING_BYTES` override bounded at
    /// `RAM - MEMORY_PLAN_OVERRIDE_RAM_RESERVE_BYTES`.
    ///
    /// The Metal `recommendedMaxWorkingSetSize` is deliberately NOT part of
    /// this bound. It is a vendor hint, not an allocator limit: a Flash Next
    /// 6-bit pack measured a 109.8 GiB peak RSS with no swap growth on an
    /// M5 Max whose recommendation is 107.5 GiB, and taking the minimum would
    /// have paged a pack that in fact runs resident. The recommendation is
    /// still reported, and [`Self::exceeds_recommended_working_set`] flags a
    /// plan that goes above it.
    pub fn ceiling_bytes(&self) -> Option<u64> {
        match self.ceiling_override_bytes {
            Some(override_bytes) => Some(match self.host_ram_bytes {
                Some(ram) => {
                    override_bytes.min(ram.saturating_sub(MEMORY_PLAN_OVERRIDE_RAM_RESERVE_BYTES))
                }
                None => override_bytes,
            }),
            None => self
                .host_ram_bytes
                .map(|ram| ram.saturating_sub(memory_plan_os_reserve_bytes(ram))),
        }
    }

    /// Whether the override, not the OS reserve, decided the ceiling.
    pub fn ceiling_is_overridden(&self) -> bool {
        self.ceiling_override_bytes.is_some()
    }

    /// True when the priced working set is above the Metal
    /// `recommendedMaxWorkingSetSize`.
    ///
    /// Advisory only: the decision keeps the RAM-derived ceiling, and the log
    /// line carries both numbers. `false` when either side is unknown.
    pub fn exceeds_recommended_working_set(&self) -> bool {
        self.metal_recommended_max_working_set_bytes
            .is_some_and(|recommended| self.resident_working_set_bytes() > recommended)
    }

    /// Everything that stays resident regardless of the expert decision.
    /// Prefill scratch, the MLX buffer cache, and the runtime reserve are
    /// part of the fixed cost: streaming cannot free them.
    pub fn fixed_bytes(&self) -> u64 {
        self.resident_non_expert_bytes
            .saturating_add(self.kv_bytes)
            .saturating_add(self.state_bytes)
            .saturating_add(self.prefill_scratch_bytes)
            .saturating_add(self.mlx_buffer_cache_bytes)
            .saturating_add(self.runtime_reserve_bytes)
    }

    /// Fixed part plus fully resident experts. The n-gram table is excluded:
    /// it is disk/virtual.
    pub fn resident_working_set_bytes(&self) -> u64 {
        self.fixed_bytes().saturating_add(self.expert_bytes)
    }

    /// Decision ladder: resident if the whole working set fits, stream if the
    /// fixed part plus the minimum scratch fits, refuse otherwise.
    pub fn decide(&self) -> MemoryPlanDecision {
        let fixed = self.fixed_bytes();
        let working = self.resident_working_set_bytes();
        let Some(ceiling) = self.ceiling_bytes() else {
            // Unknown capacity cannot prove residency: page instead of
            // risking an OOM, and leave the byte cap to the caller.
            return MemoryPlanDecision::Stream {
                working_set_bytes: working,
                expert_cache_bytes: 0,
                ceiling_bytes: None,
            };
        };
        if working <= ceiling {
            return MemoryPlanDecision::Resident {
                working_set_bytes: working,
                ceiling_bytes: ceiling,
            };
        }
        if fixed <= ceiling {
            return MemoryPlanDecision::Stream {
                working_set_bytes: working,
                expert_cache_bytes: ceiling - fixed,
                ceiling_bytes: Some(ceiling),
            };
        }
        MemoryPlanDecision::Refuse {
            fixed_bytes: fixed,
            ceiling_bytes: ceiling,
        }
    }

    /// One structured English log line: every component, the ceiling, the
    /// Metal recommendation, and the decision. Contains no paths or host
    /// names.
    ///
    /// When the priced working set is above the Metal recommendation the line
    /// carries a `warning=working-set-exceeds-metal-recommended(...)` field
    /// with both numbers; the decision itself is unchanged.
    pub fn log_line(&self) -> String {
        let decision = self.decide();
        let ceiling = self
            .ceiling_bytes()
            .map_or_else(|| "unknown".to_string(), bytes_label);
        let os_reserve = self.host_ram_bytes.map_or_else(
            || "unknown".to_string(),
            |ram| bytes_label(memory_plan_os_reserve_bytes(ram)),
        );
        let host_ram = self
            .host_ram_bytes
            .map_or_else(|| "unknown".to_string(), bytes_label);
        let metal = self
            .metal_recommended_max_working_set_bytes
            .map_or_else(|| "unknown".to_string(), bytes_label);
        let ceiling_source = if self.ceiling_is_overridden() {
            "override"
        } else if self.host_ram_bytes.is_some() {
            "ram"
        } else {
            "unknown"
        };
        let above_recommended = self.exceeds_recommended_working_set();
        let warning = if above_recommended {
            let recommended = self.metal_recommended_max_working_set_bytes;
            format!(
                " warning=working-set-exceeds-metal-recommended(working-set={} metal-recommended={})",
                bytes_label(self.resident_working_set_bytes()),
                recommended.map_or_else(|| "unknown".to_string(), bytes_label),
            )
        } else {
            String::new()
        };
        let (decision_label, expert_cache) = match decision {
            MemoryPlanDecision::Resident {
                working_set_bytes, ..
            } => (
                "resident",
                format!("working-set={}", bytes_label(working_set_bytes)),
            ),
            MemoryPlanDecision::Stream {
                working_set_bytes,
                expert_cache_bytes,
                ..
            } => (
                "stream",
                format!(
                    "working-set={} expert-cache={}",
                    bytes_label(working_set_bytes),
                    bytes_label(expert_cache_bytes)
                ),
            ),
            MemoryPlanDecision::Refuse {
                fixed_bytes,
                ceiling_bytes,
            } => (
                "refuse",
                format!(
                    "fixed={} ceiling={}",
                    bytes_label(fixed_bytes),
                    bytes_label(ceiling_bytes)
                ),
            ),
        };
        format!(
            "memory plan: resident-non-expert={} expert={} ngram-table={}(not-resident) \
             kv={} hybrid-state={} prefill-scratch={} mlx-buffer-cache={} runtime-reserve={} \
             host-ram={} metal-recommended={} metal-recommended-exceeded={} os-reserve={} \
             ceiling-source={} ceiling={} {}{} decision={}",
            bytes_label(self.resident_non_expert_bytes),
            bytes_label(self.expert_bytes),
            bytes_label(self.ngram_table_bytes),
            bytes_label(self.kv_bytes),
            bytes_label(self.state_bytes),
            bytes_label(self.prefill_scratch_bytes),
            bytes_label(self.mlx_buffer_cache_bytes),
            bytes_label(self.runtime_reserve_bytes),
            host_ram,
            metal,
            above_recommended,
            os_reserve,
            ceiling_source,
            ceiling,
            expert_cache,
            warning,
            decision_label,
        )
    }
}

/// Integer-only byte label (`10.51GiB`), so log lines never inherit float
/// rounding drift. Hundredths are rounded half-up, which turns a value one
/// byte under a boundary into the value operators quote (`115.20GiB`, not
/// `115.19GiB`).
fn bytes_label(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    let (unit_bytes, unit) = if bytes >= GIB {
        (GIB, "GiB")
    } else if bytes >= MIB {
        (MIB, "MiB")
    } else if bytes >= KIB {
        (KIB, "KiB")
    } else {
        return format!("{bytes}B");
    };
    let whole = bytes / unit_bytes;
    let hundredths = ((bytes % unit_bytes) * 100 + unit_bytes / 2) / unit_bytes;
    format!("{whole}.{hundredths:02}{unit}")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn footprint_estimate_adds_runtime_factor_and_floor() {
        let weights = 16 * 1024 * 1024 * 1024u64; // 16 GiB on disk
        let legacy = estimated_footprint_bytes(weights, None);
        assert_eq!(
            legacy,
            weights + weights / 8 + LOAD_FOOTPRINT_FIXED_FLOOR_BYTES
        );

        let kv_bytes = 3 * 1024 * 1024 * 1024u64;
        let with_kv = estimated_footprint_bytes(weights, Some(kv_bytes));
        assert_eq!(
            with_kv,
            weights + weights / 8 + kv_bytes + LOAD_FOOTPRINT_RUNTIME_FLOOR_BYTES
        );
    }

    #[test]
    fn footprint_estimate_saturates_instead_of_overflowing() {
        // The runtime factor alone saturates, and every addend after it is
        // saturating too — admission must never see a wrapped (tiny) peak.
        assert_eq!(estimated_footprint_bytes(u64::MAX, None), u64::MAX);
        assert_eq!(
            estimated_footprint_bytes(u64::MAX, Some(u64::MAX)),
            u64::MAX
        );
    }

    fn kv_geometry(value: serde_json::Value) -> ManifestKvGeometry {
        serde_json::from_value(value).expect("geometry should parse")
    }

    #[test]
    fn kv_pool_bytes_charges_dense_layers_at_pool() {
        let geometry = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2
        }));
        let per_token_layer = 2 * 128 * KV_CACHE_BYTES_PER_HEAD_ELEMENT;
        assert_eq!(
            estimated_kv_pool_bytes(&geometry, 16384),
            Some(4 * 16384 * per_token_layer)
        );
    }

    #[test]
    fn kv_pool_bytes_charges_sliding_layers_at_pool_and_skips_shared() {
        let geometry = kv_geometry(serde_json::json!({
            "layer_count": 6, "attention_head_dim": 128, "kv_head_count": 2,
            "global_head_dim": 256, "global_kv_head_count": 1,
            "sliding_window_size": 512,
            "layer_types": [
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ],
            "kv_shared_source_layers": {"3": 0}
        }));
        let pool = 16384u64;
        let sliding_per_token = 2 * 128 * KV_CACHE_BYTES_PER_HEAD_ELEMENT;
        let full_per_token = 256 * KV_CACHE_BYTES_PER_HEAD_ELEMENT;
        // Layers 0, 1, 4 are sliding: rings bound KV per request, so the
        // pool-wide worst case (many concurrent ≤window requests) is the
        // pool at the sliding head dim — the window never caps admission.
        // Layer 3 shares layer 0's KV and charges nothing; layers 2 and 5
        // are full attention at the pool, using the ISWA global head dim.
        let expected = 3 * pool * sliding_per_token + 2 * pool * full_per_token;
        assert_eq!(estimated_kv_pool_bytes(&geometry, pool), Some(expected));
    }

    #[test]
    fn kv_pool_bytes_honors_explicit_global_kv_heads_and_legacy_width() {
        let explicit = kv_geometry(serde_json::json!({
            "layer_count": 1, "attention_head_dim": 256, "kv_head_count": 8,
            "global_head_dim": 512, "global_kv_head_count": 1,
            "layer_types": ["full_attention"]
        }));
        let legacy = kv_geometry(serde_json::json!({
            "layer_count": 1, "attention_head_dim": 256, "kv_head_count": 8,
            "global_head_dim": 512,
            "layer_types": ["full_attention"]
        }));
        let pool = 1000;
        assert_eq!(
            estimated_kv_pool_bytes(&explicit, pool),
            Some(pool * 512 * KV_CACHE_BYTES_PER_HEAD_ELEMENT)
        );
        assert_eq!(
            estimated_kv_pool_bytes(&legacy, pool),
            Some(pool * (8 * 256) * KV_CACHE_BYTES_PER_HEAD_ELEMENT)
        );
    }

    #[test]
    fn kv_pool_bytes_infers_legacy_gemma4_global_heads_from_k_projection() {
        let geometry = kv_geometry(serde_json::json!({
            "layer_count": 2,
            "attention_head_dim": 256,
            "kv_head_count": 1,
            "global_head_dim": 512,
            "layer_types": ["sliding_attention", "full_attention"],
            "tensors": [{
                "role": "attention_k",
                "layer_index": 1,
                "shape": [512, 192]
            }]
        }));
        let pool = 1000;
        let expected_per_token = (256 + 512) * KV_CACHE_BYTES_PER_HEAD_ELEMENT;
        assert_eq!(
            estimated_kv_pool_bytes(&geometry, pool),
            Some(pool * expected_per_token)
        );
    }

    #[test]
    fn kv_pool_bytes_charges_only_hybrid_full_attention_layers() {
        let explicit = kv_geometry(serde_json::json!({
            "model_family": "qwen3_5",
            "layer_count": 8, "attention_head_dim": 256, "kv_head_count": 4,
            "linear_attention": {"full_attention_interval": 4}
        }));
        let per_token_layer = 4 * 256 * KV_CACHE_BYTES_PER_HEAD_ELEMENT;
        // Layers 3 and 7 (every 4th) keep KV; linear layers hold per-request
        // state covered by the runtime floor.
        assert_eq!(
            estimated_kv_pool_bytes(&explicit, 1000),
            Some(2 * 1000 * per_token_layer)
        );

        let family_default = kv_geometry(serde_json::json!({
            "model_family": "qwen3_5",
            "layer_count": 8, "attention_head_dim": 256, "kv_head_count": 4,
            "linear_attention": {"num_key_heads": 16}
        }));
        assert_eq!(
            estimated_kv_pool_bytes(&family_default, 1000),
            Some(2 * 1000 * per_token_layer)
        );
    }

    #[test]
    fn kv_pool_bytes_fails_open_on_unknown_geometry() {
        // An MLA pack whose latent width is missing is charged at the
        // equivalent full-attention width, never guessed at zero.
        let mla_unknown_width = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2,
            "mla_attention": {"kv_lora_rank": 512}
        }));
        assert_eq!(
            estimated_kv_pool_bytes(&mla_unknown_width, 1000),
            Some(4 * 1000 * 2 * 128 * KV_CACHE_BYTES_PER_HEAD_ELEMENT)
        );

        let diffusion = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2,
            "diffusion": {"canvas_size": 64}
        }));
        assert_eq!(estimated_kv_pool_bytes(&diffusion, 1000), None);

        let zero_layers = kv_geometry(serde_json::json!({
            "layer_count": 0, "attention_head_dim": 128, "kv_head_count": 2
        }));
        assert_eq!(estimated_kv_pool_bytes(&zero_layers, 1000), None);

        // Linear attention enabled on a family with no default interval and
        // no explicit interval: geometry is unknowable.
        let unresolved_hybrid = kv_geometry(serde_json::json!({
            "model_family": "not_a_hybrid_family",
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2,
            "linear_attention": {"num_key_heads": 16}
        }));
        assert_eq!(estimated_kv_pool_bytes(&unresolved_hybrid, 1000), None);

        // A sliding annotation charges the pool at the sliding head dim
        // whether or not the manifest carries a window (rings are a
        // per-request bound, so the window never lowers admission).
        let sliding_without_window = kv_geometry(serde_json::json!({
            "layer_count": 1, "attention_head_dim": 128, "kv_head_count": 2,
            "layer_types": ["sliding_attention"]
        }));
        assert_eq!(
            estimated_kv_pool_bytes(&sliding_without_window, 1000),
            Some(1000 * 2 * 128 * KV_CACHE_BYTES_PER_HEAD_ELEMENT)
        );
    }

    #[test]
    fn kv_pool_bytes_prices_the_mla_latent_cache() {
        // GLM/DeepSeek V3 MLA: one compressed latent (`kv_lora_rank`) plus the
        // decoupled RoPE slice per token per layer, single head.
        let mla = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2,
            "mla_attention": {"kv_lora_rank": 512, "qk_rope_head_dim": 64}
        }));
        assert_eq!(
            estimated_kv_pool_bytes(&mla, 1_000),
            Some(4 * 1_000 * (512 + 64) * MLA_CACHE_BYTES_PER_ELEMENT)
        );

        // DeepSeek V4: a single fused latent row of `head_dim` per token.
        let v4 = kv_geometry(serde_json::json!({
            "model_family": "deepseek_v4",
            "layer_count": 8, "attention_head_dim": 128, "kv_head_count": 1,
            "deepseek_v4": {"attention": {"head_dim": 576, "qk_rope_head_dim": 64}}
        }));
        assert_eq!(
            estimated_kv_pool_bytes(&v4, 1_000),
            Some(8 * 1_000 * 576 * MLA_CACHE_BYTES_PER_ELEMENT)
        );

        // Shared-KV layers allocate none of their own latent rows.
        let shared = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2,
            "mla_attention": {"kv_lora_rank": 512, "qk_rope_head_dim": 64},
            "kv_shared_source_layers": {"3": 2}
        }));
        assert_eq!(
            estimated_kv_pool_bytes(&shared, 1_000),
            Some(3 * 1_000 * (512 + 64) * MLA_CACHE_BYTES_PER_ELEMENT)
        );

        // The latent width does not scale with the conventional KV head count:
        // MLA keeps one compressed row per token, so a wider head table must
        // not inflate the latent charge.
        let wider_heads = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 32,
            "mla_attention": {"kv_lora_rank": 512, "qk_rope_head_dim": 64}
        }));
        assert_eq!(
            estimated_kv_pool_bytes(&wider_heads, 1_000),
            Some(4 * 1_000 * (512 + 64) * MLA_CACHE_BYTES_PER_ELEMENT)
        );
    }

    #[test]
    fn kv_pool_bytes_honors_layer_types_over_sentinel_hybrid_interval() {
        // Nemotron-H writes full_attention_interval=1 as a sentinel to disable
        // the Qwen gated-delta path; layer_types carries the real layer kinds.
        // The interval rule alone would charge every layer for per-token KV
        // (idx % 1 == 0 == interval - 1 is always true).
        let geometry = kv_geometry(serde_json::json!({
            "model_family": "nemotron_h",
            "layer_count": 8, "attention_head_dim": 128, "kv_head_count": 2,
            "linear_attention": {"full_attention_interval": 1, "num_key_heads": 8},
            "layer_types": [
                "mamba", "mamba", "moe", "attention",
                "mamba", "mamba", "moe", "attention"
            ]
        }));
        let per_token_layer = 2 * 128 * KV_CACHE_BYTES_PER_HEAD_ELEMENT;
        assert_eq!(
            estimated_kv_pool_bytes(&geometry, 1000),
            Some(2 * 1000 * per_token_layer)
        );
    }

    #[test]
    fn kv_pool_bytes_is_zero_for_a_zero_token_pool() {
        // A pool of zero tokens still has knowable geometry; the estimate is
        // exactly zero rather than unknown.
        let geometry = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2
        }));
        assert_eq!(estimated_kv_pool_bytes(&geometry, 0), Some(0));
    }

    #[test]
    fn kv_pool_bytes_saturates_per_layer_products() {
        // pool_tokens × bytes-per-token overflows u64; every layer saturates
        // and the running total stays at u64::MAX instead of wrapping.
        let geometry = kv_geometry(serde_json::json!({
            "layer_count": 2, "attention_head_dim": 128, "kv_head_count": 2
        }));
        assert_eq!(estimated_kv_pool_bytes(&geometry, u64::MAX), Some(u64::MAX));
    }
    #[test]
    fn zero_global_head_dimension_is_unknown_without_panicking() {
        let geometry = kv_geometry(serde_json::json!({
            "layer_count": 1, "attention_head_dim": 128, "kv_head_count": 2,
            "global_head_dim": 0, "global_kv_head_count": 1
        }));
        assert_eq!(estimated_kv_pool_bytes(&geometry, 1000), None);
    }

    // Priced memory plan -----------------------------------------------------

    const GIB: u64 = 1024 * 1024 * 1024;

    /// Metal `recommendedMaxWorkingSetSize` an M5 Max with 128 GiB unified
    /// memory reports (measured 2026-10-24). It is advisory: the plan records
    /// it for the warning and never clamps the ceiling to it, because a 6-bit
    /// Flash Next session ran resident at a higher peak RSS with no swap
    /// growth.
    const M5_MAX_RECOMMENDED_WORKING_SET_BYTES: u64 = 115_448_725_504;

    /// Flash Next 6-bit pack: 98.1 GB of experts, 11.3 GB of resident
    /// non-expert weights, and a 57.6 GB n-gram table served from disk. The
    /// table is disk/virtual, so the priced working set is 105.88 GiB.
    fn flash_next_6bit_plan(host_ram_gi_b: u64) -> MemoryPlan {
        flash_next_6bit_plan_with_metal(host_ram_gi_b, M5_MAX_RECOMMENDED_WORKING_SET_BYTES)
    }

    /// The same pack priced against an explicit Metal recommendation, so the
    /// value the host actually reports can be varied without touching the
    /// weight bytes.
    fn flash_next_6bit_plan_with_metal(host_ram_gi_b: u64, metal_bytes: u64) -> MemoryPlan {
        MemoryPlan {
            resident_non_expert_bytes: 11_278_347_768,
            expert_bytes: 98_146_713_600,
            ngram_table_bytes: 57_600_276_480,
            kv_bytes: 805_306_368,
            // QSA index/pooled 125,829,120 + GDN state 115,458,048.
            state_bytes: 125_829_120 + 115_458_048,
            prefill_scratch_bytes: 0,
            host_ram_bytes: Some(host_ram_gi_b * GIB),
            metal_recommended_max_working_set_bytes: Some(metal_bytes),
            ..MemoryPlan::default()
        }
    }

    fn mxfp4_plan(host_ram_gi_b: u64) -> MemoryPlan {
        MemoryPlan {
            resident_non_expert_bytes: 10_455_888_888,
            expert_bytes: 64_172_851_200,
            kv_bytes: 805_306_368,
            state_bytes: 125_829_120 + 115_458_048,
            host_ram_bytes: Some(host_ram_gi_b * GIB),
            metal_recommended_max_working_set_bytes: Some(M5_MAX_RECOMMENDED_WORKING_SET_BYTES),
            ..MemoryPlan::default()
        }
    }

    /// `RAM - max(8 GiB, RAM/10)` on a 128 GiB host: 115.20 GiB.
    const CEILING_128_GIB_HOST_BYTES: u64 = 123_695_058_125;

    #[test]
    fn priced_plan_keeps_flash_next_6bit_resident_on_128gib() {
        let plan = flash_next_6bit_plan(128);
        let decision = plan.decide();
        assert!(
            decision.is_resident(),
            "6-bit Flash Next must stay resident: {decision:?} in {}",
            plan.log_line()
        );
        // The 57.6 GB table is priced but never charged to the working set.
        assert_eq!(
            plan.resident_working_set_bytes(),
            plan.fixed_bytes() + plan.expert_bytes
        );
        assert_eq!(plan.fixed_bytes(), 15_546_166_776);
        assert_eq!(plan.resident_working_set_bytes(), 113_692_880_376);
        // The ceiling comes from RAM alone and sits ABOVE the Metal
        // recommendation: that is the whole point of not taking the minimum.
        assert_eq!(plan.ceiling_bytes(), Some(CEILING_128_GIB_HOST_BYTES));
        assert_eq!(
            plan.ceiling_bytes(),
            Some(128 * GIB - memory_plan_os_reserve_bytes(128 * GIB))
        );
        assert!(plan.ceiling_bytes() > plan.metal_recommended_max_working_set_bytes);
        assert!(plan.resident_working_set_bytes() < CEILING_128_GIB_HOST_BYTES);
        // A resident decision reports the expert cache as absent.
        assert_eq!(decision.expert_cache_bytes(), None);
        let line = plan.log_line();
        for component in [
            "resident-non-expert=10.50GiB",
            "expert=91.41GiB",
            "ngram-table=53.64GiB(not-resident)",
            "host-ram=128.00GiB",
            "metal-recommended=107.52GiB",
            "metal-recommended-exceeded=false",
            "os-reserve=12.80GiB",
            "ceiling-source=ram",
            "ceiling=115.20GiB",
            "working-set=105.88GiB",
            "decision=resident",
        ] {
            assert!(line.contains(component), "missing {component:?} in {line}");
        }
        assert!(!line.contains("warning="), "unexpected warning in {line}");
        // Dropping the disk-served table from the pricing cannot change the
        // decision, which is what makes it safe to exclude.
        let without_ngram = MemoryPlan {
            ngram_table_bytes: 0,
            ..plan
        };
        assert_eq!(without_ngram.decide(), decision);
    }

    /// The priced working set is 105.88 GiB and the M5 Max recommends
    /// 107.52 GiB, so this pack does not warn on that host. A pack whose
    /// working set is above the recommendation keeps its decision, and the
    /// log line carries both numbers.
    #[test]
    fn priced_plan_warns_above_the_metal_recommendation_without_changing_the_decision() {
        let fits = flash_next_6bit_plan(128);
        assert!(!fits.exceeds_recommended_working_set());
        assert!(fits.log_line().contains("metal-recommended-exceeded=false"));

        let above = flash_next_6bit_plan_with_metal(128, 100 * GIB);
        assert!(above.exceeds_recommended_working_set());
        assert!(
            above.decide().is_resident(),
            "the advisory warning must not page a pack that fits RAM: {}",
            above.log_line()
        );
        let line = above.log_line();
        assert!(line.contains("metal-recommended=100.00GiB"), "{line}");
        assert!(line.contains("metal-recommended-exceeded=true"), "{line}");
        assert!(
            line.contains(
                "warning=working-set-exceeds-metal-recommended(\
                 working-set=105.88GiB metal-recommended=100.00GiB)"
            ),
            "{line}"
        );
        assert!(line.contains("decision=resident"), "{line}");
    }

    #[test]
    fn priced_plan_keeps_mxfp4_resident() {
        let plan = mxfp4_plan(128);
        let decision = plan.decide();
        assert!(
            decision.is_resident(),
            "MXFP4 Flash Next must stay resident: {}",
            plan.log_line()
        );
        // 73.48 GiB of working set against a 115.20 GiB ceiling, below the
        // 107.52 GiB recommendation, so no advisory warning fires either.
        assert_eq!(plan.resident_working_set_bytes(), 78_896_559_096);
        assert!(!plan.exceeds_recommended_working_set());
        assert!(plan.log_line().contains("ceiling=115.20GiB"));
        assert!(plan.log_line().contains("decision=resident"));
    }

    #[test]
    fn priced_plan_streams_when_experts_do_not_fit() {
        // Both packs on a 64 GiB host: the fixed part fits the 56 GiB
        // ceiling, the experts do not, so the plan pages them with the
        // remainder.
        for (plan, expected_cache) in [
            (flash_next_6bit_plan(64), 44_583_375_368u64),
            (mxfp4_plan(64), 45_405_834_248u64),
        ] {
            let ceiling = plan.ceiling_bytes().expect("64 GiB host has a ceiling");
            assert_eq!(
                ceiling,
                64 * GIB - memory_plan_os_reserve_bytes(64 * GIB),
                "the 8 GiB reserve floor binds on a 64 GiB host"
            );
            assert!(plan.fixed_bytes() <= ceiling);
            assert!(plan.resident_working_set_bytes() > ceiling);
            let decision = plan.decide();
            assert!(
                matches!(
                    decision,
                    MemoryPlanDecision::Stream {
                        expert_cache_bytes,
                        ceiling_bytes: Some(reported),
                        ..
                    } if reported == ceiling && expert_cache_bytes == ceiling - plan.fixed_bytes()
                ),
                "expected a stream decision against the {ceiling} byte ceiling, got {decision:?}"
            );
            assert_eq!(decision.expert_cache_bytes(), Some(expected_cache));
            assert!(plan.log_line().contains("decision=stream"));
        }
    }

    #[test]
    fn priced_plan_refuses_when_fixed_part_does_not_fit() {
        // 60 GiB of resident non-expert weights plus the default 3 GiB of
        // runtime reserves cannot fit a 64 GiB host's 56 GiB ceiling even
        // with every expert streamed.
        let plan = MemoryPlan {
            resident_non_expert_bytes: 60 * GIB,
            expert_bytes: 10 * GIB,
            host_ram_bytes: Some(64 * GIB),
            metal_recommended_max_working_set_bytes: Some(112 * GIB),
            ..MemoryPlan::default()
        };
        let ceiling = plan.ceiling_bytes().expect("known ceiling");
        assert_eq!(ceiling, 56 * GIB);
        assert!(plan.fixed_bytes() > ceiling);
        let decision = plan.decide();
        assert!(
            matches!(
                decision,
                MemoryPlanDecision::Refuse {
                    fixed_bytes,
                    ceiling_bytes,
                } if fixed_bytes == plan.fixed_bytes() && ceiling_bytes == ceiling
            ),
            "expected a refusal, got {decision:?}"
        );
        assert!(plan.log_line().contains("decision=refuse"));
    }

    #[test]
    fn priced_plan_pages_when_capacity_is_unknown() {
        // Neither probe succeeded and no override was given: residency cannot
        // be proven, so the plan pages without imposing a byte cap.
        let plan = MemoryPlan {
            resident_non_expert_bytes: 1,
            expert_bytes: 1,
            ..MemoryPlan::default()
        };
        assert_eq!(plan.ceiling_bytes(), None);
        let decision = plan.decide();
        assert!(
            matches!(
                decision,
                MemoryPlanDecision::Stream {
                    ceiling_bytes: None,
                    expert_cache_bytes: 0,
                    ..
                }
            ),
            "expected an uncapped stream decision, got {decision:?}"
        );
        assert!(plan.log_line().contains("ceiling-source=unknown"));
    }

    #[test]
    fn ceiling_override_parses_only_positive_integers() {
        assert_eq!(parse_memory_plan_ceiling_override(None), None);
        for raw in ["", "   ", "0", "-1", "abc", "1.5", "115GiB"] {
            assert_eq!(
                parse_memory_plan_ceiling_override(Some(raw)),
                None,
                "{raw:?} must not become a ceiling"
            );
        }
        assert_eq!(
            parse_memory_plan_ceiling_override(Some("107374182400")),
            Some(100 * GIB)
        );
        // Surrounding whitespace is trimmed before the parse.
        assert_eq!(
            parse_memory_plan_ceiling_override(Some(" 133143986176\n")),
            Some(124 * GIB)
        );
    }

    #[test]
    fn ceiling_override_still_keeps_four_gib_of_ram() {
        // A 100 GiB override on a 128 GiB host stands: the operator can move
        // the ceiling in either direction.
        let tightened = MemoryPlan {
            ceiling_override_bytes: Some(100 * GIB),
            ..flash_next_6bit_plan(128)
        };
        assert_eq!(tightened.ceiling_bytes(), Some(100 * GIB));
        assert!(tightened.ceiling_is_overridden());
        let line = tightened.log_line();
        assert!(line.contains("ceiling-source=override"), "{line}");
        assert!(line.contains("ceiling=100.00GiB"), "{line}");
        // 105.88 GiB no longer fits the tighter ceiling, so the experts page
        // with the remainder instead of the pack being refused.
        let decision = tightened.decide();
        assert_eq!(
            decision.expert_cache_bytes(),
            Some(100 * GIB - tightened.fixed_bytes())
        );
        assert!(line.contains("decision=stream"), "{line}");

        // An override above `RAM - 4 GiB` is clamped there: the last 4 GiB
        // stay with the kernel and the window server.
        let raised = MemoryPlan {
            ceiling_override_bytes: Some(200 * GIB),
            ..flash_next_6bit_plan(128)
        };
        assert_eq!(raised.ceiling_bytes(), Some(128 * GIB - 4 * GIB));
        assert!(raised.decide().is_resident());

        // A host smaller than the fixed 4 GiB reserve saturates at zero
        // instead of wrapping into a huge ceiling.
        let tiny = MemoryPlan {
            host_ram_bytes: Some(2 * GIB),
            ceiling_override_bytes: Some(8 * GIB),
            ..MemoryPlan::default()
        };
        assert_eq!(tiny.ceiling_bytes(), Some(0));

        // With RAM unknown an explicit override is the only capacity the plan
        // has, and it is used as given.
        let unknown_ram = MemoryPlan {
            ceiling_override_bytes: Some(96 * GIB),
            ..MemoryPlan::default()
        };
        assert_eq!(unknown_ram.ceiling_bytes(), Some(96 * GIB));
        assert!(unknown_ram.log_line().contains("ceiling-source=override"));
        assert!(unknown_ram.log_line().contains("os-reserve=unknown"));
    }

    #[test]
    fn memory_plan_ceiling_and_os_reserve_use_integer_arithmetic() {
        // 8 GiB floor dominates on small hosts; a tenth of RAM dominates on
        // large ones.
        assert_eq!(memory_plan_os_reserve_bytes(32 * GIB), 8 * GIB);
        assert_eq!(memory_plan_os_reserve_bytes(64 * GIB), 8 * GIB);
        assert_eq!(memory_plan_os_reserve_bytes(80 * GIB), 8 * GIB);
        assert_eq!(memory_plan_os_reserve_bytes(128 * GIB), 128 * GIB / 10);
        assert_eq!(memory_plan_os_reserve_bytes(u64::MAX), u64::MAX / 10);
        // The Metal recommendation never bounds the ceiling, however low it
        // is: only RAM and an explicit override do.
        let with_low_recommendation = MemoryPlan {
            host_ram_bytes: Some(128 * GIB),
            metal_recommended_max_working_set_bytes: Some(64 * GIB),
            ..MemoryPlan::default()
        };
        assert_eq!(
            with_low_recommendation.ceiling_bytes(),
            Some(CEILING_128_GIB_HOST_BYTES)
        );
        let no_recommendation = MemoryPlan {
            host_ram_bytes: Some(128 * GIB),
            ..MemoryPlan::default()
        };
        assert_eq!(
            no_recommendation.ceiling_bytes(),
            Some(CEILING_128_GIB_HOST_BYTES)
        );
        // Saturating sums never wrap to a tiny working set.
        let huge = MemoryPlan {
            resident_non_expert_bytes: u64::MAX,
            expert_bytes: u64::MAX,
            ..MemoryPlan::default()
        };
        assert_eq!(huge.fixed_bytes(), u64::MAX);
        assert_eq!(huge.resident_working_set_bytes(), u64::MAX);
        // A RAM figure below the reserve saturates to a zero ceiling.
        let too_small = MemoryPlan {
            host_ram_bytes: Some(1),
            ..MemoryPlan::default()
        };
        assert_eq!(too_small.ceiling_bytes(), Some(0));
    }
}
