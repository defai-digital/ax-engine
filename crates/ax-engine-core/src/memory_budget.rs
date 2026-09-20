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

use crate::model::{NativeDiffusionConfig, NativeLinearAttentionConfig, NativeMlaAttentionConfig};

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
/// MLA latent caches and diffusion follow different math, and a zero dim or
/// an unresolvable hybrid interval means the manifest predates this
/// estimator — in which case the caller falls back to the flat floor baked
/// into [`estimated_footprint_bytes`] rather than guessing.
///
/// The result is a pool-wide worst case for load admission. Returning `None`
/// (the unknown-geometry fallback) is NOT proof that a request or the whole
/// working set fits, and the returned bound is not a hard allocator limit.
pub fn estimated_kv_pool_bytes(geometry: &ManifestKvGeometry, pool_tokens: u64) -> Option<u64> {
    if geometry.mla_attention.is_enabled() || geometry.diffusion.is_enabled() {
        return None;
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
            // Hybrid linear-attention layers keep per-request state, not a
            // per-token cache; the runtime floor covers them.
            if u64::from(layer_idx) % interval != interval - 1 {
                continue;
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
        let mla = kv_geometry(serde_json::json!({
            "layer_count": 4, "attention_head_dim": 128, "kv_head_count": 2,
            "mla_attention": {"kv_lora_rank": 512}
        }));
        assert_eq!(estimated_kv_pool_bytes(&mla, 1000), None);

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
}
