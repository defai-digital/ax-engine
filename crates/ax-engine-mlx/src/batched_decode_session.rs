//! Continuous batched-decode session — the slot lifecycle over a
//! [`BatchedKvCache`] that a runner drives to decode many requests together.
//!
//! Requests join via [`BatchedDecodeSession::add`] (seeded from their batch=1
//! prefill KV), advance together one token per [`BatchedDecodeSession::step`],
//! and leave via [`BatchedDecodeSession::remove`]. The active set is kept a
//! contiguous prefix of the cache, and the session's `slot → request` and
//! `slot → current-token` vectors are swap-removed in lockstep with the cache's
//! swap-remove, so the bookkeeping always mirrors the cache.
//!
//! [`BatchedDecodeSession::step`] decodes the whole cohort greedily (GPU
//! `argmax`). For mixed greedy/sampled cohorts the runner drives
//! [`BatchedDecodeSession::step_logits`] instead and resolves each row's token
//! itself with the request's own RNG (see [`crate::batched_sampling`]). Scope is
//! the same as [`crate::model::decode_batched_forward`]: full-attention dense
//! families, ragged positions supported.

use std::sync::OnceLock;

use mlx_sys::{MlxArray, slice};

use crate::batched_decode_certification::BatchedDecodeCertificationStatus;
use crate::batched_kv_cache::BatchedKvCache;
use crate::batched_linear_state::BatchedLinearState;
use crate::batched_sampling::argmax_batched;
use crate::kv_cache::MlxKVCache;
use crate::model::{ModelConfig, decode_batched_forward};
use crate::weights::{LayerWeights, ModelWeights};

/// `AX_MLX_BATCHED_DECODE` — opt in to routing eligible greedy dense-decode
/// requests through a shared batched forward. **Default: OFF.** Experimental:
/// the batched path holds KV in the session rather than each request's
/// `MlxKVCache`; the core keeps its logical block ledger, while runner state is
/// released whenever a request is preempted or reaches a terminal state.
pub fn batched_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("AX_MLX_BATCHED_DECODE").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

/// `AX_MLX_BATCHED_DECODE_SAMPLING` — additionally admit **host-sampled**
/// requests (temperature with top-k/top-p, or a repetition penalty) into the
/// batched cohort. **Default: OFF**, and only meaningful when
/// [`batched_decode_enabled`] is on.
///
/// Kept separate from the master switch because sampled equivalence is more
/// sensitive than greedy equivalence: greedy compares only the winning logit,
/// while sampling reads the full logit tail (softmax / top-k / top-p). Batched
/// GEMM reduction order can perturb either result, so parity is an empirical
/// question only the sequential-oracle probe
/// (`batched_decode_e2e_probe` with `AX_SAMPLING`) can answer on real weights.
/// Until it does, sampled batching stays behind an additional opt-in.
pub fn batched_decode_sampling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("AX_MLX_BATCHED_DECODE_SAMPLING").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

/// `AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED` — permit the experimental
/// batched forward even when its numerical-equivalence matrix has not passed.
/// **Default: OFF.** This is a diagnostic override, not a production setting.
pub fn batched_decode_allow_uncertified() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

/// Batch-size buckets for the batched decode forward (Phase 3 "B-buckets").
///
/// As requests join/leave, the active cohort size B fluctuates and each distinct
/// B is a distinct forward graph shape. Snapping B up to a bucket (and padding
/// the cohort to it) bounds the number of shapes MLX ever plans/encodes — the
/// batch-size analogue of the embed `max_len` buckets — and is the compile-cache
/// key once the batched forward is compiled per shape.
///
/// Powers of two: fewest distinct shapes, at most `active-1` pad rows.
///
/// Staged primitive: the helper is built + oracle-tested but **not yet wired**.
/// Wiring it means padding the cohort (KV cache + linear state + mask) to the
/// bucket each step, whose payoff only materializes once the batched forward is
/// compiled per shape — on the current eager path the pad compute is a likely
/// regression, so padding is deferred (see `docs/performance/README.md`). Landed
/// now, like `BatchedLinearState` was, so the bucket math is ready + verified.
#[allow(dead_code)]
pub(crate) const DEFAULT_DECODE_BATCH_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

/// Smallest bucket `>= active`; returns `active` unchanged when it exceeds every
/// bucket (unbounded tail) or `buckets` is empty. Pure — the env/caching wrapper
/// is [`decode_batch_bucket`].
#[allow(dead_code)]
fn snap_batch_to_buckets(active: usize, buckets: &[usize]) -> usize {
    buckets
        .iter()
        .copied()
        .find(|&b| active <= b)
        .unwrap_or(active)
}

/// `AX_MLX_DECODE_BATCH_BUCKETS` — snap the active batched-decode cohort size up
/// to a fixed bucket (padding the cohort) for graph-shape stability.
///
/// **Default: OFF.** Unlike the embed `max_len` buckets (default-on, where the
/// path is compiled so bucketing yields compile-cache hits), the batched decode
/// forward is still eager, so bucketing trades pad compute for shape stability
/// with no compile-cache payoff yet — an empirical question the A/B probe must
/// answer before any default-flip. `off`/`0` disable; `on`/`1` use the default
/// list; `2,4,8,16` a custom list.
#[allow(dead_code)] // staged: wired when the batched forward is compiled per shape
pub(crate) fn decode_batch_bucket(active: usize) -> usize {
    static BUCKETS: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    let buckets = BUCKETS.get_or_init(|| match std::env::var("AX_MLX_DECODE_BATCH_BUCKETS") {
        Err(_) => None, // default OFF
        Ok(raw) => {
            let t = raw.trim().to_ascii_lowercase();
            if t.is_empty() || matches!(t.as_str(), "0" | "false" | "off" | "no") {
                return None;
            }
            if matches!(t.as_str(), "1" | "true" | "on" | "yes" | "default") {
                return Some(DEFAULT_DECODE_BATCH_BUCKETS.to_vec());
            }
            let mut b: Vec<usize> = t
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .filter(|v| *v > 0)
                .collect();
            if b.is_empty() {
                return None;
            }
            b.sort_unstable();
            b.dedup();
            Some(b)
        }
    });
    match buckets.as_ref() {
        Some(b) => snap_batch_to_buckets(active, b),
        None => active,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BatchedDecodeCapabilities {
    certification_status: BatchedDecodeCertificationStatus,
    has_mtp: bool,
    is_diffusion: bool,
    has_sliding_window: bool,
    /// OR over layers: any full/sliding (non-linear, non-MLA) attention layer.
    has_full_attention: bool,
    has_moe: bool,
    /// Weight-level: every MoE layer is compatible with `moe_router_qwen3`
    /// (no gemma4 expert scale, no MXFP4). Independent of linear attention.
    batched_qwen3_moe_router: bool,
    has_linear_attention: bool,
    has_mla: bool,
    has_layer_gating: bool,
    /// Every layer has a usable attention projection layout for its kind
    /// (linear / MLA / split-QKV / packed-QKV), not a naive all-split-QKVO check.
    has_complete_attention_projections: bool,
}

/// True when a layer's attention projections are sufficient for the batched
/// path for that attention kind. Linear and MLA layers do not carry split
/// Q/K/V/O; requiring them blocked hybrid Qwen3-Next eligibility (Issue 1).
pub(crate) fn layer_has_usable_attention_projections(weights: &LayerWeights) -> bool {
    if weights.linear_attn.is_some() {
        return true;
    }
    if weights.glm_mla_attn.is_some() {
        // MLA has its own layout; structural gates still reject MLA for batched
        // decode. Completeness is independent of structural eligibility.
        return true;
    }
    let has_o = weights.o_proj.is_some();
    let has_split =
        weights.q_proj.is_some() && weights.k_proj.is_some() && weights.v_proj.is_some();
    let has_packed = weights.qkv_packed.is_some();
    // Q-only layers that share KV from a source layer (assistant shared-KV).
    let has_q_shared_kv =
        weights.q_proj.is_some() && weights.k_proj.is_none() && weights.v_proj.is_none();
    has_o && (has_split || has_packed || has_q_shared_kv)
}

/// Weight-level fail-closed check for the only MoE router the batched FFN runs
/// (`moe_router_qwen3` + non-MXFP4 expert stacks). Gemma4 and GPT-OSS leave
/// distinctive tensors that must never reach `ffn_batched`.
fn layer_moe_is_batched_qwen3_compatible(weights: &LayerWeights) -> bool {
    if weights.router_proj.is_none() {
        return true;
    }
    weights.mxfp4_gate_up_exps.is_none()
        && weights.mxfp4_down_exps.is_none()
        && weights.router_expert_scale.is_none()
        && weights.router_combined_scale.is_none()
        && (weights.gate_up_exps_packed.is_some()
            || weights.gate_exps.is_some()
            || weights.down_exps.is_some())
}

impl BatchedDecodeCapabilities {
    pub(crate) fn from_loaded_model(
        has_mtp: bool,
        is_diffusion: bool,
        layer_windows: &[Option<usize>],
        layers: &[LayerWeights],
        certification_status: BatchedDecodeCertificationStatus,
    ) -> Self {
        let has_moe = layers.iter().any(|weights| weights.router_proj.is_some());
        let has_linear_attention = layers.iter().any(|weights| weights.linear_attn.is_some());
        let has_mla = layers.iter().any(|weights| weights.glm_mla_attn.is_some());
        // Match StructuralCapabilities::from_layers: any non-linear, non-MLA
        // layer participates as full/sliding attention (OR, not "pure dense").
        // Projection completeness is a separate gate — incomplete dense layers
        // still count as full-attention kind.
        // Empty layer list is the synthetic "structure-only" test path and keeps
        // the historical pure-dense default when no hybrid markers exist.
        let has_full_attention = if layers.is_empty() {
            !has_linear_attention && !has_mla
        } else {
            layers
                .iter()
                .any(|weights| weights.linear_attn.is_none() && weights.glm_mla_attn.is_none())
        };
        Self {
            certification_status,
            has_mtp,
            is_diffusion,
            has_sliding_window: layer_windows.iter().any(Option::is_some),
            has_full_attention,
            has_moe,
            batched_qwen3_moe_router: has_moe
                && layers.iter().all(layer_moe_is_batched_qwen3_compatible),
            has_linear_attention,
            has_mla,
            has_layer_gating: layers
                .iter()
                .any(|weights| weights.per_layer_gate.is_some() || weights.layer_scalar.is_some()),
            has_complete_attention_projections: layers.is_empty()
                || layers.iter().all(layer_has_usable_attention_projections),
        }
    }

    pub(crate) fn eligible(self, allow_uncertified: bool) -> bool {
        self.rejection_reasons(allow_uncertified).is_empty()
    }

    pub(crate) fn rejection_reasons(self, allow_uncertified: bool) -> Vec<&'static str> {
        let mut reasons = Vec::new();
        if self.has_mtp {
            reasons.push("mtp");
        }
        // Structural gates via ADR-038 StructuralCapabilities (not family names).
        // Order matches the historical runner contract so route telemetry stays stable:
        // mtp → structural(diffusion first) → cert → remaining structure.
        // Attention *kind* is independent of whether projections are complete —
        // incomplete projections are a separate rejection
        // (`missing_attention_projection`).
        let structural = ax_engine_core::StructuralCapabilities {
            has_full_attention: self.has_full_attention,
            has_sliding_window: self.has_sliding_window,
            has_linear_attention: self.has_linear_attention,
            has_mla: self.has_mla,
            has_moe: self.has_moe,
            batched_qwen3_moe_router: self.batched_qwen3_moe_router,
            has_layer_gating: self.has_layer_gating,
            is_diffusion: self.is_diffusion,
            is_encoder_embed: false,
            is_multimodal_capable: false,
        };
        let structural_reasons = structural.batched_decode_structural_rejections();
        // Emit diffusion first (legacy position), then defer other structural
        // reasons until after certification for telemetry stability.
        if structural_reasons.contains(&"diffusion") {
            reasons.push("diffusion");
        }
        if !self.certification_status.is_certified() && !allow_uncertified {
            reasons.push(self.certification_status.route_reason());
        }
        for reason in structural_reasons {
            if reason == "diffusion" {
                continue;
            }
            reasons.push(reason);
        }
        if !self.has_complete_attention_projections {
            reasons.push("missing_attention_projection");
        }
        reasons
    }
}

/// A batched decode cohort with continuous join/leave.
pub struct BatchedDecodeSession {
    cache: BatchedKvCache,
    /// Per-row gated-delta recurrent state for hybrid (linear-attention) models,
    /// indexed by linear-layer order. `None` for pure full-attention models;
    /// lazily allocated on the first [`Self::add`] whose prefill reveals
    /// linear-attention layers. Swap-removed in lockstep with `cache`.
    lin_state: Option<BatchedLinearState>,
    /// `slot → request id`; `slot_req.len() == cache.batch()`.
    slot_req: Vec<u64>,
    /// `slot → current (last-produced) token`.
    cur: Vec<u32>,
}

impl BatchedDecodeSession {
    /// A session holding up to `max_batch` concurrent requests across
    /// `num_layers` layers, starting empty.
    pub fn new(num_layers: usize, max_batch: usize) -> Self {
        Self {
            cache: BatchedKvCache::with_capacity(num_layers, max_batch),
            lin_state: None,
            slot_req: Vec::with_capacity(max_batch),
            cur: Vec::with_capacity(max_batch),
        }
    }

    /// Number of active requests.
    pub fn len(&self) -> usize {
        self.cache.batch()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Max concurrent requests.
    pub fn capacity(&self) -> usize {
        self.cache.capacity()
    }

    /// Request ids currently decoding, in slot order.
    pub fn active_ids(&self) -> &[u64] {
        &self.slot_req
    }

    /// Admit `id` into the cohort, seeding its KV from a completed batch=1
    /// `prefill` cache and taking `first_token` as its current token (the token
    /// produced by prefill, to be fed on the next `step`). Panics if full or if
    /// `id` is already present.
    pub fn add(&mut self, id: u64, prefill: &MlxKVCache, first_token: u32) {
        self.add_with_seed_len(id, prefill, first_token, None);
    }

    /// Like [`Self::add`], but seed only the first `seed_len` KV tokens of each
    /// layer (default: the whole cache).
    ///
    /// The runner's per-request `MlxKVCache` is **warmed**: its last token is
    /// `first_token`'s KV, already appended by generation-state init. Re-feeding
    /// `first_token` would double it, so the runner passes `seed_len =
    /// cache.seq_len() - 1` — seed everything but that last token, then the first
    /// `step` re-appends it, reproducing the state exactly. Callers whose
    /// `first_token` is NOT yet in the cache (a fresh prefill, e.g. the probes)
    /// pass `None` and seed the whole cache.
    pub fn add_with_seed_len(
        &mut self,
        id: u64,
        prefill: &MlxKVCache,
        first_token: u32,
        seed_len: Option<usize>,
    ) {
        assert!(
            !self.slot_req.contains(&id),
            "request {id} already in session"
        );
        // Lazily provision the linear-attention recurrent-state store the first
        // time a hybrid model's prefill reveals gated-delta layers. Capacity
        // mirrors the KV cache; the store is indexed by linear-layer order.
        if self.lin_state.is_none() {
            let num_linear = (0..self.cache.num_layers())
                .filter(|&l| matches!(prefill.linear_state(l), (Some(_), Some(_))))
                .count();
            if num_linear > 0 {
                self.lin_state = Some(BatchedLinearState::with_capacity(
                    num_linear,
                    self.cache.capacity(),
                ));
            }
        }

        let slot = self.cache.add_active_row();
        // Linear-attention layers' recurrent state, in linear-layer order, seeded
        // into `lin_state` after the loop (one `add_row` per joining request).
        let mut lin_convs: Vec<MlxArray> = Vec::new();
        let mut lin_recs: Vec<MlxArray> = Vec::new();
        for layer in 0..self.cache.num_layers() {
            if let Some((k, v)) = prefill.peek_layer_kv(layer) {
                // Full-attention layer: seed KV (optionally the first `seed_len`).
                let full = k.shape()[2];
                let keep = seed_len
                    .map(|n| n.min(full as usize) as i32)
                    .unwrap_or(full);
                let (k, v) = if keep < full {
                    let heads = k.shape()[1];
                    let dim = k.shape()[3];
                    let ones = [1, 1, 1, 1];
                    (
                        slice(&k, &[0, 0, 0, 0], &[1, heads, keep, dim], &ones, None),
                        slice(&v, &[0, 0, 0, 0], &[1, heads, keep, dim], &ones, None),
                    )
                } else {
                    (k, v)
                };
                self.cache.seed_row_layer(layer, slot, &k, &v);
            } else if let (Some(conv), Some(rec)) = prefill.linear_state(layer) {
                // Linear-attention layer: the conv1d + recurrent state is a fixed
                // per-row tensor (no length-varying KV), so no `seed_len` slice.
                lin_convs.push(conv.clone());
                lin_recs.push(rec.clone());
            } else {
                panic!(
                    "batched decode: layer {layer} has neither full-attention KV nor linear state"
                );
            }
        }
        self.cache.materialize();
        if let Some(lin) = self.lin_state.as_mut() {
            let lin_slot = lin.add_row(&lin_convs, &lin_recs);
            debug_assert_eq!(lin_slot, slot, "linear-state slot must track the KV slot");
        }
        self.slot_req.push(id);
        self.cur.push(first_token);
    }

    /// Remove `id` from the cohort (e.g. it hit EOS / max tokens). The cache
    /// swaps its last active row into the freed slot; the session mirrors that
    /// with `swap_remove` so `slot_req`/`cur` stay aligned. No-op returning
    /// `false` if `id` is not active.
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(slot) = self.slot_req.iter().position(|&x| x == id) else {
            return false;
        };
        // Vec::swap_remove moves the LAST element into `slot` — exactly the
        // move BatchedKvCache::remove_active_row performs on the KV buffers.
        self.cache.remove_active_row(slot);
        if let Some(lin) = self.lin_state.as_mut() {
            // Same swap-remove on the recurrent state so its rows stay aligned.
            lin.remove_row(slot);
        }
        self.slot_req.swap_remove(slot);
        self.cur.swap_remove(slot);
        true
    }

    /// Override the token that will be fed for request `id` on the next
    /// [`Self::step`] — the engine's scheduler is the source of truth for which
    /// token each request decodes. Returns `false` if `id` is not active.
    pub fn set_current(&mut self, id: u64, token: u32) -> bool {
        match self.slot_req.iter().position(|&x| x == id) {
            Some(slot) => {
                self.cur[slot] = token;
                true
            }
            None => false,
        }
    }

    /// Decode one token for every active request (greedy). Returns `(id, token)`
    /// per active request, in slot order, and updates each slot's current token.
    /// Empty when the cohort is empty.
    pub fn step(&mut self, cfg: &ModelConfig, weights: &ModelWeights) -> Vec<(u64, u32)> {
        if self.cache.batch() == 0 {
            return Vec::new();
        }
        let logits = decode_batched_forward(
            cfg,
            weights,
            &self.cur,
            &mut self.cache,
            self.lin_state.as_mut(),
        );
        let toks = argmax_batched(&logits);
        self.cur.copy_from_slice(&toks);
        self.slot_req
            .iter()
            .zip(&toks)
            .map(|(&id, &t)| (id, t))
            .collect()
    }

    /// Run one batched forward and return the raw logits without sampling, for
    /// the runner's mixed greedy/sampled path: the caller resolves each row's
    /// token itself (GPU `argmax` for greedy rows, host `sample_categorical` for
    /// sampled rows, using the request's own RNG). Returns `(id-per-slot,
    /// logits[B, 1, vocab])` in slot order, or `None` when the cohort is empty.
    ///
    /// Unlike [`Self::step`] this does **not** advance `self.cur` — the runner is
    /// the source of truth for each request's next fed token and calls
    /// [`Self::set_current`] before the next step. Feeding the cohort therefore
    /// stays identical whether a step goes through `step` or `step_logits`.
    pub fn step_logits(
        &mut self,
        cfg: &ModelConfig,
        weights: &ModelWeights,
    ) -> Option<(Vec<u64>, MlxArray)> {
        if self.cache.batch() == 0 {
            return None;
        }
        let logits = decode_batched_forward(
            cfg,
            weights,
            &self.cur,
            &mut self.cache,
            self.lin_state.as_mut(),
        );
        Some((self.slot_req.clone(), logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_eligibility_guards() {
        let no_windows: [Option<usize>; 0] = [];
        let uncertified = BatchedDecodeCapabilities::from_loaded_model(
            false,
            false,
            &no_windows,
            &[],
            BatchedDecodeCertificationStatus::Missing,
        );
        assert!(!uncertified.eligible(false));
        assert!(uncertified.eligible(true));
        assert_eq!(
            uncertified.rejection_reasons(false),
            vec!["certification_missing"]
        );
        assert!(
            BatchedDecodeCapabilities::from_loaded_model(
                false,
                false,
                &no_windows,
                &[],
                BatchedDecodeCertificationStatus::Certified,
            )
            .eligible(false)
        );
        assert!(
            !BatchedDecodeCapabilities::from_loaded_model(
                true,
                false,
                &no_windows,
                &[],
                BatchedDecodeCertificationStatus::Missing,
            )
            .eligible(true)
        );
        assert!(
            !BatchedDecodeCapabilities::from_loaded_model(
                false,
                true,
                &no_windows,
                &[],
                BatchedDecodeCertificationStatus::Missing,
            )
            .eligible(true)
        );
        assert!(
            !BatchedDecodeCapabilities::from_loaded_model(
                false,
                false,
                &[Some(4096)],
                &[],
                BatchedDecodeCertificationStatus::Missing,
            )
            .eligible(true)
        );
    }

    #[test]
    fn model_rejection_reasons_preserve_all_independent_gates() {
        let reasons = BatchedDecodeCapabilities::from_loaded_model(
            true,
            true,
            &[Some(4096)],
            &[],
            BatchedDecodeCertificationStatus::Missing,
        )
        .rejection_reasons(false);

        assert_eq!(
            reasons,
            vec![
                "mtp",
                "diffusion",
                "certification_missing",
                "sliding_window"
            ]
        );
    }

    fn stub_layer_weights() -> crate::weights::LayerWeights {
        use mlx_sys::{MlxDtype, zeros};
        crate::weights::LayerWeights {
            attn_norm: zeros(&[16], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            ffn_norm: zeros(&[16], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: None,
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        }
    }

    fn dense_split_layer() -> crate::weights::LayerWeights {
        use crate::weights::QuantizedWeight;
        use mlx_sys::{MlxDtype, zeros};
        let mut layer = stub_layer_weights();
        // Minimal stubs — only `.is_some()` is checked for eligibility.
        let dummy = QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None);
        layer.q_proj = Some(dummy.clone());
        layer.k_proj = Some(dummy.clone());
        layer.v_proj = Some(dummy.clone());
        layer.o_proj = Some(dummy);
        layer
    }

    #[test]
    fn incomplete_projections_do_not_emit_spurious_no_attention() {
        // One dense layer with missing Q/K/V/O projections.
        // Attention *kind* is still full; incompleteness is a separate gate.
        let incomplete = stub_layer_weights();

        let reasons = BatchedDecodeCapabilities::from_loaded_model(
            false,
            false,
            &[],
            &[incomplete],
            BatchedDecodeCertificationStatus::Certified,
        )
        .rejection_reasons(false);

        assert!(
            reasons.contains(&"missing_attention_projection"),
            "expected missing_attention_projection, got {reasons:?}"
        );
        assert!(
            !reasons.contains(&"no_attention"),
            "incomplete dense projections must not be labeled no_attention: {reasons:?}"
        );
    }

    #[test]
    fn hybrid_linear_layers_do_not_fail_projection_completeness() {
        // Qwen3-Next hybrid: full-attention layer + linear-attention layer.
        // Linear layers load with split Q/K/V/O all None — that must not emit
        // missing_attention_projection (production runner bug fix).
        use crate::weights::LinearAttentionWeights;
        use mlx_sys::{MlxDtype, zeros};

        let full = dense_split_layer();
        let mut linear = stub_layer_weights();
        linear.linear_attn = Some(LinearAttentionWeights {
            in_proj_qkv: None,
            in_proj_z: None,
            in_proj_a: None,
            in_proj_b: None,
            in_proj_qkvz: None,
            in_proj_ba: None,
            conv1d_bias: None,
            d: None,
            conv1d_dense: zeros(&[1, 1, 1], MlxDtype::Float32, None),
            dt_bias: zeros(&[1], MlxDtype::Float32, None),
            a_log: zeros(&[1], MlxDtype::Float32, None),
            norm: zeros(&[1], MlxDtype::Float32, None),
            out_proj: crate::weights::QuantizedWeight::new(
                zeros(&[1, 1], MlxDtype::Float32, None),
                None,
                None,
            ),
        });

        let caps = BatchedDecodeCapabilities::from_loaded_model(
            false,
            false,
            &[],
            &[full, linear],
            BatchedDecodeCertificationStatus::Missing,
        );
        let without_allow = caps.rejection_reasons(false);
        assert!(
            !without_allow.contains(&"missing_attention_projection"),
            "hybrid must not fail projection completeness: {without_allow:?}"
        );
        assert!(
            without_allow.contains(&"certification_missing"),
            "uncertified hybrid still needs ALLOW or cert: {without_allow:?}"
        );
        assert!(
            caps.eligible(true),
            "hybrid + ALLOW_UNCERTIFIED must be eligible, got {:?}",
            caps.rejection_reasons(true)
        );
        // Hybrid must not emit no_attention (full+linear both present).
        assert!(
            !caps.rejection_reasons(true).contains(&"no_attention"),
            "hybrid must report attention structure, got {:?}",
            caps.rejection_reasons(true)
        );
    }

    #[test]
    fn hybrid_qwen3_moe_admitted_only_with_compatible_router_weights() {
        use crate::weights::QuantizedWeight;
        use mlx_sys::{MlxDtype, zeros};

        let mut full = dense_split_layer();
        let dummy = QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None);
        full.router_proj = Some(dummy.clone());
        full.gate_up_exps_packed = Some(dummy.clone());
        full.down_exps = Some(dummy);

        let caps = BatchedDecodeCapabilities::from_loaded_model(
            false,
            false,
            &[],
            &[full],
            BatchedDecodeCertificationStatus::Missing,
        );
        assert!(
            !caps.rejection_reasons(true).contains(&"moe"),
            "compatible MoE must not be moe-rejected under ALLOW: {:?}",
            caps.rejection_reasons(true)
        );
        assert!(
            caps.eligible(true),
            "qwen3-style MoE under ALLOW must be eligible: {:?}",
            caps.rejection_reasons(true)
        );

        // Gemma4-style expert scale must fail closed.
        let mut gemma_moe = dense_split_layer();
        let dummy = QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None);
        gemma_moe.router_proj = Some(dummy);
        gemma_moe.router_expert_scale = Some(zeros(&[8], MlxDtype::Float32, None));
        let gemma_caps = BatchedDecodeCapabilities::from_loaded_model(
            false,
            false,
            &[],
            &[gemma_moe],
            BatchedDecodeCertificationStatus::Missing,
        );
        assert!(
            gemma_caps.rejection_reasons(true).contains(&"moe"),
            "gemma4 MoE markers must reject even under ALLOW: {:?}",
            gemma_caps.rejection_reasons(true)
        );
    }

    #[test]
    fn uncertified_path_still_requires_allow_flag() {
        let full = dense_split_layer();
        let caps = BatchedDecodeCapabilities::from_loaded_model(
            false,
            false,
            &[],
            &[full],
            BatchedDecodeCertificationStatus::Missing,
        );
        assert!(!caps.eligible(false));
        assert!(caps.eligible(true));
        assert_eq!(caps.rejection_reasons(false), vec!["certification_missing"]);
    }

    #[test]
    fn snap_batch_to_buckets_rounds_up_to_the_next_bucket() {
        let d = DEFAULT_DECODE_BATCH_BUCKETS;
        assert_eq!(snap_batch_to_buckets(0, d), 1);
        assert_eq!(snap_batch_to_buckets(1, d), 1);
        assert_eq!(snap_batch_to_buckets(2, d), 2);
        assert_eq!(snap_batch_to_buckets(3, d), 4);
        assert_eq!(snap_batch_to_buckets(5, d), 8);
        assert_eq!(snap_batch_to_buckets(8, d), 8);
        assert_eq!(snap_batch_to_buckets(9, d), 16);
        // Beyond the largest bucket: identity (no unbounded padding).
        assert_eq!(snap_batch_to_buckets(65, d), 65);
        // Custom list.
        assert_eq!(snap_batch_to_buckets(10, &[2, 4, 8, 16]), 16);
        // Empty list: identity.
        assert_eq!(snap_batch_to_buckets(7, &[]), 7);
    }

    #[test]
    fn decode_batch_bucket_is_identity_by_default() {
        // Default OFF (no AX_MLX_DECODE_BATCH_BUCKETS in the test env): no padding.
        for b in [1usize, 3, 5, 7, 13, 100] {
            assert_eq!(decode_batch_bucket(b), b);
        }
    }
}
