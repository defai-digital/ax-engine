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

use crate::batched_kv_cache::BatchedKvCache;
use crate::batched_sampling::argmax_batched;
use crate::kv_cache::MlxKVCache;
use crate::model::{ModelConfig, decode_batched_forward};
use crate::weights::{LayerWeights, ModelWeights};

/// `AX_MLX_BATCHED_DECODE` — opt in to routing eligible greedy dense-decode
/// requests through a shared batched forward. **Default: OFF.** Experimental:
/// the batched path holds KV in the session rather than each request's
/// `MlxKVCache`, so it is not yet reconciled with the core's KV block
/// accounting and does not handle preemption of session-resident requests.
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
/// Kept separate from the master switch because sampled token-exactness is not
/// guaranteed by construction the way greedy's is: greedy only needs the batched
/// forward's per-row `argmax` to match the single forward, but sampling reads the
/// full logit tail (softmax / top-k / top-p), so a last-bit difference between
/// the batched GEMM's reduction order and the single-row path could flip a
/// sample. That parity is an empirical question only the sequential-oracle probe
/// (`batched_decode_e2e_probe` with `AX_SAMPLING`) can answer on real weights.
/// Until it does, sampled batching stays opt-in so it cannot perturb the proven
/// greedy path.
pub fn batched_decode_sampling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("AX_MLX_BATCHED_DECODE_SAMPLING").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

/// Whether a model can use the batched dense-decode path. Pure so it is
/// unit-testable. Conservative: only dense full-attention `qwen3` with no
/// speculative/compression/sliding features (the layers
/// [`crate::model::families::standard::layer_forward_batched`] supports).
#[allow(clippy::too_many_arguments)]
pub fn model_batched_eligible(
    model_family: &str,
    has_mtp: bool,
    is_diffusion: bool,
    kv_compression_on: bool,
    layer_windows: &[Option<usize>],
    layers: &[LayerWeights],
) -> bool {
    if has_mtp || is_diffusion || kv_compression_on {
        return false;
    }
    if model_family != "qwen3" {
        return false;
    }
    if layer_windows.iter().any(|w| w.is_some()) {
        return false;
    }
    layers.iter().all(|w| {
        w.router_proj.is_none()
            && w.linear_attn.is_none()
            && w.glm_mla_attn.is_none()
            && w.per_layer_gate.is_none()
            && w.layer_scalar.is_none()
            && w.q_proj.is_some()
            && w.k_proj.is_some()
            && w.v_proj.is_some()
            && w.o_proj.is_some()
    })
}

/// A batched decode cohort with continuous join/leave.
pub struct BatchedDecodeSession {
    cache: BatchedKvCache,
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
    /// cache.seq_len - 1` — seed everything but that last token, then the first
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
        let slot = self.cache.add_active_row();
        for layer in 0..self.cache.num_layers() {
            let (k, v) = prefill
                .peek_layer_kv(layer)
                .expect("prefilled full-attention layer KV");
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
        let logits = decode_batched_forward(cfg, weights, &self.cur, &mut self.cache);
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
        let logits = decode_batched_forward(cfg, weights, &self.cur, &mut self.cache);
        Some((self.slot_req.clone(), logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_eligibility_guards() {
        let no_windows: [Option<usize>; 0] = [];
        // Dense qwen3 with no special features and no layers → eligible (vacuous).
        assert!(model_batched_eligible(
            "qwen3",
            false,
            false,
            false,
            &no_windows,
            &[]
        ));
        // Each disqualifier flips it off, independent of the layer set.
        assert!(!model_batched_eligible(
            "qwen3",
            true,
            false,
            false,
            &no_windows,
            &[]
        )); // MTP
        assert!(!model_batched_eligible(
            "qwen3",
            false,
            true,
            false,
            &no_windows,
            &[]
        )); // diffusion
        assert!(!model_batched_eligible(
            "qwen3",
            false,
            false,
            true,
            &no_windows,
            &[]
        )); // kv compression
        assert!(!model_batched_eligible(
            "gemma4",
            false,
            false,
            false,
            &no_windows,
            &[]
        )); // wrong family
        assert!(!model_batched_eligible(
            "qwen3",
            false,
            false,
            false,
            &[Some(4096)],
            &[]
        )); // sliding window
    }
}
