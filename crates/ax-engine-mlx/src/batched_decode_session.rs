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
//! Greedy only for now (the batched decode's proven path); per-request sampling
//! via [`crate::batched_sampling::sample_batched_host`] is a follow-up. Scope is
//! the same as [`crate::model::decode_batched_forward`]: full-attention dense
//! families, ragged positions supported.

use crate::batched_kv_cache::BatchedKvCache;
use crate::batched_sampling::argmax_batched;
use crate::kv_cache::MlxKVCache;
use crate::model::{ModelConfig, decode_batched_forward};
use crate::weights::ModelWeights;

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
        assert!(
            !self.slot_req.contains(&id),
            "request {id} already in session"
        );
        let slot = self.cache.add_active_row();
        for layer in 0..self.cache.num_layers() {
            let (k, v) = prefill
                .peek_layer_kv(layer)
                .expect("prefilled full-attention layer KV");
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
}
