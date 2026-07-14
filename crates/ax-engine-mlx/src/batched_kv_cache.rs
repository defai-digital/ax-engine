//! Batched (multi-row) full-attention KV cache — Phase 1 of batched MLX decode.
//!
//! ## Why this exists
//!
//! The production [`crate::kv_cache::MlxKVCache`] stores a single sequence
//! (`[1, n_kv_heads, seq_len, head_dim]`, `k_shape[0] == 1` asserted). The MLX
//! runner therefore decodes one request per forward pass and re-reads every
//! weight matrix from DRAM per request. Phase 0
//! (`.internal/analysis/batched-decode/`) proved that stacking concurrent
//! requests into one `[B, 1, hidden]` forward amortizes those weight reads
//! (~1.9× at B=2, ~2.4× at B=4 on M5 Max). To do that, the KV cache must hold
//! **B independent sequences of possibly-different lengths** in one set of
//! buffers.
//!
//! This module is that storage layer, built and tested **in isolation** — it is
//! deliberately NOT wired into the runner yet (that is Phase 2). Its single
//! correctness contract is the token-exact oracle in the tests: for any sequence
//! of seeds/appends, **row `r` of the batched cache holds byte-identical KV to a
//! single-sequence [`MlxKVCache`] fed the same tokens.**
//!
//! ## Layout
//!
//! Per layer: `k`, `v` of shape `[batch, n_kv_heads, capacity, head_dim]`, the
//! natural SDPA batch layout. Each row tracks its own logical length in
//! [`BatchedKvCache::lengths`]; rows may differ (a batch formed from prompts of
//! different lengths is ragged by a constant offset that grows with decode).
//! `layer_view` returns `[batch, n_kv_heads, max_len, head_dim]`; positions
//! beyond a row's length are stale/zero and MUST be masked by the caller's
//! attention (Phase 1 step 2) — this module only guarantees the `[0..len]`
//! region of each row.
//!
//! ## Growth
//!
//! Like `MlxKVCache`, buffers are pre-allocated in `KV_CHUNK_TOKENS` blocks and
//! written with `slice_update`; growth (reallocate + copy) happens at most once
//! per chunk across the whole batch.
//!
//! ## Out of scope (deliberately)
//!
//! MLA / linear-attention / TurboQuant / sliding-window / rollback / disk
//! serialization, and runner-side slot compaction. This is the full-attention
//! foundation those layer on top of; [`BatchedKvCache::reset_row`] is the
//! primitive the runner's slot reuse will build on. Decode advances all rows in
//! lockstep (`advance_all`); ragged lengths come only from differing seed
//! lengths.

use mlx_sys::{MlxArray, MlxDtype, contiguous, eval, slice, slice_update, zeros};

use crate::kv_cache::{KV_CHUNK_TOKENS, MlxKVCache};

fn chunk_ceiling(n: usize) -> usize {
    n.div_ceil(KV_CHUNK_TOKENS) * KV_CHUNK_TOKENS
}

/// Per-layer batched backing buffers.
#[derive(Clone)]
struct BatchedLayerKv {
    /// `[batch, n_kv_heads, capacity, head_dim]`.
    k: MlxArray,
    v: MlxArray,
    n_kv_heads: i32,
    head_dim: i32,
    capacity: usize,
    dtype: MlxDtype,
}

/// A batched full-attention KV cache holding `batch` independent sequences.
///
/// See the module docs for the layout, growth strategy, and correctness
/// contract. Layers are allocated lazily on first write, so the cache does not
/// need to know `n_kv_heads`/`head_dim`/`dtype` up front.
pub struct BatchedKvCache {
    num_layers: usize,
    /// Allocated buffer width (rows). The KV buffers are `[allocated, ...]`.
    allocated: usize,
    /// Logically active rows — always the contiguous prefix `[0..active)`, so a
    /// batch slice is contiguous. Decode operates on these rows only.
    active: usize,
    layers: Vec<Option<BatchedLayerKv>>,
    /// Logical token count per row; `lengths.len() == allocated`.
    lengths: Vec<usize>,
}

impl BatchedKvCache {
    /// A cache with all `batch` rows active from the start (fixed cohort). Rows
    /// start empty (length 0); buffers are allocated on the first seed/append.
    pub fn new(num_layers: usize, batch: usize) -> Self {
        assert!(batch > 0, "BatchedKvCache requires batch >= 1");
        Self {
            num_layers,
            allocated: batch,
            active: batch,
            layers: (0..num_layers).map(|_| None).collect(),
            lengths: vec![0; batch],
        }
    }

    /// A cache that can hold up to `max_batch` rows but starts with **none**
    /// active — for continuous batching, where requests join via
    /// [`Self::add_active_row`] and leave via [`Self::remove_active_row`].
    pub fn with_capacity(num_layers: usize, max_batch: usize) -> Self {
        assert!(max_batch > 0, "BatchedKvCache requires max_batch >= 1");
        Self {
            num_layers,
            allocated: max_batch,
            active: 0,
            layers: (0..num_layers).map(|_| None).collect(),
            lengths: vec![0; max_batch],
        }
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of active rows (the batch width decode runs over).
    pub fn batch(&self) -> usize {
        self.active
    }

    /// Maximum rows this cache can hold.
    pub fn capacity(&self) -> usize {
        self.allocated
    }

    /// Reserve a fresh active row (the contiguous next slot) and return its
    /// index; the caller then seeds it via [`Self::seed_row_layer`]. Panics if
    /// the cache is already at capacity.
    pub fn add_active_row(&mut self) -> usize {
        assert!(
            self.active < self.allocated,
            "BatchedKvCache at capacity {}",
            self.allocated
        );
        let slot = self.active;
        self.lengths[slot] = 0;
        self.active += 1;
        slot
    }

    /// Remove active row `slot`, keeping the active set a contiguous prefix by
    /// moving the last active row into `slot` (its KV is copied for every
    /// allocated layer). Returns the previous index of the row now living at
    /// `slot` (`Some(old_last)` when a move happened, `None` when `slot` was
    /// already the last active row) so the caller can update its slot→request
    /// map.
    pub fn remove_active_row(&mut self, slot: usize) -> Option<usize> {
        assert!(slot < self.active, "remove_active_row: slot out of range");
        let last = self.active - 1;
        let moved = if slot != last {
            let len = self.lengths[last];
            for layer in self.layers.iter_mut().flatten() {
                Self::copy_row(layer, last, slot, len);
            }
            self.lengths[slot] = len;
            Some(last)
        } else {
            None
        };
        self.lengths[last] = 0;
        self.active = last;
        moved
    }

    /// Copy row `src`'s first `len` tokens into row `dst` (same layer buffer).
    fn copy_row(layer: &mut BatchedLayerKv, src: usize, dst: usize, len: usize) {
        if len == 0 {
            return;
        }
        let ones = [1, 1, 1, 1];
        let (src, dst) = (src as i32, dst as i32);
        let take = |buf: &MlxArray| {
            slice(
                buf,
                &[src, 0, 0, 0],
                &[src + 1, layer.n_kv_heads, len as i32, layer.head_dim],
                &ones,
                None,
            )
        };
        let (k_row, v_row) = (take(&layer.k), take(&layer.v));
        let stop = [dst + 1, layer.n_kv_heads, len as i32, layer.head_dim];
        layer.k = slice_update(&layer.k, &k_row, &[dst, 0, 0, 0], &stop, &ones, None);
        layer.v = slice_update(&layer.v, &v_row, &[dst, 0, 0, 0], &stop, &ones, None);
    }

    /// Logical token count of row `r`.
    pub fn row_len(&self, row: usize) -> usize {
        self.lengths[row]
    }

    /// Per-row logical lengths for the active rows — the input a batched
    /// attention mask is built from. `lengths().len() == batch()`.
    pub fn lengths(&self) -> &[usize] {
        &self.lengths[..self.active]
    }

    /// Max logical length across active rows (the token extent of `layer_view`).
    pub fn max_len(&self) -> usize {
        self.lengths[..self.active]
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// Clear row `r`'s logical length so its slot can be reused. The backing
    /// bytes are left in place and overwritten by the next seed/append into
    /// that row (positions `[0..len]` are always rewritten before they are read
    /// back), so no wipe is needed. This is the primitive the runner's slot
    /// reuse (Phase 2) builds on.
    pub fn reset_row(&mut self, row: usize) {
        self.lengths[row] = 0;
    }

    /// Materialize all allocated K/V buffers so seed sources may be released.
    pub fn materialize(&self) {
        let arrays = self
            .layers
            .iter()
            .flatten()
            .flat_map(|layer| [&layer.k, &layer.v])
            .collect::<Vec<_>>();
        if !arrays.is_empty() {
            eval(&arrays);
        }
    }

    /// Advance every row's logical length by `n`. Call once per decode step
    /// after all layers have been appended, mirroring the single-sequence
    /// cache's `seq_len += 1`.
    pub fn advance_all(&mut self, n: usize) {
        for len in &mut self.lengths[..self.active] {
            *len += n;
        }
    }

    fn validate_kv(&self, k: &MlxArray, v: &MlxArray, expect_batch: i32) -> (i32, i32, MlxDtype) {
        let ks = k.shape();
        let vs = v.shape();
        assert_eq!(ks, vs, "batched KV write requires matching K/V shapes");
        assert_eq!(
            ks.len(),
            4,
            "batched KV write expects [batch, heads, tokens, dim]"
        );
        assert_eq!(ks[0], expect_batch, "batched KV write batch dim mismatch");
        assert!(
            ks[1] > 0 && ks[2] > 0 && ks[3] > 0,
            "batched KV write requires positive heads, tokens, dim"
        );
        assert_eq!(
            k.dtype(),
            v.dtype(),
            "batched KV write requires matching dtypes"
        );
        (ks[1], ks[3], k.dtype())
    }

    /// Allocate (or grow) layer `layer`'s buffers so their token capacity is at
    /// least `min_capacity`, preserving any existing contents.
    fn ensure_layer(
        &mut self,
        layer: usize,
        n_kv_heads: i32,
        head_dim: i32,
        dtype: MlxDtype,
        min_capacity: usize,
    ) {
        assert!(layer < self.num_layers, "layer {layer} out of bounds");
        let batch = self.allocated as i32;
        match &mut self.layers[layer] {
            None => {
                let capacity = chunk_ceiling(min_capacity.max(1));
                let shape = [batch, n_kv_heads, capacity as i32, head_dim];
                self.layers[layer] = Some(BatchedLayerKv {
                    k: zeros(&shape, dtype, None),
                    v: zeros(&shape, dtype, None),
                    n_kv_heads,
                    head_dim,
                    capacity,
                    dtype,
                });
            }
            Some(existing) => {
                assert_eq!(
                    existing.n_kv_heads, n_kv_heads,
                    "n_kv_heads changed for layer"
                );
                assert_eq!(existing.head_dim, head_dim, "head_dim changed for layer");
                assert_eq!(existing.dtype, dtype, "dtype changed for layer");
                if min_capacity > existing.capacity {
                    let new_capacity = chunk_ceiling(min_capacity);
                    let shape = [batch, n_kv_heads, new_capacity as i32, head_dim];
                    let old_cap = existing.capacity as i32;
                    let copy_stop = [batch, n_kv_heads, old_cap, head_dim];
                    let ones = [1, 1, 1, 1];
                    let k_new = slice_update(
                        &zeros(&shape, dtype, None),
                        &existing.k,
                        &[0, 0, 0, 0],
                        &copy_stop,
                        &ones,
                        None,
                    );
                    let v_new = slice_update(
                        &zeros(&shape, dtype, None),
                        &existing.v,
                        &[0, 0, 0, 0],
                        &copy_stop,
                        &ones,
                        None,
                    );
                    existing.k = k_new;
                    existing.v = v_new;
                    existing.capacity = new_capacity;
                }
            }
        }
    }

    /// Write one row's `[1, n_kv_heads, tokens, head_dim]` update into the
    /// backing buffer at `[row, :, pos.., :]`.
    fn write_row_block(
        layer: &mut BatchedLayerKv,
        row: usize,
        pos: usize,
        tokens: usize,
        new_k: &MlxArray,
        new_v: &MlxArray,
    ) {
        let row = row as i32;
        let start = [row, 0, pos as i32, 0];
        let stop = [
            row + 1,
            layer.n_kv_heads,
            (pos + tokens) as i32,
            layer.head_dim,
        ];
        let ones = [1, 1, 1, 1];
        layer.k = slice_update(&layer.k, new_k, &start, &stop, &ones, None);
        layer.v = slice_update(&layer.v, new_v, &start, &stop, &ones, None);
    }

    /// Seed row `row` of `layer` with a single-sequence prefill KV block
    /// `[1, n_kv_heads, len, head_dim]` (as produced by a batch=1 prefill),
    /// written at token positions `[0..len]`.
    ///
    /// Sets the row's logical length to `len`; asserts consistency if the row
    /// was already seeded to a different length by another layer this step.
    pub fn seed_row_layer(&mut self, layer: usize, row: usize, new_k: &MlxArray, new_v: &MlxArray) {
        assert!(
            row < self.allocated,
            "row {row} out of bounds for batch {}",
            self.allocated
        );
        let (n_kv_heads, head_dim, dtype) = self.validate_kv(new_k, new_v, 1);
        let len = new_k.shape()[2] as usize;
        assert!(len > 0, "seed requires at least one token");
        self.ensure_layer(layer, n_kv_heads, head_dim, dtype, len);
        let layer_kv = self.layers[layer]
            .as_mut()
            .expect("layer allocated by ensure_layer");
        Self::write_row_block(layer_kv, row, 0, len, new_k, new_v);
        self.lengths[row] = len;
    }

    /// Append one decode token per row to `layer` and return the attention view
    /// `[batch, n_kv_heads, max_len + 1, head_dim]` spanning through the
    /// just-written positions — mirroring the single-sequence `MlxKVCache::append`,
    /// whose returned view already includes the current token.
    ///
    /// `new_k`/`new_v` are `[batch, n_kv_heads, 1, head_dim]`; row `r`'s token is
    /// written at its current logical length `lengths[r]`. The view's key axis is
    /// `max(lengths) + 1`, so a row shorter than the max has stale/padding slots
    /// in `[lengths[r] + 1 .. max_len + 1]` that the caller MUST mask (build the
    /// mask with `valid_lengths[r] = lengths[r] + 1`, `key_len = max_len + 1`).
    ///
    /// Does NOT advance lengths — call [`Self::advance_all`]`(1)` once after all
    /// layers, mirroring the single cache's per-step `seq_len` bump.
    pub fn append_decode_layer(
        &mut self,
        layer: usize,
        new_k: &MlxArray,
        new_v: &MlxArray,
    ) -> (MlxArray, MlxArray) {
        let (n_kv_heads, head_dim, dtype) = self.validate_kv(new_k, new_v, self.active as i32);
        assert_eq!(
            new_k.shape()[2],
            1,
            "append_decode_layer writes exactly one token per row"
        );
        let view_len = self.max_len() + 1;
        self.ensure_layer(layer, n_kv_heads, head_dim, dtype, view_len);
        let ones = [1, 1, 1, 1];
        // Per-row write, because rows may sit at different logical lengths. A
        // fused scatter is a Phase 2 perf follow-up; correctness first.
        for row in 0..self.active {
            let pos = self.lengths[row];
            let r = row as i32;
            let row_k = slice(
                new_k,
                &[r, 0, 0, 0],
                &[r + 1, n_kv_heads, 1, head_dim],
                &ones,
                None,
            );
            let row_v = slice(
                new_v,
                &[r, 0, 0, 0],
                &[r + 1, n_kv_heads, 1, head_dim],
                &ones,
                None,
            );
            let layer_kv = self.layers[layer]
                .as_mut()
                .expect("layer allocated by ensure_layer");
            Self::write_row_block(layer_kv, row, pos, 1, &row_k, &row_v);
        }
        let layer_kv = self.layers[layer]
            .as_ref()
            .expect("layer allocated by ensure_layer");
        let stop = [self.active as i32, n_kv_heads, view_len as i32, head_dim];
        (
            slice(&layer_kv.k, &[0, 0, 0, 0], &stop, &ones, None),
            slice(&layer_kv.v, &[0, 0, 0, 0], &stop, &ones, None),
        )
    }

    /// The batched attention view of `layer`: `[batch, n_kv_heads, max_len,
    /// head_dim]`. Positions beyond a row's length are stale/zero and must be
    /// masked by the caller using [`Self::lengths`]. Returns `None` if the layer
    /// has not been written yet.
    pub fn layer_view(&self, layer: usize) -> Option<(MlxArray, MlxArray)> {
        let layer_kv = self.layers[layer].as_ref()?;
        let max_len = self.max_len();
        if max_len == 0 {
            return None;
        }
        let start = [0, 0, 0, 0];
        let stop = [
            self.active as i32,
            layer_kv.n_kv_heads,
            max_len as i32,
            layer_kv.head_dim,
        ];
        let ones = [1, 1, 1, 1];
        Some((
            slice(&layer_kv.k, &start, &stop, &ones, None),
            slice(&layer_kv.v, &start, &stop, &ones, None),
        ))
    }

    /// A single row's valid KV view `[1, n_kv_heads, row_len, head_dim]` — used
    /// by the token-exact oracle to compare row `r` against a single-sequence
    /// cache. Returns `None` for an empty row or unwritten layer.
    pub fn row_view(&self, layer: usize, row: usize) -> Option<(MlxArray, MlxArray)> {
        let layer_kv = self.layers[layer].as_ref()?;
        let len = self.lengths[row];
        if len == 0 {
            return None;
        }
        let r = row as i32;
        let start = [r, 0, 0, 0];
        let stop = [r + 1, layer_kv.n_kv_heads, len as i32, layer_kv.head_dim];
        let ones = [1, 1, 1, 1];
        Some((
            slice(&layer_kv.k, &start, &stop, &ones, None),
            slice(&layer_kv.v, &start, &stop, &ones, None),
        ))
    }

    /// Extract row `row`'s valid KV into a fresh single-sequence [`MlxKVCache`],
    /// byte-identical to the cache a per-request decode of that row would hold —
    /// the inverse of seeding a row from a single cache
    /// ([`crate::batched_decode_session::BatchedDecodeSession::add_with_seed_len`]).
    ///
    /// This is the mechanism a **preempted** session member needs to resume on
    /// the per-item decode path (Gate 2 of the batched-decode promotion): its KV
    /// lives in the session, so evicting it requires writing that KV back into
    /// its own `MlxKVCache`. Each layer's `[1, n_kv_heads, row_len, head_dim]`
    /// view is materialized contiguous and stored logically; `seq_len` is set to
    /// the row length. An empty row yields an empty cache (`seq_len == 0`).
    ///
    /// NOTE: the KV copy is exact (unit-tested), but the runner still owns the
    /// resume convention — which `seq_len`/feed-token to restore so the per-item
    /// path does not double the current token. That wiring is validated by the
    /// preemption harness on real weights, not here.
    // Not yet wired into the runner (Gate 2). Exercised only by the writeback
    // unit test, mirroring how the validity mask was proven before Phase 2.
    #[allow(dead_code)]
    pub fn writeback_row(&self, row: usize) -> MlxKVCache {
        let mut cache = MlxKVCache::new(self.num_layers());
        let len = self.row_len(row);
        if len == 0 {
            return cache;
        }
        for layer in 0..self.num_layers() {
            if let Some((k, v)) = self.row_view(layer, row) {
                let k = contiguous(&k, None);
                let v = contiguous(&v, None);
                eval(&[&k, &v]);
                cache.set_layer_kv_logical(layer, k, v, len);
            }
        }
        cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::MlxKVCache;
    use mlx_sys::{contiguous, eval};

    /// Materialize a (possibly strided) slice view into row-contiguous order so
    /// `data_f32()` reads logical order — mirrors `kv_cache`'s contiguous() call
    /// before `data_raw`.
    fn dense(a: &MlxArray) -> MlxArray {
        contiguous(a, None)
    }

    /// Row-major f32 contents of a (row) view.
    fn view_data(a: &MlxArray) -> Vec<f32> {
        let a = dense(a);
        eval(&[&a]);
        a.data_f32().to_vec()
    }

    /// Build a `[1, heads, tokens, dim]` f32 array from `data`. The caller must
    /// keep `data` alive until the array is materialized (`from_raw_data`
    /// borrows), which every test below does by holding its Vecs in `keep`.
    fn arr(data: &[f32], heads: i32, tokens: i32, dim: i32) -> MlxArray {
        assert_eq!(data.len(), (heads * tokens * dim) as usize);
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            &[1, heads, tokens, dim],
            MlxDtype::Float32,
        )
    }

    /// Deterministic distinct value for (row, token, head, dim) so cross-row or
    /// cross-token corruption is caught by the equality assertions.
    fn val(row: usize, token: usize, head: usize, dim: usize) -> f32 {
        ((row + 1) as f32) * 1000.0
            + (token as f32) * 10.0
            + (head as f32) * 0.1
            + (dim as f32) * 0.001
    }

    fn seq_block(
        row: usize,
        start_token: usize,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(heads * tokens * dim);
        // Layout [1, heads, tokens, dim]: head-major, then token, then dim.
        for h in 0..heads {
            for t in 0..tokens {
                for d in 0..dim {
                    out.push(val(row, start_token + t, h, d));
                }
            }
        }
        out
    }

    fn assert_arrays_eq(a: &MlxArray, b: &MlxArray) {
        let a = dense(a);
        let b = dense(b);
        eval(&[&a, &b]);
        assert_eq!(a.shape(), b.shape(), "shape mismatch");
        assert_eq!(a.data_f32(), b.data_f32(), "data mismatch");
    }

    /// THE oracle: a batched cache of B ragged-length rows holds byte-identical
    /// KV to B independent single-sequence `MlxKVCache`s fed the same tokens,
    /// across prefill seeding + several lockstep decode steps + a growth event.
    #[test]
    fn batched_rows_match_single_sequence_caches() {
        let heads = 2usize;
        let dim = 4usize;
        let layers = 3usize;
        // Ragged prefill lengths, one past a 256-token chunk (exercises a row
        // whose seed spans two chunks alongside short rows).
        let prefill = [5usize, 260usize, 1usize];
        let batch = prefill.len();

        let mut keep: Vec<Vec<f32>> = Vec::new();
        let mut batched = BatchedKvCache::new(layers, batch);
        let mut refs: Vec<MlxKVCache> = (0..batch).map(|_| MlxKVCache::new(layers)).collect();

        // Prefill: seed each row from its own single-sequence prefill KV.
        for (row, &len) in prefill.iter().enumerate() {
            for layer in 0..layers {
                let kd = seq_block(row, 0, len, heads, dim);
                let vd: Vec<f32> = kd.iter().map(|x| x + 0.5).collect();
                keep.push(kd);
                keep.push(vd);
                let k = arr(
                    keep[keep.len() - 2].as_slice(),
                    heads as i32,
                    len as i32,
                    dim as i32,
                );
                let v = arr(
                    keep[keep.len() - 1].as_slice(),
                    heads as i32,
                    len as i32,
                    dim as i32,
                );
                batched.seed_row_layer(layer, row, &k, &v);
                refs[row].append(layer, k, v);
            }
            refs[row].set_seq_len(len);
        }

        // Decode: 4 lockstep steps. Each step builds a [batch, heads, 1, dim]
        // token per layer; row r's token is at its own current length.
        let decode_steps = 4usize;
        for step in 0..decode_steps {
            for layer in 0..layers {
                // Batched input [batch, heads, 1, dim].
                let mut batched_data = Vec::with_capacity(batch * heads * dim);
                for (row, &prefill_len) in prefill.iter().enumerate().take(batch) {
                    let tok = prefill_len + step; // this row's current length
                    for h in 0..heads {
                        for d in 0..dim {
                            batched_data.push(val(row, tok, h, d));
                        }
                    }
                }
                keep.push(batched_data);
                let bk_slice = keep.last().unwrap().as_slice();
                let bk = MlxArray::from_raw_data(
                    bk_slice.as_ptr().cast(),
                    std::mem::size_of_val(bk_slice),
                    &[batch as i32, heads as i32, 1, dim as i32],
                    MlxDtype::Float32,
                );
                let bv_data: Vec<f32> = bk_slice.iter().map(|x| x + 0.5).collect();
                keep.push(bv_data);
                let bv_slice = keep.last().unwrap().as_slice();
                let bv = MlxArray::from_raw_data(
                    bv_slice.as_ptr().cast(),
                    std::mem::size_of_val(bv_slice),
                    &[batch as i32, heads as i32, 1, dim as i32],
                    MlxDtype::Float32,
                );
                batched.append_decode_layer(layer, &bk, &bv);

                // Reference: each row appends its own [1, heads, 1, dim] slice.
                for row in 0..batch {
                    let tok = prefill[row] + step;
                    let rk = seq_block(row, tok, 1, heads, dim);
                    let rv: Vec<f32> = rk.iter().map(|x| x + 0.5).collect();
                    keep.push(rk);
                    keep.push(rv);
                    let k = arr(keep[keep.len() - 2].as_slice(), heads as i32, 1, dim as i32);
                    let v = arr(keep[keep.len() - 1].as_slice(), heads as i32, 1, dim as i32);
                    refs[row].append(layer, k, v);
                }
            }
            batched.advance_all(1);
            for row in 0..batch {
                refs[row].set_seq_len(prefill[row] + step + 1);
            }
        }

        // Assert every row of every layer matches its reference cache exactly.
        for layer in 0..layers {
            for row in 0..batch {
                let expected_len = prefill[row] + decode_steps;
                assert_eq!(batched.row_len(row), expected_len);
                let (bk, bv) = batched.row_view(layer, row).expect("row has data");
                let (rk, rv) = refs[row].peek_layer_kv(layer).expect("ref has data");
                assert_arrays_eq(&bk, &rk);
                assert_arrays_eq(&bv, &rv);
            }
        }
    }

    /// `writeback_row` reconstructs a single-sequence `MlxKVCache` byte-identical
    /// to the per-request cache — the inverse of seeding, and the KV half of
    /// Gate-2 eviction (a preempted session member resuming per-item). Seeds
    /// ragged rows + lockstep decode, then writes each row back and compares its
    /// logical layer KV and `seq_len` to the reference single-sequence cache.
    #[test]
    fn writeback_row_reconstructs_single_sequence_cache() {
        let (heads, dim, layers) = (2usize, 4usize, 2usize);
        let prefill = [3usize, 6usize];
        let batch = prefill.len();
        let mut keep: Vec<Vec<f32>> = Vec::new();
        let mut batched = BatchedKvCache::new(layers, batch);
        let mut refs: Vec<MlxKVCache> = (0..batch).map(|_| MlxKVCache::new(layers)).collect();

        // Seed each row from its own single-sequence prefill KV.
        for (row, &len) in prefill.iter().enumerate() {
            for layer in 0..layers {
                let kd = seq_block(row, 0, len, heads, dim);
                let vd: Vec<f32> = kd.iter().map(|x| x + 0.5).collect();
                keep.push(kd);
                keep.push(vd);
                let k = arr(
                    keep[keep.len() - 2].as_slice(),
                    heads as i32,
                    len as i32,
                    dim as i32,
                );
                let v = arr(
                    keep[keep.len() - 1].as_slice(),
                    heads as i32,
                    len as i32,
                    dim as i32,
                );
                batched.seed_row_layer(layer, row, &k, &v);
                refs[row].append(layer, k, v);
            }
            refs[row].set_seq_len(len);
        }

        // Two lockstep decode steps.
        let steps = 2usize;
        for step in 0..steps {
            for layer in 0..layers {
                let mut bd = Vec::with_capacity(batch * heads * dim);
                for (row, &plen) in prefill.iter().enumerate().take(batch) {
                    let tok = plen + step;
                    for h in 0..heads {
                        for d in 0..dim {
                            bd.push(val(row, tok, h, d));
                        }
                    }
                }
                keep.push(bd);
                let bk_slice = keep.last().unwrap().as_slice();
                let bk = MlxArray::from_raw_data(
                    bk_slice.as_ptr().cast(),
                    std::mem::size_of_val(bk_slice),
                    &[batch as i32, heads as i32, 1, dim as i32],
                    MlxDtype::Float32,
                );
                let bv_data: Vec<f32> = bk_slice.iter().map(|x| x + 0.5).collect();
                keep.push(bv_data);
                let bv_slice = keep.last().unwrap().as_slice();
                let bv = MlxArray::from_raw_data(
                    bv_slice.as_ptr().cast(),
                    std::mem::size_of_val(bv_slice),
                    &[batch as i32, heads as i32, 1, dim as i32],
                    MlxDtype::Float32,
                );
                batched.append_decode_layer(layer, &bk, &bv);
                for row in 0..batch {
                    let tok = prefill[row] + step;
                    let rk = seq_block(row, tok, 1, heads, dim);
                    let rv: Vec<f32> = rk.iter().map(|x| x + 0.5).collect();
                    keep.push(rk);
                    keep.push(rv);
                    let k = arr(keep[keep.len() - 2].as_slice(), heads as i32, 1, dim as i32);
                    let v = arr(keep[keep.len() - 1].as_slice(), heads as i32, 1, dim as i32);
                    refs[row].append(layer, k, v);
                }
            }
            batched.advance_all(1);
            for row in 0..batch {
                refs[row].set_seq_len(prefill[row] + step + 1);
            }
        }

        // Writeback each row and compare to its single-sequence reference.
        for row in 0..batch {
            let expected_len = prefill[row] + steps;
            let wb = batched.writeback_row(row);
            assert_eq!(wb.seq_len(), expected_len, "row {row} writeback seq_len");
            for layer in 0..layers {
                let (wk, wv) = wb.peek_layer_kv(layer).expect("writeback has data");
                let (rk, rv) = refs[row].peek_layer_kv(layer).expect("ref has data");
                assert_arrays_eq(&wk, &rk);
                assert_arrays_eq(&wv, &rv);
            }
        }
    }

    /// Growth mid-decode must not corrupt any already-written row.
    #[test]
    fn growth_preserves_all_rows() {
        let (heads, dim, batch) = (1usize, 2usize, 2usize);
        let mut keep: Vec<Vec<f32>> = Vec::new();
        let mut cache = BatchedKvCache::new(1, batch);

        // Seed both rows near the chunk boundary so the first decode grows.
        let seed_len = KV_CHUNK_TOKENS - 1;
        for row in 0..batch {
            let kd = seq_block(row, 0, seed_len, heads, dim);
            let vd: Vec<f32> = kd.iter().map(|x| x + 0.5).collect();
            keep.push(kd);
            keep.push(vd);
            let k = arr(
                keep[keep.len() - 2].as_slice(),
                heads as i32,
                seed_len as i32,
                dim as i32,
            );
            let v = arr(
                keep[keep.len() - 1].as_slice(),
                heads as i32,
                seed_len as i32,
                dim as i32,
            );
            cache.seed_row_layer(0, row, &k, &v);
        }

        // Two decode steps: the second crosses the 256 boundary → growth.
        for step in 0..2usize {
            let mut bd = Vec::with_capacity(batch * heads * dim);
            for row in 0..batch {
                for h in 0..heads {
                    for d in 0..dim {
                        bd.push(val(row, seed_len + step, h, d));
                    }
                }
            }
            keep.push(bd);
            let s = keep.last().unwrap().as_slice();
            let bk = MlxArray::from_raw_data(
                s.as_ptr().cast(),
                std::mem::size_of_val(s),
                &[batch as i32, heads as i32, 1, dim as i32],
                MlxDtype::Float32,
            );
            cache.append_decode_layer(0, &bk, &bk);
            cache.advance_all(1);
        }

        // Every position of every row must read back its authored value.
        for row in 0..batch {
            let (bk, _bv) = cache.row_view(0, row).expect("row has data");
            let bk = dense(&bk);
            eval(&[&bk]);
            let data = bk.data_f32();
            let total_tokens = seed_len + 2;
            assert_eq!(data.len(), heads * total_tokens * dim);
            for h in 0..heads {
                for t in 0..total_tokens {
                    for d in 0..dim {
                        let idx = (h * total_tokens + t) * dim + d;
                        assert_eq!(data[idx], val(row, t, h, d), "row {row} tok {t} corrupted");
                    }
                }
            }
        }
    }

    /// `reset_row` frees a slot; re-seeding it must not disturb the other row.
    #[test]
    fn reset_row_isolates_slot_reuse() {
        let (heads, dim) = (1usize, 2usize);
        let mut keep: Vec<Vec<f32>> = Vec::new();
        let mut cache = BatchedKvCache::new(1, 2);

        for row in 0..2usize {
            let kd = seq_block(row, 0, 3, heads, dim);
            keep.push(kd);
            let k = arr(keep.last().unwrap().as_slice(), heads as i32, 3, dim as i32);
            cache.seed_row_layer(0, row, &k, &k);
        }

        // Reuse row 0 for a fresh, shorter sequence (distinct "row 5" values).
        cache.reset_row(0);
        assert_eq!(cache.row_len(0), 0);
        let kd = seq_block(5, 0, 2, heads, dim);
        keep.push(kd);
        let k = arr(keep.last().unwrap().as_slice(), heads as i32, 2, dim as i32);
        cache.seed_row_layer(0, 0, &k, &k);

        // Row 0 now holds the new sequence; row 1 is untouched.
        let (r0, _) = cache.row_view(0, 0).expect("row 0");
        let r0 = dense(&r0);
        eval(&[&r0]);
        assert_eq!(cache.row_len(0), 2);
        for h in 0..heads {
            for t in 0..2 {
                for d in 0..dim {
                    let idx = (h * 2 + t) * dim + d;
                    assert_eq!(r0.data_f32()[idx], val(5, t, h, d));
                }
            }
        }
        let (r1, _) = cache.row_view(0, 1).expect("row 1");
        let r1 = dense(&r1);
        eval(&[&r1]);
        for h in 0..heads {
            for t in 0..3 {
                for d in 0..dim {
                    let idx = (h * 3 + t) * dim + d;
                    assert_eq!(r1.data_f32()[idx], val(1, t, h, d), "row 1 disturbed");
                }
            }
        }
    }

    /// `layer_view` spans the ragged max length; `lengths` reports each row.
    #[test]
    fn layer_view_spans_ragged_max_len() {
        let (heads, dim) = (1usize, 2usize);
        let mut keep: Vec<Vec<f32>> = Vec::new();
        let mut cache = BatchedKvCache::new(1, 2);
        for (row, len) in [3usize, 7usize].into_iter().enumerate() {
            let kd = seq_block(row, 0, len, heads, dim);
            keep.push(kd);
            let k = arr(
                keep.last().unwrap().as_slice(),
                heads as i32,
                len as i32,
                dim as i32,
            );
            cache.seed_row_layer(0, row, &k, &k);
        }
        assert_eq!(cache.lengths(), &[3, 7]);
        assert_eq!(cache.max_len(), 7);
        let (k, _v) = cache.layer_view(0).expect("view");
        eval(&[&k]);
        assert_eq!(k.shape(), vec![2, heads as i32, 7, dim as i32]);
    }

    /// `append_decode_layer` returns a view `[B, H, max_len+1, D]` that includes
    /// the token it just wrote (mirroring single-cache `append`), so attention in
    /// the same layer can see the current token before `advance_all`.
    #[test]
    fn append_returns_view_including_current_token() {
        let (heads, dim, batch) = (1usize, 2usize, 2usize);
        let mut keep: Vec<Vec<f32>> = Vec::new();
        let mut cache = BatchedKvCache::new(1, batch);
        // Seed both rows to equal length 2.
        for row in 0..batch {
            let kd = seq_block(row, 0, 2, heads, dim);
            keep.push(kd);
            let k = arr(keep.last().unwrap().as_slice(), heads as i32, 2, dim as i32);
            cache.seed_row_layer(0, row, &k, &k);
        }
        // Append one token per row at position 2 (distinct "token 2" values).
        let mut bd = Vec::new();
        for row in 0..batch {
            for h in 0..heads {
                for d in 0..dim {
                    bd.push(val(row, 2, h, d));
                }
            }
        }
        keep.push(bd);
        let s = keep.last().unwrap().as_slice();
        let bk = MlxArray::from_raw_data(
            s.as_ptr().cast(),
            std::mem::size_of_val(s),
            &[batch as i32, heads as i32, 1, dim as i32],
            MlxDtype::Float32,
        );
        let (view_k, _view_v) = cache.append_decode_layer(0, &bk, &bk);
        let view_k = dense(&view_k);
        eval(&[&view_k]);
        // View spans max_len(=2) + 1 = 3 keys.
        assert_eq!(
            view_k.shape(),
            vec![batch as i32, heads as i32, 3, dim as i32]
        );
        // Position 2 of each row is the just-appended token.
        let data = view_k.data_f32();
        for row in 0..batch {
            for h in 0..heads {
                for d in 0..dim {
                    let idx = ((row * heads + h) * 3 + 2) * dim + d;
                    assert_eq!(
                        data[idx],
                        val(row, 2, h, d),
                        "row {row} current token missing"
                    );
                }
            }
        }
    }

    /// Continuous composition: `with_capacity` starts empty; `add_active_row`
    /// grows the active prefix; `remove_active_row` swaps the last active row
    /// into the freed slot (copying its KV), keeping the prefix contiguous.
    #[test]
    fn continuous_composition_add_remove_swaps_rows() {
        let (heads, dim) = (2usize, 4usize);
        let mut keep: Vec<Vec<f32>> = Vec::new();
        let mut cache = BatchedKvCache::with_capacity(1, 4);
        assert_eq!(cache.batch(), 0);
        assert_eq!(cache.capacity(), 4);

        // Add 3 rows with distinct "row ids" 10,11,12, length 3.
        for id in [10usize, 11, 12] {
            let slot = cache.add_active_row();
            let kd = seq_block(id, 0, 3, heads, dim);
            keep.push(kd);
            let k = arr(keep.last().unwrap(), heads as i32, 3, dim as i32);
            cache.seed_row_layer(0, slot, &k, &k);
        }
        assert_eq!(cache.batch(), 3);
        assert_eq!(cache.lengths(), &[3, 3, 3]);
        let row2_before = view_data(&cache.row_view(0, 2).expect("row2").0); // id 12

        // Remove the MIDDLE row (slot 1, id 11): last row (slot 2, id 12) moves
        // into slot 1.
        let moved = cache.remove_active_row(1);
        assert_eq!(moved, Some(2));
        assert_eq!(cache.batch(), 2);
        // Slot 1 now byte-identical to the old last row (id 12).
        assert_eq!(
            view_data(&cache.row_view(0, 1).expect("new slot1").0),
            row2_before
        );
        // Slot 0 (id 10) untouched.
        let (r0, _) = cache.row_view(0, 0).expect("row0");
        let r0 = view_data(&r0);
        for h in 0..heads {
            for t in 0..3 {
                for d in 0..dim {
                    assert_eq!(
                        r0[(h * 3 + t) * dim + d],
                        val(10, t, h, d),
                        "row0 disturbed"
                    );
                }
            }
        }

        // Add a new row (id 20) — reuses freed slot 2.
        let slot = cache.add_active_row();
        assert_eq!(slot, 2);
        let kd = seq_block(20, 0, 3, heads, dim);
        keep.push(kd);
        let k = arr(keep.last().unwrap(), heads as i32, 3, dim as i32);
        cache.seed_row_layer(0, slot, &k, &k);
        assert_eq!(cache.batch(), 3);

        // Decode one token into all active rows; view spans max_len+1 = 4.
        let mut bd = Vec::new();
        for row in 0..3usize {
            for h in 0..heads {
                for d in 0..dim {
                    bd.push(val(100 + row, 3, h, d));
                }
            }
        }
        keep.push(bd);
        let s = keep.last().unwrap().as_slice();
        let bk = MlxArray::from_raw_data(
            s.as_ptr().cast(),
            std::mem::size_of_val(s),
            &[3, heads as i32, 1, dim as i32],
            MlxDtype::Float32,
        );
        let (view_k, _) = cache.append_decode_layer(0, &bk, &bk);
        cache.advance_all(1);
        assert_eq!(cache.lengths(), &[4, 4, 4]);
        let vd = view_data(&view_k);
        // Each active row's freshly-appended token sits at position 3.
        for row in 0..3usize {
            for h in 0..heads {
                for d in 0..dim {
                    let idx = ((row * heads + h) * 4 + 3) * dim + d;
                    assert_eq!(vd[idx], val(100 + row, 3, h, d), "row {row} decode token");
                }
            }
        }
    }
}
