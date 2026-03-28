//! CPU RAM KV cache.
//!
//! Stores K and V vectors for all layers in f32 CPU RAM.
//! Used when `CpuBackend` or `HybridCpuDecode` is selected.
//!
//! # v2 changes vs v1 `KvCache`
//! - Removed `advance_by()` — there is no GPU path to sync with.
//! - Removed Q8 quantized storage (deferred to v2.1).
//! - Removed paged KV / prefix cache integration (deferred to v2.1).
//! - `append()` now calls `ensure_capacity()` AND increments `seq_len` atomically,
//!   matching the GPU path's atomicity guarantee.

use super::page::{
    KvCacheConfig, KvDtype, PageAllocator, initial_token_capacity, planned_capacity_for_needed,
    recommended_page_size,
};

#[derive(Debug, Clone)]
pub struct CpuKvSnapshot {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    seq_len: usize,
    capacity: usize,
    max_seq_len: usize,
    token_stride: usize,
    page_size: usize,
    allocator: PageAllocator,
}

/// CPU-resident KV cache for all transformer layers.
///
/// Layout per layer: contiguous `[max_seq_len, n_kv_heads * head_dim]` f32 values.
/// Memory grows by `page_size` tokens on demand.
#[derive(Debug)]
pub struct CpuKv {
    /// Per-layer key cache (f32).
    k: Vec<Vec<f32>>,
    /// Per-layer value cache (f32).
    v: Vec<Vec<f32>>,
    /// Number of tokens currently stored.
    seq_len: usize,
    /// Current allocated capacity in tokens (grows by page_size).
    capacity: usize,
    /// Maximum sequence length (hard limit).
    max_seq_len: usize,
    /// Stride per token: n_kv_heads * head_dim.
    token_stride: usize,
    /// Growth increment in tokens.
    page_size: usize,
    /// Allocation statistics.
    allocator: PageAllocator,
}

impl CpuKv {
    /// Create a new CPU KV cache with default page size.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let page_size = recommended_page_size(n_kv_heads, head_dim);
        Self::with_config(&KvCacheConfig {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            page_size,
            dtype: KvDtype::F32,
        })
    }

    /// Create a CPU KV cache from explicit config.
    pub fn with_config(cfg: &KvCacheConfig) -> Self {
        let stride = cfg
            .n_kv_heads
            .checked_mul(cfg.head_dim)
            .expect("CPU KV token stride overflow");
        let page_size = cfg.page_size.max(1);
        let initial_cap = initial_token_capacity(page_size, cfg.max_seq_len);
        let layer_elems = initial_cap
            .checked_mul(stride)
            .expect("CPU KV initial allocation overflow");

        let mut allocator = PageAllocator::new();
        allocator.record_grow(1);

        Self {
            k: (0..cfg.n_layers)
                .map(|_| vec![0.0f32; layer_elems])
                .collect(),
            v: (0..cfg.n_layers)
                .map(|_| vec![0.0f32; layer_elems])
                .collect(),
            seq_len: 0,
            capacity: initial_cap,
            max_seq_len: cfg.max_seq_len,
            token_stride: stride,
            page_size,
            allocator,
        }
    }

    /// Number of tokens currently stored.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Current allocated capacity in tokens.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Stride per token (n_kv_heads * head_dim).
    pub fn token_stride(&self) -> usize {
        self.token_stride
    }

    /// Append a token's K and V vectors for a given layer, then advance seq_len.
    ///
    /// In v2, `append` is atomic: it appends the data AND increments seq_len in one
    /// call. Callers must call `append` for ALL layers before reading back via `k_slice`.
    ///
    /// In v1, `append` did not increment seq_len; callers had to call `advance()` separately.
    /// That pattern led to subtle bugs when seq_len and buffer contents fell out of sync.
    pub fn append_and_advance(&mut self, layer: usize, k_new: &[f32], v_new: &[f32]) {
        debug_assert_eq!(k_new.len(), self.token_stride);
        debug_assert_eq!(v_new.len(), self.token_stride);
        assert!(
            self.seq_len < self.max_seq_len,
            "CPU KV cache full: seq_len={} >= max={}",
            self.seq_len,
            self.max_seq_len
        );

        self.ensure_capacity();

        let offset = self.token_offset(self.seq_len);
        self.k[layer][offset..offset + self.token_stride].copy_from_slice(k_new);
        self.v[layer][offset..offset + self.token_stride].copy_from_slice(v_new);

        // Only advance after ALL layers have been written for this token.
        // Callers must call append_and_advance in layer order 0..n_layers,
        // then call finalize_token() once after the last layer.
        // See `finalize_token()`.
    }

    /// Append a batch of token-major K/V rows for one layer without advancing seq_len.
    ///
    /// `k_batch` and `v_batch` are laid out as `[n_tokens, token_stride]`.
    /// Callers must write every participating layer before calling `finalize_batch()`.
    pub fn append_batch(
        &mut self,
        layer: usize,
        k_batch: &[f32],
        v_batch: &[f32],
        n_tokens: usize,
    ) {
        assert!(self.seq_len + n_tokens <= self.max_seq_len);
        assert_eq!(k_batch.len(), n_tokens * self.token_stride);
        assert_eq!(v_batch.len(), n_tokens * self.token_stride);

        self.ensure_capacity_for(self.seq_len + n_tokens);
        let offset = self.token_offset(self.seq_len);
        let end = offset + n_tokens * self.token_stride;
        self.k[layer][offset..end].copy_from_slice(k_batch);
        self.v[layer][offset..end].copy_from_slice(v_batch);
    }

    /// Finalize storage of one token. Call after `append_and_advance` for all layers.
    pub fn finalize_token(&mut self) {
        assert!(self.seq_len < self.max_seq_len);
        self.seq_len += 1;
    }

    /// Finalize storage of a batch of tokens after all layer writes complete.
    pub fn finalize_batch(&mut self, n_tokens: usize) {
        assert!(self.seq_len + n_tokens <= self.max_seq_len);
        self.seq_len += n_tokens;
    }

    /// Reset the KV cache to zero tokens (keeps allocated memory).
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Rewind seq_len to `pos` without freeing memory.
    ///
    /// Used by speculative decoding to roll back the KV cache after a draft
    /// token is rejected. Data beyond `pos` is considered invalid but the
    /// allocated pages are retained to avoid reallocation on the next fill.
    pub fn truncate_to(&mut self, pos: usize) {
        assert!(
            pos <= self.seq_len,
            "truncate_to: pos={pos} > seq_len={}",
            self.seq_len
        );
        self.seq_len = pos;
    }

    /// K slice for layer up to `n` tokens (read-only view for attention).
    ///
    /// Returns `&k[layer][0..n * token_stride]`.
    pub fn k_slice(&self, layer: usize, n: usize) -> &[f32] {
        assert!(
            n <= self.seq_len,
            "k_slice: n={n} > seq_len={}",
            self.seq_len
        );
        let end = self.token_offset(n);
        assert!(
            end <= self.k[layer].len(),
            "k_slice: end={end} > capacity={} (capacity={} tokens)",
            self.k[layer].len(),
            self.capacity
        );
        &self.k[layer][..end]
    }

    /// K slice for a layer that already wrote the current in-progress token.
    ///
    /// CPU decode paths append K/V for one layer and immediately run attention
    /// before `finalize_token()` increments the global `seq_len`. In that
    /// window, the current layer is allowed to read `seq_len + 1` tokens.
    pub fn k_slice_including_current(&self, layer: usize, n: usize) -> &[f32] {
        assert!(
            n <= self.seq_len + 1,
            "k_slice_including_current: n={n} > seq_len+1={}",
            self.seq_len + 1
        );
        let end = self.token_offset(n);
        assert!(
            end <= self.k[layer].len(),
            "k_slice_including_current: end={end} > capacity={} (capacity={} tokens)",
            self.k[layer].len(),
            self.capacity
        );
        &self.k[layer][..end]
    }

    /// V slice for layer up to `n` tokens (read-only view for attention).
    pub fn v_slice(&self, layer: usize, n: usize) -> &[f32] {
        assert!(
            n <= self.seq_len,
            "v_slice: n={n} > seq_len={}",
            self.seq_len
        );
        let end = self.token_offset(n);
        assert!(
            end <= self.v[layer].len(),
            "v_slice: end={end} > capacity={} (capacity={} tokens)",
            self.v[layer].len(),
            self.capacity
        );
        &self.v[layer][..end]
    }

    /// V slice for a layer that already wrote the current in-progress token.
    pub fn v_slice_including_current(&self, layer: usize, n: usize) -> &[f32] {
        assert!(
            n <= self.seq_len + 1,
            "v_slice_including_current: n={n} > seq_len+1={}",
            self.seq_len + 1
        );
        let end = self.token_offset(n);
        assert!(
            end <= self.v[layer].len(),
            "v_slice_including_current: end={end} > capacity={} (capacity={} tokens)",
            self.v[layer].len(),
            self.capacity
        );
        &self.v[layer][..end]
    }

    /// K pointer for layer at token `seq_idx` (for single-token decode QKV write).
    pub fn k_token_mut(&mut self, layer: usize, seq_idx: usize) -> &mut [f32] {
        assert!(seq_idx < self.max_seq_len);
        self.ensure_capacity_for(seq_idx + 1);
        let offset = self.token_offset(seq_idx);
        &mut self.k[layer][offset..offset + self.token_stride]
    }

    /// V pointer for layer at token `seq_idx`.
    pub fn v_token_mut(&mut self, layer: usize, seq_idx: usize) -> &mut [f32] {
        assert!(seq_idx < self.max_seq_len);
        self.ensure_capacity_for(seq_idx + 1);
        let offset = self.token_offset(seq_idx);
        &mut self.v[layer][offset..offset + self.token_stride]
    }

    /// Allocation statistics.
    pub fn page_stats(&self) -> &PageAllocator {
        &self.allocator
    }

    /// Snapshot the full CPU KV state, including allocated pages.
    pub fn snapshot(&self) -> CpuKvSnapshot {
        CpuKvSnapshot {
            k: self.k.clone(),
            v: self.v.clone(),
            seq_len: self.seq_len,
            capacity: self.capacity,
            max_seq_len: self.max_seq_len,
            token_stride: self.token_stride,
            page_size: self.page_size,
            allocator: self.allocator.clone(),
        }
    }

    /// Restore a previously captured CPU KV snapshot.
    pub fn restore(&mut self, snapshot: &CpuKvSnapshot) {
        assert_eq!(
            self.k.len(),
            snapshot.k.len(),
            "cpu kv snapshot layer count mismatch"
        );
        assert_eq!(
            self.v.len(),
            snapshot.v.len(),
            "cpu kv snapshot layer count mismatch"
        );
        assert_eq!(
            self.max_seq_len, snapshot.max_seq_len,
            "cpu kv snapshot max_seq_len mismatch"
        );
        assert_eq!(
            self.token_stride, snapshot.token_stride,
            "cpu kv snapshot token_stride mismatch"
        );
        assert_eq!(
            self.page_size, snapshot.page_size,
            "cpu kv snapshot page_size mismatch"
        );

        self.k = snapshot.k.clone();
        self.v = snapshot.v.clone();
        self.seq_len = snapshot.seq_len;
        self.capacity = snapshot.capacity;
        self.allocator = snapshot.allocator.clone();
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Grow if current capacity < seq_len + 1.
    fn ensure_capacity(&mut self) {
        self.ensure_capacity_for(self.seq_len + 1);
    }

    /// Grow if current capacity < needed tokens.
    fn ensure_capacity_for(&mut self, needed: usize) {
        if self.capacity >= needed {
            return;
        }
        let new_cap =
            planned_capacity_for_needed(self.capacity, needed, self.page_size, self.max_seq_len)
                .unwrap_or_else(|| {
                    panic!("CPU KV: needed={needed} > max_seq_len={}", self.max_seq_len)
                });
        let new_elems = new_cap
            .checked_mul(self.token_stride)
            .expect("CPU KV growth overflow");
        for layer_k in &mut self.k {
            layer_k.resize(new_elems, 0.0);
        }
        for layer_v in &mut self.v {
            layer_v.resize(new_elems, 0.0);
        }
        let pages_added = (new_cap - self.capacity) / self.page_size;
        self.allocator.record_grow(pages_added.max(1));
        self.capacity = new_cap;
    }

    fn token_offset(&self, token_idx: usize) -> usize {
        token_idx
            .checked_mul(self.token_stride)
            .expect("CPU KV token offset overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_kv_append_finalize_roundtrip() {
        let n_layers = 2;
        let n_kv_heads = 2;
        let head_dim = 4;
        let max_seq_len = 16;
        let mut kv = CpuKv::new(n_layers, n_kv_heads, head_dim, max_seq_len);

        let stride = n_kv_heads * head_dim;
        let k0: Vec<f32> = (0..stride).map(|i| i as f32).collect();
        let v0: Vec<f32> = (0..stride).map(|i| (i + 100) as f32).collect();

        for layer in 0..n_layers {
            kv.append_and_advance(layer, &k0, &v0);
        }
        kv.finalize_token();

        assert_eq!(kv.seq_len(), 1);
        assert_eq!(kv.k_slice(0, 1), k0.as_slice());
        assert_eq!(kv.v_slice(0, 1), v0.as_slice());
    }

    #[test]
    fn cpu_kv_append_batch_finalize_roundtrip() {
        let mut kv = CpuKv::new(1, 2, 4, 16);
        let stride = kv.token_stride();
        let n_tokens = 3usize;
        let k_batch: Vec<f32> = (0..n_tokens * stride).map(|i| i as f32).collect();
        let v_batch: Vec<f32> = (0..n_tokens * stride).map(|i| (i + 100) as f32).collect();

        kv.append_batch(0, &k_batch, &v_batch, n_tokens);
        kv.finalize_batch(n_tokens);

        assert_eq!(kv.seq_len(), n_tokens);
        assert_eq!(kv.k_slice(0, n_tokens), k_batch.as_slice());
        assert_eq!(kv.v_slice(0, n_tokens), v_batch.as_slice());
    }

    #[test]
    fn test_truncate_to_resets_seq_len() {
        let mut kv = CpuKv::new(1, 2, 4, 64);
        let stride = 8usize;
        let k = vec![1.0f32; stride];
        let v = vec![2.0f32; stride];

        // Append 5 tokens
        for _ in 0..5 {
            kv.append_and_advance(0, &k, &v);
            kv.finalize_token();
        }
        assert_eq!(kv.seq_len(), 5);
        let cap_before = kv.capacity();

        // Truncate to 3 — no realloc, seq_len updated
        kv.truncate_to(3);
        assert_eq!(kv.seq_len(), 3);
        assert_eq!(kv.capacity(), cap_before, "capacity must not change");
    }

    #[test]
    fn cpu_kv_grows_on_demand() {
        // Use max_seq_len large enough that initial_cap < max_seq_len.
        // With n_kv_heads=1, head_dim=4, stride=4, page_size=512 (MAX_PAGE_SIZE),
        // so initial_cap = min(512, 1024) = 512.
        let mut kv = CpuKv::new(1, 1, 4, 1024);
        // Initial capacity is page_size (512 for tiny stride)
        let initial_cap = kv.capacity();

        // Fill past initial capacity
        let stride = 4usize;
        let k = vec![1.0f32; stride];
        let v = vec![2.0f32; stride];
        for _ in 0..(initial_cap + 1) {
            kv.append_and_advance(0, &k, &v);
            kv.finalize_token();
        }
        assert!(kv.capacity() > initial_cap);
        assert_eq!(kv.seq_len(), initial_cap + 1);
    }

    #[test]
    fn cpu_kv_including_current_slice_allows_in_progress_token() {
        let mut kv = CpuKv::new(1, 1, 2, 8);
        let k = vec![1.0f32, 2.0];
        let v = vec![3.0f32, 4.0];

        kv.append_and_advance(0, &k, &v);

        assert_eq!(kv.seq_len(), 0);
        assert_eq!(kv.k_slice_including_current(0, 1), k.as_slice());
        assert_eq!(kv.v_slice_including_current(0, 1), v.as_slice());
    }

    #[test]
    fn cpu_kv_snapshot_restore_round_trip() {
        let mut kv = CpuKv::new(1, 1, 2, 16);
        let k0 = vec![1.0f32, 2.0];
        let v0 = vec![3.0f32, 4.0];
        let k1 = vec![5.0f32, 6.0];
        let v1 = vec![7.0f32, 8.0];

        kv.append_and_advance(0, &k0, &v0);
        kv.finalize_token();
        let snapshot = kv.snapshot();

        kv.append_and_advance(0, &k1, &v1);
        kv.finalize_token();
        assert_eq!(kv.seq_len(), 2);

        kv.restore(&snapshot);
        assert_eq!(kv.seq_len(), 1);
        assert_eq!(kv.k_slice(0, 1), k0.as_slice());
        assert_eq!(kv.v_slice(0, 1), v0.as_slice());
    }
}
