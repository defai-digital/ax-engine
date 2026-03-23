//! GPU-resident KV cache backed by Metal shared-memory buffers.
//!
//! On Apple Silicon UMA, MetalBuffers are zero-copy: the CPU writes via
//! `contents()` pointer, and the GPU reads the same physical pages directly.
//!
//! # v2 changes vs v1 `GpuKvCache`
//! - Removed `advance_by()`. In v2 the GPU path is the only owner of KV data.
//!   There is no CPU mirror to keep in sync. The forward pass calls
//!   `append_batch(n_tokens)` which appends data AND advances seq_len atomically.
//! - Removed paged KV / prefix cache integration (deferred to v2.1).

use ax_metal::{MetalBuffer, MetalDevice};

/// Storage format for GPU KV cache buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKvDtype {
    F32,
    F16,
}

impl GpuKvDtype {
    #[inline]
    pub fn elem_size(self) -> usize {
        match self {
            Self::F32 => std::mem::size_of::<f32>(),
            Self::F16 => std::mem::size_of::<half::f16>(),
        }
    }
}

/// GPU-resident KV cache for all transformer layers.
///
/// Each layer has a pair of MetalBuffers (K and V) that grow on demand.
/// Layout per buffer: `[capacity * kv_stride]` values, where
/// `kv_stride = n_kv_heads * head_dim`.
pub struct GpuKv {
    /// Per-layer key cache MetalBuffers.
    k_bufs: Vec<MetalBuffer>,
    /// Per-layer value cache MetalBuffers.
    v_bufs: Vec<MetalBuffer>,
    /// Number of tokens currently stored.
    seq_len: usize,
    /// Current allocated capacity in tokens.
    capacity: usize,
    /// Stride per token: n_kv_heads * head_dim.
    kv_stride: usize,
    /// Growth increment in tokens.
    page_size: usize,
    /// Maximum sequence length (hard limit).
    max_seq_len: usize,
    /// KV storage dtype.
    dtype: GpuKvDtype,
}

impl GpuKv {
    /// Create a new GPU KV cache (f32).
    pub fn new(
        device: &MetalDevice,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        page_size: usize,
    ) -> anyhow::Result<Self> {
        Self::new_with_dtype(
            device,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            page_size,
            GpuKvDtype::F32,
        )
    }

    /// Create a new GPU KV cache with explicit storage dtype (f32 or f16).
    pub fn new_with_dtype(
        device: &MetalDevice,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        page_size: usize,
        dtype: GpuKvDtype,
    ) -> anyhow::Result<Self> {
        let page_size = page_size.max(1);
        let initial_cap = page_size.min(max_seq_len);
        let kv_stride = n_kv_heads
            .checked_mul(head_dim)
            .expect("GPU KV stride overflow");
        let buf_bytes = initial_cap
            .checked_mul(kv_stride)
            .and_then(|elems| elems.checked_mul(dtype.elem_size()))
            .expect("GPU KV allocation overflow");

        let mut k_bufs = Vec::with_capacity(n_layers);
        let mut v_bufs = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_bufs.push(MetalBuffer::new(device.device(), buf_bytes)?);
            v_bufs.push(MetalBuffer::new(device.device(), buf_bytes)?);
        }

        tracing::info!(
            n_layers,
            kv_stride,
            initial_cap,
            kv_dtype = ?dtype,
            buf_mb = (2 * n_layers * buf_bytes) / (1024 * 1024),
            "GPU KV cache allocated"
        );

        Ok(Self {
            k_bufs,
            v_bufs,
            seq_len: 0,
            capacity: initial_cap,
            kv_stride,
            page_size,
            max_seq_len,
            dtype,
        })
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// Number of tokens currently stored.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Current allocated capacity in tokens.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Stride per token (n_kv_heads * head_dim).
    pub fn kv_stride(&self) -> usize {
        self.kv_stride
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.k_bufs.len()
    }

    /// KV storage dtype.
    pub fn dtype(&self) -> GpuKvDtype {
        self.dtype
    }

    /// True when KV buffers are stored as f16.
    pub fn is_f16(&self) -> bool {
        self.dtype == GpuKvDtype::F16
    }

    /// Get the K buffer for a layer (for GPU binding).
    pub fn k_buffer(&self, layer: usize) -> &MetalBuffer {
        &self.k_bufs[layer]
    }

    /// Get the V buffer for a layer (for GPU binding).
    pub fn v_buffer(&self, layer: usize) -> &MetalBuffer {
        &self.v_bufs[layer]
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    /// Append one token's K and V for a single layer via CPU UMA write.
    ///
    /// This writes into the MetalBuffer via the UMA-coherent `contents()` pointer.
    /// The GPU sees this data on its next read without any explicit copy.
    ///
    /// Must call `finalize_token()` after appending to all layers.
    pub fn append_layer(&mut self, layer: usize, k_new: &[f32], v_new: &[f32]) {
        debug_assert_eq!(k_new.len(), self.kv_stride);
        debug_assert_eq!(v_new.len(), self.kv_stride);
        assert!(
            self.seq_len < self.capacity,
            "GPU KV append_layer: seq_len={} >= capacity={} (call ensure_capacity first)",
            self.seq_len,
            self.capacity,
        );

        let offset_elems = self
            .seq_len
            .checked_mul(self.kv_stride)
            .expect("GPU KV offset overflow");
        match self.dtype {
            GpuKvDtype::F32 => {
                let elem_size = std::mem::size_of::<f32>();
                let offset_bytes = offset_elems
                    .checked_mul(elem_size)
                    .expect("GPU KV byte offset overflow");
                let copy_bytes = self
                    .kv_stride
                    .checked_mul(elem_size)
                    .expect("GPU KV copy size overflow");
                let end_bytes = offset_bytes
                    .checked_add(copy_bytes)
                    .expect("GPU KV end offset overflow");
                assert!(
                    end_bytes <= self.k_bufs[layer].len(),
                    "GPU KV K write out of bounds: end={} > len={}",
                    end_bytes,
                    self.k_bufs[layer].len()
                );
                assert!(
                    end_bytes <= self.v_bufs[layer].len(),
                    "GPU KV V write out of bounds: end={} > len={}",
                    end_bytes,
                    self.v_bufs[layer].len()
                );
                unsafe {
                    let k_dst =
                        (self.k_bufs[layer].contents().as_ptr() as *mut u8).add(offset_bytes);
                    std::ptr::copy_nonoverlapping(k_new.as_ptr() as *const u8, k_dst, copy_bytes);
                    let v_dst =
                        (self.v_bufs[layer].contents().as_ptr() as *mut u8).add(offset_bytes);
                    std::ptr::copy_nonoverlapping(v_new.as_ptr() as *const u8, v_dst, copy_bytes);
                }
            }
            GpuKvDtype::F16 => {
                let end_elems = offset_elems
                    .checked_add(self.kv_stride)
                    .expect("GPU KV f16 end offset overflow");
                let k_capacity = self.k_bufs[layer].len() / std::mem::size_of::<half::f16>();
                let v_capacity = self.v_bufs[layer].len() / std::mem::size_of::<half::f16>();
                assert!(
                    end_elems <= k_capacity,
                    "GPU KV K f16 write out of bounds: end={} > len={}",
                    end_elems,
                    k_capacity
                );
                assert!(
                    end_elems <= v_capacity,
                    "GPU KV V f16 write out of bounds: end={} > len={}",
                    end_elems,
                    v_capacity
                );
                let k_ptr = self.k_bufs[layer].contents().as_ptr() as *mut half::f16;
                let v_ptr = self.v_bufs[layer].contents().as_ptr() as *mut half::f16;
                unsafe {
                    for i in 0..self.kv_stride {
                        *k_ptr.add(offset_elems + i) = half::f16::from_f32(k_new[i]);
                        *v_ptr.add(offset_elems + i) = half::f16::from_f32(v_new[i]);
                    }
                }
            }
        }
    }

    /// Advance seq_len by one. Call after `append_layer` for all layers.
    pub fn finalize_token(&mut self) {
        assert!(self.seq_len < self.max_seq_len);
        self.seq_len += 1;
    }

    /// Reset the KV cache to zero tokens (keeps allocated Metal buffers).
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Rewind seq_len to `pos` without reallocating Metal buffers.
    ///
    /// Used by speculative decoding to roll back the KV cache after a draft
    /// token is rejected. On Apple Silicon UMA the Metal buffers retain the
    /// old data beyond `pos`; they will be overwritten on the next fill.
    pub fn truncate_to(&mut self, pos: usize) {
        assert!(
            pos <= self.seq_len,
            "truncate_to: pos={pos} > seq_len={}",
            self.seq_len
        );
        self.seq_len = pos;
    }

    /// Advance seq_len by `n` after a batched GPU prefill that wrote KV directly.
    ///
    /// Unlike v1's `advance_by`, this is only called from `forward_batch_gpu_unified`
    /// which writes KV data to GPU buffers directly (not via `append_layer`). There is
    /// no CPU mirror to sync — this is the sole KV owner.
    pub fn finalize_batch(&mut self, n: usize) {
        assert!(
            self.seq_len + n <= self.capacity,
            "GPU KV finalize_batch: seq_len({}) + n({n}) > capacity({})",
            self.seq_len,
            self.capacity,
        );
        assert!(
            self.seq_len + n <= self.max_seq_len,
            "GPU KV finalize_batch: seq_len({}) + n({n}) > max_seq_len({})",
            self.seq_len,
            self.max_seq_len,
        );
        self.seq_len += n;
    }

    /// Ensure capacity for at least `needed` tokens.
    ///
    /// Must be called BEFORE encoding a command buffer that appends tokens,
    /// because MetalBuffer reallocation cannot happen inside `execute_sync`.
    pub fn ensure_capacity(&mut self, device: &MetalDevice, needed: usize) -> anyhow::Result<()> {
        if needed <= self.capacity {
            return Ok(());
        }

        let mut new_cap = self.capacity;
        while new_cap < needed {
            let prev = new_cap;
            new_cap = (new_cap + self.page_size).min(self.max_seq_len);
            if new_cap == prev {
                anyhow::bail!(
                    "GPU KV cannot grow: needed={needed} > max={}",
                    self.max_seq_len
                );
            }
        }

        let elem_size = self.dtype.elem_size();
        let new_bytes = new_cap
            .checked_mul(self.kv_stride)
            .and_then(|elems| elems.checked_mul(elem_size))
            .expect("GPU KV growth overflow");
        let old_bytes = self
            .seq_len
            .checked_mul(self.kv_stride)
            .and_then(|elems| elems.checked_mul(elem_size))
            .expect("GPU KV existing size overflow");

        for i in 0..self.k_bufs.len() {
            let new_k = MetalBuffer::new(device.device(), new_bytes)?;
            let new_v = MetalBuffer::new(device.device(), new_bytes)?;
            if old_bytes > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.k_bufs[i].contents().as_ptr() as *const u8,
                        new_k.contents().as_ptr() as *mut u8,
                        old_bytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        self.v_bufs[i].contents().as_ptr() as *const u8,
                        new_v.contents().as_ptr() as *mut u8,
                        old_bytes,
                    );
                }
            }
            self.k_bufs[i] = new_k;
            self.v_bufs[i] = new_v;
        }

        tracing::debug!(old_cap = self.capacity, new_cap, "GPU KV cache grew");
        self.capacity = new_cap;
        Ok(())
    }
}
