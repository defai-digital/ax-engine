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

use ax_engine_metal::{MetalBuffer, MetalDevice};

use super::page::{initial_token_capacity, planned_capacity_for_needed};

/// Storage format for GPU KV cache buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKvDtype {
    F32,
    F16,
    /// Q8_0: block-quantized int8 with per-block f16 scale.
    /// Block size 32 values → 34 bytes (2 byte scale + 32 byte i8 data).
    /// ~1.88× smaller than F16, ~3.76× smaller than F32.
    Q8_0,
}

/// Q8_0 block size: 32 values per block.
pub const Q8_0_BLOCK_VALUES: usize = 32;
/// Q8_0 block byte size: half(2) + i8×32(32) = 34 bytes.
pub const Q8_0_BLOCK_BYTES: usize = 34;

impl GpuKvDtype {
    /// Per-element byte size (only valid for F32/F16).
    /// For Q8_0, use `row_bytes()` instead.
    #[inline]
    pub fn elem_size(self) -> usize {
        match self {
            Self::F32 => std::mem::size_of::<f32>(),
            Self::F16 => std::mem::size_of::<half::f16>(),
            Self::Q8_0 => panic!("Q8_0 is block-quantized; use row_bytes() instead of elem_size()"),
        }
    }

    /// Bytes per token row for a given kv_stride (n_kv_heads × head_dim).
    #[inline]
    pub fn row_bytes(self, kv_stride: usize) -> usize {
        match self {
            Self::F32 => kv_stride * std::mem::size_of::<f32>(),
            Self::F16 => kv_stride * std::mem::size_of::<half::f16>(),
            Self::Q8_0 => {
                debug_assert!(
                    kv_stride.is_multiple_of(Q8_0_BLOCK_VALUES),
                    "kv_stride must be multiple of Q8_0 block size (32)"
                );
                (kv_stride / Q8_0_BLOCK_VALUES) * Q8_0_BLOCK_BYTES
            }
        }
    }
}

/// GPU-resident KV cache for all transformer layers.
///
/// Each layer has a pair of MetalBuffers (K and V) that grow on demand.
/// Layout per buffer: `[capacity * kv_stride]` values, where
/// `kv_stride = n_kv_heads * head_dim` (may differ per layer for Gemma4).
pub struct GpuKv {
    /// Per-layer key cache MetalBuffers.
    k_bufs: Vec<MetalBuffer>,
    /// Per-layer value cache MetalBuffers.
    v_bufs: Vec<MetalBuffer>,
    /// Number of tokens currently stored.
    seq_len: usize,
    /// Current allocated capacity in tokens.
    capacity: usize,
    /// Maximum stride across all layers (used for batch scratch sizing).
    kv_stride: usize,
    /// Per-layer strides (n_kv_heads * head_dim per layer).
    /// For uniform models all values are identical; for Gemma4 SWA/global
    /// layers differ.
    kv_strides: Vec<usize>,
    /// Growth increment in tokens.
    page_size: usize,
    /// Maximum sequence length (hard limit).
    max_seq_len: usize,
    /// KV storage dtype.
    dtype: GpuKvDtype,
}

impl std::fmt::Debug for GpuKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let per_layer = self.kv_strides.windows(2).any(|w| w[0] != w[1]);
        let mut s = f.debug_struct("GpuKv");
        s.field("n_layers", &self.k_bufs.len())
            .field("seq_len", &self.seq_len)
            .field("capacity", &self.capacity)
            .field("kv_stride", &self.kv_stride);
        if per_layer {
            s.field("kv_strides", &self.kv_strides);
        }
        s.field("page_size", &self.page_size)
            .field("max_seq_len", &self.max_seq_len)
            .field("dtype", &self.dtype)
            .finish()
    }
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
        let kv_stride = n_kv_heads
            .checked_mul(head_dim)
            .expect("GPU KV stride overflow");
        let strides = vec![kv_stride; n_layers];
        Self::new_with_per_layer_strides(device, n_layers, strides, max_seq_len, page_size, dtype)
    }

    /// Create a new GPU KV cache with per-layer strides.
    ///
    /// For models like Gemma4 where SWA and global attention layers have
    /// different n_kv_heads × head_dim products, each layer gets a buffer
    /// sized to its own stride instead of the global maximum.
    pub fn new_with_per_layer_strides(
        device: &MetalDevice,
        n_layers: usize,
        kv_strides: Vec<usize>,
        max_seq_len: usize,
        page_size: usize,
        dtype: GpuKvDtype,
    ) -> anyhow::Result<Self> {
        assert_eq!(kv_strides.len(), n_layers);
        let page_size = page_size.max(1);
        let initial_cap = initial_token_capacity(page_size, max_seq_len);
        let kv_stride = kv_strides.iter().copied().max().unwrap_or(0);

        let mut k_bufs = Vec::with_capacity(n_layers);
        let mut v_bufs = Vec::with_capacity(n_layers);
        let mut total_bytes = 0usize;
        for &stride in &kv_strides {
            let row_bytes = dtype.row_bytes(stride);
            let buf_bytes = initial_cap
                .checked_mul(row_bytes)
                .expect("GPU KV allocation overflow");
            k_bufs.push(MetalBuffer::new(device.device(), buf_bytes)?);
            v_bufs.push(MetalBuffer::new(device.device(), buf_bytes)?);
            total_bytes += 2 * buf_bytes;
        }

        let per_layer = kv_strides.windows(2).any(|w| w[0] != w[1]);
        if per_layer {
            tracing::info!(
                n_layers,
                kv_stride_max = kv_stride,
                ?kv_strides,
                initial_cap,
                kv_dtype = ?dtype,
                buf_mb = total_bytes / (1024 * 1024),
                "GPU KV cache allocated (per-layer strides)"
            );
        } else {
            tracing::info!(
                n_layers,
                kv_stride,
                initial_cap,
                kv_dtype = ?dtype,
                buf_mb = total_bytes / (1024 * 1024),
                "GPU KV cache allocated"
            );
        }

        Ok(Self {
            k_bufs,
            v_bufs,
            seq_len: 0,
            capacity: initial_cap,
            kv_stride,
            kv_strides,
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

    /// Maximum stride across all layers (n_kv_heads * head_dim).
    /// Use `kv_stride_for_layer()` when the per-layer stride matters.
    pub fn kv_stride(&self) -> usize {
        self.kv_stride
    }

    /// Stride for a specific layer. For uniform models this equals `kv_stride()`.
    /// For Gemma4, SWA and global layers have different strides.
    pub fn kv_stride_for_layer(&self, layer: usize) -> usize {
        self.kv_strides[layer]
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

    /// True when KV buffers are stored as Q8_0 (block-quantized int8).
    pub fn is_q8(&self) -> bool {
        self.dtype == GpuKvDtype::Q8_0
    }

    /// Get the K buffer for a layer (for GPU binding).
    pub fn k_buffer(&self, layer: usize) -> &MetalBuffer {
        &self.k_bufs[layer]
    }

    /// Get the V buffer for a layer (for GPU binding).
    pub fn v_buffer(&self, layer: usize) -> &MetalBuffer {
        &self.v_bufs[layer]
    }

    /// Read a single token row from GPU KV into CPU f32 slices.
    ///
    /// Hybrid decode paths may read the just-written token at `idx == seq_len`
    /// before `finalize_token()` advances the logical length, so this permits
    /// one pending token in addition to finalized rows.
    pub fn read_layer_token_into(
        &self,
        layer: usize,
        token_idx: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) {
        let stride = self.kv_strides[layer];
        assert_eq!(k_out.len(), stride);
        assert_eq!(v_out.len(), stride);
        assert!(
            token_idx <= self.seq_len,
            "GPU KV read token out of bounds (idx={token_idx}, seq_len={})",
            self.seq_len
        );
        assert!(
            token_idx < self.capacity,
            "GPU KV read token out of capacity (idx={token_idx}, capacity={})",
            self.capacity
        );

        let row_bytes = self.dtype.row_bytes(stride);
        let offset_bytes = token_idx * row_bytes;
        match self.dtype {
            GpuKvDtype::F32 => unsafe {
                let offset_elems = token_idx * stride;
                let k_src =
                    &self.k_bufs[layer].as_slice::<f32>()[offset_elems..offset_elems + stride];
                let v_src =
                    &self.v_bufs[layer].as_slice::<f32>()[offset_elems..offset_elems + stride];
                k_out.copy_from_slice(k_src);
                v_out.copy_from_slice(v_src);
            },
            GpuKvDtype::F16 => unsafe {
                let offset_elems = token_idx * stride;
                let capacity_elems = self
                    .capacity
                    .checked_mul(stride)
                    .expect("GPU KV F16 capacity element count overflow");
                let k_src = std::slice::from_raw_parts(
                    self.k_bufs[layer].contents().as_ptr().cast::<half::f16>(),
                    capacity_elems,
                );
                let v_src = std::slice::from_raw_parts(
                    self.v_bufs[layer].contents().as_ptr().cast::<half::f16>(),
                    capacity_elems,
                );
                for i in 0..stride {
                    k_out[i] = k_src[offset_elems + i].to_f32();
                    v_out[i] = v_src[offset_elems + i].to_f32();
                }
            },
            GpuKvDtype::Q8_0 => {
                let k_src = unsafe {
                    (self.k_bufs[layer].contents().as_ptr() as *const u8).add(offset_bytes)
                };
                let v_src = unsafe {
                    (self.v_bufs[layer].contents().as_ptr() as *const u8).add(offset_bytes)
                };
                dequantize_row_q8_0(k_src, k_out, stride / Q8_0_BLOCK_VALUES);
                dequantize_row_q8_0(v_src, v_out, stride / Q8_0_BLOCK_VALUES);
            }
        }
    }

    /// Read the finalized prefix for one layer into CPU f32 slices.
    pub fn read_layer_prefix_into(
        &self,
        layer: usize,
        n_tokens: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) {
        self.read_layer_range_into(layer, 0, n_tokens, k_out, v_out);
    }

    /// Read a finalized token range for one layer into CPU f32 slices.
    pub fn read_layer_range_into(
        &self,
        layer: usize,
        start_token: usize,
        n_tokens: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) {
        let stride = self.kv_strides[layer];
        assert!(
            start_token + n_tokens <= self.seq_len,
            "GPU KV read range out of bounds (start={start_token}, n_tokens={n_tokens}, seq_len={})",
            self.seq_len
        );
        let prefix_elems = n_tokens
            .checked_mul(stride)
            .expect("GPU KV range element count overflow");
        assert_eq!(k_out.len(), prefix_elems);
        assert_eq!(v_out.len(), prefix_elems);
        let start_elems = start_token
            .checked_mul(stride)
            .expect("GPU KV range start overflow");
        let end_elems = start_elems
            .checked_add(prefix_elems)
            .expect("GPU KV range end overflow");

        match self.dtype {
            GpuKvDtype::F32 => unsafe {
                let k_src = &self.k_bufs[layer].as_slice::<f32>()[start_elems..end_elems];
                let v_src = &self.v_bufs[layer].as_slice::<f32>()[start_elems..end_elems];
                k_out.copy_from_slice(k_src);
                v_out.copy_from_slice(v_src);
            },
            GpuKvDtype::F16 => unsafe {
                let k_src = std::slice::from_raw_parts(
                    self.k_bufs[layer].contents().as_ptr().cast::<half::f16>(),
                    self.capacity * stride,
                );
                let v_src = std::slice::from_raw_parts(
                    self.v_bufs[layer].contents().as_ptr().cast::<half::f16>(),
                    self.capacity * stride,
                );
                for i in 0..prefix_elems {
                    k_out[i] = k_src[start_elems + i].to_f32();
                    v_out[i] = v_src[start_elems + i].to_f32();
                }
            },
            GpuKvDtype::Q8_0 => {
                let row_bytes = self.dtype.row_bytes(stride);
                let n_blocks_per_row = stride / Q8_0_BLOCK_VALUES;
                let k_base = self.k_bufs[layer].contents().as_ptr() as *const u8;
                let v_base = self.v_bufs[layer].contents().as_ptr() as *const u8;
                for t in 0..n_tokens {
                    let src_byte_off = (start_token + t) * row_bytes;
                    let dst_elem_off = t * stride;
                    dequantize_row_q8_0(
                        unsafe { k_base.add(src_byte_off) },
                        &mut k_out[dst_elem_off..dst_elem_off + stride],
                        n_blocks_per_row,
                    );
                    dequantize_row_q8_0(
                        unsafe { v_base.add(src_byte_off) },
                        &mut v_out[dst_elem_off..dst_elem_off + stride],
                        n_blocks_per_row,
                    );
                }
            }
        }
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    /// Append one token's K and V for a single layer via CPU UMA write.
    ///
    /// This writes into the MetalBuffer via the UMA-coherent `contents()` pointer.
    /// The GPU sees this data on its next read without any explicit copy.
    ///
    /// Must call `finalize_token()` after appending to all layers.
    pub fn append_layer(&mut self, layer: usize, k_new: &[f32], v_new: &[f32]) {
        let stride = self.kv_strides[layer];
        debug_assert_eq!(k_new.len(), stride);
        debug_assert_eq!(v_new.len(), stride);
        assert!(
            self.seq_len < self.max_seq_len,
            "GPU KV append_layer: seq_len={} >= max_seq_len={}",
            self.seq_len,
            self.max_seq_len,
        );
        assert!(
            self.seq_len < self.capacity,
            "GPU KV append_layer: seq_len={} >= capacity={} (call ensure_capacity first)",
            self.seq_len,
            self.capacity,
        );

        let row_bytes = self.dtype.row_bytes(stride);
        let offset_bytes = self
            .seq_len
            .checked_mul(row_bytes)
            .expect("GPU KV offset overflow");
        let end_bytes = offset_bytes + row_bytes;
        assert!(
            end_bytes <= self.k_bufs[layer].len(),
            "GPU KV K write out of bounds: end={end_bytes} > len={}",
            self.k_bufs[layer].len()
        );
        assert!(
            end_bytes <= self.v_bufs[layer].len(),
            "GPU KV V write out of bounds: end={end_bytes} > len={}",
            self.v_bufs[layer].len()
        );
        match self.dtype {
            GpuKvDtype::F32 => unsafe {
                let k_dst = (self.k_bufs[layer].contents().as_ptr() as *mut u8).add(offset_bytes);
                std::ptr::copy_nonoverlapping(k_new.as_ptr() as *const u8, k_dst, row_bytes);
                let v_dst = (self.v_bufs[layer].contents().as_ptr() as *mut u8).add(offset_bytes);
                std::ptr::copy_nonoverlapping(v_new.as_ptr() as *const u8, v_dst, row_bytes);
            },
            GpuKvDtype::F16 => {
                let offset_elems = self.seq_len * stride;
                let k_ptr = self.k_bufs[layer].contents().as_ptr() as *mut half::f16;
                let v_ptr = self.v_bufs[layer].contents().as_ptr() as *mut half::f16;
                unsafe {
                    for i in 0..stride {
                        *k_ptr.add(offset_elems + i) = half::f16::from_f32(k_new[i]);
                        *v_ptr.add(offset_elems + i) = half::f16::from_f32(v_new[i]);
                    }
                }
            }
            GpuKvDtype::Q8_0 => {
                let n_blocks = stride / Q8_0_BLOCK_VALUES;
                let k_dst = unsafe {
                    (self.k_bufs[layer].contents().as_ptr() as *mut u8).add(offset_bytes)
                };
                let v_dst = unsafe {
                    (self.v_bufs[layer].contents().as_ptr() as *mut u8).add(offset_bytes)
                };
                quantize_row_q8_0(k_new, k_dst, n_blocks);
                quantize_row_q8_0(v_new, v_dst, n_blocks);
            }
        }
    }

    /// Advance seq_len by one. Call after `append_layer` for all layers.
    pub fn finalize_token(&mut self) {
        assert!(self.seq_len < self.max_seq_len);
        self.seq_len += 1;
    }

    /// Append a batch of token-major K/V rows for a single layer without advancing seq_len.
    ///
    /// The caller must invoke `finalize_batch(n_tokens)` after appending all layers.
    pub fn append_layer_batch(
        &mut self,
        layer: usize,
        k_batch: &[f32],
        v_batch: &[f32],
        n_tokens: usize,
    ) {
        let stride = self.kv_strides[layer];
        debug_assert_eq!(k_batch.len(), n_tokens * stride);
        debug_assert_eq!(v_batch.len(), n_tokens * stride);
        assert!(
            self.seq_len + n_tokens <= self.capacity,
            "GPU KV append_layer_batch: seq_len({}) + n_tokens({n_tokens}) > capacity({})",
            self.seq_len,
            self.capacity,
        );

        let row_bytes = self.dtype.row_bytes(stride);
        let offset_bytes = self
            .seq_len
            .checked_mul(row_bytes)
            .expect("GPU KV batch offset overflow");
        let total_bytes = n_tokens
            .checked_mul(row_bytes)
            .expect("GPU KV batch size overflow");
        let end_bytes = offset_bytes + total_bytes;
        assert!(
            end_bytes <= self.k_bufs[layer].len(),
            "GPU KV K batch write out of bounds: end={end_bytes} > len={}",
            self.k_bufs[layer].len()
        );
        assert!(
            end_bytes <= self.v_bufs[layer].len(),
            "GPU KV V batch write out of bounds: end={end_bytes} > len={}",
            self.v_bufs[layer].len()
        );
        match self.dtype {
            GpuKvDtype::F32 => unsafe {
                let k_dst = (self.k_bufs[layer].contents().as_ptr() as *mut u8).add(offset_bytes);
                std::ptr::copy_nonoverlapping(k_batch.as_ptr() as *const u8, k_dst, total_bytes);
                let v_dst = (self.v_bufs[layer].contents().as_ptr() as *mut u8).add(offset_bytes);
                std::ptr::copy_nonoverlapping(v_batch.as_ptr() as *const u8, v_dst, total_bytes);
            },
            GpuKvDtype::F16 => {
                let offset_elems = self
                    .seq_len
                    .checked_mul(stride)
                    .expect("GPU KV F16 batch append offset overflow");
                let total_elems = n_tokens
                    .checked_mul(stride)
                    .expect("GPU KV F16 batch append count overflow");
                let k_ptr = self.k_bufs[layer].contents().as_ptr() as *mut half::f16;
                let v_ptr = self.v_bufs[layer].contents().as_ptr() as *mut half::f16;
                unsafe {
                    for i in 0..total_elems {
                        *k_ptr.add(offset_elems + i) = half::f16::from_f32(k_batch[i]);
                        *v_ptr.add(offset_elems + i) = half::f16::from_f32(v_batch[i]);
                    }
                }
            }
            GpuKvDtype::Q8_0 => {
                let n_blocks = stride / Q8_0_BLOCK_VALUES;
                let k_dst = unsafe {
                    (self.k_bufs[layer].contents().as_ptr() as *mut u8).add(offset_bytes)
                };
                let v_dst = unsafe {
                    (self.v_bufs[layer].contents().as_ptr() as *mut u8).add(offset_bytes)
                };
                for t in 0..n_tokens {
                    let src_off = t * stride;
                    let dst_off = t * row_bytes;
                    quantize_row_q8_0(
                        &k_batch[src_off..src_off + stride],
                        unsafe { k_dst.add(dst_off) },
                        n_blocks,
                    );
                    quantize_row_q8_0(
                        &v_batch[src_off..src_off + stride],
                        unsafe { v_dst.add(dst_off) },
                        n_blocks,
                    );
                }
            }
        }
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

        let new_cap =
            planned_capacity_for_needed(self.capacity, needed, self.page_size, self.max_seq_len)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GPU KV cannot grow: needed={needed} > max={}",
                        self.max_seq_len
                    )
                })?;

        // Allocate ALL new buffers into temp vecs first so that a mid-loop
        // allocation failure does not leave self with mixed old/new buffers.
        let n_layers = self.k_bufs.len();
        let mut new_k_bufs = Vec::with_capacity(n_layers);
        let mut new_v_bufs = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let stride = self.kv_strides[i];
            let row_bytes = self.dtype.row_bytes(stride);
            let new_bytes = new_cap
                .checked_mul(row_bytes)
                .expect("GPU KV growth overflow");
            let old_bytes = self
                .seq_len
                .checked_mul(row_bytes)
                .expect("GPU KV existing size overflow");
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
            new_k_bufs.push(new_k);
            new_v_bufs.push(new_v);
        }

        // All allocations succeeded — swap atomically.
        self.k_bufs = new_k_bufs;
        self.v_bufs = new_v_bufs;

        tracing::debug!(old_cap = self.capacity, new_cap, "GPU KV cache grew");
        self.capacity = new_cap;
        Ok(())
    }
}

/// Quantize a row of f32 values into Q8_0 blocks at `dst`.
///
/// `n_blocks` = number of Q8_0 blocks (each block covers 32 values).
/// `src` must have `n_blocks * 32` elements.
/// `dst` must have space for `n_blocks * 34` bytes.
fn quantize_row_q8_0(src: &[f32], dst: *mut u8, n_blocks: usize) {
    debug_assert_eq!(src.len(), n_blocks * Q8_0_BLOCK_VALUES);
    for b in 0..n_blocks {
        let block_src = &src[b * Q8_0_BLOCK_VALUES..(b + 1) * Q8_0_BLOCK_VALUES];
        let amax = block_src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        let block_dst = unsafe { dst.add(b * Q8_0_BLOCK_BYTES) };
        // Write scale (2 bytes)
        unsafe {
            std::ptr::copy_nonoverlapping(&d_f16 as *const half::f16 as *const u8, block_dst, 2);
        }
        // Write quantized values (32 bytes)
        for (i, &val) in block_src.iter().enumerate() {
            let q = (val * id).round().clamp(-128.0, 127.0) as i8;
            unsafe { *block_dst.add(2 + i) = q as u8 };
        }
    }
}

/// Dequantize Q8_0 blocks at `src` into f32 values.
fn dequantize_row_q8_0(src: *const u8, dst: &mut [f32], n_blocks: usize) {
    debug_assert_eq!(dst.len(), n_blocks * Q8_0_BLOCK_VALUES);
    for b in 0..n_blocks {
        let block_src = unsafe { src.add(b * Q8_0_BLOCK_BYTES) };
        let d_f16: half::f16 = unsafe { std::ptr::read_unaligned(block_src as *const half::f16) };
        let d = d_f16.to_f32();
        for i in 0..Q8_0_BLOCK_VALUES {
            let q = unsafe { *block_src.add(2 + i) } as i8;
            dst[b * Q8_0_BLOCK_VALUES + i] = q as f32 * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_kv_read_layer_token_into_allows_pending_token_before_finalize() {
        let device = MetalDevice::new().expect("metal device");
        let mut kv = GpuKv::new(&device, 1, 1, 2, 16, 8).expect("gpu kv");

        kv.append_layer(0, &[1.0, 2.0], &[3.0, 4.0]);

        let mut k = vec![0.0; 2];
        let mut v = vec![0.0; 2];
        kv.read_layer_token_into(0, 0, &mut k, &mut v);

        assert_eq!(k, vec![1.0, 2.0]);
        assert_eq!(v, vec![3.0, 4.0]);
    }
}
