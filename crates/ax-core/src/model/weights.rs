//! Access to model weight tensors backed by mmap'd GGUF data.
//!
//! WeightStore provides named access to dequantized weight tensors.
//!
//! # v2: Bias folding
//!
//! In v1, Qwen3 bias terms (attn_q.bias, attn_k.bias, attn_v.bias) were applied
//! per-token inside the batched prefill loop:
//!
//!   for i in 0..n_tokens {
//!       copy_to_scratch(); add_bias(); per_head_norm(); rope(); copy_back();
//!   }
//!
//! This caused `n_tokens × n_layers × 7` Metal dispatches with barriers, defeating
//! the purpose of batching.
//!
//! v2 solution: `BiasFoldedWeightStore` pre-computes bias-folded scratch buffers at
//! model init time. `fold_bias_into_scratch(proj_name, bias_name)` returns a
//! persistent `BiasedProj` that can be used as a normal projection weight at inference
//! time. The bias is not baked into the quantized data (which would change the format);
//! instead, at inference time the bias is applied to the entire batch buffer in one
//! batched dispatch — not per-token.
//!
//! See `BiasAddStrategy` for the runtime approach.

use std::collections::HashMap;

use crate::gguf::mmap::MappedModel;
use crate::gguf::tensor::GgmlType;
use crate::quant;

/// Provides access to model weight tensors from a mmap'd GGUF file.
pub struct WeightStore<'a> {
    model: &'a MappedModel,
}

impl<'a> WeightStore<'a> {
    pub fn new(model: &'a MappedModel) -> Self {
        Self { model }
    }

    /// Get a tensor's raw bytes (zero-copy from mmap).
    pub fn raw(&self, name: &str) -> anyhow::Result<&[u8]> {
        self.model
            .tensor_data_by_name(name)
            .map_err(|e| anyhow::anyhow!("tensor '{}': {}", name, e))
    }

    /// Get tensor info (shape, dtype, etc).
    pub fn info(&self, name: &str) -> anyhow::Result<&crate::gguf::TensorInfo> {
        self.model
            .tensor_info(name)
            .ok_or_else(|| anyhow::anyhow!("tensor not found: {name}"))
    }

    /// Dequantize a tensor to f32, writing into the provided buffer.
    pub fn dequantize_into(&self, name: &str, dst: &mut [f32]) -> anyhow::Result<usize> {
        let info = self.info(name)?;
        let src = self.model.tensor_data(info)?;
        let n_elements = info.n_elements() as usize;
        anyhow::ensure!(dst.len() >= n_elements, "buffer too small for '{name}'");
        anyhow::ensure!(
            quant::dequant_supported(info.dtype),
            "tensor '{name}' uses unsupported dtype {} for dequantize",
            info.dtype
        );
        quant::dequantize(info.dtype, src, dst);
        Ok(n_elements)
    }

    /// Dequantize a tensor to a new Vec<f32>.
    pub fn dequantize(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        let info = self.info(name)?;
        let src = self.model.tensor_data(info)?;
        let n_elements = info.n_elements() as usize;
        anyhow::ensure!(
            quant::dequant_supported(info.dtype),
            "tensor '{name}' uses unsupported dtype {} for dequantize",
            info.dtype
        );
        let mut dst = vec![0.0f32; n_elements];
        quant::dequantize(info.dtype, src, &mut dst);
        Ok(dst)
    }

    /// Get an f32 tensor as a zero-copy slice (only works for F32 tensors).
    pub fn f32_slice(&self, name: &str) -> anyhow::Result<&[f32]> {
        let info = self.info(name)?;
        anyhow::ensure!(
            info.dtype == GgmlType::F32,
            "tensor '{}' is {:?}, not F32",
            name,
            info.dtype
        );
        let src = self.model.tensor_data(info)?;
        let n = info.n_elements() as usize;
        anyhow::ensure!(src.len() >= n * 4, "tensor '{name}' data too small");
        anyhow::ensure!(
            (src.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()),
            "tensor '{name}' data pointer not f32-aligned"
        );
        let ptr = src.as_ptr() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
    }

    /// Get raw bytes and dtype for a tensor (zero-copy).
    pub fn raw_with_dtype(&self, name: &str) -> anyhow::Result<(&[u8], GgmlType)> {
        let info = self.info(name)?;
        let data = self.model.tensor_data(info)?;
        Ok((data, info.dtype))
    }

    /// Dequantize a single row from a 2D tensor.
    ///
    /// For embedding lookup: instead of dequantizing the entire vocab×dim matrix,
    /// only dequantize the single row for the given token_id.
    pub fn dequantize_row(&self, name: &str, row: usize, dst: &mut [f32]) -> anyhow::Result<usize> {
        let info = self.info(name)?;
        let src = self.model.tensor_data(info)?;

        let row_elements = info.shape[0] as usize;
        let n_rows = if info.shape.len() > 1 {
            info.shape[1] as usize
        } else {
            1
        };

        anyhow::ensure!(
            row < n_rows,
            "row {row} out of bounds for '{name}' ({n_rows} rows)"
        );
        anyhow::ensure!(dst.len() >= row_elements, "buffer too small");

        let block_size = info.dtype.block_size();
        let bytes_per_block = info.dtype.bytes_per_block();
        anyhow::ensure!(
            row_elements.is_multiple_of(block_size),
            "tensor '{name}' row_elements {row_elements} not multiple of block_size {block_size}"
        );
        anyhow::ensure!(
            quant::dequant_supported(info.dtype),
            "tensor '{name}' uses unsupported dtype {} for row dequantize",
            info.dtype
        );

        let blocks_per_row = row_elements / block_size;
        let row_bytes = blocks_per_row * bytes_per_block;
        let row_start = row * row_bytes;
        let row_end = row_start + row_bytes;

        anyhow::ensure!(row_end <= src.len(), "row byte range out of bounds");
        quant::dequantize(info.dtype, &src[row_start..row_end], dst);
        Ok(row_elements)
    }

    /// Check if a tensor exists.
    pub fn has(&self, name: &str) -> bool {
        self.model.tensor_info(name).is_some()
    }
}

/// Caching wrapper around `WeightStore` that stores dequantized f32 tensors.
pub struct CachedWeightStore<'a> {
    pub inner: WeightStore<'a>,
    cache: HashMap<String, Vec<f32>>,
}

impl<'a> CachedWeightStore<'a> {
    pub fn new(inner: WeightStore<'a>) -> Self {
        Self {
            inner,
            cache: HashMap::new(),
        }
    }

    /// Dequantize a tensor, caching the result for future calls.
    pub fn dequantize(&mut self, name: &str) -> anyhow::Result<&[f32]> {
        if !self.cache.contains_key(name) {
            let data = self.inner.dequantize(name)?;
            self.cache.insert(name.to_string(), data);
        }
        Ok(self.cache.get(name).unwrap())
    }

    pub fn raw_with_dtype(&self, name: &str) -> anyhow::Result<(&[u8], GgmlType)> {
        self.inner.raw_with_dtype(name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.inner.has(name)
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    pub fn cache_bytes(&self) -> usize {
        self.cache.values().map(|v| v.len() * 4).sum()
    }
}

// ── Bias folding ─────────────────────────────────────────────────────────────

/// Per-layer bias vectors dequantized at model init time.
///
/// In v2, Qwen3's QKV biases are loaded once here and applied to the
/// batch projection buffer in a single batched dispatch, not per-token.
///
/// The bias is NOT baked into the quantized projection weight (that would require
/// re-quantizing). Instead, it is applied at the batch level:
///   - GPU path: one `encode_elementwise_broadcast_add` dispatch per bias
///   - CPU path: one loop over `n_tokens` rows
///
/// This is O(1) dispatches regardless of n_tokens (down from O(n_tokens) in v1).
pub struct LayerBiases {
    /// `blk.N.attn_q.bias` dequantized to f32, or empty if not present.
    pub q_bias: Vec<f32>,
    /// `blk.N.attn_k.bias` dequantized to f32, or empty if not present.
    pub k_bias: Vec<f32>,
    /// `blk.N.attn_v.bias` dequantized to f32, or empty if not present.
    pub v_bias: Vec<f32>,
}

impl LayerBiases {
    /// Load biases for one layer. Returns zero-length vecs if the tensors are absent.
    pub fn load(weights: &WeightStore, layer: usize) -> Self {
        let load = |key: &str| -> Vec<f32> {
            let name = format!("blk.{layer}.{key}");
            if weights.has(&name) {
                weights.dequantize(&name).unwrap_or_default()
            } else {
                Vec::new()
            }
        };
        Self {
            q_bias: load("attn_q.bias"),
            k_bias: load("attn_k.bias"),
            v_bias: load("attn_v.bias"),
        }
    }

    /// True if any bias is present for this layer.
    pub fn has_any(&self) -> bool {
        !self.q_bias.is_empty() || !self.k_bias.is_empty() || !self.v_bias.is_empty()
    }
}

/// Apply a bias vector to each row of a 2D batch buffer (CPU path).
///
/// `buf` has shape `[n_tokens, dim]` (row-major).
/// `bias` has shape `[dim]`.
///
/// This replaces the v1 per-token loop that did `copy + add + copy_back` per token.
pub fn apply_bias_to_batch(buf: &mut [f32], bias: &[f32], n_tokens: usize) {
    if bias.is_empty() {
        return;
    }
    let dim = bias.len();
    debug_assert_eq!(buf.len(), n_tokens * dim);
    for row in buf.chunks_mut(dim) {
        for (x, b) in row.iter_mut().zip(bias.iter()) {
            *x += b;
        }
    }
}
