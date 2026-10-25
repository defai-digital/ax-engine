//! Request-owned QSA block selection for Flash Next (`qwen4_exp`).
//!
//! Shared indexer weights project index queries and raw keys. Each request
//! owns the raw key history. Completed causal blocks are mean-pooled in
//! float32, restored to the input dtype, RMS-normalized, and rotated at the
//! block start with the official QSA rotary rounding. Queries are
//! RMS-normalized and rotated at their absolute positions the same way. The score of a block is the sum over query heads of
//! `relu(dot(query_head, block_key)) / sqrt(index_head_dim)`. Each query keeps
//! `token_budget / compression_ratio` complete blocks plus the current
//! partial block, and never a future key.
//!
//! Rotary width is the main-attention RoPE dimension, not an index-head
//! fraction. Selected token indices are returned for a later KV gather; this
//! module does not build a dense T-by-T mask.
//!
//! Cache updates are staged locally and published only when selection
//! succeeds. The input cache is never mutated.
//!
//! `AX_MLX_FLASH_NEXT_QSA_TILED_SCORES` (default off) scores a long prefill
//! chunk as query tiles and keeps only the selected indices. A tile runs
//! only for an MLX 0.32.4 `applegpu_g14d` (M2) or `applegpu_g17s` (M5)
//! GEMM shape whose tests match the monolithic score bits: the proven head
//! counts, and query widths 1024, 512, or 256 plus a folded tail of at least
//! two queries. The production indexer is 4 heads of dim 128. Every other
//! shape keeps one score tensor and the 256 MiB workspace check. When prefill
//! device selection is also on, tiling still wins for a chunk that does not
//! fit that monolithic workspace, and the blocks are then chosen on the host.
//!
//! `AX_MLX_FLASH_NEXT_QSA_PREFILL_DEVICE_SELECT` (default on) chooses prefill
//! blocks on the device. The single-query switch
//! `AX_MLX_FLASH_NEXT_QSA_DEVICE_SELECT` stays on for decode and is a separate
//! kill switch. Short verifier windows stay on
//! `AX_MLX_FLASH_NEXT_QSA_DEVICE_SELECT_MULTI`. Set a switch to `0` to restore
//! the host sort for that path. Prefill reads one scalar flag per chunk: a
//! causal non-finite score, or a row with fewer finite causal blocks than the
//! block budget, selects that chunk on the host (same order, same error). The
//! block-key kernel is compiled at Flash Next load so the first request does
//! not pay that compile. A failed build is logged once and the host sort is
//! used for later prefills. Decode and short windows do not read that flag
//! and do not synchronize. By default they argsort negated scores and do not
//! launch or compile the block-key kernel. A causal NaN is mapped to -inf
//! first. That argsort compares floats. Metal flushes subnormals in the
//! compare, so a subnormal can tie with zero. Decode does not promise host
//! parity for a subnormal score. On a short window a future column is NaN,
//! and that NaN has to sort after +inf. Equal scores, signed zero included,
//! keep the lower block id only while the argsort returns ascending indices
//! inside a tie. That tie order and the NaN order are MLX implementation
//! details, not a documented API. `mlx_argsort_pins_decode_qsa_tie_and_nan_order`
//! fails when a bump breaks them. `AX_MLX_FLASH_NEXT_QSA_DECODE_BLOCK_KEY=1`
//! orders a unique block key instead (the prefill finite-score encoding, a
//! causal NaN in the -inf class, future blocks after every causal key) and
//! does not depend on argsort stability. The key is built from the IEEE bits,
//! so a subnormal keeps the host order. A missing block-key kernel stays on
//! the argsort. A non-finite causal score does not raise on those paths.

#[cfg(test)]
use mlx_sys::{
    bitwise_and, bitwise_invert, bitwise_or, bitwise_xor, equal, eval, left_shift, right_shift,
    view,
};
use std::sync::OnceLock;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxClosure, MlxDtype, MlxMetalKernel,
    MlxQuantizationMode, add, arange, argpartition_axis, argsort_axis, astype, broadcast_to,
    concatenate, contiguous, cos, divide, expand_dims_axes, gpu_device_architecture, less, matmul,
    maximum, minimum, multiply, negative, not_equal, outer, quantized_matmul_with_mode, reshape,
    rms_norm, sin, slice, slice_last_dim, subtract, sum_axis, take_along_axis, transpose,
    where_cond, zeros,
};
use thiserror::Error;

use crate::model::shared::qwen4_exp_fused::{self, Stretch};
use crate::weights::QuantizedWeight;

/// Multimodal (MRoPE) positions of one qwen4_exp request with image input.
///
/// Inside the prompt, a token's `(t, h, w)` comes from the image layout;
/// past it every axis is `physical + delta`. Rotary frequency `i` reads its
/// position from the axis the interleaved `sections` assign to it (axis
/// `i % 3` within that axis' section, else the temporal axis).
#[derive(Debug, PartialEq)]
pub(crate) struct Qwen4ExpMediaPositions {
    prompt: Vec<[i32; 3]>,
    delta: i32,
    sections: [usize; 3],
}

impl Qwen4ExpMediaPositions {
    pub(crate) fn new(prompt: Vec<[i32; 3]>, delta: i32, sections: &[usize]) -> Result<Self> {
        let sections: [usize; 3] = sections
            .try_into()
            .map_err(|_| QsaError::InvalidScalar("mrope_section must have three entries"))?;
        if prompt.is_empty() {
            return Err(QsaError::InvalidScalar("media positions are empty"));
        }
        Ok(Self {
            prompt,
            delta,
            sections,
        })
    }

    /// `(t, h, w)` of the token at physical position `physical`.
    pub(crate) fn at(&self, physical: usize) -> [i32; 3] {
        match self.prompt.get(physical) {
            Some(position) => *position,
            None => {
                let position = i32::try_from(physical)
                    .unwrap_or(i32::MAX)
                    .saturating_add(self.delta);
                [position; 3]
            }
        }
    }

    /// Source axis of each of the `half` rotary frequencies.
    fn source_axes(&self, half: usize) -> Vec<usize> {
        let mut axes = vec![0usize; half];
        for (axis, section) in self.sections.iter().copied().enumerate().skip(1) {
            let mut index = axis;
            while index < section.saturating_mul(3) && index < half {
                axes[index] = axis;
                index += 3;
            }
        }
        axes
    }
}

thread_local! {
    /// Media positions of the qwen4_exp request whose forward is running on
    /// this thread; `None` for text-only requests.
    static MEDIA_SCOPE: std::cell::RefCell<Option<std::sync::Arc<Qwen4ExpMediaPositions>>> =
        const { std::cell::RefCell::new(None) };
}

/// Restores the previous media scope on drop.
#[must_use]
pub(crate) struct Qwen4ExpMediaScope {
    previous: Option<std::sync::Arc<Qwen4ExpMediaPositions>>,
}

impl Drop for Qwen4ExpMediaScope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        MEDIA_SCOPE.with(|current| *current.borrow_mut() = previous);
    }
}

/// Make `media` the rotary positions of every QSA / attention rotary call
/// while the guard lives. `None` keeps plain 1D rotary.
pub(crate) fn scoped_media(
    media: Option<std::sync::Arc<Qwen4ExpMediaPositions>>,
) -> Qwen4ExpMediaScope {
    let previous = MEDIA_SCOPE.with(|current| std::mem::replace(&mut *current.borrow_mut(), media));
    Qwen4ExpMediaScope { previous }
}

fn current_media() -> Option<std::sync::Arc<Qwen4ExpMediaPositions>> {
    MEDIA_SCOPE.with(|current| current.borrow().clone())
}

type Result<T> = std::result::Result<T, QsaError>;

#[derive(Debug, Error)]
pub enum QsaError {
    #[error("qwen4_exp QSA evaluation failed during {stage} ({context}): {detail}")]
    Evaluation {
        stage: &'static str,
        context: String,
        detail: String,
    },
    #[error("qwen4_exp QSA nonfinite score at batch {batch}, query {query}, block {block}")]
    NonFiniteScore {
        batch: usize,
        query: usize,
        block: usize,
    },
    #[error(
        "qwen4_exp QSA geometry is invalid: query_heads={query_heads}, key_heads={key_heads}, head_dim={head_dim}, rotary_dim={rotary_dim}, ratio={compress_ratio}, budget={token_budget}, hidden={hidden_size}"
    )]
    InvalidGeometry {
        query_heads: usize,
        key_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        compress_ratio: usize,
        token_budget: usize,
        hidden_size: usize,
    },
    #[error("qwen4_exp QSA {0} must be finite and positive")]
    InvalidScalar(&'static str),
    #[error("qwen4_exp QSA {tensor} shape {actual:?} does not match {expected}")]
    TensorShape {
        tensor: &'static str,
        expected: String,
        actual: Vec<i32>,
    },
    #[error("qwen4_exp QSA {tensor} dtype {actual:?} is not a floating dtype")]
    NonFloatingDtype {
        tensor: &'static str,
        actual: MlxDtype,
    },
    #[error("qwen4_exp QSA {tensor} has invalid quantization group_size={group_size} bits={bits}")]
    InvalidQuantization {
        tensor: &'static str,
        group_size: i32,
        bits: i32,
    },
    #[error("qwen4_exp QSA {tensor} must not carry a linear bias")]
    UnexpectedLinearBias { tensor: &'static str },
    #[error(
        "qwen4_exp QSA position offset {offset} does not match cached key length {cached_tokens}"
    )]
    PositionMismatch { offset: usize, cached_tokens: usize },
}

/// Validated indexer geometry. `rotary_dim` is the main-attention RoPE width.
#[derive(Clone, Copy, Debug)]
pub struct QsaConfig {
    query_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    compress_ratio: i32,
    token_budget: i32,
    block_topk: i32,
    hidden_size: i32,
    rms_eps: f32,
    rope_base: f32,
}

impl QsaConfig {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        query_heads: usize,
        key_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        compress_ratio: usize,
        token_budget: usize,
        hidden_size: usize,
        rms_eps: f32,
        rope_base: f32,
    ) -> Result<Self> {
        let invalid = || QsaError::InvalidGeometry {
            query_heads,
            key_heads,
            head_dim,
            rotary_dim,
            compress_ratio,
            token_budget,
            hidden_size,
        };
        if key_heads != 1
            || query_heads == 0
            || head_dim == 0
            || rotary_dim == 0
            || !rotary_dim.is_multiple_of(2)
            || rotary_dim > head_dim
            || compress_ratio == 0
            || token_budget == 0
            || !token_budget.is_multiple_of(compress_ratio)
            || hidden_size == 0
        {
            return Err(invalid());
        }
        if !rms_eps.is_finite() || rms_eps <= 0.0 {
            return Err(QsaError::InvalidScalar("rms_eps"));
        }
        if !rope_base.is_finite() || rope_base <= 0.0 {
            return Err(QsaError::InvalidScalar("rope_base"));
        }
        let query_heads = i32::try_from(query_heads).map_err(|_| invalid())?;
        let head_dim = i32::try_from(head_dim).map_err(|_| invalid())?;
        let rotary_dim = i32::try_from(rotary_dim).map_err(|_| invalid())?;
        let compress_ratio = i32::try_from(compress_ratio).map_err(|_| invalid())?;
        let token_budget = i32::try_from(token_budget).map_err(|_| invalid())?;
        let hidden_size = i32::try_from(hidden_size).map_err(|_| invalid())?;
        query_heads
            .checked_add(1)
            .and_then(|h| h.checked_mul(head_dim))
            .ok_or_else(invalid)?;
        Ok(Self {
            query_heads,
            head_dim,
            rotary_dim,
            compress_ratio,
            token_budget,
            block_topk: token_budget / compress_ratio,
            hidden_size,
            rms_eps,
            rope_base,
        })
    }

    pub(crate) fn head_dim(self) -> usize {
        self.head_dim as usize
    }

    pub(crate) fn rotary_dim(self) -> usize {
        self.rotary_dim as usize
    }

    pub(crate) fn compress_ratio(self) -> usize {
        self.compress_ratio as usize
    }

    pub(crate) fn block_topk(self) -> usize {
        self.block_topk as usize
    }

    pub(crate) fn hidden_size(self) -> usize {
        self.hidden_size as usize
    }

    fn query_width(self) -> i32 {
        self.query_heads * self.head_dim
    }

    fn proj_out(self) -> i32 {
        self.query_width() + self.head_dim
    }

    fn max_keep(self) -> usize {
        self.token_budget as usize + self.compress_ratio as usize - 1
    }

    /// Longest context for which every query keeps every visible token: block
    /// `i` is complete once `4 * (i + 1)` tokens are visible, so up to
    /// `block_topk` complete blocks plus a partial tail always fit the budget.
    pub(crate) fn dense_context_limit(self) -> usize {
        self.max_keep()
    }
}

/// Shared index q/k projection and sanitized RMS gains (1 + raw delta).
pub struct QsaIndexerWeights {
    pub qk_proj: QuantizedWeight,
    pub q_norm: MlxArray,
    pub k_norm: MlxArray,
}

/// Pooled, RMS-normalized, rotated keys of the complete blocks `[0, n_blocks)`
/// of the cache that owns them: exactly what [`pooled_block_keys`] returns for
/// `n_blocks` blocks, at `[batch, n_blocks, head_dim]`.
///
/// Derived data: a complete block never changes once it is complete, so this
/// only has to be extended when a new block completes. Never serialized.
#[derive(Clone)]
pub(crate) struct PooledBlocks {
    blocks: MlxArray,
    n_blocks: i32,
}

/// Request-owned raw index keys `[batch, tokens, head_dim]`, plus the derived
/// pooled block keys of every complete block of those keys.
#[derive(Clone)]
pub struct QsaIndexKeyCache {
    keys: Option<MlxArray>,
    pooled: Option<PooledBlocks>,
}

impl QsaIndexKeyCache {
    pub(crate) fn empty() -> Self {
        Self {
            keys: None,
            pooled: None,
        }
    }

    /// Crate-private restore path for durable snapshots. The wire decoder
    /// hands back a plain optional tensor; shape/dtype legality against a
    /// specific model's geometry is checked afterward, once the state is
    /// paired with the model it is being rebound to. The pooled blocks are
    /// derived data and are rebuilt from the restored keys.
    pub(crate) fn from_serialized(keys: Option<MlxArray>) -> Self {
        Self { keys, pooled: None }
    }

    pub(crate) fn keys(&self) -> Option<&MlxArray> {
        self.keys.as_ref()
    }

    /// Pooled keys of the complete blocks of [`Self::keys`], when cached.
    pub(crate) fn pooled_blocks(&self) -> Option<&MlxArray> {
        self.pooled.as_ref().map(|pooled| &pooled.blocks)
    }

    pub(crate) fn token_count(&self) -> Result<usize> {
        let Some(keys) = &self.keys else {
            return Ok(0);
        };
        let shape = keys.shape();
        if shape.len() != 3 || shape[1] < 0 {
            return Err(QsaError::TensorShape {
                tensor: "index key cache",
                expected: "[batch, tokens, head_dim]".to_string(),
                actual: shape,
            });
        }
        Ok(shape[1] as usize)
    }
}

/// Selected gather indices plus the cache to adopt after a successful step.
///
/// A multi-query chunk selects on the host (`tokens`). A single-query step
/// (decode, singleton verification) selects on the device: how many tokens a
/// query keeps depends only on its position, so `device_tokens` has an exact
/// host-known length and no read-back, and therefore no pipeline stall, is
/// needed. Test builds materialize host tokens from it on demand.
pub struct QsaSelection {
    #[cfg(test)]
    gather_indices: MlxArray,
    tokens: Vec<Vec<Vec<i32>>>,
    device_tokens: Option<MlxArray>,
    /// Multi-query device selections: per `(batch, query)` row the packed
    /// device tensor holds `block_len` selected-block tokens, then -1 padding
    /// up to the block segment width, then `tail_len` visible-tail tokens.
    /// Empty for host or singleton-device selections.
    device_row_segments: Vec<(i32, i32)>,
    /// Multi-query device selections: width of the trailing visible-tail
    /// segment in the packed device rows (`compress_ratio - 1`). Zero unless
    /// the selection came from the multi-query device path.
    device_tail_width: i32,
    #[cfg(test)]
    test_host_tokens: std::sync::OnceLock<Vec<Vec<Vec<i32>>>>,
    next_cache: QsaIndexKeyCache,
}

impl QsaSelection {
    #[cfg(test)]
    /// Host selections: `[batch, queries, token_budget + ratio - 1]` int32
    /// indices, `-1` padded. Device selections: `[batch, 1, kept]` exact length.
    pub(crate) fn gather_indices(&self) -> &MlxArray {
        &self.gather_indices
    }

    /// `[batch, 1, kept]` int32 token indices chosen on the device, if any.
    /// Multi-query device selections are `[batch, seq, max_keep]` padded.
    pub(crate) fn device_tokens(&self) -> Option<&MlxArray> {
        self.device_tokens.as_ref()
    }

    /// One query row of device-chosen indices as an exact-length `[kept]`
    /// int32 tensor: the singleton path slices its row; the multi-query window
    /// path joins the row's block and tail segments, dropping -1 padding.
    pub(crate) fn device_tokens_for_query(&self, batch: usize, query: usize) -> Option<MlxArray> {
        let device = self.device_tokens.as_ref()?;
        let shape = device.shape();
        let (b, s) = (batch as i32, query as i32);
        // Singleton decode stores an exact-length row. A segmented row (short
        // window or prefill), including a one-query chunk, pads with -1 and
        // records how much of the block and tail segments are real.
        if self.device_row_segments.is_empty() {
            if shape[1] != 1 || query != 0 {
                return None;
            }
            let kept = shape[2];
            return Some(reshape(
                &slice(device, &[b, 0, 0], &[b + 1, 1, kept], &[1, 1, 1], None),
                &[kept],
                None,
            ));
        }
        let (block_len, tail_len) = self
            .device_row_segments
            .get(batch * shape[1] as usize + query)
            .copied()?;
        let width = shape[2];
        let blocks_width = width - self.device_tail_width;
        let row = slice(device, &[b, s, 0], &[b + 1, s + 1, width], &[1, 1, 1], None);
        let blocks = slice(&row, &[0, 0, 0], &[1, 1, block_len], &[1, 1, 1], None);
        let tail = slice(
            &row,
            &[0, 0, blocks_width],
            &[1, 1, blocks_width + tail_len],
            &[1, 1, 1],
            None,
        );
        let kept = block_len + tail_len;
        let joined = match (block_len > 0, tail_len > 0) {
            (true, true) => concatenate(&[&blocks, &tail], -1, None),
            (true, false) => blocks,
            (false, true) => tail,
            (false, false) => return None,
        };
        Some(reshape(&joined, &[kept], None))
    }

    /// Multi-query device selection (short window or prefill), as opposed to a
    /// singleton row whose length is exact.
    pub(crate) fn device_multi_query(&self) -> bool {
        self.device_tokens.is_some() && !self.device_row_segments.is_empty()
    }

    /// Host-known selected length. Device rows use the closed-form segment
    /// lengths, so a prefill attend does not read the score tensor back.
    pub(crate) fn selected_len(&self, batch: usize, query: usize) -> usize {
        if let Some(device) = &self.device_tokens {
            let seq = device.shape().get(1).copied().unwrap_or(0);
            if seq > 0
                && let Some(&(block_len, tail_len)) = self
                    .device_row_segments
                    .get(batch.saturating_mul(seq as usize).saturating_add(query))
            {
                return (block_len.max(0) + tail_len.max(0)) as usize;
            }
        }
        self.tokens_for_query(batch, query).len()
    }

    /// `[batch, seq, max_keep]` int32 indices with the visible tokens packed at
    /// the front of each row and `-1` after them. Same order as the host gather
    /// buffer, so batched attention can slice a sub-batch without a copy to
    /// the host.
    pub(crate) fn compact_device_indices(&self) -> Option<MlxArray> {
        let device = self.device_tokens.as_ref()?;
        let shape = device.shape();
        if shape.len() != 3 || shape[1] <= 0 || shape[2] <= 0 {
            return None;
        }
        let seq = shape[1] as usize;
        if self.device_row_segments.len() < seq {
            return None;
        }
        let blocks_width = shape[2] - self.device_tail_width;
        if blocks_width < 0 {
            return None;
        }
        let block_lens: Vec<i32> = self.device_row_segments[..seq]
            .iter()
            .map(|row| row.0)
            .collect();
        let tail_lens: Vec<i32> = self.device_row_segments[..seq]
            .iter()
            .map(|row| row.1)
            .collect();
        Some(compact_segmented_tokens(
            device,
            &block_lens,
            &tail_lens,
            blocks_width,
        ))
    }

    pub(crate) fn tokens_for_query(&self, batch: usize, query: usize) -> &[i32] {
        #[cfg(test)]
        if self.device_tokens.is_some() {
            return &self.test_host_tokens.get_or_init(|| {
                let shape = self.device_tokens.as_ref().unwrap().shape();
                (0..shape[0] as usize)
                    .map(|b| {
                        (0..shape[1] as usize)
                            .map(|s| {
                                let row = self.device_tokens_for_query(b, s).unwrap();
                                let row = astype(&row, MlxDtype::Uint32, None);
                                eval(&[&row]);
                                row.data_u32().iter().map(|&token| token as i32).collect()
                            })
                            .collect::<Vec<Vec<i32>>>()
                    })
                    .collect()
            })[batch][query];
        }
        self.tokens
            .get(batch)
            .and_then(|queries| queries.get(query))
            .map_or(&[], Vec::as_slice)
    }

    #[cfg(test)]
    pub(crate) fn next_cache(&self) -> &QsaIndexKeyCache {
        &self.next_cache
    }

    /// Test-only: a host selection from explicit ordered token rows, so a
    /// caller can exercise attention over a hand-built selection without
    /// running the indexer. The rows must already be in the block-then-tail
    /// order [`tokens_from_score_data`] produces.
    #[cfg(test)]
    pub(crate) fn from_host_tokens(tokens: Vec<Vec<Vec<i32>>>) -> Self {
        let batch = tokens.len() as i32;
        let rows = tokens.first().map_or(0, Vec::len) as i32;
        let longest = tokens
            .iter()
            .flat_map(|queries| queries.iter().map(Vec::len))
            .max()
            .unwrap_or(0) as i32;
        let mut indices = vec![-1_i32; (batch * rows * longest).max(1) as usize];
        for (b, queries) in tokens.iter().enumerate() {
            for (s, chosen) in queries.iter().enumerate() {
                let start = (b as i32 * rows + s as i32) * longest;
                indices[start as usize..start as usize + chosen.len()].copy_from_slice(chosen);
            }
        }
        let gather_indices = MlxArray::from_raw_data(
            indices.as_ptr().cast(),
            std::mem::size_of_val(indices.as_slice()),
            &[batch, rows, longest],
            MlxDtype::Int32,
        );
        Self {
            gather_indices,
            tokens,
            device_tokens: None,
            device_row_segments: Vec::new(),
            device_tail_width: 0,
            test_host_tokens: std::sync::OnceLock::new(),
            next_cache: QsaIndexKeyCache::empty(),
        }
    }

    /// Test-only: a multi-query device selection in the packed layout the
    /// prefill selector produces, so a caller can exercise attention over
    /// device rows without running the indexer. `device` is
    /// `[1, seq, width]` int32 with each row's block tokens, `-1` padding, and
    /// its visible tail at `width - tail_width`; `segments` holds the
    /// `(block_len, tail_len)` of each query. The host token lists stay empty,
    /// exactly as they do for a real device selection.
    #[cfg(test)]
    pub(crate) fn from_device_rows(
        device: MlxArray,
        segments: Vec<(i32, i32)>,
        tail_width: i32,
    ) -> Self {
        Self {
            gather_indices: device.clone(),
            tokens: Vec::new(),
            device_tokens: Some(device),
            device_row_segments: segments,
            device_tail_width: tail_width,
            test_host_tokens: std::sync::OnceLock::new(),
            next_cache: QsaIndexKeyCache::empty(),
        }
    }

    pub(crate) fn into_next_cache(self) -> QsaIndexKeyCache {
        self.next_cache
    }
}

pub struct QsaIndexer {
    config: QsaConfig,
    qk_proj: QuantizedWeight,
    /// Raw gains; production reads the pre-cast f32 copies below.
    #[cfg_attr(not(test), allow(dead_code))]
    q_norm: MlxArray,
    #[cfg_attr(not(test), allow(dead_code))]
    k_norm: MlxArray,
    /// The gains widened to f32 once in the constructor: the cast is a constant
    /// of the layer, so every RMS call skips rebuilding it.
    q_norm_f32: MlxArray,
    k_norm_f32: MlxArray,
}

impl QsaIndexer {
    pub(crate) fn new(config: QsaConfig, weights: QsaIndexerWeights) -> Result<Self> {
        validate_projection(
            "index_qk_proj",
            &weights.qk_proj,
            config.proj_out(),
            config.hidden_size,
        )?;
        validate_gain("q_norm", &weights.q_norm, config.head_dim)?;
        validate_gain("k_norm", &weights.k_norm, config.head_dim)?;
        let q_norm_f32 = astype(&weights.q_norm, MlxDtype::Float32, None);
        let k_norm_f32 = astype(&weights.k_norm, MlxDtype::Float32, None);
        Ok(Self {
            config,
            qk_proj: weights.qk_proj,
            q_norm: weights.q_norm,
            k_norm: weights.k_norm,
            q_norm_f32,
            k_norm_f32,
        })
    }

    pub(crate) fn config(&self) -> QsaConfig {
        self.config
    }

    pub(crate) fn projection_quantization_mode(&self) -> MlxQuantizationMode {
        self.qk_proj.mlx_quantization_mode()
    }

    /// Append the new chunk's raw index keys without scoring any block.
    ///
    /// When the whole context fits the token budget every query keeps every
    /// visible token, so selection is the identity and attention can run densely
    /// (see [`QsaConfig::dense_context_limit`]); only the key history still has
    /// to advance so a later, longer context can score its blocks.
    pub(crate) fn append_keys_with_projection(
        &self,
        hidden: &MlxArray,
        cache: &QsaIndexKeyCache,
        position_offset: usize,
        projection: impl FnOnce(&MlxArray, &QuantizedWeight) -> MlxArray,
    ) -> Result<QsaIndexKeyCache> {
        let cfg = self.config;
        let (batch, seq) = rank3_dims("hidden", hidden, cfg.hidden_size)?;
        ensure_floating("hidden", hidden.dtype())?;
        let cached_tokens = cache.token_count()?;
        if position_offset != cached_tokens {
            return Err(QsaError::PositionMismatch {
                offset: position_offset,
                cached_tokens,
            });
        }
        if let Some(keys) = cache.keys() {
            let shape = keys.shape();
            if shape[0] != batch || shape[2] != cfg.head_dim {
                return Err(QsaError::TensorShape {
                    tensor: "index key cache",
                    expected: format!("[{batch}, tokens, {}]", cfg.head_dim),
                    actual: shape,
                });
            }
            ensure_floating("index key cache", keys.dtype())?;
            if keys.dtype() != hidden.dtype() {
                return Err(QsaError::InvalidScalar("index key cache dtype mismatch"));
            }
        }
        let projected = projection(hidden, &self.qk_proj);
        let raw_keys = slice_last_dim(&projected, cfg.query_width(), cfg.proj_out(), None);
        let raw_keys = reshape(&raw_keys, &[batch, seq, cfg.head_dim], None);
        let staged = match cache.keys() {
            Some(past) => contiguous(&concatenate(&[past, &raw_keys], 1, None), None),
            None => contiguous(&raw_keys, None),
        };
        Ok(QsaIndexKeyCache {
            keys: Some(staged),
            pooled: None,
        })
    }

    /// Project the new hidden chunk, append raw keys, and return gather indices
    /// for every new query. `cache` is not modified; adopt [`QsaSelection::next_cache`]
    /// only after the rest of the step succeeds.
    pub(crate) fn select(
        &self,
        hidden: &MlxArray,
        cache: &QsaIndexKeyCache,
        position_offset: usize,
    ) -> Result<QsaSelection> {
        self.select_with_projection(hidden, cache, position_offset, project)
    }

    /// The draft graph supplies its single-row projection contract here.
    /// Ordinary index selection retains its existing shared projection.
    pub(crate) fn select_with_projection(
        &self,
        hidden: &MlxArray,
        cache: &QsaIndexKeyCache,
        position_offset: usize,
        projection: impl FnOnce(&MlxArray, &QuantizedWeight) -> MlxArray,
    ) -> Result<QsaSelection> {
        let single_query = hidden.shape().get(1) == Some(&1);
        let seq = hidden.shape().get(1).copied().unwrap_or(0);
        let short_multi = !single_query
            && device_select_multi_enabled()
            && seq <= crate::fastpath::QSA_DEVICE_SELECT_MULTI_MAX_QUERIES;
        // Short windows stay on the multi-query switch. The prefill switch
        // covers only chunks past that cap, so `MULTI=0` still restores the
        // host sort for verifier windows.
        let prefill_device = !single_query
            && seq > crate::fastpath::QSA_DEVICE_SELECT_MULTI_MAX_QUERIES
            && prefill_device_select_enabled();
        let device_select =
            (single_query && device_select_enabled()) || short_multi || prefill_device;
        self.select_with_mode(hidden, cache, position_offset, projection, device_select)
    }

    pub(crate) fn select_with_mode(
        &self,
        hidden: &MlxArray,
        cache: &QsaIndexKeyCache,
        position_offset: usize,
        projection: impl FnOnce(&MlxArray, &QuantizedWeight) -> MlxArray,
        device_select: bool,
    ) -> Result<QsaSelection> {
        let cfg = self.config;
        let (batch, seq) = rank3_dims("hidden", hidden, cfg.hidden_size)?;
        let total_tokens = position_offset
            .checked_add(seq as usize)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or(QsaError::InvalidScalar("total context length"))?;
        let n_blocks_est = total_tokens / cfg.compress_ratio();
        // Device selection keeps one score tensor when that tensor fits. A
        // chunk that would miss the monolithic workspace check uses the tiled
        // host path instead, which is the path device selection off would take
        // when tiled scores are on. Shapes the tile contract refuses still hit
        // the workspace check below.
        let tiled_plan = i32::try_from(n_blocks_est).ok().and_then(|n_blocks| {
            ScoreTilePlan::try_new(
                batch,
                seq,
                cfg.query_heads,
                cfg.head_dim,
                n_blocks,
                hidden.dtype(),
            )
        });
        let monolithic_fits = monolithic_scores_fit(batch, seq, cfg.query_heads, n_blocks_est);
        let prefer_tiled_host = device_select && tiled_plan.is_some() && !monolithic_fits;
        let score_plan = if device_select && !prefer_tiled_host {
            None
        } else {
            tiled_plan
        };
        let select_on_device = device_select && !prefer_tiled_host;
        if score_plan.is_none() {
            ensure_score_workspace(batch, seq, cfg.query_heads, n_blocks_est)?;
        }
        ensure_floating("hidden", hidden.dtype())?;
        let cached_tokens = cache.token_count()?;
        if position_offset != cached_tokens {
            return Err(QsaError::PositionMismatch {
                offset: position_offset,
                cached_tokens,
            });
        }
        if let Some(keys) = cache.keys() {
            let shape = keys.shape();
            if shape[0] != batch || shape[2] != cfg.head_dim {
                return Err(QsaError::TensorShape {
                    tensor: "index key cache",
                    expected: format!("[{batch}, tokens, {}]", cfg.head_dim),
                    actual: shape,
                });
            }
            ensure_floating("index key cache", keys.dtype())?;
            if keys.dtype() != hidden.dtype() {
                return Err(QsaError::InvalidScalar("index key cache dtype mismatch"));
            }
        }

        let projected = projection(hidden, &self.qk_proj);
        let queries = slice_last_dim(&projected, 0, cfg.query_width(), None);
        let raw_keys = slice_last_dim(&projected, cfg.query_width(), cfg.proj_out(), None);
        let queries = reshape(&queries, &[batch, seq, cfg.query_heads, cfg.head_dim], None);
        let raw_keys = reshape(&raw_keys, &[batch, seq, cfg.head_dim], None);
        let queries = rms_with_scale_f32(&queries, &self.q_norm_f32, cfg.rms_eps);
        let queries = apply_qsa_rotary_offset(
            &queries,
            cfg.rotary_dim,
            cfg.rope_base,
            i32::try_from(position_offset)
                .map_err(|_| QsaError::InvalidScalar("position offset"))?,
            1,
        );

        #[cfg(test)]
        crate::model::qwen4_exp::profiling::mark("qsa_i_proj", &[&queries, &raw_keys]);
        let staged_keys = match cache.keys() {
            Some(past) => contiguous(&concatenate(&[past, &raw_keys], 1, None), None),
            None => contiguous(&raw_keys, None),
        };
        #[cfg(test)]
        crate::model::qwen4_exp::profiling::mark("qsa_j_stage", &[&staged_keys]);
        let total = staged_keys.shape()[1];
        let n_complete = total / cfg.compress_ratio;
        // Media rotary reads 3D positions the text cache does not cover, so
        // those blocks are both recomputed and never adopted.
        let media_active = current_media().is_some();
        let mut next_pooled: Option<PooledBlocks> = None;
        let mut tiled_tokens = None;
        let scores = if n_complete > 0 {
            let blocks = pooled_blocks_for(
                cache,
                &staged_keys,
                n_complete,
                cfg,
                &self.k_norm_f32,
                media_active,
            )?;
            #[cfg(test)]
            crate::model::qwen4_exp::profiling::mark("qsa_k_pool", &[&blocks]);
            // The plan was built from the estimated block count. A staged
            // count that disagrees is re-planned. When that plan is refused,
            // the monolithic scores run only if they fit the 256 MiB check.
            let tiled = if let Some(plan) = score_plan
                .as_ref()
                .filter(|plan| plan.matches_blocks(n_complete))
            {
                Some(select_tokens_tiled(
                    cfg,
                    batch,
                    position_offset,
                    n_complete,
                    &queries,
                    &blocks,
                    plan,
                )?)
            } else if score_plan.is_some() {
                match ScoreTilePlan::try_new(
                    batch,
                    seq,
                    cfg.query_heads,
                    cfg.head_dim,
                    n_complete,
                    hidden.dtype(),
                ) {
                    Some(plan) => Some(select_tokens_tiled(
                        cfg,
                        batch,
                        position_offset,
                        n_complete,
                        &queries,
                        &blocks,
                        &plan,
                    )?),
                    None => {
                        ensure_score_workspace(batch, seq, cfg.query_heads, n_complete as usize)?;
                        None
                    }
                }
            } else {
                None
            };
            let scores = if tiled.is_none() {
                let scores = block_scores(&queries, &blocks);
                #[cfg(test)]
                crate::model::qwen4_exp::profiling::mark("qsa_l_scores", &[&scores]);
                Some(scores)
            } else {
                None
            };
            // Only complete blocks are pooled; the next step extends them
            // instead of recomputing the whole history.
            if !media_active {
                next_pooled = Some(PooledBlocks {
                    blocks,
                    n_blocks: n_complete,
                });
            }
            tiled_tokens = tiled;
            scores
        } else {
            None
        };

        if select_on_device {
            if seq == 1 {
                // No score read-back. A causal non-finite score stays on this
                // path and does not raise.
                let device_tokens =
                    select_tokens_device(cfg, batch, position_offset, n_complete, scores.as_ref())?;
                return Ok(QsaSelection {
                    #[cfg(test)]
                    gather_indices: device_tokens.clone(),
                    tokens: Vec::new(),
                    device_tokens: Some(device_tokens),
                    device_row_segments: Vec::new(),
                    device_tail_width: 0,
                    #[cfg(test)]
                    test_host_tokens: std::sync::OnceLock::new(),
                    next_cache: QsaIndexKeyCache {
                        keys: Some(staged_keys),
                        pooled: next_pooled,
                    },
                });
            }
            // Short verifier windows select on the device when the block tally
            // is complete. Longer prefill chunks use the prefill device path
            // below when that switch is on; anything else keeps the host path.
            if seq <= crate::fastpath::QSA_DEVICE_SELECT_MULTI_MAX_QUERIES
                && n_complete >= cfg.block_topk() as i32
            {
                // No score read-back. Future columns are not candidates.
                let (device_tokens, segments) = select_tokens_device_multi(
                    cfg,
                    batch,
                    seq,
                    position_offset,
                    n_complete,
                    scores.as_ref(),
                )?;
                return Ok(QsaSelection {
                    #[cfg(test)]
                    gather_indices: device_tokens.clone(),
                    tokens: Vec::new(),
                    device_tokens: Some(device_tokens),
                    device_row_segments: segments,
                    device_tail_width: cfg.compress_ratio() as i32 - 1,
                    #[cfg(test)]
                    test_host_tokens: std::sync::OnceLock::new(),
                    next_cache: QsaIndexKeyCache {
                        keys: Some(staged_keys),
                        pooled: next_pooled,
                    },
                });
            }
        }
        if select_on_device
            && seq > crate::fastpath::QSA_DEVICE_SELECT_MULTI_MAX_QUERIES
            && prefill_device_select_enabled()
        {
            // One flag word comes back so a non-finite causal score raises the
            // host error. Rows that fit the block budget and are finite stay
            // on the device; anything else uses the host sort for this chunk.
            let (device_tokens, segments) = select_tokens_device_prefill(
                cfg,
                batch,
                seq,
                position_offset,
                n_complete,
                scores.as_ref(),
            )?;
            let selection = QsaSelection {
                #[cfg(test)]
                gather_indices: device_tokens.clone(),
                tokens: Vec::new(),
                device_tokens: Some(device_tokens),
                device_row_segments: segments,
                device_tail_width: cfg.compress_ratio() as i32 - 1,
                #[cfg(test)]
                test_host_tokens: std::sync::OnceLock::new(),
                next_cache: QsaIndexKeyCache {
                    keys: Some(staged_keys),
                    pooled: next_pooled,
                },
            };
            #[cfg(test)]
            record_device_selection_if_capturing(&selection);
            return Ok(selection);
        }

        let tokens = if let Some(tokens) = tiled_tokens {
            tokens
        } else {
            select_tokens(
                cfg,
                batch,
                seq,
                position_offset,
                n_complete,
                scores.as_ref(),
            )?
        };
        #[cfg(test)]
        record_host_selection(&tokens);
        let gather_indices = pack_gather_indices(cfg, batch, seq, &tokens)?;
        let mut owned = vec![&staged_keys, &gather_indices];
        if let Some(scores) = &scores {
            owned.push(scores);
        }
        mlx_sys::try_eval(&owned).map_err(|detail| QsaError::Evaluation {
            stage: "selected-key cache and gather-index materialization",
            context: format!(
                "keys={:?} gather={:?} scores={:?}",
                staged_keys.shape(),
                gather_indices.shape(),
                scores.as_ref().map(MlxArray::shape),
            ),
            detail,
        })?;

        Ok(QsaSelection {
            #[cfg(test)]
            gather_indices,
            tokens,
            device_tokens: None,
            device_row_segments: Vec::new(),
            device_tail_width: 0,
            #[cfg(test)]
            test_host_tokens: std::sync::OnceLock::new(),
            next_cache: QsaIndexKeyCache {
                keys: Some(staged_keys),
                pooled: next_pooled,
            },
        })
    }
}

fn project(x: &MlxArray, weight: &QuantizedWeight) -> MlxArray {
    if let Some(scales) = &weight.scales {
        let mode = weight.mlx_quantization_mode();
        let biases = match mode {
            MlxQuantizationMode::Affine => weight.biases.as_ref(),
            _ => None,
        };
        quantized_matmul_with_mode(
            x,
            &weight.weight,
            scales,
            biases,
            true,
            Some(weight.group_size),
            Some(weight.bits),
            mode,
            None,
        )
    } else {
        matmul(x, &transpose(&weight.weight, &[1, 0], None), None)
    }
}

/// RMS norm with a gain cast per call, kept for callers without a pre-cast
/// gain and for the differential tests of [`rms_with_scale_f32`].
#[cfg_attr(not(test), allow(dead_code))]
fn rms_with_gain(x: &MlxArray, weight: &MlxArray, eps: f32) -> MlxArray {
    let dtype = x.dtype();
    let x32 = astype(x, MlxDtype::Float32, None);
    let scale = astype(weight, MlxDtype::Float32, None);
    astype(&rms_norm(&x32, Some(&scale), eps, None), dtype, None)
}

/// RMS norm with the gain already widened to f32: callers holding the layer's
/// cached `*_norm_f32` skip the per-call `astype` of the gain, and the result
/// is bit-identical to [`rms_with_gain`].
fn rms_with_scale_f32(x: &MlxArray, scale_f32: &MlxArray, eps: f32) -> MlxArray {
    let dtype = x.dtype();
    let x32 = astype(x, MlxDtype::Float32, None);
    astype(&rms_norm(&x32, Some(scale_f32), eps, None), dtype, None)
}

/// Identity of one QSA rotary cos/sin table. The floats compare by raw bits:
/// equal-bit arguments rebuild byte-identical tables.
#[derive(Clone, Copy, PartialEq)]
struct TrigKey {
    /// `f64::to_bits` of the first position.
    start: u64,
    /// `f64::to_bits` of the position step.
    step: u64,
    count: i32,
    rotary_dim: i32,
    /// `f32::to_bits` of the rotary base.
    rope_base: u32,
    dtype: MlxDtype,
    ndim: usize,
    seq_axis: i32,
}

/// One cached cos/sin pair, already reshaped to the broadcast shape of the
/// caller's tensor, plus the half tables the compiled rotary stretches read.
struct TrigEntry {
    key: TrigKey,
    tables: RotaryTables,
}

/// Full rotary tables of one [`TrigKey`] and their `rotary_dim / 2` halves.
///
/// The full tables broadcast like the caller's tensor with the last axis
/// `rotary_dim`; the halves keep the same shape with the last axis `h`, so
/// they broadcast against the half views of the rotated tensor.
#[derive(Clone)]
struct RotaryTables {
    cos: MlxArray,
    sin: MlxArray,
    cos_h: MlxArray,
    sin_h: MlxArray,
}

thread_local! {
    /// Rotary cos/sin tables shared by every QSA layer of one decode step.
    /// `MlxArray` clone is a refcount bump, so hits are cheap.
    static TRIG_CACHE: std::cell::RefCell<Vec<TrigEntry>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// FIFO capacity of [`TRIG_CACHE`]; a decode step needs one entry per distinct
/// `(offset, dtype, shape)` rotary table and seldom more.
const TRIG_CACHE_ENTRIES: usize = 8;

fn rope_cache_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = ROPE_CACHE_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    crate::fastpath::flash_next_qsa_rope_cache_enabled()
}

#[cfg(test)]
thread_local! {
    static ROPE_CACHE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Run `action` with the rotary table cache forced on or off on this thread.
#[cfg(test)]
pub(crate) fn with_rope_cache<T>(enabled: bool, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ROPE_CACHE_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(ROPE_CACHE_OVERRIDE.with(|value| value.replace(Some(enabled))));
    action()
}

#[cfg(test)]
fn reset_rope_cache_for_test() {
    TRIG_CACHE.with(|cache| cache.borrow_mut().clear());
}

#[cfg(test)]
fn rope_cache_len_for_test() -> usize {
    TRIG_CACHE.with(|cache| cache.borrow().len())
}

fn pooled_cache_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = POOLED_CACHE_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    crate::fastpath::flash_next_qsa_pooled_cache_enabled()
}

#[cfg(test)]
thread_local! {
    static POOLED_CACHE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Run `action` with the pooled block key cache forced on or off on this
/// thread.
#[cfg(test)]
pub(crate) fn with_pooled_cache<T>(enabled: bool, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            POOLED_CACHE_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(POOLED_CACHE_OVERRIDE.with(|value| value.replace(Some(enabled))));
    action()
}

/// Broadcast shape of the cos/sin table for `x`: ones except the sequence axis
/// (`count`) and the last axis (`rotary_dim`).
fn trig_shape(x: &MlxArray, rotary_dim: i32, seq_axis: i32) -> Vec<i32> {
    let mut shape = vec![1i32; x.ndim()];
    shape[seq_axis as usize] = x.shape()[seq_axis as usize];
    shape[x.ndim() - 1] = rotary_dim;
    shape
}

/// Official Flash Next QSA rotary.
///
/// Inverse frequencies and angles stay float32. Cos and sin are cast to the
/// projection dtype once. `rotate_half` and the products/sum then run in that
/// dtype on the first `rotary_dim` channels; later channels pass through.
/// Certification notes must record this as a production-math correction with
/// M2 evidence.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn apply_qsa_rotary(
    x: &MlxArray,
    rotary_dim: i32,
    rope_base: f32,
    positions: &MlxArray,
    seq_axis: i32,
) -> MlxArray {
    let (cos, sin) = qsa_rotary_cos_sin(positions, rotary_dim, rope_base, x.dtype());
    apply_rotary_trig(x, rotary_dim, cos, sin, seq_axis)
}

/// Rotate the first `rotary_dim` channels of `x` with `[seq, rotary_dim]`
/// cos/sin already in `x`'s dtype; later channels pass through.
fn apply_rotary_trig(
    x: &MlxArray,
    rotary_dim: i32,
    cos: MlxArray,
    sin: MlxArray,
    seq_axis: i32,
) -> MlxArray {
    let trig = trig_shape(x, rotary_dim, seq_axis);
    let cos = reshape(&cos, &trig, None);
    let sin = reshape(&sin, &trig, None);
    apply_rotary_trig_shaped(x, rotary_dim, &cos, &sin)
}

/// Rotate the first `rotary_dim` channels with cos/sin already reshaped to
/// [`trig_shape`]; later channels pass through.
fn apply_rotary_trig_shaped(
    x: &MlxArray,
    rotary_dim: i32,
    cos: &MlxArray,
    sin: &MlxArray,
) -> MlxArray {
    let shape = x.shape();
    let ndim = shape.len();
    let head_dim = shape[ndim - 1];
    let rope_part = slice_last_dim(x, 0, rotary_dim, None);
    let rotated = add(
        &multiply(&rope_part, cos, None),
        &multiply(&rotate_half_last(&rope_part), sin, None),
        None,
    );
    if rotary_dim < head_dim {
        concatenate(
            &[&rotated, &slice_last_dim(x, rotary_dim, head_dim, None)],
            -1,
            None,
        )
    } else {
        rotated
    }
}

/// Key component for a compiled stretch whose inputs carry the dtype.
///
/// The rotary closure bodies bake in no scalar and read their dtype from the
/// inputs, but MLX still traces one body per input dtype, so the registry key
/// must select it: a past key without the dtype reused a closure across a
/// dtype switch. Stretches that bake the dtype into the closure (the QSA
/// output gate, the MoE routing tail) also key on it.
pub(crate) fn dtype_key(dtype: MlxDtype) -> i64 {
    match dtype {
        MlxDtype::Float32 => 0,
        MlxDtype::Float16 => 1,
        MlxDtype::Bfloat16 => 2,
        _ => 3,
    }
}

/// Closure body of the lower rotary half: `x1 * cos_h + (-x2) * sin_h`, with
/// the explicit negative kept so each product and the sum round exactly like
/// the eager ops. No scalar is baked in; the dtype comes from the inputs.
fn qsa_rope_lo_closure() -> Option<MlxClosure> {
    qwen4_exp_fused::compile(|inputs| {
        let (x1, x2, cos_h, sin_h) = (inputs.get(0), inputs.get(1), inputs.get(2), inputs.get(3));
        vec![add(
            &multiply(&x1, &cos_h, None),
            &multiply(&negative(&x2, None), &sin_h, None),
            None,
        )]
    })
}

/// Closure body of the upper rotary half: `x2 * cos_h + x1 * sin_h`.
fn qsa_rope_hi_closure() -> Option<MlxClosure> {
    qwen4_exp_fused::compile(|inputs| {
        let (x1, x2, cos_h, sin_h) = (inputs.get(0), inputs.get(1), inputs.get(2), inputs.get(3));
        vec![add(
            &multiply(&x2, &cos_h, None),
            &multiply(&x1, &sin_h, None),
            None,
        )]
    })
}

/// Rotate the first `rotary_dim` channels of `x` with a cached table pair,
/// through the two compiled half stretches when they may apply.
///
/// The halves read `x1 = x[..., :h]` and `x2 = x[..., h:rotary_dim]` against
/// the half tables and equal the eager halves bit for bit; the half-table
/// slices, the final concatenate and the tail pass-through stay eager. The
/// row gate is the caller's sequence axis (a pooled block step rotates
/// `n_blocks_new` rows). Both stretches must apply; anything else - an odd
/// `rotary_dim`, a row count past the fuse cap, a failed compile or apply -
/// falls back to the full eager sequence.
fn apply_rotary_trig_split(
    x: &MlxArray,
    rotary_dim: i32,
    tables: &RotaryTables,
    seq_axis: i32,
) -> MlxArray {
    if rotary_dim % 2 != 0 {
        return apply_rotary_trig_shaped(x, rotary_dim, &tables.cos, &tables.sin);
    }
    let h = rotary_dim / 2;
    let rows = x.shape()[seq_axis as usize];
    let x1 = slice_last_dim(x, 0, h, None);
    let x2 = slice_last_dim(x, h, rotary_dim, None);
    let key = [
        dtype_key(x.dtype()),
        i64::from(h),
        x.ndim() as i64,
        i64::from(seq_axis),
    ];
    let inputs = [&x1, &x2, &tables.cos_h, &tables.sin_h];
    let Some(lo) =
        qwen4_exp_fused::apply(Stretch::QsaRopeLo, rows, key, qsa_rope_lo_closure, &inputs)
    else {
        return apply_rotary_trig_shaped(x, rotary_dim, &tables.cos, &tables.sin);
    };
    let Some(hi) =
        qwen4_exp_fused::apply(Stretch::QsaRopeHi, rows, key, qsa_rope_hi_closure, &inputs)
    else {
        // The upper half failed, so drop the compiled lower half with it: the
        // full eager sequence is the only anchor the caller can trust.
        return apply_rotary_trig_shaped(x, rotary_dim, &tables.cos, &tables.sin);
    };
    let head_dim = x.shape()[x.ndim() - 1];
    if rotary_dim < head_dim {
        concatenate(
            &[&lo, &hi, &slice_last_dim(x, rotary_dim, head_dim, None)],
            -1,
            None,
        )
    } else {
        concatenate(&[&lo, &hi], -1, None)
    }
}

pub(crate) fn apply_qsa_rotary_offset(
    x: &MlxArray,
    rotary_dim: i32,
    rope_base: f32,
    offset: i32,
    seq_axis: i32,
) -> MlxArray {
    let seq = x.shape()[seq_axis as usize];
    if let Some(media) = current_media() {
        let start = usize::try_from(offset).unwrap_or(0);
        let positions: Vec<[i32; 3]> = (0..seq as usize).map(|i| media.at(start + i)).collect();
        return apply_qsa_mrope(x, rotary_dim, rope_base, &positions, &media, seq_axis);
    }
    let tables = cached_rotary_trig(
        x,
        rotary_dim,
        rope_base,
        f64::from(offset),
        1.0,
        seq,
        seq_axis,
    );
    apply_rotary_trig_split(x, rotary_dim, &tables, seq_axis)
}

fn qsa_rotary_cos_sin(
    positions: &MlxArray,
    rotary_dim: i32,
    rope_base: f32,
    dtype: MlxDtype,
) -> (MlxArray, MlxArray) {
    let half = rotary_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|index| 1.0 / rope_base.powf((2 * index) as f32 / rotary_dim as f32))
        .collect();
    let inv_freq = MlxArray::from_f32_slice(&inv);
    let theta = outer(&astype(positions, MlxDtype::Float32, None), &inv_freq, None);
    let cos_h = cos(&theta, None);
    let sin_h = sin(&theta, None);
    let cos = concatenate(&[&cos_h, &cos_h], -1, None);
    let sin = concatenate(&[&sin_h, &sin_h], -1, None);
    (astype(&cos, dtype, None), astype(&sin, dtype, None))
}

/// Cos/sin for `count` positions from `start` in steps of `step`, reshaped to
/// [`trig_shape`] of `x`, with the half tables [`apply_rotary_trig_split`]
/// reads.
///
/// The table depends only on the key's arguments, and every operation below is
/// the same one the uncached path always ran, so a cache hit is bit-identical
/// to a rebuild. The halves are sliced once per miss; hits clone them out (a
/// refcount bump, no ops).
fn cached_rotary_trig(
    x: &MlxArray,
    rotary_dim: i32,
    rope_base: f32,
    start: f64,
    step: f64,
    count: i32,
    seq_axis: i32,
) -> RotaryTables {
    let key = TrigKey {
        start: start.to_bits(),
        step: step.to_bits(),
        count,
        rotary_dim,
        rope_base: rope_base.to_bits(),
        dtype: x.dtype(),
        ndim: x.ndim(),
        seq_axis,
    };
    let enabled = rope_cache_enabled();
    if enabled {
        // Clone the tables out and drop the borrow before any MLX call.
        let cached = TRIG_CACHE.with(|cache| {
            cache
                .borrow()
                .iter()
                .find(|entry| entry.key == key)
                .map(|entry| entry.tables.clone())
        });
        if let Some(tables) = cached {
            return tables;
        }
    }
    let positions = arange(
        start,
        start + f64::from(count) * step,
        step,
        MlxDtype::Float32,
        None,
    );
    let (cos, sin) = qsa_rotary_cos_sin(&positions, rotary_dim, rope_base, x.dtype());
    let trig = trig_shape(x, rotary_dim, seq_axis);
    let cos = reshape(&cos, &trig, None);
    let sin = reshape(&sin, &trig, None);
    let half = rotary_dim / 2;
    let tables = RotaryTables {
        cos_h: slice_last_dim(&cos, 0, half, None),
        sin_h: slice_last_dim(&sin, 0, half, None),
        cos,
        sin,
    };
    if enabled {
        TRIG_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() >= TRIG_CACHE_ENTRIES {
                cache.remove(0);
            }
            cache.push(TrigEntry {
                key,
                tables: tables.clone(),
            });
        });
    }
    tables
}

/// Interleaved MRoPE with the official QSA rounding: float32 angles, cos and
/// sin cast to `x`'s dtype once. Each frequency's angle is its own axis
/// position times the inverse frequency, the same single float32 product the
/// 1D path computes, so equal axes reproduce [`apply_qsa_rotary`].
fn apply_qsa_mrope(
    x: &MlxArray,
    rotary_dim: i32,
    rope_base: f32,
    positions: &[[i32; 3]],
    media: &Qwen4ExpMediaPositions,
    seq_axis: i32,
) -> MlxArray {
    let half = (rotary_dim / 2) as usize;
    let inv: Vec<f32> = (0..half)
        .map(|index| 1.0 / rope_base.powf((2 * index) as f32 / rotary_dim as f32))
        .collect();
    let axes = media.source_axes(half);
    let mut theta = Vec::with_capacity(positions.len() * half);
    for position in positions {
        for (index, &inv_freq) in inv.iter().enumerate() {
            theta.push(position[axes[index]] as f32 * inv_freq);
        }
    }
    let theta = MlxArray::from_f32_slice(&theta);
    let theta = reshape(&theta, &[positions.len() as i32, half as i32], None);
    let cos_h = cos(&theta, None);
    let sin_h = sin(&theta, None);
    let cos_full = astype(&concatenate(&[&cos_h, &cos_h], -1, None), x.dtype(), None);
    let sin_full = astype(&concatenate(&[&sin_h, &sin_h], -1, None), x.dtype(), None);
    apply_rotary_trig(x, rotary_dim, cos_full, sin_full, seq_axis)
}

fn rotate_half_last(x: &MlxArray) -> MlxArray {
    let dim = x.shape()[x.ndim() - 1];
    let half = dim / 2;
    concatenate(
        &[
            &negative(&slice_last_dim(x, half, dim, None), None),
            &slice_last_dim(x, 0, half, None),
        ],
        -1,
        None,
    )
}

/// Pooled block keys of every complete block of `keys`.
fn pooled_block_keys(
    keys: &MlxArray,
    n_complete: i32,
    cfg: QsaConfig,
    k_norm_f32: &MlxArray,
) -> Result<MlxArray> {
    pooled_block_keys_range(keys, 0, n_complete, cfg, k_norm_f32)
}

/// Pooled block keys of the complete blocks `[first_block, n_complete)`.
///
/// Runs exactly the operations [`pooled_block_keys`] runs on the raw rows of
/// those blocks only. Every block is independent: the mean reduces over its
/// own `compress_ratio` rows, RMS normalization is per row, and the rotary
/// angle is the block's absolute position, so splitting a range is
/// bit-identical to pooling all of it at once.
fn pooled_block_keys_range(
    keys: &MlxArray,
    first_block: i32,
    n_complete: i32,
    cfg: QsaConfig,
    k_norm_f32: &MlxArray,
) -> Result<MlxArray> {
    if first_block < 0 || first_block >= n_complete {
        return Err(QsaError::InvalidScalar("pooled block range"));
    }
    let shape = keys.shape();
    let (batch, dim) = (shape[0], shape[2]);
    let ratio = cfg.compress_ratio;
    let start = first_block * ratio;
    let used = n_complete * ratio;
    let n_new = n_complete - first_block;
    let grouped = slice(keys, &[0, start, 0], &[batch, used, dim], &[1, 1, 1], None);
    let grouped = reshape(&grouped, &[batch, n_new, ratio, dim], None);
    let grouped_f32 = astype(&grouped, MlxDtype::Float32, None);
    let summed = sum_axis(&grouped_f32, 2, false, None);
    let mean = divide(&summed, &MlxArray::from_f32(ratio as f32), None);
    let pooled = astype(&mean, keys.dtype(), None);
    let pooled = rms_with_scale_f32(&pooled, k_norm_f32, cfg.rms_eps);
    Ok(rope_block_starts(
        &pooled,
        cfg.rotary_dim,
        cfg.rope_base,
        ratio,
        first_block,
    ))
}

/// Pooled block keys for `[0, n_complete)`, extending the cache's derived
/// blocks instead of recomputing the whole history when possible.
///
/// Reuse is limited to the case where it is bit-identical to a full
/// recomputation and the cache's blocks belong to the same key history:
/// matching batch, dtype, head width, and no media scope (media rotary reads
/// positions the cached table does not cover). Anything else falls back to
/// [`pooled_block_keys`].
fn pooled_blocks_for(
    cache: &QsaIndexKeyCache,
    keys: &MlxArray,
    n_complete: i32,
    cfg: QsaConfig,
    k_norm_f32: &MlxArray,
    media_active: bool,
) -> Result<MlxArray> {
    let reusable = match &cache.pooled {
        Some(cached) if pooled_cache_enabled() && !media_active => {
            reusable_pooled(cached, keys, n_complete, cfg)
        }
        _ => None,
    };
    let Some(cached) = reusable else {
        return pooled_block_keys(keys, n_complete, cfg, k_norm_f32);
    };
    if cached.n_blocks == n_complete {
        return Ok(cached.blocks.clone());
    }
    let new = pooled_block_keys_range(keys, cached.n_blocks, n_complete, cfg, k_norm_f32)?;
    Ok(concatenate(&[&cached.blocks, &new], 1, None))
}

/// The cached blocks, when they are a prefix of `keys`' complete blocks and
/// carry the same layout. `n_blocks == n_complete` means no new block.
fn reusable_pooled<'a>(
    cached: &'a PooledBlocks,
    keys: &MlxArray,
    n_complete: i32,
    cfg: QsaConfig,
) -> Option<&'a PooledBlocks> {
    let shape = cached.blocks.shape();
    if cached.n_blocks <= 0
        || cached.n_blocks > n_complete
        || shape.len() != 3
        || shape[0] != keys.shape()[0]
        || shape[1] != cached.n_blocks
        || shape[2] != cfg.head_dim
        || cached.blocks.dtype() != keys.dtype()
    {
        return None;
    }
    Some(cached)
}

/// Rotate pooled block keys at their block starts, starting at `first_block`.
fn rope_block_starts(
    blocks: &MlxArray,
    rotary_dim: i32,
    rope_base: f32,
    ratio: i32,
    first_block: i32,
) -> MlxArray {
    let n_blocks = blocks.shape()[1];
    if let Some(media) = current_media() {
        // A pooled block key rotates with its first token's 3D position.
        let positions: Vec<[i32; 3]> = (0..n_blocks as usize)
            .map(|block| media.at((first_block as usize + block) * ratio as usize))
            .collect();
        return apply_qsa_mrope(blocks, rotary_dim, rope_base, &positions, &media, 1);
    }
    let tables = cached_rotary_trig(
        blocks,
        rotary_dim,
        rope_base,
        f64::from(first_block) * f64::from(ratio),
        f64::from(ratio),
        n_blocks,
        1,
    );
    apply_rotary_trig_split(blocks, rotary_dim, &tables, 1)
}

/// Largest query tile. One tile that covers every query is the monolithic GEMM,
/// so that shape stays on the monolithic path.
const QSA_TILE_MAX_QUERIES: i32 = 1024;
/// Query widths whose f32 GEMM matched the monolithic scores. The planner
/// picks the largest of these that fits. It does not search for another width.
/// Order is preference (largest first), not an identity: folding uses
/// [`QSA_SMALLEST_PROVEN_TILE_WIDTH`].
const QSA_PROVEN_TILE_WIDTHS: [i32; 3] = [1024, 512, 256];
/// Smallest proven query width. Folding and the test-only tile cap compare
/// against this value so reordering [`QSA_PROVEN_TILE_WIDTHS`] cannot change
/// which one-row tails fold.
const QSA_SMALLEST_PROVEN_TILE_WIDTH: i32 = {
    let mut smallest = QSA_PROVEN_TILE_WIDTHS[0];
    let mut index = 1;
    while index < QSA_PROVEN_TILE_WIDTHS.len() {
        if QSA_PROVEN_TILE_WIDTHS[index] < smallest {
            smallest = QSA_PROVEN_TILE_WIDTHS[index];
        }
        index += 1;
    }
    smallest
};
/// Scratch budget for one tile: the score graph plus the f32 casts.
const QSA_TILE_WORKSPACE_BUDGET: usize = 128 * 1024 * 1024;
/// Monolithic score-workspace ceiling. Each issued tile is checked against it.
const QSA_SCORE_WORKSPACE_LIMIT: usize = 256 * 1024 * 1024;
/// GPU architectures whose tiled f32 GEMM matched the monolithic score bits.
const QSA_TILED_SCORE_ARCHES: &[&str] = &["applegpu_g14d", "applegpu_g17s"];
const QSA_TILED_MLX_VERSION: &str = "0.32.4";

/// Query tiles for one score. `query_rows` sums to the chunk length.
/// `workspace_bytes` is the largest tile's scratch estimate.
/// `dispatch_identity` names the MLX 0.32.4 dtype and tile shape.
struct ScoreTilePlan {
    query_rows: Vec<i32>,
    workspace_bytes: usize,
    dispatch_identity: u64,
    n_blocks: i32,
}

impl ScoreTilePlan {
    fn matches_blocks(&self, n_complete: i32) -> bool {
        self.n_blocks == n_complete
    }

    /// `None` keeps the monolithic scores. A plan exists only when every tile
    /// is a proven GEMM width (or the folded tail of at least two queries),
    /// the scratch estimate fits the tile budget, and at least two tiles are
    /// required.
    fn try_new(
        batch: i32,
        seq: i32,
        heads: i32,
        dim: i32,
        n_blocks: i32,
        dtype: MlxDtype,
    ) -> Option<Self> {
        if !tiled_scores_enabled()
            || batch < 1
            || seq < 2
            || heads < 1
            || dim < 1
            || n_blocks < 1
            || !tiled_score_contract(batch, heads, dim, n_blocks, dtype)
        {
            return None;
        }
        let (max_rows, budget) = tile_limits();
        let query_rows = choose_query_tiles(batch, seq, heads, dim, n_blocks, max_rows, budget)?;
        let mut workspace_bytes = 0usize;
        for &rows in &query_rows {
            let bytes = estimate_tile_workspace(batch, rows, heads, dim, n_blocks)?;
            workspace_bytes = workspace_bytes.max(bytes);
        }
        Some(Self {
            dispatch_identity: score_dispatch_identity(
                dtype,
                batch,
                heads,
                dim,
                n_blocks,
                &query_rows,
            ),
            query_rows,
            workspace_bytes,
            n_blocks,
        })
    }
}

fn tiled_scores_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = TILED_SCORES_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    crate::fastpath::flash_next_qsa_tiled_scores_enabled()
}

fn tile_limits() -> (i32, usize) {
    #[cfg(test)]
    if let Some((rows, bytes)) = TILE_LIMITS_OVERRIDE.with(std::cell::Cell::get) {
        return (rows.max(1), bytes.max(1));
    }
    (QSA_TILE_MAX_QUERIES, QSA_TILE_WORKSPACE_BUDGET)
}

/// Shapes measured against the monolithic f32 GEMM on MLX 0.32.4.
/// The production indexer is 4 heads of dim 128, and that width matches only
/// once there are at least 129 key blocks (`context / 4`). The other pairs
/// are the head counts the bit tests prove. A one-row GEMV is rejected later,
/// when the tiles are split.
fn tiled_score_contract(batch: i32, heads: i32, dim: i32, n_blocks: i32, dtype: MlxDtype) -> bool {
    if !(1..=2).contains(&batch) {
        return false;
    }
    if !matches!(
        dtype,
        MlxDtype::Float32 | MlxDtype::Float16 | MlxDtype::Bfloat16
    ) {
        return false;
    }
    let shape_ok = match (heads, dim) {
        (4, 128) => n_blocks >= 129,
        (1, 64) | (2, 32) | (2, 4) => n_blocks >= 1,
        _ => false,
    };
    shape_ok && tiled_score_runtime_matches()
}

/// Arch, MLX version, and the production indexer contract. Outer prefill
/// chunks stay at the base size when this is false.
pub(crate) fn flash_next_qsa_tiled_score_gate_open() -> bool {
    tiled_score_contract(1, 4, 128, 129, MlxDtype::Float32)
}

fn tiled_score_runtime_matches() -> bool {
    #[cfg(test)]
    if let Some(forced) = TILED_RUNTIME_OVERRIDE.with(std::cell::Cell::get) {
        return forced;
    }
    static ARCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static VERSION: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let arch_ok = if let Some(matches) = ARCH.get() {
        *matches
    } else if let Ok(arch) = gpu_device_architecture() {
        *ARCH.get_or_init(|| QSA_TILED_SCORE_ARCHES.contains(&arch.as_str()))
    } else {
        false
    };
    let version_ok = if let Some(matches) = VERSION.get() {
        *matches
    } else if let Ok(version) = mlx_sys::runtime_version() {
        *VERSION.get_or_init(|| version == QSA_TILED_MLX_VERSION)
    } else {
        false
    };
    arch_ok && version_ok
}

fn score_workspace_limit() -> usize {
    #[cfg(test)]
    if let Some(limit) = SCORE_WORKSPACE_LIMIT_OVERRIDE.with(std::cell::Cell::get) {
        return limit.max(1);
    }
    QSA_SCORE_WORKSPACE_LIMIT
}

fn monolithic_scores_fit(batch: i32, rows: i32, heads: i32, n_blocks: usize) -> bool {
    full_score_workspace_bytes(batch, rows, heads, n_blocks)
        .is_some_and(|bytes| bytes <= score_workspace_limit())
}

fn full_score_workspace_bytes(batch: i32, rows: i32, heads: i32, n_blocks: usize) -> Option<usize> {
    (batch as usize)
        .checked_mul(rows as usize)
        .and_then(|n| n.checked_mul(heads as usize))
        .and_then(|n| n.checked_mul(n_blocks))
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
}

fn ensure_score_workspace(batch: i32, rows: i32, heads: i32, n_blocks: usize) -> Result<()> {
    let bytes = full_score_workspace_bytes(batch, rows, heads, n_blocks);
    if bytes.is_none_or(|n| n > score_workspace_limit()) {
        return Err(QsaError::InvalidScalar(
            "score workspace; use a smaller prefill chunk",
        ));
    }
    Ok(())
}

/// `B * Qt * N * 4 * (2H + 2)` for the dots, ReLU, summed reduction, and the
/// divided output, plus the f32 query cast, its contiguous copy, and the key
/// cast and transpose.
fn estimate_tile_workspace(
    batch: i32,
    rows: i32,
    heads: i32,
    dim: i32,
    n_blocks: i32,
) -> Option<usize> {
    let b = i64::from(batch);
    let q = i64::from(rows);
    let h = i64::from(heads);
    let d = i64::from(dim);
    let n = i64::from(n_blocks);
    let scores = b
        .checked_mul(q)?
        .checked_mul(n)?
        .checked_mul(4)?
        .checked_mul(2 * h + 2)?;
    let q_cast = b
        .checked_mul(q)?
        .checked_mul(h)?
        .checked_mul(d)?
        .checked_mul(4)?;
    let k_cast = 2i64
        .checked_mul(b)?
        .checked_mul(n)?
        .checked_mul(d)?
        .checked_mul(4)?;
    let total = scores
        .checked_add(q_cast)?
        .checked_add(q_cast)?
        .checked_add(k_cast)?;
    usize::try_from(total).ok()
}

fn tile_within_budget(
    batch: i32,
    rows: i32,
    heads: i32,
    dim: i32,
    n_blocks: i32,
    budget: usize,
) -> bool {
    estimate_tile_workspace(batch, rows, heads, dim, n_blocks).is_some_and(|bytes| bytes <= budget)
        && full_score_workspace_bytes(batch, rows, heads, n_blocks as usize)
            .is_some_and(|bytes| bytes <= score_workspace_limit())
}

fn candidate_tile_widths(max_rows: i32) -> Vec<i32> {
    let mut widths = Vec::new();
    for &width in &QSA_PROVEN_TILE_WIDTHS {
        if width <= max_rows {
            widths.push(width);
        }
    }
    // Tests cap the tile below the proven set so a small GEMM can still be
    // checked. Production never sets that cap, so it cannot invent a width.
    #[cfg(test)]
    if TILE_LIMITS_OVERRIDE.with(std::cell::Cell::get).is_some()
        && (1..QSA_SMALLEST_PROVEN_TILE_WIDTH).contains(&max_rows)
        && !widths.contains(&max_rows)
    {
        widths.push(max_rows);
    }
    widths
}

fn choose_query_tiles(
    batch: i32,
    seq: i32,
    heads: i32,
    dim: i32,
    n_blocks: i32,
    max_rows: i32,
    budget: usize,
) -> Option<Vec<i32>> {
    for width in candidate_tile_widths(max_rows) {
        if width > seq || !tile_within_budget(batch, width, heads, dim, n_blocks, budget) {
            continue;
        }
        if let Some(rows) = split_query_tiles(seq, width, heads, |rows| {
            tile_within_budget(batch, rows, heads, dim, n_blocks, budget)
        }) {
            return Some(rows);
        }
    }
    None
}

/// Split `seq` into widths of at most `width`. A one-row tail is folded into
/// the previous tile when there is one head (a GEMV) or `width` is at least
/// [`QSA_SMALLEST_PROVEN_TILE_WIDTH`]. The folded tail is at least two queries.
fn split_query_tiles(
    seq: i32,
    width: i32,
    heads: i32,
    fits: impl Fn(i32) -> bool,
) -> Option<Vec<i32>> {
    if width < 1 || seq < 2 {
        return None;
    }
    let mut rows = Vec::new();
    let mut left = seq;
    while left > 0 {
        let take = width.min(left);
        rows.push(take);
        left -= take;
    }
    let fold_one_row_tail = heads == 1 || width >= QSA_SMALLEST_PROVEN_TILE_WIDTH;
    if fold_one_row_tail && rows.last().copied() == Some(1) {
        rows.pop();
        let last = rows.last_mut()?;
        *last = last.checked_add(1)?;
    }
    if rows.len() < 2 || rows.iter().any(|&row| row.saturating_mul(heads) < 2) {
        return None;
    }
    if rows.iter().copied().all(fits) {
        Some(rows)
    } else {
        None
    }
}

fn score_dispatch_identity(
    dtype: MlxDtype,
    batch: i32,
    heads: i32,
    dim: i32,
    n_blocks: i32,
    rows: &[i32],
) -> u64 {
    let mix = |id: u64, value: u64| id.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(value);
    let mut id = 0x0032_0004u64;
    id = mix(id, u64::try_from(dtype_key(dtype)).unwrap_or(0));
    id = mix(id, u64::try_from(batch).unwrap_or(0));
    id = mix(id, u64::try_from(heads).unwrap_or(0));
    id = mix(id, u64::try_from(dim).unwrap_or(0));
    id = mix(id, u64::try_from(n_blocks).unwrap_or(0));
    id = mix(id, rows.len() as u64);
    for &row in rows {
        id = mix(id, u64::try_from(row).unwrap_or(0));
    }
    id
}

fn block_scores(queries: &MlxArray, blocks: &MlxArray) -> MlxArray {
    let q_shape = queries.shape();
    let b_shape = blocks.shape();
    let (batch, seq, heads, dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let n_blocks = b_shape[1];
    let q32 = astype(queries, MlxDtype::Float32, None);
    let k32 = astype(blocks, MlxDtype::Float32, None);
    let q = reshape(&q32, &[batch, seq * heads, dim], None);
    let k_t = transpose(&k32, &[0, 2, 1], None);
    let dots = matmul(&q, &k_t, None);
    let dots = reshape(&dots, &[batch, seq, heads, n_blocks], None);
    let relu = maximum(&dots, &MlxArray::from_f32(0.0), None);
    let summed = sum_axis(&relu, 2, false, None);
    divide(&summed, &MlxArray::from_f32((dim as f32).sqrt()), None)
}

fn select_tokens(
    cfg: QsaConfig,
    batch: i32,
    seq: i32,
    position_offset: usize,
    n_complete: i32,
    scores: Option<&MlxArray>,
) -> Result<Vec<Vec<Vec<i32>>>> {
    let score_data = if let Some(scores) = scores {
        materialize_scores(scores, "block-score materialization")?;
        scores.data_f32().to_vec()
    } else {
        Vec::new()
    };
    tokens_from_score_data(cfg, batch, seq, 0, position_offset, n_complete, &score_data)
}

fn materialize_scores(scores: &MlxArray, stage: &'static str) -> Result<()> {
    mlx_sys::try_eval(&[scores]).map_err(|detail| QsaError::Evaluation {
        stage,
        context: format!("scores={:?}", scores.shape()),
        detail,
    })
}

/// Score descending, lower block index first. `query_base` is the chunk-local
/// index of `score_data`'s first query, so a tile reports the same query index
/// the monolithic path would.
fn tokens_from_score_data(
    cfg: QsaConfig,
    batch: i32,
    rows: i32,
    query_base: usize,
    position_offset: usize,
    n_complete: i32,
    score_data: &[f32],
) -> Result<Vec<Vec<Vec<i32>>>> {
    let batch_u = batch as usize;
    let rows_u = rows as usize;
    let n_blocks = n_complete as usize;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(batch_u)
        .map_err(|_| QsaError::InvalidScalar("token index buffer"))?;
    for b in 0..batch_u {
        let mut queries = Vec::new();
        queries
            .try_reserve_exact(rows_u)
            .map_err(|_| QsaError::InvalidScalar("token index buffer"))?;
        for local in 0..rows_u {
            let query = query_base + local;
            let abs_pos = position_offset + query;
            let visible = abs_pos + 1;
            let complete = visible / cfg.compress_ratio();
            let mut chosen = Vec::new();
            if complete > 0 {
                if complete > n_blocks {
                    return Err(QsaError::InvalidScalar("complete block count"));
                }
                let row_start = (b * rows_u + local) * n_blocks;
                let row_end = row_start + complete;
                if row_end > score_data.len() {
                    return Err(QsaError::InvalidScalar("block score length"));
                }
                let order =
                    select_blocks(&score_data[row_start..row_end], cfg.block_topk(), b, query)?;
                for block in order {
                    let start = (block * cfg.compress_ratio()) as i32;
                    for token in 0..cfg.compress_ratio() {
                        chosen.push(start + token as i32);
                    }
                }
            }
            let partial_start = complete * cfg.compress_ratio();
            for token in partial_start..visible {
                chosen.push(token as i32);
            }
            queries.push(chosen);
        }
        tokens.push(queries);
    }
    Ok(tokens)
}

/// Score and select one query tile at a time. The score array is evaluated
/// and dropped before the next tile; only indices are kept.
fn select_tokens_tiled(
    cfg: QsaConfig,
    batch: i32,
    position_offset: usize,
    n_complete: i32,
    queries: &MlxArray,
    blocks: &MlxArray,
    plan: &ScoreTilePlan,
) -> Result<Vec<Vec<Vec<i32>>>> {
    let q_shape = queries.shape();
    if q_shape.len() != 4 {
        return Err(QsaError::InvalidScalar("tiled query rank"));
    }
    let (heads, dim) = (q_shape[2], q_shape[3]);
    let batch_u = batch as usize;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(batch_u)
        .map_err(|_| QsaError::InvalidScalar("token index buffer"))?;
    for _ in 0..batch_u {
        tokens.push(Vec::new());
    }
    let mut cursor = 0i32;
    for &rows in &plan.query_rows {
        ensure_score_workspace(batch, rows, heads, n_complete as usize)?;
        let tile = slice(
            queries,
            &[0, cursor, 0, 0],
            &[batch, cursor + rows, heads, dim],
            &[1, 1, 1, 1],
            None,
        );
        let scores = block_scores(&tile, blocks);
        let summed_bytes = scores
            .shape()
            .iter()
            .try_fold(1usize, |acc, &axis| {
                acc.checked_mul(usize::try_from(axis).unwrap_or(usize::MAX))
            })
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()));
        if summed_bytes.is_none_or(|bytes| bytes > QSA_SCORE_WORKSPACE_LIMIT) {
            return Err(QsaError::InvalidScalar(
                "score workspace; use a smaller prefill chunk",
            ));
        }
        mlx_sys::try_eval(&[&scores]).map_err(|detail| QsaError::Evaluation {
            stage: "tiled block-score materialization",
            context: format!(
                "rows={rows} workspace_bytes={} dispatch={:#x} scores={:?}",
                plan.workspace_bytes,
                plan.dispatch_identity,
                scores.shape()
            ),
            detail,
        })?;
        let data = scores.data_f32().to_vec();
        drop(scores);
        drop(tile);
        let chosen = tokens_from_score_data(
            cfg,
            batch,
            rows,
            cursor as usize,
            position_offset,
            n_complete,
            &data,
        )?;
        for (batch_index, row) in chosen.into_iter().enumerate() {
            tokens[batch_index].extend(row);
        }
        cursor += rows;
    }
    Ok(tokens)
}

fn device_select_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = DEVICE_SELECT_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    crate::fastpath::flash_next_qsa_device_select_enabled()
}

fn device_select_multi_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = DEVICE_SELECT_MULTI_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    crate::fastpath::flash_next_qsa_device_select_multi_enabled()
}

fn prefill_device_select_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = PREFILL_DEVICE_SELECT_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    crate::fastpath::flash_next_qsa_prefill_device_select_enabled()
}

fn decode_block_key_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = DECODE_BLOCK_KEY_OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    crate::fastpath::flash_next_qsa_decode_block_key_enabled()
}

#[cfg(test)]
type HostSelectionLog = Vec<Vec<Vec<Vec<i32>>>>;

#[cfg(test)]
thread_local! {
    static DEVICE_SELECT_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static DEVICE_SELECT_MULTI_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static PREFILL_DEVICE_SELECT_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static TILED_SCORES_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static TILE_LIMITS_OVERRIDE: std::cell::Cell<Option<(i32, usize)>> =
        const { std::cell::Cell::new(None) };
    static TILED_RUNTIME_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static SCORE_WORKSPACE_LIMIT_OVERRIDE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static PREFILL_KEY_KERNEL_FORCE_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PREFILL_DEVICE_ORDER_HITS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static PREFILL_HOST_FALLBACK_HITS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static DECODE_DEVICE_ORDER_HITS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static DECODE_HOST_FALLBACK_HITS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static DECODE_KEY_ORDER_HITS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static DECODE_BLOCK_KEY_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static ROW_FLAG_SYNC_HITS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static HOST_SELECTION_CAPTURE: std::cell::RefCell<Option<HostSelectionLog>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `action` with single-query selection forced on or off on this thread.
#[cfg(test)]
pub(crate) fn with_device_select<T>(enabled: bool, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DEVICE_SELECT_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(DEVICE_SELECT_OVERRIDE.with(|value| value.replace(Some(enabled))));
    action()
}

/// Run `action` with multi-query window selection forced on or off on this
/// thread.
#[cfg(test)]
pub(crate) fn with_device_select_multi<T>(enabled: bool, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DEVICE_SELECT_MULTI_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(DEVICE_SELECT_MULTI_OVERRIDE.with(|value| value.replace(Some(enabled))));
    action()
}

/// Run `action` with prefill device selection forced on or off on this thread.
#[cfg(test)]
pub(crate) fn with_prefill_device_select<T>(enabled: bool, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PREFILL_DEVICE_SELECT_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore =
        Restore(PREFILL_DEVICE_SELECT_OVERRIDE.with(|value| value.replace(Some(enabled))));
    action()
}

/// Run `action` with tiled QSA scores forced on or off on this thread.
#[cfg(test)]
pub(crate) fn with_tiled_scores<T>(enabled: bool, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TILED_SCORES_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(TILED_SCORES_OVERRIDE.with(|value| value.replace(Some(enabled))));
    action()
}

/// Run `action` with a smaller query-tile cap and scratch budget.
#[cfg(test)]
pub(crate) fn with_tile_limits<T>(
    max_rows: i32,
    workspace_bytes: usize,
    action: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<(i32, usize)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TILE_LIMITS_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(
        TILE_LIMITS_OVERRIDE.with(|value| value.replace(Some((max_rows, workspace_bytes)))),
    );
    action()
}

/// Run `action` with the tiled-score GPU contract forced open or closed.
#[cfg(test)]
fn with_tiled_score_runtime<T>(enabled: bool, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TILED_RUNTIME_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(TILED_RUNTIME_OVERRIDE.with(|value| value.replace(Some(enabled))));
    action()
}

/// Run `action` with a smaller monolithic score-workspace ceiling.
#[cfg(test)]
fn with_score_workspace_limit<T>(limit: usize, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SCORE_WORKSPACE_LIMIT_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(SCORE_WORKSPACE_LIMIT_OVERRIDE.with(|value| value.replace(Some(limit))));
    action()
}

/// Run `action` with the prefill block-key kernel treated as unavailable.
#[cfg(test)]
fn with_prefill_key_kernel<T>(enabled: bool, action: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            PREFILL_KEY_KERNEL_FORCE_OFF.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(PREFILL_KEY_KERNEL_FORCE_OFF.with(|value| value.replace(!enabled)));
    action()
}

/// Run `action` with the decode unique-key order forced on or off on this thread.
#[cfg(test)]
fn with_decode_block_key<T>(enabled: bool, action: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DECODE_BLOCK_KEY_OVERRIDE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(DECODE_BLOCK_KEY_OVERRIDE.with(|value| value.replace(Some(enabled))));
    action()
}

#[cfg(test)]
fn prefill_route_hits() -> (u32, u32) {
    (
        PREFILL_DEVICE_ORDER_HITS.with(std::cell::Cell::get),
        PREFILL_HOST_FALLBACK_HITS.with(std::cell::Cell::get),
    )
}

fn note_prefill_route(device_order: bool) {
    #[cfg(test)]
    {
        if device_order {
            PREFILL_DEVICE_ORDER_HITS.with(|hits| hits.set(hits.get().saturating_add(1)));
        } else {
            PREFILL_HOST_FALLBACK_HITS.with(|hits| hits.set(hits.get().saturating_add(1)));
        }
    }
    #[cfg(not(test))]
    let _ = device_order;
}

#[cfg(test)]
fn decode_select_route_hits() -> (u32, u32) {
    (
        DECODE_DEVICE_ORDER_HITS.with(std::cell::Cell::get),
        DECODE_HOST_FALLBACK_HITS.with(std::cell::Cell::get),
    )
}

fn note_decode_select_route(device_order: bool) {
    #[cfg(test)]
    {
        if device_order {
            DECODE_DEVICE_ORDER_HITS.with(|hits| hits.set(hits.get().saturating_add(1)));
        } else {
            DECODE_HOST_FALLBACK_HITS.with(|hits| hits.set(hits.get().saturating_add(1)));
        }
    }
    #[cfg(not(test))]
    let _ = device_order;
}

fn note_decode_key_order() {
    #[cfg(test)]
    {
        DECODE_KEY_ORDER_HITS.with(|hits| hits.set(hits.get().saturating_add(1)));
    }
}

#[cfg(test)]
fn decode_key_order_hits() -> u32 {
    DECODE_KEY_ORDER_HITS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn row_flag_sync_hits() -> u32 {
    ROW_FLAG_SYNC_HITS.with(std::cell::Cell::get)
}

/// Count the prefill row-flag read. Decode does not call it.
fn note_row_flag_sync() {
    #[cfg(test)]
    {
        ROW_FLAG_SYNC_HITS.with(|hits| hits.set(hits.get().saturating_add(1)));
    }
}

#[cfg(test)]
fn record_host_selection(tokens: &[Vec<Vec<i32>>]) {
    HOST_SELECTION_CAPTURE.with(|slot| {
        if let Some(captured) = slot.borrow_mut().as_mut() {
            captured.push(tokens.to_vec());
        }
    });
}

/// Materialize device rows into the host-selection log while a capture is
/// active. No-op otherwise, so a stage profile does not copy indices back.
#[cfg(test)]
fn record_device_selection_if_capturing(selection: &QsaSelection) {
    let capturing = HOST_SELECTION_CAPTURE.with(|slot| slot.borrow().is_some());
    if !capturing {
        return;
    }
    let Some(device) = selection.device_tokens() else {
        return;
    };
    let shape = device.shape();
    if shape.len() != 3 || shape[0] < 0 || shape[1] < 0 {
        return;
    }
    let mut tokens = Vec::with_capacity(shape[0] as usize);
    for batch in 0..shape[0] as usize {
        let mut queries = Vec::with_capacity(shape[1] as usize);
        for query in 0..shape[1] as usize {
            queries.push(selection.tokens_for_query(batch, query).to_vec());
        }
        tokens.push(queries);
    }
    record_host_selection(&tokens);
}

/// Record every host QSA selection made while `action` runs.
#[cfg(test)]
pub(crate) fn capture_host_selections<T>(action: impl FnOnce() -> T) -> (T, HostSelectionLog) {
    struct Restore(Option<HostSelectionLog>);
    impl Drop for Restore {
        fn drop(&mut self) {
            HOST_SELECTION_CAPTURE.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let previous = HOST_SELECTION_CAPTURE.with(|slot| slot.borrow_mut().take());
    let _restore = Restore(previous);
    HOST_SELECTION_CAPTURE.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
    let value = action();
    let captured = HOST_SELECTION_CAPTURE.with(|slot| slot.borrow_mut().take().unwrap_or_default());
    (value, captured)
}

/// Map NaN to -inf. Infinities and finite values, including signed zero, stay.
/// No device read-back.
fn map_nan_to_neg_inf(scores: &MlxArray) -> MlxArray {
    where_cond(
        &not_equal(scores, scores, None),
        &MlxArray::from_f32(f32::NEG_INFINITY),
        scores,
        None,
    )
}

#[cfg(test)]
fn scalar_u32(value: u32) -> MlxArray {
    MlxArray::from_raw_data(
        (&value as *const u32).cast(),
        std::mem::size_of::<u32>(),
        &[1],
        MlxDtype::Uint32,
    )
}

#[cfg(test)]
fn scalar_i32(value: i32) -> MlxArray {
    MlxArray::from_raw_data(
        (&value as *const i32).cast(),
        std::mem::size_of::<i32>(),
        &[1],
        MlxDtype::Int32,
    )
}

#[cfg(test)]
fn scalar_u64(value: u64) -> MlxArray {
    MlxArray::from_raw_data(
        (&value as *const u64).cast(),
        std::mem::size_of::<u64>(),
        &[1],
        MlxDtype::Uint64,
    )
}

#[cfg(test)]
fn host_u32(data: &[u32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Uint32,
    )
}

/// Unique ascending uint64 key, one per score. Test-only.
///
/// High 32 bits are the inverted order-preserving map of the f32 bits. Low 32
/// bits are the block id. Signed zero collapses to `+0` by its bit pattern, so
/// a subnormal is not flushed into that class. A NaN (exponent all ones and a
/// non-zero mantissa) is rewritten to the -inf bits before the map, so it
/// shares the -inf class. That class is `0xff800000`. A future block uses
/// `0xffffffff`, which sorts after every causal key. MLX 0.32.4 argsorts
/// uint64 on the GPU. The two fields make every key in a row unique.
///
/// `decode_block_key_elementwise_microbench` times this builder plus an
/// argsort against the negated-score argsort. It was slower on every measured
/// decode shape, so [`decode_unique_block_ids`] does not call it. The opt-in
/// flag stays on the prefill block-key kernel.
#[cfg(test)]
fn elementwise_decode_block_keys(
    scores: &MlxArray,
    completes: &[i32],
    seq: i32,
    n_complete: i32,
) -> MlxArray {
    let scores = contiguous(scores, None);
    let mut bits = view(&scores, MlxDtype::Uint32, None);
    let neg_zero = scalar_u32(0x8000_0000);
    bits = where_cond(&equal(&bits, &neg_zero, None), &scalar_u32(0), &bits, None);
    let abs_bits = bitwise_and(&bits, &scalar_u32(0x7fff_ffff), None);
    let nan = less(&scalar_u32(0x7f80_0000), &abs_bits, None);
    bits = where_cond(&nan, &scalar_u32(0xff80_0000), &bits, None);
    // Arithmetic shift: the sign bit fills the mask, so a negative value
    // inverts every bit and a positive value only flips the sign bit.
    let mask = view(
        &right_shift(&view(&bits, MlxDtype::Int32, None), &scalar_i32(31), None),
        MlxDtype::Uint32,
        None,
    );
    let order = bitwise_xor(&bits, &bitwise_or(&mask, &neg_zero, None), None);
    let descending = bitwise_invert(&order, None);
    let block_ids = reshape(
        &arange(0.0, f64::from(n_complete), 1.0, MlxDtype::Uint32, None),
        &[1, 1, n_complete],
        None,
    );
    let complete_u: Vec<u32> = completes.iter().map(|&value| value as u32).collect();
    let causal = less(&block_ids, &host_u32(&complete_u, &[1, seq, 1]), None);
    let key32 = where_cond(&causal, &descending, &scalar_u32(0xffff_ffff), None);
    bitwise_or(
        &left_shift(
            &astype(&key32, MlxDtype::Uint64, None),
            &scalar_u64(32),
            None,
        ),
        &astype(&block_ids, MlxDtype::Uint64, None),
        None,
    )
}

/// Block ids `[batch, seq, keep_cols]`, smallest unique key first.
///
/// `None` means the caller keeps the negated-score argsort. That is the
/// default: the opt-in is off, the block-key kernel is unavailable, or the
/// row cannot be encoded (non-f32 scores, or a block id that does not fit in
/// 30 bits). The flag is checked before the kernel is built, so the default
/// decode path does not launch or compile it. The key and the row flag stay
/// on the device.
fn decode_unique_block_ids(
    scores: &MlxArray,
    batch: i32,
    seq: i32,
    n_complete: i32,
    completes: &[i32],
    keep_cols: i32,
) -> Result<Option<MlxArray>> {
    if !decode_block_key_enabled() || prefill_key_kernel().is_none() {
        return Ok(None);
    }
    if keep_cols < 1 || n_complete < keep_cols || n_complete >= 1 << 30 {
        return Ok(None);
    }
    if completes.len() != seq as usize {
        return Err(QsaError::InvalidScalar("decode block-key rows"));
    }
    if scores.dtype() != MlxDtype::Float32 || scores.shape() != [batch, seq, n_complete] {
        return Ok(None);
    }
    let (keys, _flags) = launch_prefill_block_keys(
        scores,
        &host_i32(completes, &[seq]),
        batch,
        seq,
        n_complete,
        keep_cols,
        true,
    )?;
    let order = ordered_block_ids(&keys, keep_cols, keep_cols)?;
    note_decode_key_order();
    Ok(Some(order))
}

/// Single-query block selection on the device, with no host read-back.
///
/// The query keeps `min(block_topk, complete)` blocks plus the partial tail, so
/// the number of kept tokens is fixed by its position alone. Nothing is read
/// back. A query with no complete block does not touch the scores.
///
/// The default path slices off future columns, maps a causal NaN to -inf, and
/// argsorts the negated causal scores. The axis is block-id order, so equal
/// keys (signed zero included, and a NaN tied with -inf after that map) keep
/// the lower block id only while that argsort is stable. NaN-after-+inf is the
/// other half of the same MLX 0.32.4 behaviour (stable Metal merge sort, CPU
/// `std::stable_sort`): it is not a documented API. Both properties are pinned
/// by `mlx_argsort_pins_decode_qsa_tie_and_nan_order` on the float32 shapes
/// this function and [`select_tokens_device_multi`] sort, on the GPU stream.
/// The negated-score argsort compares floats. Metal flushes subnormals in
/// that compare, so a subnormal can tie with zero. Decode does not promise
/// host parity for a subnormal score.
/// `AX_MLX_FLASH_NEXT_QSA_DECODE_BLOCK_KEY=1` replaces that argsort with the
/// prefill order key: signed zero collapses to `+0`, a causal NaN is rewritten
/// to the -inf class, the block id occupies the low 30 bits, and a future
/// block's key sorts after every causal key. Those keys are unique, so the
/// partition and argsort do not depend on sort stability. The default path
/// does not launch or compile that kernel. A missing kernel stays on the
/// negated-score argsort.
///
/// Unlike [`select_tokens_device_prefill`], decode does not raise
/// [`QsaError::NonFiniteScore`]. A non-finite causal score means the forward
/// is already broken. This path still returns a causal, deterministic
/// selection and does not synchronize the device.
fn select_tokens_device(
    cfg: QsaConfig,
    batch: i32,
    position_offset: usize,
    n_complete: i32,
    scores: Option<&MlxArray>,
) -> Result<MlxArray> {
    let ratio = cfg.compress_ratio();
    let visible = position_offset + 1;
    let complete = visible / ratio;
    if n_complete < 0 || complete > n_complete as usize {
        return Err(QsaError::InvalidScalar("complete block count"));
    }
    let keep = cfg.block_topk().min(complete);
    let mut parts: Vec<MlxArray> = Vec::with_capacity(2);
    if keep > 0 {
        let scores = scores.ok_or(QsaError::InvalidScalar("block scores"))?;
        let complete_i =
            i32::try_from(complete).map_err(|_| QsaError::InvalidScalar("complete block count"))?;
        note_decode_select_route(true);
        // The default path slices future columns off before the argsort. The
        // opt-in key path leaves them in the tensor: their keys sort after
        // every causal key, and `keep` is at most the causal count. No flag
        // is read either way.
        let order = match decode_unique_block_ids(
            scores,
            batch,
            1,
            n_complete,
            &[complete_i],
            keep as i32,
        )? {
            Some(order) => order,
            None => {
                let causal_scores = if complete == n_complete as usize {
                    scores.clone()
                } else {
                    slice(
                        scores,
                        &[0, 0, 0],
                        &[batch, 1, complete_i],
                        &[1, 1, 1],
                        None,
                    )
                };
                let causal_scores = map_nan_to_neg_inf(&causal_scores);
                let order = argsort_axis(&negative(&causal_scores, None), -1, None);
                let order = slice(
                    &order,
                    &[0, 0, 0],
                    &[batch, 1, keep as i32],
                    &[1, 1, 1],
                    None,
                );
                astype(&order, MlxDtype::Uint32, None)
            }
        };
        let ratio_scalar = MlxArray::from_raw_data(
            (&(ratio as u32) as *const u32).cast(),
            std::mem::size_of::<u32>(),
            &[1],
            MlxDtype::Uint32,
        );
        let starts = expand_dims_axes(
            &multiply(&astype(&order, MlxDtype::Uint32, None), &ratio_scalar, None),
            &[-1],
            None,
        );
        let within = arange(0.0, ratio as f64, 1.0, MlxDtype::Uint32, None);
        let tokens = add(&starts, &within, None);
        parts.push(reshape(&tokens, &[batch, 1, (keep * ratio) as i32], None));
    }
    let partial_start = complete * ratio;
    if partial_start < visible {
        let len = visible - partial_start;
        let by_batch: Vec<u32> = (0..batch)
            .flat_map(|_| (partial_start..visible).map(|token| token as u32))
            .collect();
        parts.push(MlxArray::from_raw_data(
            by_batch.as_ptr().cast(),
            std::mem::size_of_val(by_batch.as_slice()),
            &[batch, 1, len as i32],
            MlxDtype::Uint32,
        ));
    }
    let refs: Vec<&MlxArray> = parts.iter().collect();
    let joined = if refs.len() == 1 {
        refs[0].clone()
    } else {
        concatenate(&refs, 2, None)
    };
    Ok(astype(&joined, MlxDtype::Int32, None))
}

/// Multi-query block selection on the device for short verifier windows.
///
/// Per-query keep counts are closed-form in the position offset, so the host
/// uploads only two small `[seq]` vectors and reads nothing back. Columns past
/// that row's keep count are -1.
///
/// The default path maps a causal NaN to -inf and writes NaN past the causal
/// prefix, then argsorts the negated row. A future NaN sorts after every
/// numeric key, including the +inf produced by negating -inf, so a future
/// block cannot be selected. Equal causal keys keep the lower block id only
/// while that argsort returns ascending indices within a tie, signed zero
/// included. `mlx_argsort_pins_decode_qsa_tie_and_nan_order` pins both
/// properties for this float32 shape on the GPU stream. The negated-score
/// argsort compares floats. Metal flushes subnormals in that compare, so a
/// subnormal can tie with zero. Decode does not promise host parity for a
/// subnormal score.
/// `AX_MLX_FLASH_NEXT_QSA_DECODE_BLOCK_KEY=1` uses the same unique key as
/// [`select_tokens_device`]: score descending, lower block id, signed zero
/// collapsed, a causal NaN in the -inf class, and a future block after every
/// causal key. The default path does not launch or compile that kernel. A
/// missing kernel stays on the negated-score argsort.
///
/// Unlike [`select_tokens_device_prefill`], this path does not raise
/// [`QsaError::NonFiniteScore`]. A non-finite causal score means the forward
/// is already broken. Returns the packed `[batch, seq, topk*ratio + ratio-1]`
/// int32 tokens (-1 padded) and per-`(batch, query)` segment lengths
/// `(block tokens, tail tokens)`.
fn select_tokens_device_multi(
    cfg: QsaConfig,
    batch: i32,
    seq: i32,
    position_offset: usize,
    n_complete: i32,
    scores: Option<&MlxArray>,
) -> Result<(MlxArray, Vec<(i32, i32)>)> {
    let ratio = cfg.compress_ratio();
    let topk = cfg.block_topk();
    if n_complete < topk as i32 {
        return Err(QsaError::InvalidScalar("multi-query device block tally"));
    }
    // Closed-form per-row counts from position arithmetic. No GPU data is
    // read back; these vectors are the only host input. Counts are
    // batch-invariant (all rows share the window positions); segments are
    // recorded per (batch, query) for the consumer.
    let mut complete = Vec::with_capacity(seq as usize);
    let mut keep = Vec::with_capacity(seq as usize);
    let mut segments = Vec::with_capacity((batch * seq) as usize);
    for s in 0..seq {
        let visible = position_offset + s as usize + 1;
        let complete_s = visible / ratio;
        let keep_s = topk.min(complete_s);
        complete.push(complete_s as i32);
        keep.push(keep_s as i32);
    }
    for _b in 0..batch {
        for s in 0..seq as usize {
            let block_len = (keep[s] as usize * ratio) as i32;
            let tail_len = (position_offset + s + 1 - complete[s] as usize * ratio) as i32;
            segments.push((block_len, tail_len));
        }
    }
    let host_i32 = |data: &[i32], shape: &[i32]| {
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Int32,
        )
    };
    let mut parts: Vec<MlxArray> = Vec::with_capacity(2);
    let scores = scores.ok_or(QsaError::InvalidScalar("block scores"))?;
    note_decode_select_route(true);
    // Default: a NaN past the causal count sorts after every numeric key,
    // including a negated causal -inf. Opt-in: a future key sorts after every
    // causal key. Either way a future block is not selected.
    let complete_dev = host_i32(&complete, &[1, seq, 1]);
    let cols =
        match decode_unique_block_ids(scores, batch, seq, n_complete, &complete, topk as i32)? {
            Some(order) => order,
            None => {
                let block_ids = reshape(
                    &arange(0.0, n_complete as f64, 1.0, MlxDtype::Int32, None),
                    &[1, 1, n_complete],
                    None,
                );
                let valid = less(&block_ids, &complete_dev, None);
                let prepared = map_nan_to_neg_inf(scores);
                let masked = where_cond(&valid, &prepared, &MlxArray::from_f32(f32::NAN), None);
                let order = argsort_axis(&negative(&masked, None), -1, None);
                let cols = slice(
                    &order,
                    &[0, 0, 0],
                    &[batch, seq, topk as i32],
                    &[1, 1, 1],
                    None,
                );
                astype(&cols, MlxDtype::Uint32, None)
            }
        };
    let ratio_u32 = MlxArray::from_raw_data(
        (&(ratio as u32) as *const u32).cast(),
        std::mem::size_of::<u32>(),
        &[1],
        MlxDtype::Uint32,
    );
    let starts = expand_dims_axes(&multiply(&cols, &ratio_u32, None), &[-1], None);
    let within = arange(0.0, ratio as f64, 1.0, MlxDtype::Uint32, None);
    let tokens = add(&starts, &within, None);
    let tokens = astype(
        &reshape(&tokens, &[batch, seq, (topk * ratio) as i32], None),
        MlxDtype::Int32,
        None,
    );
    // Columns at or past a row's keep count came from invalid blocks: -1.
    let keep_dev = host_i32(&keep, &[1, seq, 1]);
    let col_ids = reshape(
        &arange(0.0, topk as f64, 1.0, MlxDtype::Int32, None),
        &[1, 1, topk as i32],
        None,
    );
    let col_valid = less(&col_ids, &keep_dev, None);
    let col_valid = reshape(
        &broadcast_to(
            &expand_dims_axes(&col_valid, &[-1], None),
            &[batch, seq, topk as i32, ratio as i32],
            None,
        ),
        &[batch, seq, (topk * ratio) as i32],
        None,
    );
    let neg_one = MlxArray::from_raw_data(
        (&(-1_i32) as *const i32).cast(),
        std::mem::size_of::<i32>(),
        &[1],
        MlxDtype::Int32,
    );
    let block_tokens = where_cond(&col_valid, &tokens, &neg_one, None);
    parts.push(block_tokens);
    // Visible tail: tokens complete*ratio .. visible-1, up to ratio-1 long.
    let tail_width = ratio as i32 - 1;
    let visible_dev = host_i32(
        &((position_offset + 1)..=(position_offset + seq as usize))
            .map(|v| v as i32)
            .collect::<Vec<_>>(),
        &[1, seq, 1],
    );
    let tail = add(
        &multiply(&complete_dev, &host_i32(&[ratio as i32], &[1]), None),
        &reshape(
            &arange(0.0, tail_width as f64, 1.0, MlxDtype::Int32, None),
            &[1, 1, tail_width],
            None,
        ),
        None,
    );
    let tail_valid = less(&tail, &visible_dev, None);
    // `tail` was built from [1, seq, ·] operands (the counts are
    // batch-invariant); give it the explicit batch dimension so the
    // concatenate with the [batch, seq, topk*ratio] block segment also
    // works for batch > 1.
    let tail = broadcast_to(
        &where_cond(&tail_valid, &tail, &neg_one, None),
        &[batch, seq, tail_width],
        None,
    );
    parts.push(tail);
    let refs: Vec<&MlxArray> = parts.iter().collect();
    let joined = concatenate(&refs, 2, None);
    Ok((joined, segments))
}

/// Threads per group for the block-key kernel. The launch grid is rounded up
/// to a multiple of this so Metal accepts the dispatch.
const PREFILL_KEY_THREADS: i32 = 64;

const PREFILL_KEY_HEADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Inverted order of one finite f32, matching `f32::partial_cmp` with signed
// zero collapsed to +0. `bits` is the IEEE pattern. Integer ops only: a float
// compare on this GPU flushes a subnormal into the zero class.
inline uint ax_qsa_inverted_finite_order(uint bits) {
    if ((bits & 0x7fffffffu) == 0u) {
        bits = 0u;
    }
    const uint mask = (uint)(((int)bits) >> 31);
    const uint order = bits ^ (mask | 0x80000000u);
    return ~order;
}
"#;

/// MLX puts an input shorter than this in the constant address space and a
/// longer one in the device address space, and that choice is part of the
/// compiled kernel name. Integer inputs are padded up to this so every
/// prefill length shares one pipeline.
const PREFILL_KEY_MIN_BUFFER: i32 = 8;

/// One key per `(query, block)`. High bits are the inverted order-preserving
/// score (signed zero collapsed to `+0`, matching `f32::partial_cmp`,
/// subnormals included). Low 30 bits are the block id, so an ascending sort
/// is score descending, then lower block id.
///
/// The score is read with `as_type<uint>`. `+0` and `-0` are the patterns
/// whose magnitude bits are clear. NaN and inf are the all-ones exponent.
/// The order key is built from those bits. No float compare: Metal flushes a
/// subnormal in `score == 0`, which would tie it with zero and break the
/// host order. `metal::isfinite` is the same class of float test, so the
/// prefill flag uses the exponent bits instead.
///
/// Finite causal keys stay in `0..=0x3fffffffffffffff`. A non-finite causal
/// score uses `0x4000000000000000 | block` when `limits[4]` is 0 (prefill).
/// Decode (`limits[4] != 0`) rewrites a NaN to -inf and then uses the finite
/// encoding, so `+inf` stays first, `-inf` and NaN stay last among causal
/// keys, and the two differ only by block id. A future block uses
/// `0x6000000000000000 | block`, which is greater than every causal key, so it
/// cannot win a prefix of length `completes[query]`. The block id keeps those
/// keys unique, so the argsort order is determined.
///
/// `flags[row]` is 1 when any causal score is non-finite or the number of
/// finite causal blocks is below `limits[3]` (`block_topk`). Non-finite is
/// the all-ones exponent, so a subnormal (exponent zero) stays in the finite
/// count. The budget check is the integer `finite < keep_q`. The host sort
/// handles every flagged prefill chunk. Decode writes 0 and does not read the
/// flag. `limits` is `[count, blocks, seq, keep, decode]` plus padding. Those
/// values used to be template arguments, which compiled a new pipeline for
/// every chunk shape and put that compile on the first request. The mode is a
/// buffer value, so decode reuses this pipeline.
const PREFILL_KEY_SOURCE: &str = r#"
const int count_limit = limits[0];
const int blocks_n = limits[1];
const int seq_n = limits[2];
const int keep_q = limits[3];
const int decode_keys = limits[4];
const size_t i = thread_position_in_grid.x;
if (blocks_n <= 0 || seq_n <= 0 || i >= (size_t)count_limit) return;
const size_t blocks = (size_t)blocks_n;
const size_t block = i % blocks;
const size_t query = (i / blocks) % (size_t)seq_n;
const int limit = completes[query];
const uint raw = as_type<uint>(scores[i]);
const uint magnitude = raw & 0x7fffffffu;
const uint exponent = magnitude >> 23;
const int causal = ((int)block < limit) ? 1 : 0;
long key = 0x6000000000000000L | (long)block;
if (causal != 0) {
    if (decode_keys != 0) {
        // NaN: exponent all ones and a non-zero mantissa. +inf stays.
        const uint bits = (magnitude > 0x7f800000u) ? 0xff800000u : raw;
        const uint inverted = ax_qsa_inverted_finite_order(bits);
        key = ((long)inverted << 30) | (long)block;
    } else if (exponent == 0xffu) {
        key = 0x4000000000000000L | (long)block;
    } else {
        const uint inverted = ax_qsa_inverted_finite_order(raw);
        key = ((long)inverted << 30) | (long)block;
    }
}
keys[i] = key;
if (block == 0) {
    const size_t row = i / blocks;
    if (decode_keys != 0) {
        flags[row] = 0;
    } else {
        int finite = 0;
        int bad = 0;
        for (int b = 0; b < limit; ++b) {
            const uint causal_bits = as_type<uint>(scores[row * blocks + (size_t)b]);
            if (((causal_bits >> 23) & 0xffu) == 0xffu) {
                bad = 1;
            } else {
                finite += 1;
            }
        }
        flags[row] = (bad != 0 || finite < keep_q) ? 1 : 0;
    }
}
"#;

fn prefill_key_kernel() -> Option<&'static MlxMetalKernel> {
    #[cfg(test)]
    if PREFILL_KEY_KERNEL_FORCE_OFF.with(std::cell::Cell::get) {
        return None;
    }
    static KERNEL: OnceLock<Option<MlxMetalKernel>> = OnceLock::new();
    KERNEL
        .get_or_init(|| {
            match MlxMetalKernel::try_new(
                "ax_qsa_prefill_block_key_v3",
                &["scores", "completes", "limits"],
                &["keys", "flags"],
                PREFILL_KEY_SOURCE,
                PREFILL_KEY_HEADER,
                true,
            ) {
                Ok(kernel) => Some(kernel),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "QSA prefill block-key kernel failed to build; prefill selection uses the host sort"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// Whether the block-key pipeline was built. A failed build stays `false`
/// for the process, and prefill selection uses the host sort.
pub(crate) fn prefill_block_key_kernel_available() -> bool {
    prefill_key_kernel().is_some()
}

fn host_i32(data: &[i32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Int32,
    )
}

fn launch_prefill_block_keys(
    scores: &MlxArray,
    completes: &MlxArray,
    batch: i32,
    seq: i32,
    n_blocks: i32,
    keep_q: i32,
    decode_keys: bool,
) -> Result<(MlxArray, MlxArray)> {
    let count = batch
        .checked_mul(seq)
        .and_then(|n| n.checked_mul(n_blocks))
        .ok_or(QsaError::InvalidScalar("prefill block-key count"))?;
    if count < 1 || keep_q < 1 {
        return Err(QsaError::InvalidScalar("prefill block-key count"));
    }
    let groups = count
        .checked_add(PREFILL_KEY_THREADS - 1)
        .ok_or(QsaError::InvalidScalar("prefill block-key grid"))?
        / PREFILL_KEY_THREADS;
    let grid = groups
        .checked_mul(PREFILL_KEY_THREADS)
        .ok_or(QsaError::InvalidScalar("prefill block-key grid"))?;
    let kernel =
        prefill_key_kernel().ok_or(QsaError::InvalidScalar("qsa prefill block-key kernel"))?;
    let completes = pad_prefill_key_i32(completes, seq)?;
    let limits = host_i32(
        &[
            count,
            n_blocks,
            seq,
            keep_q,
            i32::from(decode_keys),
            0,
            0,
            0,
        ],
        &[PREFILL_KEY_MIN_BUFFER],
    );
    let template: &[KernelTemplateArg<'_>] = &[];
    let mut outputs = kernel
        .try_apply_with_template(
            &[scores, &completes, &limits],
            &[
                KernelOutputSpec {
                    shape: vec![batch, seq, n_blocks],
                    dtype: MlxDtype::Int64,
                },
                KernelOutputSpec {
                    shape: vec![batch, seq],
                    dtype: MlxDtype::Int32,
                },
            ],
            template,
            (grid, 1, 1),
            (PREFILL_KEY_THREADS, 1, 1),
            None,
        )
        .map_err(|detail| QsaError::Evaluation {
            stage: "qsa prefill block-key kernel",
            context: format!("batch={batch} seq={seq} blocks={n_blocks}"),
            detail,
        })?;
    let flags = outputs
        .pop()
        .ok_or(QsaError::InvalidScalar("qsa prefill block-key kernel"))?;
    let keys = outputs
        .pop()
        .ok_or(QsaError::InvalidScalar("qsa prefill block-key kernel"))?;
    if !outputs.is_empty() {
        return Err(QsaError::InvalidScalar("qsa prefill block-key kernel"));
    }
    Ok((keys, flags))
}

/// Pad a 1-d int32 prefix out to [`PREFILL_KEY_MIN_BUFFER`] so MLX keeps it in
/// the device address space. The kernel reads only the live prefix.
fn pad_prefill_key_i32(values: &MlxArray, live: i32) -> Result<MlxArray> {
    if live >= PREFILL_KEY_MIN_BUFFER {
        return Ok(values.clone());
    }
    if live < 1 {
        return Err(QsaError::InvalidScalar("prefill block-key buffer"));
    }
    let pad = zeros(&[PREFILL_KEY_MIN_BUFFER - live], MlxDtype::Int32, None);
    Ok(concatenate(&[values, &pad], 0, None))
}

/// Whether model load should compile the prefill block-key kernel.
///
/// Non-Flash-Next packs never launch it. `AX_MLX_LOAD_KERNEL_WARMUP=0` skips
/// every load-time kernel compile. `AX_MLX_FLASH_NEXT_QSA_PREFILL_DEVICE_SELECT=0`
/// restores the host sort, which does not launch this kernel.
pub(crate) fn qsa_prefill_key_warmup_applies(
    qwen4_exp: bool,
    kernel_warmup: bool,
    prefill_device_select: bool,
) -> bool {
    qwen4_exp && kernel_warmup && prefill_device_select
}

/// Compile the prefill block-key pipeline.
///
/// One launch is enough: the limits live in a buffer, so later chunks of any
/// length reuse this pipeline instead of compiling on the first request.
/// Best-effort; the caller logs and still finishes the load on `Err`.
pub(crate) fn warm_prefill_key_kernel() -> std::result::Result<(), String> {
    let seq = PREFILL_KEY_MIN_BUFFER;
    let blocks = PREFILL_KEY_MIN_BUFFER;
    let scores = zeros(&[1, seq, blocks], MlxDtype::Float32, None);
    let completes = host_i32(&[blocks; PREFILL_KEY_MIN_BUFFER as usize], &[seq]);
    let (keys, flags) =
        launch_prefill_block_keys(&scores, &completes, 1, seq, blocks, blocks, false)
            .map_err(|error| format!("qsa prefill block-key warm-up failed: {error}"))?;
    mlx_sys::try_eval(&[&keys, &flags])
        .map_err(|error| format!("qsa prefill block-key warm-up failed: {error}"))
}

/// Smallest `keep_cols` keys, sorted ascending. Keys are unique, so the
/// partition does not have to be stable: the following argsort of those keys
/// is the host order. The result is uint32 `[batch, seq, topk]`, padded with
/// zeros when `keep_cols < topk` (those columns are masked off by the caller).
fn ordered_block_ids(keys: &MlxArray, keep_cols: i32, topk: i32) -> Result<MlxArray> {
    let shape = keys.shape();
    if shape.len() != 3 || keep_cols < 1 || topk < keep_cols {
        return Err(QsaError::InvalidScalar("prefill block order"));
    }
    let (batch, seq, n_blocks) = (shape[0], shape[1], shape[2]);
    let narrowed = if keep_cols >= n_blocks {
        argsort_axis(keys, -1, None)
    } else {
        let parted = argpartition_axis(keys, keep_cols - 1, -1, None);
        let parted = slice(
            &parted,
            &[0, 0, 0],
            &[batch, seq, keep_cols],
            &[1, 1, 1],
            None,
        );
        let chosen = take_along_axis(keys, &parted, -1, None);
        let inner = argsort_axis(&chosen, -1, None);
        take_along_axis(&parted, &inner, -1, None)
    };
    let narrowed = astype(&narrowed, MlxDtype::Uint32, None);
    if keep_cols == topk {
        return Ok(narrowed);
    }
    let pad = zeros(&[batch, seq, topk - keep_cols], MlxDtype::Uint32, None);
    Ok(concatenate(&[&narrowed, &pad], 2, None))
}

/// True when any row asked for the host sort. One scalar is read back.
/// Prefill calls this once per chunk. Decode and short multi-query selection
/// do not: they never build or read a flag.
fn prefill_row_flags_need_host(flags: &MlxArray, batch: i32, seq: i32) -> Result<bool> {
    if flags.shape() != [batch, seq] {
        return Err(QsaError::TensorShape {
            tensor: "prefill row flags",
            expected: format!("[{batch}, {seq}]"),
            actual: flags.shape(),
        });
    }
    let rows = batch
        .checked_mul(seq)
        .ok_or(QsaError::InvalidScalar("prefill row flags"))?;
    let flat = reshape(flags, &[rows], None);
    let total = sum_axis(&astype(&flat, MlxDtype::Float32, None), 0, false, None);
    note_row_flag_sync();
    mlx_sys::try_eval(&[&total]).map_err(|detail| QsaError::Evaluation {
        stage: "qsa prefill row-flag check",
        context: format!("flags={:?}", flags.shape()),
        detail,
    })?;
    let count = total
        .data_f32()
        .first()
        .copied()
        .ok_or(QsaError::InvalidScalar("prefill row flags"))?;
    Ok(count != 0.0)
}

fn expand_selected_blocks(
    order: &MlxArray,
    batch: i32,
    seq: i32,
    topk: i32,
    ratio: i32,
    keep: &[i32],
) -> MlxArray {
    let ratio_u32 = MlxArray::from_raw_data(
        (&(ratio as u32) as *const u32).cast(),
        std::mem::size_of::<u32>(),
        &[1],
        MlxDtype::Uint32,
    );
    let starts = expand_dims_axes(&multiply(order, &ratio_u32, None), &[-1], None);
    let within = arange(0.0, ratio as f64, 1.0, MlxDtype::Uint32, None);
    let tokens = astype(
        &reshape(
            &add(&starts, &within, None),
            &[batch, seq, topk * ratio],
            None,
        ),
        MlxDtype::Int32,
        None,
    );
    let col_ids = reshape(
        &arange(0.0, topk as f64, 1.0, MlxDtype::Int32, None),
        &[1, 1, topk],
        None,
    );
    let col_valid = less(&col_ids, &host_i32(keep, &[1, seq, 1]), None);
    let col_valid = reshape(
        &broadcast_to(
            &expand_dims_axes(&col_valid, &[-1], None),
            &[batch, seq, topk, ratio],
            None,
        ),
        &[batch, seq, topk * ratio],
        None,
    );
    where_cond(&col_valid, &tokens, &host_i32(&[-1], &[1]), None)
}

fn visible_tail_tokens(
    batch: i32,
    seq: i32,
    position_offset: usize,
    complete: &[i32],
    ratio: i32,
) -> Result<Option<MlxArray>> {
    let tail_width = ratio - 1;
    if tail_width <= 0 {
        return Ok(None);
    }
    let mut visible = Vec::with_capacity(seq as usize);
    for query in 0..seq as usize {
        let count = position_offset
            .checked_add(query)
            .and_then(|n| n.checked_add(1))
            .ok_or(QsaError::InvalidScalar("total context length"))?;
        visible.push(
            i32::try_from(count).map_err(|_| QsaError::InvalidScalar("total context length"))?,
        );
    }
    let tail = add(
        &multiply(
            &host_i32(complete, &[1, seq, 1]),
            &host_i32(&[ratio], &[1]),
            None,
        ),
        &reshape(
            &arange(0.0, tail_width as f64, 1.0, MlxDtype::Int32, None),
            &[1, 1, tail_width],
            None,
        ),
        None,
    );
    let tail_valid = less(&tail, &host_i32(&visible, &[1, seq, 1]), None);
    Ok(Some(broadcast_to(
        &where_cond(&tail_valid, &tail, &host_i32(&[-1], &[1]), None),
        &[batch, seq, tail_width],
        None,
    )))
}

/// Pack host rows into the segmented device layout: `block_len` tokens, `-1`
/// padding out to `blocks_width`, then `tail_len` visible-tail tokens.
fn pack_host_rows_as_device(
    host: &[Vec<Vec<i32>>],
    batch: i32,
    seq: i32,
    blocks_width: i32,
    tail_width: i32,
    segments: &[(i32, i32)],
) -> Result<MlxArray> {
    if blocks_width < 0 || tail_width < 0 || host.len() != batch as usize {
        return Err(QsaError::InvalidScalar("selected token count"));
    }
    let width = blocks_width
        .checked_add(tail_width)
        .ok_or(QsaError::InvalidScalar("selected token count"))?;
    let rows = (batch as usize)
        .checked_mul(seq as usize)
        .ok_or(QsaError::InvalidScalar("selected token count"))?;
    let stride =
        usize::try_from(width).map_err(|_| QsaError::InvalidScalar("selected token count"))?;
    let count = rows
        .checked_mul(stride)
        .ok_or(QsaError::InvalidScalar("selected token count"))?;
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(count)
        .map_err(|_| QsaError::InvalidScalar("selected token count"))?;
    packed.resize(count, -1_i32);
    let blocks_width_u = blocks_width as usize;
    for (b, queries) in host.iter().enumerate() {
        if queries.len() != seq as usize {
            return Err(QsaError::InvalidScalar("selected token count"));
        }
        for (q, row) in queries.iter().enumerate() {
            let (block_len, tail_len) = segments
                .get(b * seq as usize + q)
                .copied()
                .ok_or(QsaError::InvalidScalar("selected token count"))?;
            if block_len < 0 || tail_len < 0 {
                return Err(QsaError::InvalidScalar("selected token count"));
            }
            let block_len = block_len as usize;
            let tail_len = tail_len as usize;
            let row_len = block_len
                .checked_add(tail_len)
                .ok_or(QsaError::InvalidScalar("selected token count"))?;
            if row_len != row.len() || block_len > blocks_width_u || tail_len > tail_width as usize
            {
                return Err(QsaError::InvalidScalar("selected token count"));
            }
            let dst = (b * seq as usize + q) * stride;
            packed[dst..dst + block_len].copy_from_slice(&row[..block_len]);
            let tail_at = dst + blocks_width_u;
            packed[tail_at..tail_at + tail_len].copy_from_slice(&row[block_len..]);
        }
    }
    Ok(MlxArray::from_raw_data(
        packed.as_ptr().cast(),
        std::mem::size_of_val(packed.as_slice()),
        &[batch, seq, width],
        MlxDtype::Int32,
    ))
}

/// Prefill block selection on the device.
///
/// Same ordered token lists as [`select_tokens`]: up to `block_topk` causally
/// complete blocks, score descending and lower block id first (signed zero
/// equal, subnormals ordered from their IEEE bits), then the visible partial
/// tail. Future blocks use a key that sorts
/// after every causal block, and each row keeps at most its causal count.
/// A row with a non-finite causal score, or fewer finite causal blocks than
/// `block_topk`, selects the whole chunk on the host, including the host
/// [`QsaError::NonFiniteScore`]. A missing block-key kernel does the same.
fn select_tokens_device_prefill(
    cfg: QsaConfig,
    batch: i32,
    seq: i32,
    position_offset: usize,
    n_complete: i32,
    scores: Option<&MlxArray>,
) -> Result<(MlxArray, Vec<(i32, i32)>)> {
    let ratio = i32::try_from(cfg.compress_ratio())
        .map_err(|_| QsaError::InvalidScalar("compress ratio"))?;
    let topk =
        i32::try_from(cfg.block_topk()).map_err(|_| QsaError::InvalidScalar("block topk"))?;
    if batch < 1 || seq < 1 || ratio < 1 || topk < 1 || n_complete < 0 {
        return Err(QsaError::InvalidScalar("prefill device selection"));
    }
    if n_complete >= 1 << 30 {
        return Err(QsaError::InvalidScalar("block id"));
    }
    let mut complete = Vec::with_capacity(seq as usize);
    let mut keep = Vec::with_capacity(seq as usize);
    let mut segments = Vec::with_capacity((batch as usize) * (seq as usize));
    for query in 0..seq as usize {
        let visible = position_offset
            .checked_add(query)
            .and_then(|n| n.checked_add(1))
            .ok_or(QsaError::InvalidScalar("total context length"))?;
        let complete_q = visible / cfg.compress_ratio();
        if complete_q > n_complete as usize {
            return Err(QsaError::InvalidScalar("complete block count"));
        }
        let keep_q = cfg.block_topk().min(complete_q);
        complete.push(
            i32::try_from(complete_q)
                .map_err(|_| QsaError::InvalidScalar("complete block count"))?,
        );
        keep.push(
            i32::try_from(keep_q).map_err(|_| QsaError::InvalidScalar("selected token count"))?,
        );
    }
    for _batch in 0..batch {
        for query in 0..seq as usize {
            let block_len = keep[query]
                .checked_mul(ratio)
                .ok_or(QsaError::InvalidScalar("selected token count"))?;
            let tail_len = i32::try_from(
                position_offset + query + 1 - complete[query] as usize * cfg.compress_ratio(),
            )
            .map_err(|_| QsaError::InvalidScalar("selected token count"))?;
            segments.push((block_len, tail_len));
        }
    }
    let block_cols = topk
        .checked_mul(ratio)
        .ok_or(QsaError::InvalidScalar("selected token count"))?;
    let tail_width = (ratio - 1).max(0);
    let block_tokens = if n_complete == 0 {
        note_prefill_route(true);
        broadcast_to(&host_i32(&[-1], &[1]), &[batch, seq, block_cols], None)
    } else {
        let scores = scores.ok_or(QsaError::InvalidScalar("block scores"))?;
        if scores.dtype() != MlxDtype::Float32 {
            return Err(QsaError::InvalidScalar("block score dtype"));
        }
        let shape = scores.shape();
        if shape != [batch, seq, n_complete] {
            return Err(QsaError::TensorShape {
                tensor: "block scores",
                expected: format!("[{batch}, {seq}, {n_complete}]"),
                actual: shape,
            });
        }
        let use_host = || -> Result<(MlxArray, Vec<(i32, i32)>)> {
            note_prefill_route(false);
            let host = select_tokens(cfg, batch, seq, position_offset, n_complete, Some(scores))?;
            let packed =
                pack_host_rows_as_device(&host, batch, seq, block_cols, tail_width, &segments)?;
            Ok((packed, segments.clone()))
        };
        if prefill_key_kernel().is_none() {
            return use_host();
        }
        let (keys, flags) = launch_prefill_block_keys(
            scores,
            &host_i32(&complete, &[seq]),
            batch,
            seq,
            n_complete,
            topk,
            false,
        )?;
        if prefill_row_flags_need_host(&flags, batch, seq)? {
            return use_host();
        }
        note_prefill_route(true);
        // Every unflagged row has at least `topk` finite causal blocks, so the
        // first `topk` keys are causal. Future keys sort after every causal key.
        let keep_cols = topk.min(n_complete);
        let order = ordered_block_ids(&keys, keep_cols, topk)?;
        expand_selected_blocks(&order, batch, seq, topk, ratio, &keep)
    };
    let mut parts = vec![block_tokens];
    if let Some(tail) = visible_tail_tokens(batch, seq, position_offset, &complete, ratio)? {
        parts.push(tail);
    }
    let refs: Vec<&MlxArray> = parts.iter().collect();
    let joined = if refs.len() == 1 {
        refs[0].clone()
    } else {
        concatenate(&refs, 2, None)
    };
    Ok((joined, segments))
}

/// Pack each segmented device row into a dense valid prefix followed by `-1`,
/// matching [`pack_gather_indices`]. Lengths are the batch-0 segments; causal
/// limits do not depend on the batch index.
fn compact_segmented_tokens(
    device: &MlxArray,
    block_lens: &[i32],
    tail_lens: &[i32],
    blocks_width: i32,
) -> MlxArray {
    let shape = device.shape();
    let (batch, seq, width) = (shape[0], shape[1], shape[2]);
    let block_len = broadcast_to(
        &host_i32(block_lens, &[1, seq, 1]),
        &[batch, seq, width],
        None,
    );
    let tail_len = broadcast_to(
        &host_i32(tail_lens, &[1, seq, 1]),
        &[batch, seq, width],
        None,
    );
    let slots = broadcast_to(
        &reshape(
            &arange(0.0, width as f64, 1.0, MlxDtype::Int32, None),
            &[1, 1, width],
            None,
        ),
        &[batch, seq, width],
        None,
    );
    let in_blocks = less(&slots, &block_len, None);
    let tail_src = add(
        &subtract(&slots, &block_len, None),
        &host_i32(&[blocks_width], &[1]),
        None,
    );
    let src = where_cond(&in_blocks, &slots, &tail_src, None);
    let src = minimum(
        &maximum(&src, &host_i32(&[0], &[1]), None),
        &host_i32(&[width.saturating_sub(1)], &[1]),
        None,
    );
    let gathered = take_along_axis(device, &src, -1, None);
    let valid = less(&slots, &add(&block_len, &tail_len, None), None);
    where_cond(&valid, &gathered, &host_i32(&[-1], &[1]), None)
}

/// Preserve the attention traversal order: score descending, lower block index
/// first on ties (including signed zero). Only the retained prefix is sorted.
fn select_blocks(scores: &[f32], limit: usize, batch: usize, query: usize) -> Result<Vec<usize>> {
    let mut ascending = true;
    let mut descending = true;
    for (block, &score) in scores.iter().enumerate() {
        if !score.is_finite() {
            return Err(QsaError::NonFiniteScore {
                batch,
                query,
                block,
            });
        }
        if block > 0 {
            ascending &= scores[block - 1] <= score;
            descending &= scores[block - 1] >= score;
        }
    }
    let keep = limit.min(scores.len());
    if descending {
        return Ok((0..keep).collect());
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    let compare = |lhs: &usize, rhs: &usize| {
        // Finiteness was checked before entering either comparison algorithm.
        scores[*rhs]
            .partial_cmp(&scores[*lhs])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(lhs.cmp(rhs))
    };
    // Stable full sort detects monotone runs cheaply. Partitioning those
    // rows first would destroy that advantage, especially for equal scores.
    if ascending {
        order.sort_by(compare);
        order.truncate(keep);
        return Ok(order);
    }
    if keep < order.len() {
        order.select_nth_unstable_by(keep, compare);
        order.truncate(keep);
    }
    order.sort_unstable_by(compare);
    Ok(order)
}

fn pack_gather_indices(
    cfg: QsaConfig,
    batch: i32,
    seq: i32,
    tokens: &[Vec<Vec<i32>>],
) -> Result<MlxArray> {
    let max_keep = cfg.max_keep();
    let count = (batch as usize)
        .checked_mul(seq as usize)
        .and_then(|n| n.checked_mul(max_keep))
        .ok_or(QsaError::InvalidScalar("gather index count"))?;
    let mut packed = vec![-1_i32; count];
    for (b, queries) in tokens.iter().enumerate() {
        for (q, selected) in queries.iter().enumerate() {
            let start = (b * seq as usize + q) * max_keep;
            if selected.len() > max_keep {
                return Err(QsaError::InvalidScalar("selected token count"));
            }
            packed[start..start + selected.len()].copy_from_slice(selected);
        }
    }
    let indices = MlxArray::from_raw_data(
        packed.as_ptr() as *const u8,
        std::mem::size_of_val(packed.as_slice()),
        &[batch, seq, max_keep as i32],
        MlxDtype::Int32,
    );
    mlx_sys::try_eval(&[&indices]).map_err(|detail| QsaError::Evaluation {
        stage: "selected-token index materialization",
        context: format!("indices={:?}", indices.shape()),
        detail,
    })?;
    Ok(indices)
}

fn rank3_dims(tensor: &'static str, array: &MlxArray, last: i32) -> Result<(i32, i32)> {
    let shape = array.shape();
    if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != last {
        return Err(QsaError::TensorShape {
            tensor,
            expected: format!("[batch, seq, {last}]"),
            actual: shape,
        });
    }
    Ok((shape[0], shape[1]))
}

fn ensure_floating(tensor: &'static str, dtype: MlxDtype) -> Result<()> {
    if matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::Bfloat16
    ) {
        Ok(())
    } else {
        Err(QsaError::NonFloatingDtype {
            tensor,
            actual: dtype,
        })
    }
}

fn validate_gain(tensor: &'static str, gain: &MlxArray, dim: i32) -> Result<()> {
    let shape = gain.shape();
    if shape != [dim] {
        return Err(QsaError::TensorShape {
            tensor,
            expected: format!("[{dim}]"),
            actual: shape,
        });
    }
    ensure_floating(tensor, gain.dtype())
}

fn validate_projection(
    tensor: &'static str,
    projection: &QuantizedWeight,
    out_dim: i32,
    in_dim: i32,
) -> Result<()> {
    if projection.linear_bias.is_some() {
        return Err(QsaError::UnexpectedLinearBias { tensor });
    }
    let shape_error = |actual: Vec<i32>| QsaError::TensorShape {
        tensor,
        expected: format!("[{out_dim}, {in_dim}]"),
        actual,
    };
    let shape = projection.weight.shape();
    if shape.len() != 2 || shape[0] != out_dim {
        return Err(shape_error(shape));
    }
    let Some(scales) = &projection.scales else {
        if shape[1] != in_dim {
            return Err(shape_error(shape));
        }
        return ensure_floating(tensor, projection.weight.dtype());
    };
    // Same contract as `qwen4_exp_residual::validate_projection`: the packed
    // width, weight/scale dtypes, mode/bias pairing, and the bits/group
    // allowlist must all hold, or a truncated or mislabeled sidecar loads
    // and is misread at the first forward.
    let invalid_quantization = || QsaError::InvalidQuantization {
        tensor,
        group_size: projection.group_size,
        bits: projection.bits,
    };
    let valid_mode = match projection.mode.as_str() {
        "" | "affine" => {
            matches!(projection.bits, 2 | 3 | 4 | 5 | 6 | 8)
                && matches!(projection.group_size, 32 | 64 | 128)
                && projection.biases.as_ref().is_some_and(|bias| {
                    bias.shape() == scales.shape() && bias.dtype() == scales.dtype()
                })
                && matches!(
                    scales.dtype(),
                    MlxDtype::Float32 | MlxDtype::Float16 | MlxDtype::Bfloat16
                )
        }
        "mxfp4" | "mxfp8" => {
            projection.bits == if projection.mode == "mxfp4" { 4 } else { 8 }
                && projection.group_size == 32
                && projection.biases.is_none()
                && scales.dtype() == MlxDtype::Uint8
        }
        _ => false,
    };
    let packed_bits = in_dim.checked_mul(projection.bits);
    if !valid_mode || in_dim <= 0 || out_dim <= 0 || projection.weight.dtype() != MlxDtype::Uint32 {
        return Err(invalid_quantization());
    }
    if packed_bits.is_none_or(|bits| bits % 32 != 0 || bits / 32 != shape[1]) {
        return Err(shape_error(shape));
    }
    let scales_shape = scales.shape();
    let logical_in = scales_shape
        .get(1)
        .and_then(|groups| groups.checked_mul(projection.group_size));
    if scales_shape.len() != 2 || scales_shape[0] != out_dim || logical_in != Some(in_dim) {
        return Err(shape_error(scales_shape));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::with_isolated_peak_memory;

    fn media(prompt: Vec<[i32; 3]>, delta: i32) -> std::sync::Arc<Qwen4ExpMediaPositions> {
        std::sync::Arc::new(Qwen4ExpMediaPositions::new(prompt, delta, &[11, 11, 10]).unwrap())
    }

    #[test]
    fn media_positions_follow_the_prompt_table_then_the_delta() {
        let positions = media(
            vec![[0, 0, 0], [1, 1, 1], [1, 1, 2], [1, 2, 1], [3, 3, 3]],
            -1,
        );
        assert_eq!(positions.at(2), [1, 1, 2]);
        assert_eq!(positions.at(4), [3, 3, 3]);
        // Past the prompt every axis is physical + delta.
        assert_eq!(positions.at(5), [4, 4, 4]);
        assert_eq!(positions.at(9), [8, 8, 8]);
        assert!(Qwen4ExpMediaPositions::new(Vec::new(), 0, &[11, 11, 10]).is_err());
        assert!(Qwen4ExpMediaPositions::new(vec![[0, 0, 0]], 0, &[16, 16]).is_err());
    }

    #[test]
    fn interleaved_sections_assign_t_h_w_like_the_reference() {
        let axes = media(vec![[0; 3]], 0).source_axes(32);
        for (index, axis) in axes.iter().enumerate() {
            let expected = match index % 3 {
                1 if index < 33 => 1,
                2 if index < 30 => 2,
                _ => 0,
            };
            assert_eq!(*axis, expected, "frequency {index}");
        }
        assert_eq!(axes.iter().filter(|&&axis| axis == 1).count(), 11);
        assert_eq!(axes.iter().filter(|&&axis| axis == 2).count(), 10);
    }

    #[test]
    fn mrope_with_equal_axes_is_bit_identical_to_plain_rotary() {
        let values: Vec<f32> = (0..(2 * 3 * 64))
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let x = astype(
            &reshape(&MlxArray::from_f32_slice(&values), &[1, 2, 3, 64], None),
            MlxDtype::Bfloat16,
            None,
        );
        let plain = apply_qsa_rotary_offset(&x, 64, 10_000_000.0, 5, 2);
        let text_like: Vec<[i32; 3]> = (0..12).map(|p| [p; 3]).collect();
        let rotated = {
            let _scope = scoped_media(Some(media(text_like, 0)));
            apply_qsa_rotary_offset(&x, 64, 10_000_000.0, 5, 2)
        };
        let plain = astype(&plain, MlxDtype::Float32, None);
        let rotated = astype(&rotated, MlxDtype::Float32, None);
        eval(&[&plain, &rotated]);
        assert_eq!(plain.data_f32(), rotated.data_f32());
        // The scope ended: the plain path is back.
        assert!(current_media().is_none());
    }

    #[test]
    fn mrope_reads_each_frequency_from_its_axis() {
        let values = vec![1.0f32; 64];
        let x = reshape(&MlxArray::from_f32_slice(&values), &[1, 1, 1, 64], None);
        // One token whose H axis differs: only H frequencies may change.
        let rotated = {
            let _scope = scoped_media(Some(media(vec![[3, 9, 3]], 0)));
            apply_qsa_rotary_offset(&x, 64, 10_000_000.0, 0, 2)
        };
        let plain = apply_qsa_rotary_offset(&x, 64, 10_000_000.0, 3, 2);
        eval(&[&rotated, &plain]);
        let axes = media(vec![[0; 3]], 0).source_axes(32);
        let (rotated, plain) = (rotated.data_f32(), plain.data_f32());
        for channel in 0..64 {
            let frequency = channel % 32;
            if axes[frequency] == 1 {
                continue;
            }
            assert_eq!(rotated[channel], plain[channel], "channel {channel}");
        }
        assert_ne!(rotated, plain);
    }

    #[test]
    fn partial_selection_matches_full_sort_including_boundary_ties() {
        // Exhaust all short rows over ties, signed zero and finite extremes.
        let alphabet = [f32::MIN, -0.0, 0.0, 1.0, f32::MAX];
        for length in 0..=6 {
            for mut code in 0..alphabet.len().pow(length as u32) {
                let scores: Vec<f32> = (0..length)
                    .map(|_| {
                        let score = alphabet[code % alphabet.len()];
                        code /= alphabet.len();
                        score
                    })
                    .collect();
                let mut expected: Vec<usize> = (0..length).collect();
                expected.sort_by(|lhs, rhs| {
                    scores[*rhs]
                        .partial_cmp(&scores[*lhs])
                        .unwrap()
                        .then(lhs.cmp(rhs))
                });
                for keep in 0..=length + 1 {
                    assert_eq!(
                        select_blocks(&scores, keep, 0, 0).unwrap(),
                        expected[..keep.min(length)],
                        "scores={scores:?}, keep={keep}"
                    );
                }
            }
        }
    }

    /// RMS with a pre-widened f32 gain must equal RMS with the raw gain bit
    /// for bit in every dtype: the constructor-cached gain only removes the
    /// per-call `astype`, never changes the arithmetic.
    #[test]
    fn rms_with_scale_f32_matches_rms_with_gain() {
        for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16, MlxDtype::Float16] {
            for rows in [1, 3] {
                let x = rotary_input(&[1, rows, 32], dtype);
                let gain = astype(
                    &array_f32(&wave(32, 0.13, 0.4, 1.1), &[32]),
                    MlxDtype::Bfloat16,
                    None,
                );
                let with_gain = rms_with_gain(&x, &gain, EPS);
                let with_scale =
                    rms_with_scale_f32(&x, &astype(&gain, MlxDtype::Float32, None), EPS);
                assert_eq!(
                    raw_bits(&with_gain),
                    raw_bits(&with_scale),
                    "{dtype:?} rows {rows}: pre-widened gain diverged"
                );
            }
        }
    }

    /// The constructor caches both gains widened to f32, and the cache is
    /// exactly `astype(gain)`.
    #[test]
    fn indexer_constructor_caches_the_f32_gains() {
        let head_dim = 8;
        let indexer = QsaIndexer::new(
            QsaConfig::new(2, 1, head_dim, 8, 2, 4, 16, EPS, BASE).unwrap(),
            QsaIndexerWeights {
                qk_proj: QuantizedWeight::new(
                    array_f32(&wave((2 + 1) * head_dim * 16, 0.05, 0.3, 0.1), &[24, 16]),
                    None,
                    None,
                ),
                q_norm: astype(
                    &array_f32(&wave(head_dim, 0.21, 0.5, 1.2), &[head_dim as i32]),
                    MlxDtype::Bfloat16,
                    None,
                ),
                k_norm: astype(
                    &array_f32(&wave(head_dim, 0.07, 0.3, 0.9), &[head_dim as i32]),
                    MlxDtype::Float16,
                    None,
                ),
            },
        )
        .unwrap();
        for (name, gain, cached) in [
            ("q_norm", &indexer.q_norm, &indexer.q_norm_f32),
            ("k_norm", &indexer.k_norm, &indexer.k_norm_f32),
        ] {
            assert_eq!(cached.dtype(), MlxDtype::Float32, "{name} cache dtype");
            assert_eq!(
                raw_bits(cached),
                raw_bits(&astype(gain, MlxDtype::Float32, None)),
                "{name}: the cached f32 gain drifted from astype(gain)"
            );
        }
    }

    #[test]
    fn partial_selection_matches_full_sort_at_long_context_sizes() {
        let mut seed = 0x1234_5678_u32;
        for length in [511, 512, 513, 4096, 32768] {
            let scores: Vec<f32> = (0..length)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed % 1024) as f32
                })
                .collect();
            let mut expected: Vec<usize> = (0..length).collect();
            expected.sort_by(|lhs, rhs| {
                scores[*rhs]
                    .partial_cmp(&scores[*lhs])
                    .unwrap()
                    .then(lhs.cmp(rhs))
            });
            for keep in [1, 128, 512, length] {
                assert_eq!(
                    select_blocks(&scores, keep, 0, 0).unwrap(),
                    expected[..keep.min(length)]
                );
            }
        }
    }

    #[test]
    fn multi_query_device_selection_matches_host_order() {
        let mut seed = 0x85eb_ca6b_u32;
        for (ratio, budget) in [(2usize, 8usize), (4, 8), (4, 2048), (3, 12)] {
            let cfg = QsaConfig::new(2, 1, 4, 4, ratio, budget, 16, EPS, BASE).unwrap();
            let topk = cfg.block_topk();
            let max_keep = budget + ratio - 1;
            let mut tallies = vec![topk, topk + 1, topk * 3 + 1, 40, 130];
            tallies.retain(|&n| n >= topk);
            for n_complete in tallies {
                for seq in [2i32, 3, 4] {
                    for partial in 0..ratio {
                        let total = n_complete * ratio + partial;
                        if total < seq as usize {
                            continue;
                        }
                        let position_offset = total - seq as usize;
                        // Few distinct values force many ties, including signed zero.
                        let alphabet = [-0.0_f32, 0.0, 1.0, 2.0, 3.0];
                        let scores: Vec<f32> = (0..(seq as usize * n_complete))
                            .map(|_| {
                                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                                alphabet[(seed >> 16) as usize % alphabet.len()]
                            })
                            .collect();
                        let array = array_f32(&scores, &[1, seq, n_complete as i32]);
                        let host = select_tokens(
                            cfg,
                            1,
                            seq,
                            position_offset,
                            n_complete as i32,
                            Some(&array),
                        )
                        .unwrap();
                        let (device, segments) = select_tokens_device_multi(
                            cfg,
                            1,
                            seq,
                            position_offset,
                            n_complete as i32,
                            Some(&array),
                        )
                        .unwrap();
                        assert_eq!(device.shape(), vec![1, seq, max_keep as i32]);
                        let device = astype(&device, MlxDtype::Uint32, None);
                        eval(&[&device]);
                        let data: Vec<i32> = device
                            .data_u32()
                            .iter()
                            .map(|&token| token as i32)
                            .collect();
                        let block_width = topk * ratio;
                        for (s, expected) in host[0].iter().enumerate() {
                            let (block_len, tail_len) = segments[s];
                            let row = &data[s * max_keep..(s + 1) * max_keep];
                            let mut content = row[..block_len as usize].to_vec();
                            content.extend_from_slice(
                                &row[block_width..block_width + tail_len as usize],
                            );
                            assert_eq!(
                                content, *expected,
                                "ratio={ratio} budget={budget} blocks={n_complete} seq={seq} \
                                 partial={partial} row={s}"
                            );
                            assert_eq!(
                                expected.len(),
                                (block_len + tail_len) as usize,
                                "segment lengths must match the host row length"
                            );
                            assert!(
                                row[block_len as usize..block_width]
                                    .iter()
                                    .all(|&t| t == -1),
                                "block padding must be -1"
                            );
                            assert!(
                                row[block_width + tail_len as usize..]
                                    .iter()
                                    .all(|&t| t == -1),
                                "tail padding must be -1"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn multi_query_device_selection_requires_the_full_block_tally() {
        let cfg = QsaConfig::new(2, 1, 4, 4, 4, 2048, 16, EPS, BASE).unwrap();
        let scores = array_f32(&[1.0, 2.0], &[1, 2, 1]);
        assert!(
            select_tokens_device_multi(cfg, 1, 2, 3, 1, Some(&scores)).is_err(),
            "fewer complete blocks than topk must fail closed to the host path"
        );
    }

    #[test]
    fn multi_query_device_selection_broadcasts_the_tail_across_the_batch() {
        // The visible-tail segment is built from [1, seq, ·] operands
        // (counts are batch-invariant); without the explicit batch dimension
        // the final concatenate fails for batch > 1.
        let cfg = QsaConfig::new(2, 1, 4, 4, 4, 8, 16, EPS, BASE).unwrap();
        let (ratio, budget) = (4usize, 8usize);
        let max_keep = budget + ratio - 1;
        let n_complete = 5usize;
        let seq = 3i32;
        let batch = 2i32;
        let position_offset = n_complete * ratio + 2 - seq as usize; // partial tail
        let scores = array_f32(&[0.5f32; 2 * 3 * 5], &[batch, seq, n_complete as i32]);
        let (device, segments) = select_tokens_device_multi(
            cfg,
            batch,
            seq,
            position_offset,
            n_complete as i32,
            Some(&scores),
        )
        .unwrap();
        assert_eq!(device.shape(), vec![batch, seq, max_keep as i32]);
        assert_eq!(segments.len(), (batch * seq) as usize);
        let device = astype(&device, MlxDtype::Uint32, None);
        eval(&[&device]);
        let data = device.data_u32();
        let row_width = seq as usize * max_keep;
        assert_eq!(
            &data[..row_width],
            &data[row_width..],
            "batch rows share the batch-invariant selection"
        );
    }

    #[test]
    fn device_selection_matches_host_order_including_ties_and_partial_tails() {
        let mut seed = 0x9e37_79b9_u32;
        for (ratio, budget) in [(2usize, 8usize), (4, 8), (4, 2048), (3, 12)] {
            let cfg = QsaConfig::new(2, 1, 4, 4, ratio, budget, 16, EPS, BASE).unwrap();
            for n_complete in [1usize, 2, 3, 5, 17, 40, 130] {
                for partial in 0..ratio {
                    // Few distinct values force many ties, including signed zero.
                    let alphabet = [-0.0_f32, 0.0, 1.0, 2.0, 3.0];
                    let scores: Vec<f32> = (0..n_complete)
                        .map(|_| {
                            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                            alphabet[(seed >> 16) as usize % alphabet.len()]
                        })
                        .collect();
                    let position_offset =
                        n_complete * ratio + partial - 1 + usize::from(partial == 0);
                    let visible = position_offset + 1;
                    if visible / ratio != n_complete {
                        continue;
                    }
                    let array = array_f32(&scores, &[1, 1, n_complete as i32]);
                    let host =
                        select_tokens(cfg, 1, 1, position_offset, n_complete as i32, Some(&array))
                            .unwrap();
                    let device = select_tokens_device(
                        cfg,
                        1,
                        position_offset,
                        n_complete as i32,
                        Some(&array),
                    )
                    .unwrap();
                    let device = astype(&device, MlxDtype::Uint32, None);
                    eval(&[&device]);
                    let device: Vec<i32> = device
                        .data_u32()
                        .iter()
                        .map(|&token| token as i32)
                        .collect();
                    assert_eq!(
                        device, host[0][0],
                        "ratio={ratio} budget={budget} blocks={n_complete} partial={partial} \
                         scores={scores:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn device_selection_handles_the_first_block_and_batches() {
        let cfg = QsaConfig::new(2, 1, 4, 4, 4, 8, 16, EPS, BASE).unwrap();
        // No complete block yet: only the partial tail, for every batch row.
        let tail = select_tokens_device(cfg, 2, 1, 0, None).unwrap();
        let tail = astype(&tail, MlxDtype::Uint32, None);
        eval(&[&tail]);
        assert_eq!(tail.shape(), vec![2, 1, 2]);
        assert_eq!(tail.data_u32(), &[0, 1, 0, 1]);
        // Two batch rows with opposite score orders keep their own blocks.
        let scores = array_f32(&[1.0, 2.0, 3.0, 3.0, 2.0, 1.0], &[2, 1, 3]);
        let picked = select_tokens_device(cfg, 2, 12, 3, Some(&scores)).unwrap();
        let picked = astype(&picked, MlxDtype::Uint32, None);
        eval(&[&picked]);
        // budget 8 / ratio 4 keeps two blocks plus the one-token tail.
        assert_eq!(picked.shape(), vec![2, 1, 9]);
        let data = picked.data_u32();
        assert_eq!(&data[..9], &[8, 9, 10, 11, 4, 5, 6, 7, 12]);
        assert_eq!(&data[9..], &[0, 1, 2, 3, 4, 5, 6, 7, 12]);
        assert!(select_tokens_device(cfg, 1, 12, 2, Some(&scores)).is_err());
    }

    fn decode_i32(array: &MlxArray) -> Vec<i32> {
        let bits = astype(array, MlxDtype::Uint32, None);
        eval(&[&bits]);
        bits.data_u32().iter().map(|&token| token as i32).collect()
    }

    fn assert_single_query_matches(
        cfg: QsaConfig,
        batch: i32,
        offset: usize,
        n_complete: i32,
        scores: &[f32],
    ) {
        let array = array_f32(scores, &[batch, 1, n_complete]);
        let host = select_tokens(cfg, batch, 1, offset, n_complete, Some(&array)).unwrap();
        let device = select_tokens_device(cfg, batch, offset, n_complete, Some(&array)).unwrap();
        let data = decode_i32(&device);
        let width = host[0][0].len();
        assert_eq!(data.len(), host.len() * width);
        for (b, queries) in host.iter().enumerate() {
            assert_eq!(
                &data[b * width..(b + 1) * width],
                queries[0].as_slice(),
                "batch {b} offset {offset}"
            );
        }
    }

    fn nan_as_neg_inf(score: f32) -> f32 {
        if score.is_nan() {
            f32::NEG_INFINITY
        } else {
            score
        }
    }

    /// Score descending, then the lower block id. A NaN compares as -inf.
    fn causal_block_order(causal: &[f32], keep: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..causal.len()).collect();
        order.sort_by(|&lhs, &rhs| {
            let left = nan_as_neg_inf(causal[lhs]);
            let right = nan_as_neg_inf(causal[rhs]);
            right
                .partial_cmp(&left)
                .expect("mapped causal scores compare")
                .then(lhs.cmp(&rhs))
        });
        order.truncate(keep.min(causal.len()));
        order
    }

    fn tokens_from_causal_order(
        causal: &[f32],
        keep: usize,
        ratio: usize,
        visible: usize,
    ) -> Vec<i32> {
        let complete = visible / ratio;
        assert_eq!(causal.len(), complete);
        let mut chosen = Vec::new();
        for block in causal_block_order(causal, keep) {
            assert!(block < complete, "future block {block} entered the order");
            let start = block * ratio;
            for token in 0..ratio {
                chosen.push((start + token) as i32);
            }
        }
        for token in complete * ratio..visible {
            chosen.push(token as i32);
        }
        chosen
    }

    fn assert_single_query_rule(
        cfg: QsaConfig,
        batch: i32,
        offset: usize,
        n_complete: i32,
        scores: &[f32],
    ) {
        let sync = row_flag_sync_hits();
        let route = decode_select_route_hits();
        let array = array_f32(scores, &[batch, 1, n_complete]);
        let first = select_tokens_device(cfg, batch, offset, n_complete, Some(&array)).unwrap();
        let second = select_tokens_device(cfg, batch, offset, n_complete, Some(&array)).unwrap();
        let once = decode_i32(&first);
        let twice = decode_i32(&second);
        assert_eq!(once, twice, "single-query selection must be deterministic");
        let ratio = cfg.compress_ratio();
        let visible = offset + 1;
        let complete = visible / ratio;
        let width = n_complete as usize;
        let row_len = once.len() / batch as usize;
        for b in 0..batch as usize {
            let row = &scores[b * width..(b + 1) * width];
            let expected =
                tokens_from_causal_order(&row[..complete], cfg.block_topk(), ratio, visible);
            assert_eq!(
                &once[b * row_len..(b + 1) * row_len],
                expected.as_slice(),
                "batch {b} offset {offset}"
            );
        }
        assert_eq!(
            row_flag_sync_hits(),
            sync,
            "single-query decode added a row-flag sync"
        );
        let after = decode_select_route_hits();
        assert_eq!(
            after.1, route.1,
            "single-query decode must not fall back to the host"
        );
        if complete > 0 {
            assert!(after.0 > route.0);
        }
    }

    fn assert_multi_query_matches(
        cfg: QsaConfig,
        batch: i32,
        seq: i32,
        offset: usize,
        n_complete: i32,
        scores: &[f32],
    ) {
        let array = array_f32(scores, &[batch, seq, n_complete]);
        let host = select_tokens(cfg, batch, seq, offset, n_complete, Some(&array)).unwrap();
        let (device, segments) =
            select_tokens_device_multi(cfg, batch, seq, offset, n_complete, Some(&array)).unwrap();
        let data = decode_i32(&device);
        let width = device.shape()[2] as usize;
        let block_width = cfg.block_topk() * cfg.compress_ratio();
        for b in 0..batch as usize {
            for s in 0..seq as usize {
                let (block_len, tail_len) = segments[b * seq as usize + s];
                let row = &data[(b * seq as usize + s) * width..][..width];
                let mut content = row[..block_len as usize].to_vec();
                content.extend_from_slice(&row[block_width..block_width + tail_len as usize]);
                assert_eq!(content, host[b][s], "batch {b} query {s} offset {offset}");
            }
        }
    }

    fn assert_multi_query_rule(
        cfg: QsaConfig,
        batch: i32,
        seq: i32,
        offset: usize,
        n_complete: i32,
        scores: &[f32],
    ) {
        let sync = row_flag_sync_hits();
        let route = decode_select_route_hits();
        let array = array_f32(scores, &[batch, seq, n_complete]);
        let (first, segments) =
            select_tokens_device_multi(cfg, batch, seq, offset, n_complete, Some(&array)).unwrap();
        let (second, _) =
            select_tokens_device_multi(cfg, batch, seq, offset, n_complete, Some(&array)).unwrap();
        let once = decode_i32(&first);
        let twice = decode_i32(&second);
        assert_eq!(once, twice, "multi-query selection must be deterministic");
        let width = first.shape()[2] as usize;
        let ratio = cfg.compress_ratio();
        let block_width = cfg.block_topk() * ratio;
        let blocks = n_complete as usize;
        for b in 0..batch as usize {
            for s in 0..seq as usize {
                let visible = offset + s + 1;
                let complete = visible / ratio;
                let start = (b * seq as usize + s) * blocks;
                let expected = tokens_from_causal_order(
                    &scores[start..start + complete],
                    cfg.block_topk(),
                    ratio,
                    visible,
                );
                let (block_len, tail_len) = segments[b * seq as usize + s];
                let row = &once[(b * seq as usize + s) * width..][..width];
                let mut content = row[..block_len as usize].to_vec();
                content.extend_from_slice(&row[block_width..block_width + tail_len as usize]);
                assert_eq!(content, expected, "batch {b} query {s} offset {offset}");
                assert!(
                    row[block_len as usize..block_width]
                        .iter()
                        .all(|token| *token == -1),
                    "block padding must be -1"
                );
            }
        }
        assert_eq!(
            row_flag_sync_hits(),
            sync,
            "short multi-query selection added a row-flag sync"
        );
        let after = decode_select_route_hits();
        assert_eq!(
            after.1, route.1,
            "short multi-query selection must not fall back to the host"
        );
        assert!(after.0 > route.0);
    }

    #[test]
    fn device_selection_matches_host_on_causal_and_future_nonfinite() {
        // ratio 4, budget 8 → block_topk 2. Offset 15 sees four causal blocks.
        let cfg = QsaConfig::new(2, 1, 4, 4, 4, 8, 16, EPS, BASE).unwrap();
        let before = decode_select_route_hits();
        let sync = row_flag_sync_hits();
        // Future NaN / inf / a larger finite score must not win or raise.
        // Signed zeros in the causal prefix stay on the device argsort.
        let mut future = vec![-0.0_f32, 0.0, 1.0, 2.0, f32::NAN, 1_000.0];
        future.extend_from_slice(&[3.0, 0.5, -0.0, 4.0, f32::INFINITY, f32::NEG_INFINITY]);
        assert_single_query_matches(cfg, 2, 15, 6, &future);
        // One causal block is below the budget. Future columns are ignored.
        // The row stays on the device; decode does not fall back or sync.
        let short = [0.25_f32, f32::NAN, f32::INFINITY, 1_000.0];
        assert_single_query_matches(cfg, 1, 3, 4, &short);
        let after = decode_select_route_hits();
        assert!(
            after.0 > before.0,
            "finite causal rows, including a short prefix, stay on the device"
        );
        assert_eq!(
            after.1, before.1,
            "single-query decode must not fall back to the host"
        );
        assert_eq!(
            row_flag_sync_hits(),
            sync,
            "single-query decode added a row-flag sync"
        );
    }

    #[test]
    fn device_selection_orders_nonfinite_causal_scores_without_reading_back() {
        // Decode does not raise NonFiniteScore. NaN sorts as -inf, +inf sorts
        // first, and a future column — even a huge finite one — is not kept.
        let cfg = QsaConfig::new(2, 1, 4, 4, 4, 8, 16, EPS, BASE).unwrap();
        let causal_nan = [1.0_f32, f32::NAN, 2.0, 0.5];
        let array = array_f32(&causal_nan, &[1, 1, 4]);
        assert!(
            select_tokens(cfg, 1, 1, 15, 4, Some(&array)).is_err(),
            "the host selector still rejects a causal NaN"
        );
        assert_single_query_rule(cfg, 1, 15, 4, &causal_nan);
        assert_single_query_rule(cfg, 1, 15, 4, &[9.0, 8.0, 7.0, f32::NAN]);
        assert_single_query_rule(cfg, 1, 15, 4, &[1.0, 2.0, f32::INFINITY, 0.0]);
        assert_single_query_rule(cfg, 1, 15, 4, &[1.0, 2.0, f32::NEG_INFINITY, 4.0]);
        // Tied finite scores, then NaN tied with -inf, lower block id first.
        assert_single_query_rule(cfg, 1, 15, 4, &[2.0, f32::NAN, f32::NEG_INFINITY, 2.0]);
        let mut batched = vec![1.0_f32, 2.0, 3.0, 4.0];
        batched.extend_from_slice(&[1.0, 2.0, f32::NEG_INFINITY, 4.0]);
        assert_single_query_rule(cfg, 2, 15, 4, &batched);
        // Offset 7 sees two causal blocks. The third column is future.
        assert_single_query_rule(cfg, 1, 7, 3, &[f32::NAN, f32::NEG_INFINITY, 1_000.0]);
        assert_single_query_rule(
            cfg,
            1,
            15,
            6,
            &[1.0, f32::NAN, 2.0, 0.5, f32::INFINITY, 1_000.0],
        );
    }

    #[test]
    fn multi_query_device_selection_matches_host_on_causal_and_future_nonfinite() {
        // Both queries see exactly two causal blocks. Later columns are future.
        let cfg = QsaConfig::new(2, 1, 4, 4, 4, 8, 16, EPS, BASE).unwrap();
        let before = decode_select_route_hits();
        let sync = row_flag_sync_hits();
        let mut future = vec![3.0_f32, 1.0, f32::NAN, 1_000.0];
        future.extend_from_slice(&[0.5, 2.0, f32::INFINITY, f32::NEG_INFINITY]);
        assert_multi_query_matches(cfg, 1, 2, 7, 4, &future);
        // Each query sees one causal block. Huge and non-finite columns are
        // future and must not be selected. The short rows stay on the device.
        let mut short = vec![0.4_f32, 1_000.0, f32::NAN];
        short.extend_from_slice(&[0.2, f32::INFINITY, 50.0]);
        assert_multi_query_matches(cfg, 1, 2, 4, 3, &short);
        let after = decode_select_route_hits();
        assert!(
            after.0 > before.0,
            "future non-finite rows, including a short prefix, stay on the device"
        );
        assert_eq!(
            after.1, before.1,
            "short multi-query selection must not fall back to the host"
        );
        assert_eq!(
            row_flag_sync_hits(),
            sync,
            "short multi-query selection added a row-flag sync"
        );
    }

    #[test]
    fn multi_query_selection_orders_nonfinite_causal_scores_without_reading_back() {
        let cfg = QsaConfig::new(2, 1, 4, 4, 4, 8, 16, EPS, BASE).unwrap();
        // Query 0 causal NaN, query 1 causal +inf. Later columns are future.
        let mut causal_nan = vec![1.0_f32, f32::NAN, 0.0, 1_000.0];
        causal_nan.extend_from_slice(&[0.5, f32::INFINITY, f32::NAN, 9.0]);
        let array = array_f32(&causal_nan, &[1, 2, 4]);
        assert!(
            select_tokens(cfg, 1, 2, 7, 4, Some(&array)).is_err(),
            "the host selector still rejects a causal non-finite score"
        );
        assert_multi_query_rule(cfg, 1, 2, 7, 4, &causal_nan);
        let mut causal_inf = vec![3.0_f32, 1.0, f32::NAN, 4.0];
        causal_inf.extend_from_slice(&[f32::NEG_INFINITY, 2.0, 0.0, 0.0]);
        assert_multi_query_rule(cfg, 1, 2, 7, 4, &causal_inf);
        // Both queries see two causal blocks. A future 1000 must not replace
        // a causal NaN or -inf.
        let mut worst = vec![f32::NAN, f32::NEG_INFINITY, 1_000.0, f32::INFINITY];
        worst.extend_from_slice(&[f32::NAN, f32::NAN, 1_000.0, f32::NEG_INFINITY]);
        assert_multi_query_rule(cfg, 1, 2, 7, 4, &worst);
        let mut batched = vec![1.0_f32, 2.0, 0.0, 0.0, 0.5, 1.5, f32::NAN, 8.0];
        batched.extend_from_slice(&[1.0, f32::NAN, 0.0, 0.0, 0.25, 0.75, 0.0, 0.0]);
        assert_multi_query_rule(cfg, 2, 2, 7, 4, &batched);
    }

    /// Prefill finite-score encoding, with a causal NaN rewritten to -inf.
    /// A future block sorts after every causal key. Low 30 bits are the block id.
    fn decode_rank_key(score: f32, block: u32, causal: bool) -> u64 {
        if !causal {
            return 0x6000_0000_0000_0000 | u64::from(block);
        }
        let score = if score.is_nan() {
            f32::NEG_INFINITY
        } else {
            score
        };
        let canon = if score == 0.0 { 0.0 } else { score };
        let bits = canon.to_bits();
        let signed = i32::from_ne_bytes(bits.to_ne_bytes());
        let mask = if signed < 0 { u32::MAX } else { 0 };
        let order = bits ^ (mask | 0x8000_0000);
        let inverted = !order;
        (u64::from(inverted) << 30) | u64::from(block)
    }

    fn rank_order(scores: &[f32], causal: usize, keep: usize) -> Vec<usize> {
        let mut forward: Vec<usize> = (0..scores.len()).collect();
        forward.sort_unstable_by_key(|&block| {
            decode_rank_key(scores[block], block as u32, block < causal)
        });
        let mut pairs: Vec<(u64, usize)> = scores
            .iter()
            .enumerate()
            .rev()
            .map(|(block, score)| (decode_rank_key(*score, block as u32, block < causal), block))
            .collect();
        pairs.sort_unstable_by_key(|pair| pair.0);
        let mapped: Vec<usize> = pairs.into_iter().map(|pair| pair.1).collect();
        assert_eq!(
            mapped, forward,
            "unique keys must survive an unstable sort of a reversed block order"
        );
        forward.truncate(keep);
        forward
    }

    fn blocks_from_tokens(tokens: &[i32], ratio: usize, keep: usize) -> Vec<usize> {
        (0..keep)
            .map(|index| {
                let token = tokens[index * ratio];
                assert!(token >= 0, "selected block token {token}");
                token as usize / ratio
            })
            .collect()
    }

    #[test]
    fn decode_unique_keys_match_host_and_score_argsort_on_finite_ties() {
        let cfg = QsaConfig::new(2, 1, 4, 4, 4, 12, 16, EPS, BASE).unwrap();
        let alphabet = [-0.0_f32, 0.0, 0.0, 1.0, 1.0, -2.0, 3.0, 3.0];
        let mut scores = alphabet.to_vec();
        let mut reversed = alphabet.to_vec();
        reversed.reverse();
        scores.extend_from_slice(&reversed);
        let n_complete = 8i32;
        let offset = 31usize;
        let array = array_f32(&scores, &[2, 1, n_complete]);
        let sync = row_flag_sync_hits();
        let before = decode_key_order_hits();
        let on = with_decode_block_key(true, || {
            select_tokens_device(cfg, 2, offset, n_complete, Some(&array)).unwrap()
        });
        let off = with_decode_block_key(false, || {
            select_tokens_device(cfg, 2, offset, n_complete, Some(&array)).unwrap()
        });
        let missing = with_prefill_key_kernel(false, || {
            with_decode_block_key(true, || {
                select_tokens_device(cfg, 2, offset, n_complete, Some(&array)).unwrap()
            })
        });
        let host = select_tokens(cfg, 2, 1, offset, n_complete, Some(&array)).unwrap();
        let on_tokens = decode_i32(&on);
        let off_tokens = decode_i32(&off);
        let missing_tokens = decode_i32(&missing);
        assert_eq!(
            on_tokens, off_tokens,
            "finite scores must match the score argsort"
        );
        assert_eq!(on_tokens, missing_tokens, "a missing kernel must match too");
        let width = host[0][0].len();
        assert_eq!(&on_tokens[..width], host[0][0].as_slice());
        assert_eq!(&on_tokens[width..], host[1][0].as_slice());
        for (row, scores) in [alphabet.as_slice(), reversed.as_slice()]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                blocks_from_tokens(&on_tokens[row * width..], cfg.compress_ratio(), 3),
                rank_order(scores, scores.len(), 3),
                "row {row}"
            );
        }
        assert_eq!(decode_key_order_hits(), before + 1);
        assert_eq!(row_flag_sync_hits(), sync, "decode selection added a sync");

        let cfg = QsaConfig::new(2, 1, 4, 4, 4, 8, 16, EPS, BASE).unwrap();
        let seq = 3i32;
        let n_complete = 6i32;
        let offset = 7usize;
        let mut window = Vec::new();
        for _batch in 0..2 {
            window.extend_from_slice(&[1.0, 2.0, 50.0, 50.0, 50.0, 50.0]);
            window.extend_from_slice(&[-0.0, 0.0, 4.0, 4.0, 4.0, 4.0]);
            window.extend_from_slice(&[3.0, 1.0, 8.0, -0.0, 0.0, 8.0]);
        }
        let array = array_f32(&window, &[2, seq, n_complete]);
        let before = decode_key_order_hits();
        with_decode_block_key(true, || {
            let (on, _) =
                select_tokens_device_multi(cfg, 2, seq, offset, n_complete, Some(&array)).unwrap();
            let (off, _) = with_decode_block_key(false, || {
                select_tokens_device_multi(cfg, 2, seq, offset, n_complete, Some(&array)).unwrap()
            });
            assert_eq!(decode_i32(&on), decode_i32(&off));
            assert_multi_query_matches(cfg, 2, seq, offset, n_complete, &window);
        });
        assert_eq!(decode_key_order_hits(), before + 2);
        assert_eq!(
            row_flag_sync_hits(),
            sync,
            "short-window selection added a sync"
        );
    }

    #[test]
    fn decode_unique_keys_keep_nonfinite_causal_and_future_order() {
        let cfg = QsaConfig::new(2, 1, 4, 4, 4, 8, 16, EPS, BASE).unwrap();
        let sync = row_flag_sync_hits();
        let scores = [1.0_f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1_000.0];
        let array = array_f32(&scores, &[1, 1, 5]);
        with_decode_block_key(true, || {
            assert_single_query_rule(cfg, 1, 15, 5, &scores);
            let on = select_tokens_device(cfg, 1, 15, 5, Some(&array)).unwrap();
            let off = with_decode_block_key(false, || {
                select_tokens_device(cfg, 1, 15, 5, Some(&array)).unwrap()
            });
            assert_eq!(decode_i32(&on), decode_i32(&off));
            assert_eq!(
                blocks_from_tokens(&decode_i32(&on), cfg.compress_ratio(), 2),
                rank_order(&scores, 4, 2)
            );
        });
        let mut window = vec![1.0_f32, f32::NAN, 1_000.0, f32::INFINITY];
        window.extend_from_slice(&[f32::NEG_INFINITY, 2.0, f32::NAN, 9.0]);
        let array = array_f32(&window, &[1, 2, 4]);
        with_decode_block_key(true, || {
            assert_multi_query_rule(cfg, 1, 2, 7, 4, &window);
            let (on, _) = select_tokens_device_multi(cfg, 1, 2, 7, 4, Some(&array)).unwrap();
            let (off, _) = with_decode_block_key(false, || {
                select_tokens_device_multi(cfg, 1, 2, 7, 4, Some(&array)).unwrap()
            });
            assert_eq!(decode_i32(&on), decode_i32(&off));
        });
        assert_eq!(
            row_flag_sync_hits(),
            sync,
            "non-finite decode selection added a sync"
        );
    }

    #[test]
    fn decode_block_key_order_does_not_depend_on_argsort_stability() {
        // A stable sort of equal keys follows presentation order. Reversing
        // the axis and mapping the indices back changes the winners. Unique
        // keys do not.
        let perm = [2usize, 1, 0];
        let mut slots: Vec<usize> = (0..3).collect();
        slots.sort_by_key(|&_slot| 0u64);
        let mapped: Vec<usize> = slots.iter().map(|&slot| perm[slot]).collect();
        assert_eq!(mapped, vec![2, 1, 0]);
        assert_ne!(mapped, vec![0, 1, 2]);

        let batch = 2i32;
        let seq = 2i32;
        let n = 6i32;
        let completes = [4i32, 3];
        let mut flat = Vec::new();
        let rows = [
            [-0.0_f32, 0.0, 1.0, 1.0, 9.0, 9.0],
            [
                f32::NAN,
                f32::NEG_INFINITY,
                f32::INFINITY,
                0.5,
                1_000.0,
                f32::NAN,
            ],
        ];
        for _batch in 0..batch {
            for row in &rows {
                flat.extend_from_slice(row);
            }
        }
        let array = array_f32(&flat, &[batch, seq, n]);
        let (keys, _flags) = launch_prefill_block_keys(
            &array,
            &host_i32(&completes, &[seq]),
            batch,
            seq,
            n,
            2,
            true,
        )
        .unwrap();
        eval(&[&keys]);
        let got = keys.data_i64().to_vec();
        assert_eq!(got.len(), flat.len());
        for batch_index in 0..batch as usize {
            for (query, &limit) in completes.iter().enumerate() {
                let row_at = (batch_index * seq as usize + query) * n as usize;
                for block in 0..n as usize {
                    let expected =
                        decode_rank_key(flat[row_at + block], block as u32, block < limit as usize);
                    assert_eq!(
                        got[row_at + block],
                        i64::try_from(expected).unwrap(),
                        "batch {batch_index} query {query} block {block}"
                    );
                }
                let row = &got[row_at..row_at + n as usize];
                let mut sorted = row.to_vec();
                sorted.sort_unstable();
                for pair in sorted.windows(2) {
                    assert_ne!(pair[0], pair[1], "block keys must be unique");
                }
                let mut forward: Vec<usize> = (0..n as usize).collect();
                forward.sort_unstable_by_key(|&block| row[block]);
                let mut pairs: Vec<(i64, usize)> = row
                    .iter()
                    .copied()
                    .enumerate()
                    .rev()
                    .map(|(block, key)| (key, block))
                    .collect();
                pairs.sort_unstable_by_key(|pair| pair.0);
                let mapped: Vec<usize> = pairs.into_iter().map(|pair| pair.1).collect();
                assert_eq!(mapped, forward);
            }
        }

        let perm: Vec<i32> = (0..n).rev().collect();
        let perm_index = broadcast_to(&host_i32(&perm, &[1, 1, n]), &[batch, seq, n], None);
        let keys_perm = take_along_axis(&keys, &perm_index, -1, None);
        let order = astype(&argsort_axis(&keys, -1, None), MlxDtype::Uint32, None);
        let order_perm = astype(&argsort_axis(&keys_perm, -1, None), MlxDtype::Uint32, None);
        eval(&[&order, &order_perm]);
        let order = order.data_u32();
        let order_perm = order_perm.data_u32();
        let width = n as usize;
        for row in 0..(batch * seq) as usize {
            for column in 0..width {
                let slot = order_perm[row * width + column] as usize;
                let mapped = perm[slot];
                assert_eq!(
                    mapped as u32,
                    order[row * width + column],
                    "row {row} column {column}: permuting unique keys changed the argsort"
                );
            }
        }

        let finite = [-0.0_f32, 0.0, 2.0, -1.0, 3.0, 3.0];
        let finite_array = array_f32(&finite, &[1, 1, 6]);
        let completes = host_i32(&[4], &[1]);
        let (decode_keys, _) =
            launch_prefill_block_keys(&finite_array, &completes, 1, 1, 6, 2, true).unwrap();
        let (prefill_keys, _) =
            launch_prefill_block_keys(&finite_array, &completes, 1, 1, 6, 2, false).unwrap();
        eval(&[&decode_keys, &prefill_keys]);
        assert_eq!(
            decode_keys.data_i64(),
            prefill_keys.data_i64(),
            "finite scores must reuse the prefill key encoding"
        );
    }

    /// Elementwise decode key. Same total order as [`decode_rank_key`] for a
    /// block id below 2^30, packed as `(key32 << 32) | block_id`.
    fn elementwise_rank_key(score: f32, block: u32, causal: bool) -> u64 {
        if !causal {
            return (0xffff_ffffu64 << 32) | u64::from(block);
        }
        let mut bits = score.to_bits();
        if bits == 0x8000_0000 {
            bits = 0;
        }
        if (bits & 0x7fff_ffff) > 0x7f80_0000 {
            bits = f32::NEG_INFINITY.to_bits();
        }
        let signed = i32::from_ne_bytes(bits.to_ne_bytes());
        let mask = if signed < 0 { u32::MAX } else { 0 };
        let order = bits ^ (mask | 0x8000_0000);
        (u64::from(!order) << 32) | u64::from(block)
    }

    #[test]
    fn decode_elementwise_keys_match_host_bits_and_ignore_argsort_stability() {
        let batch = 2i32;
        let seq = 2i32;
        let n = 6i32;
        let completes = [4i32, 3];
        let rows = [
            [-0.0_f32, 0.0, 1.0, 1.0, 9.0, 9.0],
            [
                f32::NAN,
                f32::NEG_INFINITY,
                f32::INFINITY,
                f32::from_bits(0x8000_0001),
                1_000.0,
                f32::NAN,
            ],
        ];
        let mut flat = Vec::new();
        for _batch in 0..batch {
            for row in &rows {
                flat.extend_from_slice(row);
            }
        }
        let array = array_f32(&flat, &[batch, seq, n]);
        let keys = elementwise_decode_block_keys(&array, &completes, seq, n);
        let words = view(
            &reshape(&keys, &[batch * seq * n], None),
            MlxDtype::Uint32,
            None,
        );
        eval(&[&words]);
        let words = words.data_u32();
        assert_eq!(words.len(), flat.len() * 2);
        for batch_index in 0..batch as usize {
            for (query, &limit) in completes.iter().enumerate() {
                let row_at = (batch_index * seq as usize + query) * n as usize;
                let mut seen = Vec::with_capacity(n as usize);
                for block in 0..n as usize {
                    let low = words[(row_at + block) * 2];
                    let high = words[(row_at + block) * 2 + 1];
                    let packed = (u64::from(high) << 32) | u64::from(low);
                    let expected = elementwise_rank_key(
                        flat[row_at + block],
                        block as u32,
                        block < limit as usize,
                    );
                    assert_eq!(
                        packed, expected,
                        "batch {batch_index} query {query} block {block}"
                    );
                    seen.push(packed);
                }
                let mut sorted = seen.clone();
                sorted.sort_unstable();
                for pair in sorted.windows(2) {
                    assert_ne!(pair[0], pair[1], "block keys must be unique");
                }
            }
        }

        let perm: Vec<i32> = (0..n).rev().collect();
        let perm_index = broadcast_to(&host_i32(&perm, &[1, 1, n]), &[batch, seq, n], None);
        let keys_perm = take_along_axis(&keys, &perm_index, -1, None);
        let order = astype(&argsort_axis(&keys, -1, None), MlxDtype::Uint32, None);
        let order_perm = astype(&argsort_axis(&keys_perm, -1, None), MlxDtype::Uint32, None);
        eval(&[&order, &order_perm]);
        let order = order.data_u32();
        let order_perm = order_perm.data_u32();
        let width = n as usize;
        for row in 0..(batch * seq) as usize {
            for column in 0..width {
                let slot = order_perm[row * width + column] as usize;
                let mapped = perm[slot];
                assert_eq!(
                    mapped as u32,
                    order[row * width + column],
                    "row {row} column {column}: permuting unique keys changed the argsort"
                );
            }
        }

        // A negative subnormal stays distinct from zero. This sorts the
        // elementwise key, which is built from the IEEE bits. The prefill
        // device-selection test covers the block-key kernel against the host
        // order for the same class of scores.
        let cfg = QsaConfig::new(2, 1, 4, 4, 2, 4, 16, EPS, BASE).unwrap();
        let sub = f32::from_bits(0x8000_0001);
        let scores = [sub, -0.0, 0.0, f32::from_bits(0x0000_0001)];
        let array = array_f32(&scores, &[1, 1, 4]);
        let keys = elementwise_decode_block_keys(&array, &[4], 1, 4);
        let order = contiguous(
            &astype(
                &slice(
                    &argsort_axis(&keys, -1, None),
                    &[0, 0, 0],
                    &[1, 1, cfg.block_topk() as i32],
                    &[1, 1, 1],
                    None,
                ),
                MlxDtype::Uint32,
                None,
            ),
            None,
        );
        eval(&[&order]);
        let ids: Vec<usize> = order.data_u32().iter().map(|&id| id as usize).collect();
        let host = select_tokens(cfg, 1, 1, 7, 4, Some(&array)).unwrap();
        assert_eq!(
            ids,
            blocks_from_tokens(&host[0][0], cfg.compress_ratio(), cfg.block_topk())
        );
    }

    #[test]
    fn decode_block_key_elementwise_microbench() {
        let sync = row_flag_sync_hits();
        let widths = [32i32, 256, 1024, 4096, 10_000];
        for &(batch, seq) in &[(1i32, 1i32), (1, 4)] {
            for &n in &widths {
                let keep_cols = 64.min(n);
                let mut scores = Vec::with_capacity((batch * seq * n) as usize);
                for row in 0..(batch * seq) {
                    for block in 0..n {
                        let value = ((block + row) % 17) as f32;
                        scores.push(if value == 0.0 { -0.0 } else { value });
                    }
                }
                let array = array_f32(&scores, &[batch, seq, n]);
                let completes: Vec<i32> = if seq == 1 {
                    vec![n]
                } else {
                    (1..=seq).map(|query| n / 4 * query).collect()
                };
                let time_median = |build: &mut dyn FnMut() -> MlxArray| -> (f64, u64) {
                    for _ in 0..2 {
                        let value = build();
                        eval(&[&value]);
                    }
                    let mut samples = Vec::with_capacity(8);
                    let mut ops = 0u64;
                    for index in 0..8 {
                        let start_ops = mlx_sys::op_count_snapshot();
                        let started = std::time::Instant::now();
                        let value = build();
                        let built = mlx_sys::op_count_take(start_ops);
                        eval(&[&value]);
                        samples.push(started.elapsed().as_secs_f64() * 1.0e6);
                        if index == 0 {
                            ops = built;
                        }
                    }
                    samples.sort_by(|left, right| left.partial_cmp(right).unwrap());
                    (samples[samples.len() / 2], ops)
                };
                let mut default_order = || {
                    let prepared = map_nan_to_neg_inf(&array);
                    let block_ids = reshape(
                        &arange(0.0, f64::from(n), 1.0, MlxDtype::Int32, None),
                        &[1, 1, n],
                        None,
                    );
                    let valid = less(&block_ids, &host_i32(&completes, &[1, seq, 1]), None);
                    let masked = where_cond(&valid, &prepared, &MlxArray::from_f32(f32::NAN), None);
                    let order = argsort_axis(&negative(&masked, None), -1, None);
                    slice(
                        &order,
                        &[0, 0, 0],
                        &[batch, seq, keep_cols],
                        &[1, 1, 1],
                        None,
                    )
                };
                let mut elementwise_order = || {
                    let keys = elementwise_decode_block_keys(&array, &completes, seq, n);
                    let order = argsort_axis(&keys, -1, None);
                    slice(
                        &order,
                        &[0, 0, 0],
                        &[batch, seq, keep_cols],
                        &[1, 1, 1],
                        None,
                    )
                };
                let mut elementwise_partition = || {
                    let keys = elementwise_decode_block_keys(&array, &completes, seq, n);
                    ordered_block_ids(&keys, keep_cols, keep_cols).unwrap()
                };
                let mut kernel_order = || {
                    let (keys, _flags) = launch_prefill_block_keys(
                        &array,
                        &host_i32(&completes, &[seq]),
                        batch,
                        seq,
                        n,
                        keep_cols,
                        true,
                    )
                    .unwrap();
                    ordered_block_ids(&keys, keep_cols, keep_cols).unwrap()
                };
                let read_u32 = |array: &MlxArray| -> Vec<u32> {
                    let array = contiguous(&astype(array, MlxDtype::Uint32, None), None);
                    eval(&[&array]);
                    array.data_u32().to_vec()
                };
                let default_ids = read_u32(&default_order());
                let elementwise_ids = read_u32(&elementwise_order());
                let kernel_ids = read_u32(&kernel_order());
                assert_eq!(
                    elementwise_ids, default_ids,
                    "batch {batch} seq {seq} n {n}: elementwise order diverged from the argsort"
                );
                assert_eq!(
                    elementwise_ids, kernel_ids,
                    "batch {batch} seq {seq} n {n}: elementwise order diverged from the kernel key"
                );
                let (default_us, default_ops) = time_median(&mut default_order);
                let (elementwise_us, elementwise_ops) = time_median(&mut elementwise_order);
                let (partition_us, partition_ops) = time_median(&mut elementwise_partition);
                let (kernel_us, kernel_ops) = time_median(&mut kernel_order);
                eprintln!(
                    "QSA_DECODE_KEY_BENCH batch={batch} seq={seq} n={n} keep={keep_cols} default_us={default_us:.1} default_ops={default_ops} elementwise_us={elementwise_us:.1} elementwise_ops={elementwise_ops} partition_us={partition_us:.1} partition_ops={partition_ops} kernel_us={kernel_us:.1} kernel_ops={kernel_ops}"
                );
            }
        }
        assert_eq!(row_flag_sync_hits(), sync, "microbench read a row flag");
    }

    /// Shown on every failure of [`mlx_argsort_pins_decode_qsa_tie_and_nan_order`].
    const DECODE_QSA_ARGSORT_BROKEN: &str = "an MLX bump broke the decode QSA ordering assumption; AX_MLX_FLASH_NEXT_QSA_DECODE_BLOCK_KEY=1 is the stability-independent path";

    /// One float32 score row: many exact ties, both signed zeros, and a NaN
    /// that sits before a later `+inf` in block-id order.
    fn decode_qsa_argsort_fixture() -> Vec<f32> {
        vec![
            1.0,
            -0.0,
            1.0,
            0.0,
            2.5,
            -0.0,
            f32::NAN,
            2.5,
            f32::INFINITY,
            0.0,
            1.0,
            f32::NAN,
            f32::INFINITY,
            -0.0,
            -3.0,
            -3.0,
            0.0,
            f32::NEG_INFINITY,
            4.0,
            -0.0,
            1.0,
            2.5,
            -3.0,
            0.0,
            f32::NAN,
            4.0,
            f32::INFINITY,
            -0.0,
            2.5,
            -3.0,
        ]
    }

    fn rotate_row(row: &[f32], shift: usize) -> Vec<f32> {
        let shift = shift % row.len();
        let mut rotated = Vec::with_capacity(row.len());
        rotated.extend_from_slice(&row[shift..]);
        rotated.extend_from_slice(&row[..shift]);
        rotated
    }

    /// Ascending order the default decode argsort relies on. NaN is after
    /// every numeric value, including `+inf`. `+0` and `-0` tie.
    fn decode_qsa_argsort_order(left: f32, right: f32) -> std::cmp::Ordering {
        match (left.is_nan(), right.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => left
                .partial_cmp(&right)
                .expect("finite and infinite f32 values compare"),
        }
    }

    fn assert_pinned_decode_argsort(order: &[u32], values: &[f32], what: &str) {
        assert_eq!(
            order.len(),
            values.len(),
            "{DECODE_QSA_ARGSORT_BROKEN} ({what}: length)"
        );
        let mut seen = vec![false; values.len()];
        for &index in order {
            let slot = index as usize;
            assert!(
                slot < values.len() && !seen[slot],
                "{DECODE_QSA_ARGSORT_BROKEN} ({what}: index {index} is not a permutation)"
            );
            seen[slot] = true;
        }
        let mut previous: Option<(f32, u32)> = None;
        let mut longest_tie = 1usize;
        let mut tie_run = 1usize;
        for &index in order {
            let value = values[index as usize];
            if let Some((previous_value, previous_index)) = previous {
                let ordering = decode_qsa_argsort_order(previous_value, value);
                assert_ne!(
                    ordering,
                    std::cmp::Ordering::Greater,
                    "{DECODE_QSA_ARGSORT_BROKEN} ({what}: {previous_value} sorted after {value})"
                );
                if ordering == std::cmp::Ordering::Equal {
                    assert!(
                        index > previous_index,
                        "{DECODE_QSA_ARGSORT_BROKEN} ({what}: tie indices {previous_index} then {index} are not ascending, +0/-0 included)"
                    );
                    tie_run += 1;
                    longest_tie = longest_tie.max(tie_run);
                } else {
                    tie_run = 1;
                }
            }
            previous = Some((value, index));
        }
        assert!(
            longest_tie >= 4,
            "{DECODE_QSA_ARGSORT_BROKEN} ({what}: expected many exact ties, longest was {longest_tie})"
        );
        let mut zero_indices = Vec::new();
        let mut left_zeros = false;
        for &index in order {
            if values[index as usize] == 0.0 {
                assert!(
                    !left_zeros,
                    "{DECODE_QSA_ARGSORT_BROKEN} ({what}: +0/-0 is not one tie)"
                );
                zero_indices.push(index);
            } else if !zero_indices.is_empty() {
                left_zeros = true;
            }
        }
        assert!(
            zero_indices.len() >= 4,
            "{DECODE_QSA_ARGSORT_BROKEN} ({what}: expected many +0/-0 ties)"
        );
        assert!(
            zero_indices.windows(2).all(|pair| pair[0] < pair[1]),
            "{DECODE_QSA_ARGSORT_BROKEN} ({what}: +0/-0 tie indices are not ascending)"
        );
        let saw_pos_zero = zero_indices
            .iter()
            .any(|&index| values[index as usize].to_bits() == 0);
        let saw_neg_zero = zero_indices
            .iter()
            .any(|&index| values[index as usize].to_bits() == (-0.0f32).to_bits());
        assert!(
            saw_pos_zero && saw_neg_zero,
            "{DECODE_QSA_ARGSORT_BROKEN} ({what}: +0/-0 tie dropped a sign)"
        );
        let first_nan = order
            .iter()
            .position(|&index| values[index as usize].is_nan());
        let last_inf = order
            .iter()
            .rposition(|&index| values[index as usize] == f32::INFINITY);
        assert!(
            first_nan.is_some() && last_inf.is_some(),
            "{DECODE_QSA_ARGSORT_BROKEN} ({what}: row needs a NaN and a +inf)"
        );
        assert!(
            first_nan.expect("nan") > last_inf.expect("inf"),
            "{DECODE_QSA_ARGSORT_BROKEN} ({what}: NaN sorts before +inf)"
        );
    }

    fn argsort_rows_on_gpu(values: &[f32], shape: &[i32], rows: usize) {
        let stream = mlx_sys::MlxStream::default_gpu();
        let scores = array_f32(values, shape);
        assert_eq!(
            scores.dtype(),
            MlxDtype::Float32,
            "{DECODE_QSA_ARGSORT_BROKEN} (decode scores are float32)"
        );
        let order = argsort_axis(&scores, -1, Some(&stream));
        let order = astype(&order, MlxDtype::Uint32, Some(&stream));
        eval(&[&order]);
        let indices = order.data_u32().to_vec();
        let cols = values.len() / rows;
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            assert_pinned_decode_argsort(
                &indices[start..end],
                &values[start..end],
                &format!("shape {shape:?} row {row}"),
            );
        }
    }

    /// Default decode QSA selection argsorts negated float32 scores. Single-query
    /// rows are `[batch, 1, blocks]`. A short verifier window is
    /// `[batch, seq, blocks]` with `seq` at most
    /// [`crate::fastpath::QSA_DEVICE_SELECT_MULTI_MAX_QUERIES`]. Both use the
    /// GPU stream. The order is stable on exact ties, `+0` and `-0` included,
    /// and NaN sorts after `+inf`. The float compare flushes subnormals, so
    /// this pin does not cover them. Decode does not promise host parity for
    /// a subnormal score.
    #[test]
    fn mlx_argsort_pins_decode_qsa_tie_and_nan_order() {
        let row = decode_qsa_argsort_fixture();
        let cols = i32::try_from(row.len()).expect("fixture fits i32");
        let nan_before_inf = row.iter().position(|value| value.is_nan()).expect("nan")
            < row
                .iter()
                .position(|value| *value == f32::INFINITY)
                .expect("inf");
        assert!(
            nan_before_inf,
            "{DECODE_QSA_ARGSORT_BROKEN} (fixture must place a NaN before a +inf)"
        );
        argsort_rows_on_gpu(&row, &[1, 1, cols], 1);

        let seq = crate::fastpath::QSA_DEVICE_SELECT_MULTI_MAX_QUERIES as usize;
        let batch = 2usize;
        let mut window = Vec::with_capacity(batch * seq * row.len());
        for shift in 0..(batch * seq) {
            window.extend(rotate_row(&row, shift));
        }
        argsort_rows_on_gpu(&window, &[batch as i32, seq as i32, cols], batch * seq);
    }

    #[test]
    fn single_query_and_short_windows_select_on_the_device() {
        let spec = tiny(2, 4);
        let cache = QsaIndexKeyCache::empty();
        // A short multi-query window (within the multi-query cap) selects on
        // the device by default; forcing the multi flag off keeps host
        // selection, and a chunk past the cap always selects on the host.
        let window = hidden_for(&spec, 4);
        let short = with_device_select_multi(true, || run(&spec, &window, &cache, 0));
        assert!(
            short.device_tokens().is_some(),
            "short verifier windows select on the device by default"
        );
        let forced_host = with_device_select_multi(false, || run(&spec, &window, &cache, 0));
        assert!(
            forced_host.device_tokens().is_none(),
            "the multi-query kill switch restores host selection"
        );
        assert_eq!(
            short.tokens_for_query(0, 2),
            forced_host.tokens_for_query(0, 2),
            "device and host selections must agree row by row"
        );
        let chunk = hidden_for(&spec, 6);
        let long = with_prefill_device_select(false, || {
            with_device_select_multi(true, || run(&spec, &chunk, &cache, 0))
        });
        assert!(
            long.device_tokens().is_none(),
            "the prefill kill switch keeps chunks past the multi-query cap on the host"
        );
        let one = &window[..spec.hidden];
        let on = with_device_select(true, || run(&spec, one, &cache, 0));
        let off = with_device_select(false, || run(&spec, one, &cache, 0));
        assert!(on.device_tokens().is_some());
        assert!(off.device_tokens().is_none());
        assert_eq!(on.tokens_for_query(0, 0), off.tokens_for_query(0, 0));
    }

    fn prefill_cfg(ratio: usize, budget: usize) -> QsaConfig {
        QsaConfig::new(2, 1, 4, 4, ratio, budget, 16, EPS, BASE).unwrap()
    }

    fn selection_from_prefill(
        cfg: QsaConfig,
        device: MlxArray,
        segments: Vec<(i32, i32)>,
    ) -> QsaSelection {
        QsaSelection {
            gather_indices: device.clone(),
            tokens: Vec::new(),
            device_tokens: Some(device),
            device_row_segments: segments,
            device_tail_width: cfg.compress_ratio() as i32 - 1,
            test_host_tokens: std::sync::OnceLock::new(),
            next_cache: QsaIndexKeyCache::empty(),
        }
    }

    fn assert_prefill_matches_host(
        cfg: QsaConfig,
        batch: i32,
        seq: i32,
        offset: usize,
        n_complete: i32,
        scores: Option<&[f32]>,
    ) {
        let array = scores.map(|scores| array_f32(scores, &[batch, seq, n_complete]));
        let host = select_tokens(cfg, batch, seq, offset, n_complete, array.as_ref()).unwrap();
        let (device, segments) =
            select_tokens_device_prefill(cfg, batch, seq, offset, n_complete, array.as_ref())
                .unwrap();
        let selection = selection_from_prefill(cfg, device, segments);
        for (b, queries) in host.iter().enumerate() {
            for (s, chosen) in queries.iter().enumerate() {
                assert_eq!(
                    selection.tokens_for_query(b, s),
                    chosen.as_slice(),
                    "batch={b} query={s} offset={offset} blocks={n_complete}"
                );
            }
        }
        if seq < 16 {
            return;
        }
        let compact = selection.compact_device_indices().unwrap();
        let compact = astype(&compact, MlxDtype::Uint32, None);
        eval(&[&compact]);
        let width = compact.shape()[2] as usize;
        let data = compact.data_u32();
        for b in 0..batch as usize {
            for s in 0..seq as usize {
                let chosen = selection.tokens_for_query(b, s);
                let row = &data[(b * seq as usize + s) * width..][..width];
                let packed: Vec<i32> = row[..chosen.len()]
                    .iter()
                    .map(|token| *token as i32)
                    .collect();
                assert_eq!(packed, chosen, "compact prefix batch={b} query={s}");
                assert!(
                    row[chosen.len()..].iter().all(|token| *token as i32 == -1),
                    "compact tail batch={b} query={s}"
                );
            }
        }
    }

    #[test]
    fn prefill_device_selection_matches_host_on_subnormal_scores_zeros_ties_and_normals() {
        let cfg = prefill_cfg(4, 16);
        let before = prefill_route_hits();
        let pos_tiny = f32::from_bits(0x0000_0001);
        let pos_next = f32::from_bits(0x0000_0002);
        let pos_wide = f32::from_bits(0x0070_1234);
        let neg_tiny = f32::from_bits(0x8000_0001);
        let neg_next = f32::from_bits(0x8000_0002);
        // Offset 31, ratio 4, eight blocks: both queries see every block, and
        // every score is finite, so block_topk keeps the device order.
        // Flushing a subnormal into zero would pick the lower block id ahead
        // of a larger positive subnormal.
        let mut scores = vec![pos_tiny, pos_next, 0.0, -0.0, neg_tiny, neg_next, 1.0, 1.0];
        scores.extend_from_slice(&[
            pos_tiny, pos_wide, pos_wide, 0.0, -0.0, neg_tiny, -1.0, -2.0,
        ]);
        assert_prefill_matches_host(cfg, 1, 2, 31, 8, Some(&scores));
        let array = array_f32(&scores, &[1, 2, 8]);
        let (keys, _flags) =
            launch_prefill_block_keys(&array, &host_i32(&[8, 8], &[2]), 1, 2, 8, 4, false).unwrap();
        eval(&[&keys]);
        let got = keys.data_i64();
        for (index, score) in scores.iter().copied().enumerate() {
            let expected = decode_rank_key(score, (index % 8) as u32, true);
            assert_eq!(
                got[index],
                i64::try_from(expected).unwrap(),
                "block {} score bits {:08x}",
                index % 8,
                score.to_bits()
            );
        }
        let on_device = prefill_route_hits();
        assert!(
            on_device.0 > before.0,
            "finite subnormal rows stay on the device order"
        );
        assert_eq!(
            on_device.1, before.1,
            "a subnormal must stay finite and inside the causal budget"
        );

        // Two causal subnormals are finite but short of block_topk, so the
        // chunk falls back to the host. A future non-finite score stays future.
        let short = [pos_tiny, neg_tiny, 1_000.0, f32::NAN];
        assert_prefill_matches_host(cfg, 1, 1, 7, 4, Some(&short));
        let after_short = prefill_route_hits();
        assert!(
            after_short.1 > on_device.1,
            "fewer finite causal blocks than block_topk falls back to the host"
        );

        assert_prefill_error_matches(cfg, 1, 1, 15, 4, &[pos_tiny, f32::NAN, 1.0, neg_next]);
        assert_prefill_error_matches(cfg, 1, 1, 15, 4, &[neg_tiny, 0.0, f32::INFINITY, pos_wide]);
        assert_prefill_error_matches(cfg, 1, 1, 15, 4, &[f32::NEG_INFINITY, pos_tiny, -0.0, 1.0]);
    }

    #[test]
    fn prefill_device_selection_matches_host_on_random_scores_ties_and_signed_zero() {
        let cfg = prefill_cfg(4, 16);
        let before = prefill_route_hits();
        // Pinned: score 1 wins, then signed zeros by ascending block id.
        // Every row has at least block_topk finite causal blocks, so this stays
        // on the device order rather than the host fallback.
        let pinned = [-0.0_f32, 0.0, 1.0, -0.0];
        assert_prefill_matches_host(cfg, 1, 1, 15, 4, Some(&pinned));
        let tied = [0.0_f32, 0.0, -0.0, 0.0];
        assert_prefill_matches_host(cfg, 1, 1, 15, 4, Some(&tied));
        let mut seed = 0x51_5au32;
        let (batch, seq, n_blocks) = (2, 5, 9);
        let mut scores = Vec::with_capacity((batch * seq * n_blocks) as usize);
        for _ in 0..(batch * seq * n_blocks) {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let value = match (seed >> 16) % 7 {
                0 => -0.0,
                1 => 0.0,
                2 => 1.0,
                3 => 1.0,
                4 => 2.0,
                5 => -3.0,
                _ => 4.0,
            };
            scores.push(value);
        }
        assert_prefill_matches_host(cfg, batch, seq, 20, n_blocks, Some(&scores));
        let after = prefill_route_hits();
        assert!(
            after.0 > before.0,
            "finite rows with a full causal budget stay on the device order"
        );
        assert_eq!(
            after.1, before.1,
            "those rows must not fall back to the host sort"
        );
    }

    #[test]
    fn prefill_device_selection_matches_host_on_partial_tails_offsets_and_chunk_boundaries() {
        let cfg = prefill_cfg(4, 32);
        let lengths = [(1, 0), (2, 0), (4, 0), (5, 0), (16, 0), (17, 0)];
        let offsets = [0usize, 3, 4, 7, 8, 15];
        for &(seq, _) in &lengths {
            for &offset in &offsets {
                let n_complete = (offset + seq) / cfg.compress_ratio();
                let count = seq * n_complete;
                let mut seed = 0x9E37_u32
                    .wrapping_add(seq as u32)
                    .wrapping_add(offset as u32);
                let scores: Vec<f32> = (0..count)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((seed >> 16) % 5) as f32
                    })
                    .collect();
                let scores = if n_complete == 0 {
                    None
                } else {
                    Some(scores.as_slice())
                };
                assert_prefill_matches_host(cfg, 1, seq as i32, offset, n_complete as i32, scores);
            }
        }
        // No complete block: tail only, including a short chunk below the multi cap.
        assert_prefill_matches_host(cfg, 2, 3, 0, 0, None);
    }

    #[test]
    fn prefill_device_selection_rejects_visible_nonfinite_and_ignores_future() {
        let cfg = prefill_cfg(4, 16);
        // offset 3, seq 3, ratio 4: each query sees exactly block 0. Later
        // columns are future and must not win or raise, even when nonfinite.
        let n_complete = 4;
        let mut scores = vec![0.5_f32; 2 * 3 * n_complete as usize];
        // Row-major (batch, query, block). Offset 3 keeps only block 0 causal.
        scores[4] = f32::INFINITY;
        scores[13] = f32::NAN;
        scores[20] = f32::NAN;
        visible_nonfinite_at(&cfg, &scores, 0, 1, 0);
        scores[0] = f32::NEG_INFINITY;
        visible_nonfinite_at(&cfg, &scores, 0, 0, 0);
        let future = [1.0_f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let array = array_f32(&future, &[1, 1, n_complete]);
        let host = select_tokens(cfg, 1, 1, 3, n_complete, Some(&array)).unwrap();
        let (device, segments) =
            select_tokens_device_prefill(cfg, 1, 1, 3, n_complete, Some(&array)).unwrap();
        let selection = selection_from_prefill(cfg, device, segments);
        assert_eq!(selection.tokens_for_query(0, 0), host[0][0].as_slice());
    }

    fn visible_nonfinite_at(
        cfg: &QsaConfig,
        scores: &[f32],
        batch: usize,
        query: usize,
        block: usize,
    ) {
        let array = array_f32(scores, &[2, 3, 4]);
        let host = select_tokens(*cfg, 2, 3, 3, 4, Some(&array)).unwrap_err();
        let device = select_tokens_device_prefill(*cfg, 2, 3, 3, 4, Some(&array)).unwrap_err();
        assert!(
            matches!(
                host,
                QsaError::NonFiniteScore {
                    batch: got_batch,
                    query: got_query,
                    block: got_block,
                } if got_batch == batch && got_query == query && got_block == block
            ),
            "host {host}"
        );
        assert!(
            matches!(
                device,
                QsaError::NonFiniteScore {
                    batch: got_batch,
                    query: got_query,
                    block: got_block,
                } if got_batch == batch && got_query == query && got_block == block
            ),
            "device {device}"
        );
    }

    fn assert_prefill_error_matches(
        cfg: QsaConfig,
        batch: i32,
        seq: i32,
        offset: usize,
        n_complete: i32,
        scores: &[f32],
    ) {
        let array = array_f32(scores, &[batch, seq, n_complete]);
        let host = select_tokens(cfg, batch, seq, offset, n_complete, Some(&array)).unwrap_err();
        let device =
            select_tokens_device_prefill(cfg, batch, seq, offset, n_complete, Some(&array))
                .unwrap_err();
        assert_eq!(host.to_string(), device.to_string());
    }

    #[test]
    fn prefill_device_selection_matches_host_on_nonfinite_short_and_future_rows() {
        // block_topk is 2. Rows below have either a full causal budget or not.
        let cfg = prefill_cfg(4, 8);
        let before = prefill_route_hits();
        // Both queries see exactly two causal blocks. Later columns are future,
        // including NaN, infinities, and a score larger than every causal one.
        let mut future = vec![
            3.0_f32,
            1.0,
            f32::NAN,
            f32::INFINITY,
            1_000.0,
            f32::NEG_INFINITY,
        ];
        future.extend_from_slice(&[
            0.5,
            2.0,
            f32::NAN,
            f32::INFINITY,
            1_000.0,
            f32::NEG_INFINITY,
        ]);
        assert_prefill_matches_host(cfg, 1, 2, 8, 6, Some(&future));
        let mid = prefill_route_hits();
        assert!(
            mid.0 > before.0,
            "future non-finite rows stay on the device order"
        );
        assert_eq!(mid.1, before.1);

        // One causal block and a future score that would win if it were eligible.
        let short = [0.25_f32, 1_000.0, f32::NAN, f32::INFINITY];
        assert_prefill_matches_host(cfg, 1, 1, 3, 4, Some(&short));
        let after_short = prefill_route_hits();
        assert!(
            after_short.1 > mid.1,
            "fewer finite causal blocks than block_topk falls back to the host"
        );

        // No query can see a block. Every score is future.
        let all_future = [
            1_000.0_f32,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            4.0,
            -1.0,
        ];
        assert_prefill_matches_host(cfg, 1, 2, 0, 3, Some(&all_future));

        let before_err = prefill_route_hits();
        assert_prefill_error_matches(cfg, 1, 1, 15, 4, &[1.0, f32::INFINITY, 0.5, 0.2]);
        assert_prefill_error_matches(cfg, 1, 1, 15, 4, &[5.0, 4.0, 3.0, f32::NAN]);
        assert_prefill_error_matches(cfg, 1, 1, 15, 4, &[f32::NEG_INFINITY, 1.0, 2.0, 3.0]);
        let mut batched = vec![1.0_f32, 2.0, 3.0, 0.5];
        batched.extend_from_slice(&[1.0, 2.0, f32::NAN, 9.0]);
        assert_prefill_error_matches(cfg, 2, 1, 15, 4, &batched);
        let after_err = prefill_route_hits();
        assert!(
            after_err.1 > before_err.1,
            "a causal non-finite score falls back to the host error"
        );
    }

    #[test]
    fn prefill_device_selection_uses_host_when_the_block_key_kernel_is_unavailable() {
        let cfg = prefill_cfg(4, 16);
        let scores = [1.0_f32, -0.0, 0.0, 3.0, 2.0, 4.0, -1.0, 0.5];
        let before = prefill_route_hits();
        with_prefill_key_kernel(false, || {
            assert_prefill_matches_host(cfg, 1, 2, 15, 4, Some(&scores));
            assert_prefill_error_matches(cfg, 1, 1, 15, 4, &[1.0, f32::NAN, 3.0, 4.0]);
        });
        let after = prefill_route_hits();
        assert_eq!(
            after.0, before.0,
            "a missing kernel must not sort on the device"
        );
        assert!(
            after.1 > before.1,
            "a missing kernel falls back to the host sort"
        );
    }

    #[test]
    fn prefill_device_select_is_opt_in_for_long_chunks() {
        let spec = tiny(2, 4);
        let cache = QsaIndexKeyCache::empty();
        let chunk = hidden_for(&spec, 6);
        // Unset is on. `=0` (the thread-local stand-in for that kill switch)
        // is the only way back to the host sort.
        let device = run(&spec, &chunk, &cache, 0);
        assert!(
            device.device_tokens().is_some(),
            "prefill device selection is on unless AX_MLX_FLASH_NEXT_QSA_PREFILL_DEVICE_SELECT=0"
        );
        let host = with_prefill_device_select(false, || run(&spec, &chunk, &cache, 0));
        assert!(
            host.device_tokens().is_none(),
            "the kill switch restores host selection"
        );
        for query in 0..6 {
            assert_eq!(
                device.tokens_for_query(0, query),
                host.tokens_for_query(0, query),
                "query {query}"
            );
        }
    }

    #[test]
    fn prefill_key_warmup_is_flash_next_only() {
        assert!(qsa_prefill_key_warmup_applies(true, true, true));
        assert!(!qsa_prefill_key_warmup_applies(false, true, true));
        assert!(!qsa_prefill_key_warmup_applies(true, false, true));
        assert!(!qsa_prefill_key_warmup_applies(true, true, false));
    }

    #[test]
    fn prefill_key_kernel_warm_matches_host_on_shapes_other_than_the_warmup() {
        warm_prefill_key_kernel().expect("prefill key kernel warm-up");
        let cfg = prefill_cfg(4, 16);
        let scores: Vec<f32> = (0..16 * 9).map(|i| (i % 7) as f32 - 3.0).collect();
        assert_prefill_matches_host(cfg, 1, 16, 4, 9, Some(&scores));
        // Shorter than the constant-buffer cutoff. Padding keeps this on the
        // same pipeline as the warm-up launch.
        let short: Vec<f32> = (0..3 * 5).map(|i| (i as f32) * 0.5 - 1.0).collect();
        assert_prefill_matches_host(cfg, 1, 3, 1, 5, Some(&short));
        let tiny_scores = [1.0_f32, -0.0, 0.0, 2.0];
        assert_prefill_matches_host(cfg, 1, 2, 0, 2, Some(&tiny_scores));
    }

    #[test]
    fn partial_selection_reports_nonfinite_location() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                select_blocks(&[0.0, invalid, 2.0], 1, 3, 7),
                Err(QsaError::NonFiniteScore {
                    batch: 3,
                    query: 7,
                    block: 1
                })
            ));
        }
    }

    #[test]
    fn selection_rejects_nonfinite_visible_scores() {
        let cfg = QsaConfig::new(2, 1, 4, 4, 2, 4, 16, EPS, BASE).unwrap();
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let scores = array_f32(&[1.0, invalid, 3.0], &[1, 1, 3]);
            assert!(
                select_tokens(cfg, 1, 1, 5, 3, Some(&scores)).is_err(),
                "nonfinite visible score must fail selection: {invalid}"
            );
        }
    }

    #[test]
    fn selection_ignores_nonfinite_future_blocks() {
        let cfg = QsaConfig::new(2, 1, 4, 4, 2, 4, 16, EPS, BASE).unwrap();
        let scores = array_f32(&[1.0, f32::NAN, f32::INFINITY], &[1, 1, 3]);
        let selected = select_tokens(cfg, 1, 1, 2, 3, Some(&scores)).unwrap();
        assert_eq!(selected, vec![vec![vec![0, 1, 2]]]);
    }

    #[test]
    fn selection_matches_pinned_transformers_for_all_chunk_boundaries() {
        fn values(value: &serde_json::Value) -> Vec<f32> {
            match value {
                serde_json::Value::Array(items) => items.iter().flat_map(values).collect(),
                _ => vec![value.as_f64().unwrap() as f32],
            }
        }
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/flash_next/qsa.json")).unwrap();
        let weights = &fixture["weights"];
        let gain = |name: &str| {
            array_f32(
                &values(&weights[name])
                    .iter()
                    .map(|v| v + 1.0)
                    .collect::<Vec<_>>(),
                &[4],
            )
        };
        let module = QsaIndexer::new(
            QsaConfig::new(2, 1, 4, 4, 2, 4, 16, EPS, BASE).unwrap(),
            QsaIndexerWeights {
                qk_proj: QuantizedWeight::new(
                    array_f32(&values(&weights["indexer.index_qk_proj.weight"]), &[12, 16]),
                    None,
                    None,
                ),
                q_norm: gain("indexer.q_layernorm.weight"),
                k_norm: gain("indexer.k_layernorm.weight"),
            },
        )
        .unwrap();
        let data = values(&fixture["input"]);
        let expected = fixture["selected"][0].as_array().unwrap();
        for boundary in 1..=19 {
            let mut cache = QsaIndexKeyCache::empty();
            for (start, end) in [(0, boundary), (boundary, 19)] {
                if start == end {
                    continue;
                }
                let input = array_f32(&data[start * 16..end * 16], &[1, (end - start) as i32, 16]);
                let selection = module.select(&input, &cache, start).unwrap();
                for (row, expected_row) in expected.iter().enumerate().take(end).skip(start) {
                    let mut actual = selection.tokens_for_query(0, row - start).to_vec();
                    actual.sort_unstable();
                    let wanted: Vec<i32> = expected_row
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, yes)| yes.as_bool().unwrap().then_some(i as i32))
                        .collect();
                    assert_eq!(actual, wanted, "boundary {boundary}, query {row}");
                }
                cache = selection.into_next_cache();
            }
        }
        assert!(QsaConfig::new(i32::MAX as usize, 1, 4, 4, 2, 4, 16, EPS, BASE).is_err());
    }

    const EPS: f32 = 1e-6;
    const BASE: f32 = 10_000.0;
    const TOL: f32 = 2e-4;

    struct Tiny {
        query_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        ratio: usize,
        budget: usize,
        hidden: usize,
        qk: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    }

    fn tiny(rotary_dim: usize, budget: usize) -> Tiny {
        let (query_heads, head_dim, ratio, hidden) = (2usize, 4usize, 2usize, 4usize);
        let out = (query_heads + 1) * head_dim;
        Tiny {
            query_heads,
            head_dim,
            rotary_dim,
            ratio,
            budget,
            hidden,
            qk: wave(out * hidden, 0.37, 0.45, 0.08),
            q_norm: wave(head_dim, 0.21, 0.3, 0.9),
            k_norm: wave(head_dim, 0.17, 0.25, 1.05),
        }
    }

    fn wave(len: usize, freq: f32, amplitude: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + 1.0) * freq).sin() * amplitude + offset)
            .collect()
    }

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    fn eval_f32(array: &MlxArray) -> Vec<f32> {
        eval(&[array]);
        array.data_f32().to_vec()
    }

    fn eval_i32_as_f32(array: &MlxArray) -> Vec<f32> {
        let values = astype(array, MlxDtype::Float32, None);
        eval_f32(&values)
    }

    fn indexer(spec: &Tiny) -> QsaIndexer {
        let config = QsaConfig::new(
            spec.query_heads,
            1,
            spec.head_dim,
            spec.rotary_dim,
            spec.ratio,
            spec.budget,
            spec.hidden,
            EPS,
            BASE,
        )
        .unwrap();
        QsaIndexer::new(
            config,
            QsaIndexerWeights {
                qk_proj: QuantizedWeight::new(
                    array_f32(&spec.qk, &[config.proj_out(), spec.hidden as i32]),
                    None,
                    None,
                ),
                q_norm: array_f32(&spec.q_norm, &[spec.head_dim as i32]),
                k_norm: array_f32(&spec.k_norm, &[spec.head_dim as i32]),
            },
        )
        .unwrap()
    }

    fn hidden_for(spec: &Tiny, tokens: usize) -> Vec<f32> {
        let mut data = wave(tokens * spec.hidden, 0.41, 0.9, 0.12);
        for (index, value) in data.iter_mut().enumerate() {
            *value *= 0.6 + 0.2 * (index % spec.hidden) as f32;
        }
        data
    }

    fn project_host(weight: &[f32], out: usize, input: usize, x: &[f32]) -> Vec<f32> {
        (0..out)
            .map(|row| {
                (0..input)
                    .map(|col| weight[row * input + col] * x[col])
                    .sum()
            })
            .collect()
    }

    fn rms_host(x: &[f32], weight: &[f32]) -> Vec<f32> {
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        x.iter()
            .zip(weight)
            .map(|(value, gain)| value * inv * gain)
            .collect()
    }

    fn rope_host(x: &[f32], rotary_dim: usize, pos: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        let half = rotary_dim / 2;
        let log_base = (BASE as f64).ln();
        for i in 0..half {
            let inv_freq = (-(i as f64) * log_base / half as f64).exp() as f32;
            let theta = pos as f32 * inv_freq;
            let (cos, sin) = (theta.cos(), theta.sin());
            let x1 = x[i];
            let x2 = x[i + half];
            out[i] = x1 * cos - x2 * sin;
            out[i + half] = x1 * sin + x2 * cos;
        }
        out
    }

    struct HostState {
        queries: Vec<Vec<f32>>,
        raw_keys: Vec<f32>,
        scores: Vec<f32>,
        n_complete: usize,
    }

    fn host_forward(spec: &Tiny, hidden: &[f32], past_keys: &[f32], offset: usize) -> HostState {
        let tokens = hidden.len() / spec.hidden;
        let q_width = spec.query_heads * spec.head_dim;
        let mut queries = Vec::new();
        let mut new_keys = Vec::new();
        for token in 0..tokens {
            let x = &hidden[token * spec.hidden..(token + 1) * spec.hidden];
            let projected = project_host(&spec.qk, q_width + spec.head_dim, spec.hidden, x);
            let mut q_heads = Vec::new();
            for head in 0..spec.query_heads {
                let start = head * spec.head_dim;
                let normed = rms_host(&projected[start..start + spec.head_dim], &spec.q_norm);
                q_heads.extend(rope_host(&normed, spec.rotary_dim, offset + token));
            }
            queries.push(q_heads);
            new_keys.extend_from_slice(&projected[q_width..]);
        }
        let mut raw_keys = past_keys.to_vec();
        raw_keys.extend_from_slice(&new_keys);
        let total = raw_keys.len() / spec.head_dim;
        let n_complete = total / spec.ratio;
        let mut blocks = Vec::new();
        for block in 0..n_complete {
            let mut mean = vec![0.0f32; spec.head_dim];
            for inner in 0..spec.ratio {
                let row = (block * spec.ratio + inner) * spec.head_dim;
                for dim in 0..spec.head_dim {
                    mean[dim] += raw_keys[row + dim];
                }
            }
            for value in &mut mean {
                *value /= spec.ratio as f32;
            }
            let normed = rms_host(&mean, &spec.k_norm);
            blocks.push(rope_host(&normed, spec.rotary_dim, block * spec.ratio));
        }
        let scale = (spec.head_dim as f32).sqrt();
        let mut scores = Vec::new();
        for query in &queries {
            for block in &blocks {
                let mut sum = 0.0f32;
                for head in 0..spec.query_heads {
                    let q = &query[head * spec.head_dim..(head + 1) * spec.head_dim];
                    let dot: f32 = q.iter().zip(block.iter()).map(|(a, b)| a * b).sum();
                    sum += dot.max(0.0);
                }
                scores.push(sum / scale);
            }
        }
        HostState {
            queries,
            raw_keys,
            scores,
            n_complete,
        }
    }

    fn host_select(spec: &Tiny, state: &HostState, offset: usize) -> Vec<Vec<i32>> {
        let queries = state.queries.len();
        (0..queries)
            .map(|query| {
                let abs_pos = offset + query;
                let visible = abs_pos + 1;
                let complete = visible / spec.ratio;
                let mut order: Vec<usize> = (0..complete).collect();
                order.sort_by(|lhs, rhs| {
                    let left = state.scores[query * state.n_complete + lhs];
                    let right = state.scores[query * state.n_complete + rhs];
                    right
                        .partial_cmp(&left)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(lhs.cmp(rhs))
                });
                order.truncate(complete.min(spec.budget / spec.ratio));
                let mut chosen = Vec::new();
                for block in order {
                    let start = (block * spec.ratio) as i32;
                    for token in 0..spec.ratio as i32 {
                        chosen.push(start + token);
                    }
                }
                for token in complete * spec.ratio..visible {
                    chosen.push(token as i32);
                }
                chosen
            })
            .collect()
    }

    fn run(spec: &Tiny, hidden: &[f32], cache: &QsaIndexKeyCache, offset: usize) -> QsaSelection {
        let tokens = hidden.len() / spec.hidden;
        indexer(spec)
            .select(
                &array_f32(hidden, &[1, tokens as i32, spec.hidden as i32]),
                cache,
                offset,
            )
            .unwrap()
    }

    fn selected(sel: &QsaSelection, queries: usize) -> Vec<Vec<i32>> {
        (0..queries)
            .map(|query| sel.tokens_for_query(0, query).to_vec())
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= TOL * (1.0 + e.abs()),
                "{label}[{index}]: {a} vs {e} (diff {diff})"
            );
        }
    }

    #[test]
    fn causal_selection_never_keeps_future_keys() {
        let spec = tiny(2, 4);
        let hidden = hidden_for(&spec, 3);
        let sel = run(&spec, &hidden, &QsaIndexKeyCache::empty(), 0);
        let host = host_forward(&spec, &hidden, &[], 0);
        let expected = host_select(&spec, &host, 0);
        assert_eq!(selected(&sel, 3), expected);
        for (query, tokens) in expected.iter().enumerate() {
            assert!(
                tokens.iter().all(|token| *token <= query as i32),
                "query {query} kept a future key: {tokens:?}"
            );
        }
        assert_eq!(expected[0], vec![0]);
    }

    #[test]
    fn partial_block_is_always_appended() {
        let spec = tiny(2, 4);
        let hidden = hidden_for(&spec, 5);
        let sel = run(&spec, &hidden, &QsaIndexKeyCache::empty(), 0);
        let host = host_forward(&spec, &hidden, &[], 0);
        let expected = host_select(&spec, &host, 0);
        assert_eq!(selected(&sel, 5), expected);
        assert!(expected[0].contains(&0));
        assert_eq!(*expected[2].last().unwrap(), 2);
        assert_eq!(*expected[4].last().unwrap(), 4);
        assert!(!expected[2].contains(&3));
        assert!(!expected[4].contains(&5));
    }

    #[test]
    fn crossing_budget_keeps_highest_complete_blocks_and_the_tail() {
        let spec = tiny(2, 2);
        let hidden = hidden_for(&spec, 7);
        let sel = run(&spec, &hidden, &QsaIndexKeyCache::empty(), 0);
        let host = host_forward(&spec, &hidden, &[], 0);
        let expected = host_select(&spec, &host, 0);
        assert_eq!(selected(&sel, 7), expected);
        let last = &expected[6];
        assert_eq!(*last.last().unwrap(), 6);
        let complete: Vec<i32> = last[..last.len() - 1].to_vec();
        assert_eq!(complete.len(), spec.budget);
        let mut ranked: Vec<(f32, usize)> = (0..3)
            .map(|block| (host.scores[6 * 3 + block], block))
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        let best = ranked[0].1 as i32 * spec.ratio as i32;
        assert_eq!(complete, vec![best, best + 1]);
        assert_close(
            &eval_f32(sel.next_cache().keys().unwrap()),
            &host.raw_keys,
            "raw keys",
        );
    }

    #[test]
    fn chunked_selection_matches_incremental() {
        let spec = tiny(2, 4);
        let hidden = hidden_for(&spec, 6);
        let whole = run(&spec, &hidden, &QsaIndexKeyCache::empty(), 0);
        let expected = host_select(&spec, &host_forward(&spec, &hidden, &[], 0), 0);
        assert_eq!(selected(&whole, 6), expected);

        let model = indexer(&spec);
        let mut cache = QsaIndexKeyCache::empty();
        let mut step_tokens = Vec::new();
        for token in 0..6 {
            let row = &hidden[token * spec.hidden..(token + 1) * spec.hidden];
            let step = model
                .select(&array_f32(row, &[1, 1, spec.hidden as i32]), &cache, token)
                .unwrap();
            step_tokens.push(step.tokens_for_query(0, 0).to_vec());
            cache = step.into_next_cache();
        }
        assert_eq!(step_tokens, expected);
        assert_close(
            &eval_f32(cache.keys().unwrap()),
            &eval_f32(whole.next_cache().keys().unwrap()),
            "chunk vs incremental keys",
        );
    }

    #[test]
    fn main_attention_rotary_dim_is_not_the_index_head_width() {
        let spec = tiny(2, 2);
        let hidden = hidden_for(&spec, 32);
        let sel = run(&spec, &hidden, &QsaIndexKeyCache::empty(), 0);
        let expected = host_select(&spec, &host_forward(&spec, &hidden, &[], 0), 0);
        assert_eq!(selected(&sel, 32), expected);

        let full_head_spec = tiny(spec.head_dim, spec.budget);
        let full_head = host_select(
            &full_head_spec,
            &host_forward(&full_head_spec, &hidden, &[], 0),
            0,
        );
        assert_ne!(
            expected, full_head,
            "oracle must distinguish main RoPE width from the index head"
        );
        assert_eq!(spec.rotary_dim, 2);
        assert_eq!(spec.head_dim, 4);
    }

    #[test]
    fn clone_isolation_does_not_publish_failed_or_foreign_updates() {
        let spec = tiny(2, 4);
        let hidden = hidden_for(&spec, 4);
        let first = run(
            &spec,
            &hidden[..2 * spec.hidden],
            &QsaIndexKeyCache::empty(),
            0,
        );
        let original = first.next_cache().clone();
        let snapshot = eval_f32(original.keys().unwrap());
        let rest = &hidden[2 * spec.hidden..];
        let published = indexer(&spec)
            .select(
                &array_f32(rest, &[1, 2, spec.hidden as i32]),
                first.next_cache(),
                2,
            )
            .unwrap();
        assert_close(
            &eval_f32(first.next_cache().keys().unwrap()),
            &snapshot,
            "source cache after a later select",
        );
        assert_close(
            &eval_f32(original.keys().unwrap()),
            &snapshot,
            "cloned cache after a later select",
        );
        assert_ne!(eval_f32(published.next_cache().keys().unwrap()), snapshot);
        assert!(
            QsaConfig::new(2, 2, 4, 2, 2, 2, 4, EPS, BASE).is_err(),
            "key heads must stay 1"
        );
        assert!(
            QsaConfig::new(2, 1, 4, 6, 2, 2, 4, EPS, BASE).is_err(),
            "main rotary dim must fit the index head"
        );
        let packed = eval_i32_as_f32(published.gather_indices());
        assert!(packed.iter().any(|value| *value < 0.0));
        assert!(
            published
                .tokens_for_query(0, 1)
                .iter()
                .all(|token| *token <= 3)
        );
    }

    /// Deterministic rotary input: a slow sine of an arange ramp, so every
    /// value sits in `[-1, 1]` without any RNG.
    fn rotary_input(shape: &[i32], dtype: MlxDtype) -> MlxArray {
        let len: i32 = shape.iter().product();
        let ramp = arange(0.0, f64::from(len), 1.0, MlxDtype::Float32, None);
        let wave = sin(&multiply(&ramp, &MlxArray::from_f32(0.017), None), None);
        astype(&reshape(&wave, shape, None), dtype, None)
    }

    /// Widen to float32, evaluate, and compare the raw bit patterns: the cache
    /// must reproduce the uncached table exactly, not approximately.
    fn raw_bits(array: &MlxArray) -> Vec<u32> {
        let widened = astype(array, MlxDtype::Float32, None);
        eval(&[&widened]);
        widened
            .data_f32()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    #[test]
    fn rope_cache_matches_uncached_bits() {
        reset_rope_cache_for_test();
        for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16, MlxDtype::Float16] {
            for offset in [0, 1, 5, 4093] {
                for seq in [1, 3] {
                    for (shape, seq_axis) in [(vec![1, 4, seq, 256], 2), (vec![1, seq, 4, 128], 1)]
                    {
                        let x = rotary_input(&shape, dtype);
                        let uncached = with_rope_cache(false, || {
                            apply_qsa_rotary_offset(&x, 64, 10_000_000.0, offset, seq_axis)
                        });
                        let miss = with_rope_cache(true, || {
                            apply_qsa_rotary_offset(&x, 64, 10_000_000.0, offset, seq_axis)
                        });
                        let hit = with_rope_cache(true, || {
                            apply_qsa_rotary_offset(&x, 64, 10_000_000.0, offset, seq_axis)
                        });
                        let expected = raw_bits(&uncached);
                        let label =
                            format!("dtype={dtype:?} offset={offset} seq={seq} axis={seq_axis}");
                        assert_eq!(raw_bits(&miss), expected, "miss: {label}");
                        assert_eq!(raw_bits(&hit), expected, "hit: {label}");
                    }
                }
            }
        }
    }

    #[test]
    fn rope_cache_hit_skips_table_ops() {
        reset_rope_cache_for_test();
        let x = rotary_input(&[1, 4, 1, 256], MlxDtype::Bfloat16);
        let start = mlx_sys::op_count_snapshot();
        let _first = with_rope_cache(true, || apply_qsa_rotary_offset(&x, 64, 10_000_000.0, 7, 2));
        let first = mlx_sys::op_count_take(start);
        let start = mlx_sys::op_count_snapshot();
        let _second = with_rope_cache(true, || apply_qsa_rotary_offset(&x, 64, 10_000_000.0, 7, 2));
        let second = mlx_sys::op_count_take(start);
        assert!(second + 8 <= first, "first={first} second={second}");
    }

    #[test]
    fn rope_cache_keys_do_not_cross_contaminate() {
        reset_rope_cache_for_test();
        let cases: Vec<(&str, MlxArray, i32, f32, i32, i32)> = vec![
            (
                "baseline",
                rotary_input(&[1, 4, 2, 256], MlxDtype::Float32),
                64,
                10_000_000.0,
                3,
                2,
            ),
            (
                "dtype",
                rotary_input(&[1, 4, 2, 256], MlxDtype::Bfloat16),
                64,
                10_000_000.0,
                3,
                2,
            ),
            (
                "rotary-dim",
                rotary_input(&[1, 4, 2, 256], MlxDtype::Float32),
                32,
                10_000_000.0,
                3,
                2,
            ),
            (
                "rope-base",
                rotary_input(&[1, 4, 2, 256], MlxDtype::Float32),
                64,
                1_000_000.0,
                3,
                2,
            ),
            (
                "seq-axis",
                rotary_input(&[1, 2, 4, 128], MlxDtype::Float32),
                64,
                10_000_000.0,
                3,
                1,
            ),
            (
                "offset",
                rotary_input(&[1, 4, 2, 256], MlxDtype::Float32),
                64,
                10_000_000.0,
                9,
                2,
            ),
        ];
        let uncached: Vec<Vec<u32>> = cases
            .iter()
            .map(|(_, x, rotary_dim, base, offset, seq_axis)| {
                let rotated = with_rope_cache(false, || {
                    apply_qsa_rotary_offset(x, *rotary_dim, *base, *offset, *seq_axis)
                });
                raw_bits(&rotated)
            })
            .collect();
        for (index, (name, x, rotary_dim, base, offset, seq_axis)) in cases.iter().enumerate() {
            let cached = with_rope_cache(true, || {
                apply_qsa_rotary_offset(x, *rotary_dim, *base, *offset, *seq_axis)
            });
            assert_eq!(raw_bits(&cached), uncached[index], "{name}");
        }
        assert_eq!(rope_cache_len_for_test(), cases.len());
        // Interleave two keys: every hit must reuse only its own entry.
        for _ in 0..2 {
            for index in 0..2 {
                let (name, x, rotary_dim, base, offset, seq_axis) = &cases[index];
                let cached = with_rope_cache(true, || {
                    apply_qsa_rotary_offset(x, *rotary_dim, *base, *offset, *seq_axis)
                });
                assert_eq!(raw_bits(&cached), uncached[index], "{name} interleaved");
            }
        }
        assert_eq!(rope_cache_len_for_test(), cases.len());
    }

    #[test]
    fn rope_cache_evicts_oldest_beyond_capacity() {
        reset_rope_cache_for_test();
        let x = rotary_input(&[1, 4, 1, 256], MlxDtype::Bfloat16);
        let call = |offset: i32| {
            with_rope_cache(true, || {
                apply_qsa_rotary_offset(&x, 64, 10_000_000.0, offset, 2)
            })
        };
        let start = mlx_sys::op_count_snapshot();
        let first = call(0);
        let first_ops = mlx_sys::op_count_take(start);
        for offset in 1..=TRIG_CACHE_ENTRIES as i32 {
            let _ = call(offset);
        }
        assert_eq!(rope_cache_len_for_test(), TRIG_CACHE_ENTRIES);
        let expected = raw_bits(&first);
        let start = mlx_sys::op_count_snapshot();
        let rebuilt = call(0);
        assert_eq!(mlx_sys::op_count_take(start), first_ops);
        assert_eq!(raw_bits(&rebuilt), expected);
    }

    #[test]
    fn rope_cache_is_bypassed_under_media_scope() {
        reset_rope_cache_for_test();
        let x = rotary_input(&[1, 2, 3, 64], MlxDtype::Bfloat16);
        let text_like: Vec<[i32; 3]> = (0..12).map(|p| [p; 3]).collect();
        let uncached = with_rope_cache(false, || {
            let _scope = scoped_media(Some(media(text_like.clone(), 0)));
            apply_qsa_rotary_offset(&x, 64, 10_000_000.0, 5, 2)
        });
        let before = rope_cache_len_for_test();
        let cached = with_rope_cache(true, || {
            let _scope = scoped_media(Some(media(text_like, 0)));
            apply_qsa_rotary_offset(&x, 64, 10_000_000.0, 5, 2)
        });
        assert_eq!(rope_cache_len_for_test(), before);
        assert_eq!(raw_bits(&cached), raw_bits(&uncached));
    }

    #[test]
    fn block_starts_cache() {
        reset_rope_cache_for_test();
        let blocks = rotary_input(&[1, 5, 64], MlxDtype::Bfloat16);
        let uncached =
            with_rope_cache(false, || rope_block_starts(&blocks, 64, 10_000_000.0, 4, 0));
        let miss = with_rope_cache(true, || rope_block_starts(&blocks, 64, 10_000_000.0, 4, 0));
        let hit = with_rope_cache(true, || rope_block_starts(&blocks, 64, 10_000_000.0, 4, 0));
        assert_eq!(raw_bits(&miss), raw_bits(&uncached));
        assert_eq!(raw_bits(&hit), raw_bits(&uncached));
        assert_eq!(rope_cache_len_for_test(), 1);
        let longer = rotary_input(&[1, 6, 64], MlxDtype::Bfloat16);
        let other = with_rope_cache(true, || rope_block_starts(&longer, 64, 10_000_000.0, 4, 0));
        let other_uncached =
            with_rope_cache(false, || rope_block_starts(&longer, 64, 10_000_000.0, 4, 0));
        assert_eq!(raw_bits(&other), raw_bits(&other_uncached));
        assert_eq!(rope_cache_len_for_test(), 2);
    }

    #[test]
    fn explicit_positions_match_the_offset_path() {
        reset_rope_cache_for_test();
        let x = rotary_input(&[1, 4, 2, 256], MlxDtype::Bfloat16);
        let positions = arange(5.0, 7.0, 1.0, MlxDtype::Float32, None);
        let explicit = apply_qsa_rotary(&x, 64, 10_000_000.0, &positions, 2);
        let offset_path =
            with_rope_cache(true, || apply_qsa_rotary_offset(&x, 64, 10_000_000.0, 5, 2));
        assert_eq!(raw_bits(&explicit), raw_bits(&offset_path));
    }

    /// `concat(full[0..first], range(first..complete))` must reproduce
    /// `full[0..complete]` bit for bit.
    fn assert_pooled_split(
        keys: &MlxArray,
        full: &MlxArray,
        cfg: QsaConfig,
        k_norm_f32: &MlxArray,
        first: i32,
        n_complete: i32,
        label: &str,
    ) {
        let new = pooled_block_keys_range(keys, first, n_complete, cfg, k_norm_f32).unwrap();
        let joined = if first == 0 {
            new
        } else {
            let head = slice(
                full,
                &[0, 0, 0],
                &[1, first, cfg.head_dim],
                &[1, 1, 1],
                None,
            );
            concatenate(&[&head, &new], 1, None)
        };
        let reference = slice(
            full,
            &[0, 0, 0],
            &[1, n_complete, cfg.head_dim],
            &[1, 1, 1],
            None,
        );
        assert_eq!(raw_bits(&joined), raw_bits(&reference), "{label}");
    }

    #[test]
    fn pooled_range_matches_full_recompute() {
        reset_rope_cache_for_test();
        let cfg = QsaConfig::new(1, 1, 128, 64, 4, 8, 128, EPS, BASE).unwrap();
        let k_norm = array_f32(&wave(128, 0.13, 0.2, 1.0), &[128]);
        let splits = [(0, 1), (1, 2), (5, 6), (37, 40), (100, 101)];
        for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16, MlxDtype::Float16] {
            let keys = rotary_input(&[1, 4 * 101, 128], dtype);
            let full = pooled_block_keys(&keys, 101, cfg, &k_norm).unwrap();
            assert_eq!(full.shape(), vec![1, 101, 128]);
            for (first, n_complete) in splits {
                let label = format!("dtype={dtype:?} split={first}|{n_complete}");
                assert_pooled_split(&keys, &full, cfg, &k_norm, first, n_complete, &label);
            }
        }
    }

    /// Two selections must agree on everything the caller can observe: the
    /// chosen tokens, the packed gather indices, the next raw keys, and the
    /// pooled blocks that feed the block scores.
    fn assert_same_selection(
        reference: &QsaSelection,
        actual: &QsaSelection,
        queries: usize,
        label: &str,
    ) {
        for query in 0..queries {
            assert_eq!(
                actual.tokens_for_query(0, query),
                reference.tokens_for_query(0, query),
                "{label}: query {query} tokens"
            );
        }
        assert_eq!(
            eval_i32_as_f32(actual.gather_indices()),
            eval_i32_as_f32(reference.gather_indices()),
            "{label}: gather indices"
        );
        let actual_cache = actual.next_cache();
        let reference_cache = reference.next_cache();
        assert_eq!(
            raw_bits(actual_cache.keys().unwrap()),
            raw_bits(reference_cache.keys().unwrap()),
            "{label}: next keys"
        );
        match (
            actual_cache.pooled_blocks(),
            reference_cache.pooled_blocks(),
        ) {
            (Some(actual), Some(reference)) => assert_eq!(
                raw_bits(actual),
                raw_bits(reference),
                "{label}: pooled blocks"
            ),
            (None, None) => {}
            _ => panic!("{label}: pooled block presence differs"),
        }
    }

    /// Drive `steps` selections of `width` rows from a `prefix`-token cache,
    /// once with the pooled cache off (full recomputation) and once with it on.
    /// Returns how many steps completed no new block, i.e. could reuse the
    /// cache untouched.
    fn assert_pooled_selection_matches_full(
        spec: &Tiny,
        prefix: usize,
        width: usize,
        steps: usize,
        device_select: bool,
    ) -> usize {
        let model = indexer(spec);
        let cfg = model.config();
        let ratio = cfg.compress_ratio;
        let hidden = hidden_for(spec, prefix + width * steps);
        let prefix_input = array_f32(
            &hidden[..prefix * spec.hidden],
            &[1, prefix as i32, spec.hidden as i32],
        );
        let mut reference_cache = with_pooled_cache(false, || {
            with_rope_cache(false, || {
                model
                    .select_with_mode(&prefix_input, &QsaIndexKeyCache::empty(), 0, project, false)
                    .unwrap()
            })
        })
        .into_next_cache();
        let mut cached_cache = reference_cache.clone();
        let mut reused = 0usize;
        for step in 0..steps {
            let offset = prefix + step * width;
            let start = offset * spec.hidden;
            let input = array_f32(
                &hidden[start..start + width * spec.hidden],
                &[1, width as i32, spec.hidden as i32],
            );
            let mark = mlx_sys::op_count_snapshot();
            let reference = with_pooled_cache(false, || {
                with_rope_cache(false, || {
                    model
                        .select_with_mode(&input, &reference_cache, offset, project, device_select)
                        .unwrap()
                })
            });
            let full_ops = mlx_sys::op_count_take(mark);
            let mark = mlx_sys::op_count_snapshot();
            let actual = with_pooled_cache(true, || {
                with_rope_cache(false, || {
                    model
                        .select_with_mode(&input, &cached_cache, offset, project, device_select)
                        .unwrap()
                })
            });
            let cached_ops = mlx_sys::op_count_take(mark);
            let label = format!("prefix={prefix} width={width} device={device_select} step={step}");
            assert_same_selection(&reference, &actual, width, &label);
            // The pooled blocks are the block-score input, so they must also
            // equal a from-scratch pooling of this step's own keys.
            let next_cache = actual.next_cache();
            let keys = next_cache.keys().unwrap();
            let n_complete = keys.shape()[1] / ratio;
            if n_complete > 0 {
                let expected = pooled_block_keys(keys, n_complete, cfg, &model.k_norm_f32).unwrap();
                assert_eq!(
                    raw_bits(next_cache.pooled_blocks().unwrap()),
                    raw_bits(&expected),
                    "{label}: pooled blocks"
                );
            }
            let new_blocks = (offset as i32 + width as i32) / ratio - offset as i32 / ratio;
            if new_blocks == 0 && n_complete > 0 {
                reused += 1;
                assert!(
                    cached_ops < full_ops,
                    "{label}: cached={cached_ops} full={full_ops}"
                );
            }
            reference_cache = reference.into_next_cache();
            cached_cache = actual.into_next_cache();
        }
        reused
    }

    #[test]
    fn selection_with_pooled_cache_equals_full() {
        reset_rope_cache_for_test();
        let spec = tiny(2, 4);
        let host = assert_pooled_selection_matches_full(&spec, 8, 1, 40, false);
        assert!(host > 0, "single-query host path never reused blocks");
        let device = assert_pooled_selection_matches_full(&spec, 8, 1, 40, true);
        assert!(device > 0, "single-query device path never reused blocks");
        assert_pooled_selection_matches_full(&spec, 8, 3, 13, false);
        assert_pooled_selection_matches_full(&spec, 8, 3, 13, true);
    }

    #[test]
    fn pooled_cache_is_bypassed_under_media_and_restore() {
        reset_rope_cache_for_test();
        // rotary_dim 4 puts the H axis of the 3D positions into the rotation,
        // so an MRoPE block key really differs from its text rotation.
        let spec = tiny(4, 4);
        let model = indexer(&spec);
        let cfg = model.config();
        let hidden = hidden_for(&spec, 12);
        let warm = with_pooled_cache(false, || {
            model
                .select(
                    &array_f32(&hidden[..8 * spec.hidden], &[1, 8, spec.hidden as i32]),
                    &QsaIndexKeyCache::empty(),
                    0,
                )
                .unwrap()
        })
        .into_next_cache();
        assert_eq!(
            warm.pooled_blocks().unwrap().shape(),
            vec![1, 4, spec.head_dim as i32]
        );
        let next = array_f32(
            &hidden[8 * spec.hidden..12 * spec.hidden],
            &[1, 4, spec.hidden as i32],
        );
        let positions: Vec<[i32; 3]> = (0..16).map(|p| [p, p + 1, p + 2]).collect();
        let under_media = |enabled: bool| {
            with_pooled_cache(enabled, || {
                let _scope = scoped_media(Some(media(positions.clone(), 0)));
                model.select(&next, &warm, 8).unwrap()
            })
        };
        let reference = under_media(false);
        let actual = under_media(true);
        assert_same_selection(&reference, &actual, 4, "media scope");
        assert!(
            actual.next_cache().pooled_blocks().is_none(),
            "an MRoPE step must not publish text-rotated blocks"
        );
        // The gate exists because the media rotation really does differ.
        let warmed_keys = actual.next_cache().keys().unwrap();
        let n_complete = warmed_keys.shape()[1] / cfg.compress_ratio;
        let text_blocks =
            pooled_block_keys(warmed_keys, n_complete, cfg, &model.k_norm_f32).unwrap();
        let media_blocks = {
            let _scope = scoped_media(Some(media(positions.clone(), 0)));
            pooled_block_keys(warmed_keys, n_complete, cfg, &model.k_norm_f32).unwrap()
        };
        assert_ne!(
            raw_bits(&text_blocks),
            raw_bits(&media_blocks),
            "the media block rotation must differ from the text one"
        );

        // A restored cache carries no derived blocks, so its first step
        // recomputes them and still matches the warm cache exactly.
        let restored = QsaIndexKeyCache::from_serialized(Some(warm.keys().unwrap().clone()));
        assert!(restored.pooled_blocks().is_none());
        let text_reference = with_pooled_cache(false, || model.select(&next, &warm, 8).unwrap());
        let text_actual = with_pooled_cache(true, || model.select(&next, &restored, 8).unwrap());
        assert_same_selection(&text_reference, &text_actual, 4, "restored cache");
        // And the text step that follows a media step recovers the cache: it
        // finds no derived blocks, recomputes them all, and publishes them.
        let follow = array_f32(
            &hidden[11 * spec.hidden..12 * spec.hidden],
            &[1, 1, spec.hidden as i32],
        );
        let follow_reference = with_pooled_cache(false, || {
            model.select(&follow, actual.next_cache(), 12).unwrap()
        });
        let recovered = with_pooled_cache(true, || {
            model.select(&follow, actual.next_cache(), 12).unwrap()
        });
        assert_same_selection(&follow_reference, &recovered, 1, "after a media step");
        assert!(
            recovered.next_cache().pooled_blocks().is_some(),
            "a text step after a media step must publish blocks again"
        );
    }

    #[test]
    fn pooled_cache_forks_do_not_share_blocks() {
        reset_rope_cache_for_test();
        let spec = tiny(2, 4);
        let model = indexer(&spec);
        let cfg = model.config();
        let hidden = hidden_for(&spec, 12);
        // A second token stream of the same shape with different rows, so the
        // two successors really pool different block keys.
        let other: Vec<f32> = hidden.iter().map(|value| value * 0.5 + 0.25).collect();
        let base = with_pooled_cache(false, || {
            model
                .select(
                    &array_f32(&hidden[..6 * spec.hidden], &[1, 6, spec.hidden as i32]),
                    &QsaIndexKeyCache::empty(),
                    0,
                )
                .unwrap()
        })
        .into_next_cache();
        let base_blocks = base.pooled_blocks().unwrap().clone();
        assert_eq!(base_blocks.shape(), vec![1, 3, spec.head_dim as i32]);
        let base_bits = raw_bits(&base_blocks);
        // Both successors start from the same cache and advance the same two
        // positions; 8 tokens complete a fourth block out of their own rows.
        let advance = |source: &[f32]| {
            let mut cache = base.clone();
            for position in 6..8 {
                let row = array_f32(
                    &source[position * spec.hidden..(position + 1) * spec.hidden],
                    &[1, 1, spec.hidden as i32],
                );
                let selection =
                    with_pooled_cache(true, || model.select(&row, &cache, position).unwrap());
                cache = selection.into_next_cache();
                let keys = cache.keys().unwrap();
                let n_complete = keys.shape()[1] / cfg.compress_ratio;
                let blocks = cache.pooled_blocks().unwrap();
                assert_eq!(blocks.shape(), vec![1, n_complete, cfg.head_dim]);
                if n_complete == 3 {
                    // No new block completed: the cached ones are reused as is.
                    assert_eq!(raw_bits(blocks), base_bits.as_slice());
                }
                let expected = pooled_block_keys(keys, n_complete, cfg, &model.k_norm_f32).unwrap();
                assert_eq!(raw_bits(blocks), raw_bits(&expected), "position {position}");
            }
            cache
        };
        let left = advance(&hidden);
        let right = advance(&other);
        let left_blocks = left.pooled_blocks().unwrap();
        let right_blocks = right.pooled_blocks().unwrap();
        assert_eq!(left_blocks.shape(), vec![1, 4, cfg.head_dim]);
        assert_eq!(right_blocks.shape(), vec![1, 4, cfg.head_dim]);
        // The shared parent never changes.
        assert_eq!(
            raw_bits(base.pooled_blocks().unwrap()),
            base_bits.as_slice()
        );
        // Both successors keep the parent's three blocks and differ only in the
        // block their own tokens completed.
        let prefix =
            |blocks: &MlxArray| slice(blocks, &[0, 0, 0], &[1, 3, cfg.head_dim], &[1, 1, 1], None);
        assert_eq!(raw_bits(&prefix(left_blocks)), base_bits.as_slice());
        assert_eq!(raw_bits(&prefix(right_blocks)), base_bits.as_slice());
        let fourth =
            |blocks: &MlxArray| slice(blocks, &[0, 3, 0], &[1, 4, cfg.head_dim], &[1, 1, 1], None);
        assert_ne!(
            raw_bits(&fourth(left_blocks)),
            raw_bits(&fourth(right_blocks)),
            "each fork pools its own block"
        );
    }

    /// One adversarial f32 sample from a cyclic table: NaN payloads,
    /// infinities, signed zeros, subnormals, units and large finite values.
    fn adversarial_value(index: usize) -> f32 {
        const TABLE: [f32; 12] = [
            f32::NAN,
            f32::from_bits(0x7fc0_0001), // quiet NaN, payload 1
            f32::from_bits(0xffc0_abcd), // negative quiet NaN, payload
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::from_bits(1),           // smallest subnormal
            f32::from_bits(0x0070_1234), // subnormal
            1.0,
            -1.0,
            1.0e30,
        ];
        TABLE[index % TABLE.len()]
    }

    /// `shape` adversarial rotary input of `dtype`, built through an f32
    /// widening (the same path the projections take into the model dtype).
    fn adversarial_rotary_input(shape: &[i32], dtype: MlxDtype, offset: usize) -> MlxArray {
        let len: i32 = shape.iter().product();
        let data: Vec<f32> = (0..len as usize)
            .map(|index| adversarial_value(index + offset))
            .collect();
        astype(
            &MlxArray::from_raw_data(
                data.as_ptr().cast(),
                std::mem::size_of_val(data.as_slice()),
                shape,
                MlxDtype::Float32,
            ),
            dtype,
            None,
        )
    }

    /// Full tables from the real rotary builder at `offset`, bypassing the
    /// cache so a case never reads another case's entry.
    fn real_tables(x: &MlxArray, rotary_dim: i32, offset: i32, seq_axis: i32) -> RotaryTables {
        let seq = x.shape()[seq_axis as usize];
        with_rope_cache(false, || {
            cached_rotary_trig(
                x,
                rotary_dim,
                10_000_000.0,
                f64::from(offset),
                1.0,
                seq,
                seq_axis,
            )
        })
    }

    /// Adversarial tables (NaN/inf/subnormal entries) in the production
    /// layout: the full table is the half table duplicated along the last
    /// axis (`concat(h, h)`), so the two halves are equal by construction.
    fn adversarial_tables(
        rotary_dim: i32,
        count: i32,
        ndim: usize,
        seq_axis: i32,
        dtype: MlxDtype,
    ) -> RotaryTables {
        let half = rotary_dim / 2;
        let mut trig = vec![1i32; ndim];
        trig[seq_axis as usize] = count;
        trig[ndim - 1] = half;
        let len = trig.iter().product::<i32>() as usize;
        let build_half = |seed: usize| {
            let data: Vec<f32> = (0..len)
                .map(|index| adversarial_value(index + seed))
                .collect();
            astype(
                &MlxArray::from_raw_data(
                    data.as_ptr().cast(),
                    std::mem::size_of_val(data.as_slice()),
                    &trig,
                    MlxDtype::Float32,
                ),
                dtype,
                None,
            )
        };
        let cos_h = build_half(0);
        let sin_h = build_half(len);
        RotaryTables {
            cos: concatenate(&[&cos_h, &cos_h], -1, None),
            sin: concatenate(&[&sin_h, &sin_h], -1, None),
            cos_h,
            sin_h,
        }
    }

    /// Raw bits of a rotary output: f32 natively, the half dtypes through the
    /// lossless f32 widening.
    fn rope_bits(array: &MlxArray) -> Vec<u32> {
        if array.dtype() == MlxDtype::Float32 {
            eval(&[array]);
            array
                .data_f32()
                .iter()
                .map(|value| value.to_bits())
                .collect()
        } else {
            raw_bits(array)
        }
    }

    /// The two outputs must carry the same dtype before any widening, then
    /// compare bit for bit (NaN payloads included).
    fn assert_rope_bits(actual: &MlxArray, expected: &MlxArray, label: &str) {
        assert_eq!(
            actual.dtype(),
            expected.dtype(),
            "{label}: dtype must match before the bit comparison"
        );
        assert_eq!(rope_bits(actual), rope_bits(expected), "{label}");
    }

    /// The original inline op sequence of [`apply_rotary_trig_shaped`], spelled
    /// out as the third anchor of the differential test.
    fn inline_rotary_shaped(
        x: &MlxArray,
        rotary_dim: i32,
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> MlxArray {
        let shape = x.shape();
        let ndim = shape.len();
        let head_dim = shape[ndim - 1];
        let rope_part = slice_last_dim(x, 0, rotary_dim, None);
        let rotated = add(
            &multiply(&rope_part, cos, None),
            &multiply(&rotate_half_last(&rope_part), sin, None),
            None,
        );
        if rotary_dim < head_dim {
            concatenate(
                &[&rotated, &slice_last_dim(x, rotary_dim, head_dim, None)],
                -1,
                None,
            )
        } else {
            rotated
        }
    }

    /// Both compiled rotary halves against forced eager and the original
    /// inline ops, bit for bit, on adversarial inputs: three dtypes, both
    /// layouts and sequence axes, partial and full rotary width, every fuse
    /// row count, real tables at several offsets and adversarial tables.
    #[test]
    fn qsa_rope_stretch_matches_eager_on_adversarial_inputs() {
        qwen4_exp_fused::reset_registry_for_test();
        reset_rope_cache_for_test();
        for (dtype_index, dtype) in [MlxDtype::Float32, MlxDtype::Bfloat16, MlxDtype::Float16]
            .into_iter()
            .enumerate()
        {
            for (head_dim, rotary_dim) in [(256i32, 64i32), (64, 64)] {
                for (seq_axis, heads) in [(2i32, 4i32), (1, 4)] {
                    for rows in 1..=8i32 {
                        let (shape, seq) = if seq_axis == 2 {
                            (vec![1, heads, rows, head_dim], rows)
                        } else {
                            (vec![1, rows, heads, head_dim], rows)
                        };
                        let x = adversarial_rotary_input(&shape, dtype, dtype_index);
                        let mut cases: Vec<(String, RotaryTables)> = [0, 5, 4093]
                            .into_iter()
                            .map(|offset| {
                                (
                                    format!("offset {offset}"),
                                    real_tables(&x, rotary_dim, offset, seq_axis),
                                )
                            })
                            .collect();
                        cases.push((
                            "adversarial tables".to_string(),
                            adversarial_tables(rotary_dim, seq, 4, seq_axis, dtype),
                        ));
                        for (case, tables) in &cases {
                            let label = format!(
                                "{dtype:?} head {head_dim} axis {seq_axis} rows {rows}: {case}"
                            );
                            let eager = {
                                let _eager = qwen4_exp_fused::force_for_test(Some(false));
                                apply_rotary_trig_split(&x, rotary_dim, tables, seq_axis)
                            };
                            let inline =
                                inline_rotary_shaped(&x, rotary_dim, &tables.cos, &tables.sin);
                            assert_rope_bits(
                                &eager,
                                &inline,
                                &format!("{label}: eager drifted from the inline ops"),
                            );
                            let lo_before = qwen4_exp_fused::applied_for_test(Stretch::QsaRopeLo);
                            let hi_before = qwen4_exp_fused::applied_for_test(Stretch::QsaRopeHi);
                            let fused = {
                                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                                apply_rotary_trig_split(&x, rotary_dim, tables, seq_axis)
                            };
                            assert!(
                                qwen4_exp_fused::applied_for_test(Stretch::QsaRopeLo) > lo_before,
                                "{label}: QsaRopeLo never applied"
                            );
                            assert!(
                                qwen4_exp_fused::applied_for_test(Stretch::QsaRopeHi) > hi_before,
                                "{label}: QsaRopeHi never applied"
                            );
                            assert_rope_bits(
                                &fused,
                                &eager,
                                &format!("{label}: fused diverged from eager"),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Rows past the fuse cap never apply either stretch and still get the
    /// eager result.
    #[test]
    fn qsa_rope_stretches_stay_eager_above_the_row_cap() {
        qwen4_exp_fused::reset_registry_for_test();
        reset_rope_cache_for_test();
        for rows in [9i32, 16] {
            let x = rotary_input(&[1, 4, rows, 256], MlxDtype::Bfloat16);
            let tables = real_tables(&x, 64, 0, 2);
            let eager = {
                let _eager = qwen4_exp_fused::force_for_test(Some(false));
                apply_rotary_trig_split(&x, 64, &tables, 2)
            };
            let lo_before = qwen4_exp_fused::applied_for_test(Stretch::QsaRopeLo);
            let hi_before = qwen4_exp_fused::applied_for_test(Stretch::QsaRopeHi);
            let fused = {
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                apply_rotary_trig_split(&x, 64, &tables, 2)
            };
            assert_eq!(
                qwen4_exp_fused::applied_for_test(Stretch::QsaRopeLo),
                lo_before,
                "rows {rows} must stay above the fuse cap (QsaRopeLo)"
            );
            assert_eq!(
                qwen4_exp_fused::applied_for_test(Stretch::QsaRopeHi),
                hi_before,
                "rows {rows} must stay above the fuse cap (QsaRopeHi)"
            );
            assert_rope_bits(
                &fused,
                &eager,
                &format!("rows {rows} must equal the eager result"),
            );
        }
    }

    /// The registry is never reset across dtypes: the key must select one
    /// closure per dtype, and a revisited dtype must reuse its entry without
    /// stale specialisation.
    #[test]
    fn qsa_rope_registry_key_survives_dtype_switches() {
        qwen4_exp_fused::reset_registry_for_test();
        reset_rope_cache_for_test();
        for dtype in [
            MlxDtype::Float32,
            MlxDtype::Bfloat16,
            MlxDtype::Float16,
            MlxDtype::Bfloat16,
        ] {
            let x = rotary_input(&[1, 4, 1, 256], dtype);
            let tables = real_tables(&x, 64, 7, 2);
            let eager = {
                let _eager = qwen4_exp_fused::force_for_test(Some(false));
                apply_rotary_trig_split(&x, 64, &tables, 2)
            };
            let lo_before = qwen4_exp_fused::applied_for_test(Stretch::QsaRopeLo);
            let hi_before = qwen4_exp_fused::applied_for_test(Stretch::QsaRopeHi);
            let fused = {
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                apply_rotary_trig_split(&x, 64, &tables, 2)
            };
            assert!(
                qwen4_exp_fused::applied_for_test(Stretch::QsaRopeLo) > lo_before
                    && qwen4_exp_fused::applied_for_test(Stretch::QsaRopeHi) > hi_before,
                "{dtype:?}: the fused path must run after a dtype switch"
            );
            assert_rope_bits(
                &fused,
                &eager,
                &format!("{dtype:?} must match eager after a dtype switch"),
            );
        }
    }

    /// An injected apply failure of either stretch tombstones exactly that
    /// stretch, falls back to the full eager sequence bit for bit, and the
    /// tombstone holds on the next call.
    #[test]
    fn qsa_rope_failure_injection_falls_back_to_eager() {
        qwen4_exp_fused::reset_registry_for_test();
        reset_rope_cache_for_test();
        let x = rotary_input(&[1, 4, 1, 256], MlxDtype::Bfloat16);
        let tables = real_tables(&x, 64, 3, 2);
        let eager = {
            let _eager = qwen4_exp_fused::force_for_test(Some(false));
            apply_rotary_trig_split(&x, 64, &tables, 2)
        };
        let eager_bits = rope_bits(&eager);
        for stretch in [Stretch::QsaRopeLo, Stretch::QsaRopeHi] {
            let other = if stretch == Stretch::QsaRopeLo {
                Stretch::QsaRopeHi
            } else {
                Stretch::QsaRopeLo
            };
            qwen4_exp_fused::reset_registry_for_test();
            let injected = {
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                qwen4_exp_fused::fail_next_apply_for_test(stretch);
                apply_rotary_trig_split(&x, 64, &tables, 2)
            };
            assert_eq!(
                rope_bits(&injected),
                eager_bits,
                "{}: the result with an injected failure diverged",
                stretch.name()
            );
            assert!(
                qwen4_exp_fused::tombstoned_for_test(stretch),
                "{}: the injected failure must tombstone",
                stretch.name()
            );
            assert!(
                !qwen4_exp_fused::tombstoned_for_test(other),
                "{}: the other stretch must not tombstone",
                stretch.name()
            );
            // The tombstone holds on the next call, still on the eager bits.
            let after = {
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                apply_rotary_trig_split(&x, 64, &tables, 2)
            };
            assert_eq!(
                rope_bits(&after),
                eager_bits,
                "{}: the post-tombstone result diverged",
                stretch.name()
            );
        }
        qwen4_exp_fused::reset_registry_for_test();
    }

    /// The rotary cache hit path returns the same full and half tables as the
    /// miss, and the halves equal the slices of the full tables - the exact
    /// assumption the split path makes.
    #[test]
    fn rotary_cache_hit_returns_equal_full_and_half_tables() {
        reset_rope_cache_for_test();
        let x = rotary_input(&[1, 4, 3, 256], MlxDtype::Bfloat16);
        let miss = with_rope_cache(true, || {
            cached_rotary_trig(&x, 64, 10_000_000.0, 5.0, 1.0, 3, 2)
        });
        let hit = with_rope_cache(true, || {
            cached_rotary_trig(&x, 64, 10_000_000.0, 5.0, 1.0, 3, 2)
        });
        assert_eq!(rope_cache_len_for_test(), 1, "the second call must hit");
        for (name, miss_table, hit_table) in [
            ("cos", &miss.cos, &hit.cos),
            ("sin", &miss.sin, &hit.sin),
            ("cos_h", &miss.cos_h, &hit.cos_h),
            ("sin_h", &miss.sin_h, &hit.sin_h),
        ] {
            assert_eq!(rope_bits(miss_table), rope_bits(hit_table), "{name}");
        }
        let half = 32i32;
        assert_eq!(
            rope_bits(&miss.cos_h),
            rope_bits(&slice_last_dim(&miss.cos, 0, half, None)),
            "cos_h must slice cos"
        );
        assert_eq!(
            rope_bits(&miss.sin_h),
            rope_bits(&slice_last_dim(&miss.sin, 0, half, None)),
            "sin_h must slice sin"
        );
    }

    fn cast_tensor(data: &[f32], shape: &[i32], dtype: MlxDtype) -> MlxArray {
        let array = array_f32(data, shape);
        if dtype == MlxDtype::Float32 {
            array
        } else {
            astype(&array, dtype, None)
        }
    }

    fn score_bits(queries: &MlxArray, blocks: &MlxArray) -> Vec<u32> {
        let scores = block_scores(queries, blocks);
        eval(&[&scores]);
        scores
            .data_f32()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    /// Host-only assembly. Production never concatenates score tiles.
    fn tiled_score_bits(queries: &MlxArray, blocks: &MlxArray, rows: &[i32]) -> Vec<u32> {
        let shape = queries.shape();
        let (batch, seq, heads, dim) = (shape[0], shape[1], shape[2], shape[3]);
        let n_blocks = blocks.shape()[1] as usize;
        let mut bits = vec![0u32; batch as usize * seq as usize * n_blocks];
        let mut cursor = 0i32;
        for &width in rows {
            let tile = slice(
                queries,
                &[0, cursor, 0, 0],
                &[batch, cursor + width, heads, dim],
                &[1, 1, 1, 1],
                None,
            );
            let scores = block_scores(&tile, blocks);
            eval(&[&scores]);
            let data = scores.data_f32();
            for b in 0..batch as usize {
                for local in 0..width as usize {
                    let src = (b * width as usize + local) * n_blocks;
                    let dst = (b * seq as usize + cursor as usize + local) * n_blocks;
                    for col in 0..n_blocks {
                        bits[dst + col] = data[src + col].to_bits();
                    }
                }
            }
            cursor += width;
        }
        assert_eq!(cursor, seq);
        bits
    }

    fn assert_same_bits(actual: &[u32], expected: &[u32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual, expected,
                "score bit mismatch at {index}: {:08x} vs {:08x}",
                actual, expected
            );
        }
    }

    fn planned(
        batch: i32,
        seq: i32,
        heads: i32,
        dim: i32,
        n_blocks: i32,
        dtype: MlxDtype,
    ) -> Option<ScoreTilePlan> {
        with_tiled_scores(true, || {
            ScoreTilePlan::try_new(batch, seq, heads, dim, n_blocks, dtype)
        })
    }

    fn wave_pair(
        batch: i32,
        seq: i32,
        heads: i32,
        dim: i32,
        n_blocks: i32,
        dtype: MlxDtype,
    ) -> (MlxArray, MlxArray) {
        let queries = wave((batch * seq * heads * dim) as usize, 0.17, 0.55, 0.04);
        let mut blocks = wave((batch * n_blocks * dim) as usize, 0.11, 0.35, -0.03);
        if n_blocks >= 2 {
            let dim_u = dim as usize;
            let n_u = n_blocks as usize;
            for b in 0..batch as usize {
                let base = b * n_u * dim_u;
                blocks.copy_within(base..base + dim_u, base + dim_u);
                blocks[base + dim_u] += 1.0e-4;
            }
        }
        (
            cast_tensor(&queries, &[batch, seq, heads, dim], dtype),
            cast_tensor(&blocks, &[batch, n_blocks, dim], dtype),
        )
    }

    struct Geom {
        heads: usize,
        dim: usize,
        ratio: usize,
        budget: usize,
        hidden: usize,
    }

    fn geom_indexer(spec: &Geom) -> QsaIndexer {
        geom_indexer_dtype(spec, MlxDtype::Float32)
    }

    fn geom_indexer_dtype(spec: &Geom, dtype: MlxDtype) -> QsaIndexer {
        let config = QsaConfig::new(
            spec.heads,
            1,
            spec.dim,
            2,
            spec.ratio,
            spec.budget,
            spec.hidden,
            EPS,
            BASE,
        )
        .unwrap();
        let out = (spec.heads + 1) * spec.dim;
        QsaIndexer::new(
            config,
            QsaIndexerWeights {
                qk_proj: QuantizedWeight::new(
                    cast_tensor(
                        &wave(out * spec.hidden, 0.37, 0.45, 0.08),
                        &[config.proj_out(), spec.hidden as i32],
                        dtype,
                    ),
                    None,
                    None,
                ),
                q_norm: cast_tensor(&wave(spec.dim, 0.21, 0.3, 0.9), &[spec.dim as i32], dtype),
                k_norm: cast_tensor(&wave(spec.dim, 0.17, 0.25, 1.05), &[spec.dim as i32], dtype),
            },
        )
        .unwrap()
    }

    fn hidden_tensor(spec: &Geom, batch: usize, tokens: usize, salt: f32) -> MlxArray {
        let mut data = wave(batch * tokens * spec.hidden, 0.41 + salt, 0.9, 0.12);
        for (index, value) in data.iter_mut().enumerate() {
            *value *= 0.6 + 0.15 * (index % spec.hidden) as f32;
        }
        array_f32(&data, &[batch as i32, tokens as i32, spec.hidden as i32])
    }

    fn host_selection(
        indexer: &QsaIndexer,
        hidden: &MlxArray,
        cache: &QsaIndexKeyCache,
        offset: usize,
        tiled: bool,
        limits: Option<(i32, usize)>,
    ) -> Vec<Vec<Vec<i32>>> {
        let run = || {
            indexer
                .select_with_mode(hidden, cache, offset, project, false)
                .unwrap()
                .tokens
                .clone()
        };
        match limits {
            Some((rows, budget)) => {
                with_tile_limits(rows, budget, || with_tiled_scores(tiled, run))
            }
            None => with_tiled_scores(tiled, run),
        }
    }

    #[test]
    fn qsa_tiled_score_bits_match_monolithic() {
        let arch = gpu_device_architecture().unwrap();
        let version = mlx_sys::runtime_version().unwrap();
        if !QSA_TILED_SCORE_ARCHES.contains(&arch.as_str()) || version != QSA_TILED_MLX_VERSION {
            eprintln!("skipping: {arch} / MLX {version} is not a proven tiled-score target");
            return;
        }

        let cases = [
            (1, 1025, 4, 128, 129, MlxDtype::Float32, None),
            (
                2,
                5,
                4,
                128,
                129,
                MlxDtype::Float32,
                Some((2, QSA_TILE_WORKSPACE_BUDGET)),
            ),
            (
                1,
                5,
                4,
                128,
                129,
                MlxDtype::Float16,
                Some((2, QSA_TILE_WORKSPACE_BUDGET)),
            ),
            (
                1,
                5,
                4,
                128,
                129,
                MlxDtype::Bfloat16,
                Some((2, QSA_TILE_WORKSPACE_BUDGET)),
            ),
            (
                1,
                5,
                1,
                64,
                7,
                MlxDtype::Float32,
                Some((2, QSA_TILE_WORKSPACE_BUDGET)),
            ),
            (
                1,
                4,
                2,
                32,
                1,
                MlxDtype::Float32,
                Some((1, QSA_TILE_WORKSPACE_BUDGET)),
            ),
            (
                1,
                4,
                2,
                4,
                3,
                MlxDtype::Bfloat16,
                Some((1, QSA_TILE_WORKSPACE_BUDGET)),
            ),
        ];
        for (batch, seq, heads, dim, n_blocks, dtype, limits) in cases {
            let (queries, blocks) = wave_pair(batch, seq, heads, dim, n_blocks, dtype);
            let plan = match limits {
                Some((rows, budget)) => with_tile_limits(rows, budget, || {
                    planned(batch, seq, heads, dim, n_blocks, dtype)
                }),
                None => planned(batch, seq, heads, dim, n_blocks, dtype),
            }
            .unwrap_or_else(|| {
                panic!(
                    "proven shape {batch}x{seq} h{heads} d{dim} n{n_blocks} {dtype:?} did not tile"
                )
            });
            assert!(
                plan.query_rows.len() > 1,
                "proven shape must issue more than one tile: {:?}",
                plan.query_rows
            );
            assert_eq!(plan.query_rows.iter().sum::<i32>(), seq);
            let budget = limits
                .map(|(_, budget)| budget)
                .unwrap_or(QSA_TILE_WORKSPACE_BUDGET);
            assert!(plan.workspace_bytes <= budget);
            assert!(plan.dispatch_identity != 0);
            let again = match limits {
                Some((rows, tile_budget)) => with_tile_limits(rows, tile_budget, || {
                    planned(batch, seq, heads, dim, n_blocks, dtype)
                }),
                None => planned(batch, seq, heads, dim, n_blocks, dtype),
            }
            .unwrap();
            assert_eq!(again.dispatch_identity, plan.dispatch_identity);
            let monolithic = score_bits(&queries, &blocks);
            let tiled = tiled_score_bits(&queries, &blocks, &plan.query_rows);
            assert_same_bits(&tiled, &monolithic);
        }

        assert!(
            with_tiled_scores(false, || {
                ScoreTilePlan::try_new(1, 1025, 4, 128, 129, MlxDtype::Float32)
            })
            .is_none()
        );
        assert!(planned(1, 4, 4, 128, 128, MlxDtype::Float32).is_none());
        assert!(
            with_tile_limits(2, QSA_TILE_WORKSPACE_BUDGET, || {
                planned(1, 8, 4, 128, 128, MlxDtype::Float32)
            })
            .is_none(),
            "K=128 with fewer than 129 blocks stays monolithic"
        );
        assert!(
            planned(1, 1025, 8, 128, 129, MlxDtype::Float32).is_none(),
            "an unproven head count stays monolithic"
        );
        assert!(
            planned(1, 1025, 4, 64, 129, MlxDtype::Float32).is_none(),
            "dim 64 is not a proven 4-head GEMM"
        );
        assert!(planned(3, 6, 2, 64, 4, MlxDtype::Float32).is_none());
        assert!(planned(1, 6, 2, 96, 4, MlxDtype::Float32).is_none());
        assert!(
            with_tile_limits(1, QSA_TILE_WORKSPACE_BUDGET, || {
                planned(1, 4, 1, 64, 4, MlxDtype::Float32)
            })
            .is_none(),
            "a one-row GEMV is not a proven tile"
        );
        let wide = planned(1, 8, 4, 64, 4, MlxDtype::Float32);
        assert!(
            wide.is_none(),
            "one tile of the whole chunk stays on the monolithic GEMM"
        );
        assert_production_tiled_score_sweep();
    }

    #[test]
    fn one_row_tail_folds_at_the_smallest_proven_width() {
        let mut ordered = QSA_PROVEN_TILE_WIDTHS;
        ordered.sort_unstable();
        assert_eq!(ordered[0], QSA_SMALLEST_PROVEN_TILE_WIDTH);
        let fits = |_| true;
        let width = QSA_SMALLEST_PROVEN_TILE_WIDTH;
        let folded = split_query_tiles(width * 2 + 1, width, 4, fits).expect("proven width folds");
        assert_eq!(folded, vec![width, width + 1]);
        let narrow = width - 1;
        let unfolded = split_query_tiles(narrow * 2 + 1, narrow, 4, fits).expect("narrow tiles");
        assert_eq!(unfolded.last().copied(), Some(1));
        let gemv = split_query_tiles(narrow * 2 + 1, narrow, 1, fits).expect("one head folds");
        assert_ne!(gemv.last().copied(), Some(1));
    }

    /// Production matrix: 4 heads, dim 128, f32 GEMM after the cast. Every
    /// listed sequence and block count appears, every width the planner can
    /// choose is bit-compared, and the largest shape runs once.
    fn assert_production_tiled_score_sweep() {
        let shapes = [
            (1025, 129),
            (1025, 512),
            (1025, 2048),
            (1025, 8193),
            (1025, 10000),
            (2048, 129),
            (2048, 8193),
            (3000, 129),
            (4096, 2048),
            (5000, 512),
            (8192, 129),
            (8192, 10000),
        ];
        let previous_limit = mlx_sys::set_cache_limit(0);
        struct RestoreCache(usize);
        impl Drop for RestoreCache {
            fn drop(&mut self) {
                mlx_sys::set_cache_limit(self.0);
            }
        }
        let _restore_cache = RestoreCache(previous_limit);
        let mut saw_width = [false; 3];
        for (seq, n_blocks) in shapes {
            let (queries, blocks) = wave_pair(1, seq, 4, 128, n_blocks, MlxDtype::Float32);
            let plan = planned(1, seq, 4, 128, n_blocks, MlxDtype::Float32)
                .expect("production shape did not tile");
            assert_eq!(plan.query_rows.iter().sum::<i32>(), seq);
            assert!(plan.workspace_bytes <= QSA_TILE_WORKSPACE_BUDGET);
            assert_proven_tile_rows(&plan.query_rows);
            for (index, &width) in QSA_PROVEN_TILE_WIDTHS.iter().enumerate() {
                if plan.query_rows.contains(&width) {
                    saw_width[index] = true;
                }
            }
            let monolithic = score_bits(&queries, &blocks);
            let tiled = tiled_score_bits(&queries, &blocks, &plan.query_rows);
            assert_same_bits(&tiled, &monolithic);
            let offset = 7usize;
            assert_ordered_indices_match(&monolithic, &tiled, seq, n_blocks, offset);
        }
        assert!(
            saw_width.iter().all(|seen| *seen),
            "planner widths {:?}, seen {saw_width:?}",
            QSA_PROVEN_TILE_WIDTHS
        );
    }

    fn assert_proven_tile_rows(rows: &[i32]) {
        assert!(rows.len() >= 2, "production plan needs two tiles: {rows:?}");
        for (index, &width) in rows.iter().enumerate() {
            let tail = index + 1 == rows.len();
            if QSA_PROVEN_TILE_WIDTHS.contains(&width) {
                continue;
            }
            assert!(
                tail && width >= 2,
                "unproven tile width {width} in {rows:?}"
            );
        }
    }

    fn assert_ordered_indices_match(
        monolithic: &[u32],
        tiled: &[u32],
        seq: i32,
        n_blocks: i32,
        offset: usize,
    ) {
        let cfg = QsaConfig::new(4, 1, 128, 2, 4, 16, 8, EPS, BASE).unwrap();
        let n_u = n_blocks as usize;
        let seq_u = seq as usize;
        let queries: Vec<usize> = if seq_u <= 1025 {
            (0..seq_u).collect()
        } else {
            [0, 1, seq_u / 2, seq_u.saturating_sub(1)]
                .into_iter()
                .filter(|query| *query < seq_u)
                .collect()
        };
        let mut compared = 0usize;
        for query in queries {
            let visible = offset + query + 1;
            if visible / 4 > n_u {
                continue;
            }
            let row = query * n_u;
            let host_scores: Vec<f32> = monolithic[row..row + n_u]
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect();
            let tile_scores: Vec<f32> = tiled[row..row + n_u]
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect();
            let host =
                tokens_from_score_data(cfg, 1, 1, query, offset, n_blocks, &host_scores).unwrap();
            let other =
                tokens_from_score_data(cfg, 1, 1, query, offset, n_blocks, &tile_scores).unwrap();
            assert_eq!(other, host, "offset={offset} query={query} seq={seq}");
            compared += 1;
        }
        assert!(compared > 0, "offset={offset} seq={seq} compared nothing");
    }

    #[test]
    fn qsa_tiled_indices_match_including_ties() {
        let spec = Geom {
            heads: 4,
            dim: 64,
            ratio: 4,
            budget: 8,
            hidden: 8,
        };
        let indexer = geom_indexer(&spec);
        let cache = QsaIndexKeyCache::empty();
        let seq = 12;
        let hidden = hidden_tensor(&spec, 2, seq, 0.0);
        let limits = Some((2, QSA_TILE_WORKSPACE_BUDGET));
        let off = host_selection(&indexer, &hidden, &cache, 0, false, limits);
        let on = host_selection(&indexer, &hidden, &cache, 0, true, limits);
        assert_eq!(on, off);
        assert_eq!(on.len(), 2);
        assert_eq!(on[0].len(), seq);
        assert_eq!(on[1].len(), seq);
        let kept = spec.budget / spec.ratio;
        let mut saw_boundary = false;
        for queries in &on {
            for (query, tokens) in queries.iter().enumerate() {
                let complete = (query + 1) / spec.ratio;
                if complete > kept {
                    saw_boundary = true;
                    let blocks = tokens.len() / spec.ratio;
                    assert!(blocks <= kept + 1, "kept more than top-k plus the tail");
                }
            }
        }
        assert!(saw_boundary, "the chunk must cross the top-k boundary");

        let tied = array_f32(
            &vec![0.25f32; 2 * seq * spec.hidden],
            &[2, seq as i32, spec.hidden as i32],
        );
        let tied_off = host_selection(&indexer, &tied, &cache, 0, false, limits);
        let tied_on = host_selection(&indexer, &tied, &cache, 0, true, limits);
        assert_eq!(tied_on, tied_off);
        let (tie_batch, tie_seq, tie_heads, tie_dim, tie_blocks) = (1, 4, 4, 64, 4);
        let mut tie_keys = wave((tie_blocks * tie_dim) as usize, 0.11, 0.35, -0.03);
        let tie_dim_u = tie_dim as usize;
        for block in 1..tie_blocks as usize {
            tie_keys.copy_within(0..tie_dim_u, block * tie_dim_u);
        }
        let tie_queries = cast_tensor(
            &wave((tie_seq * tie_heads * tie_dim) as usize, 0.17, 0.55, 0.04),
            &[tie_batch, tie_seq, tie_heads, tie_dim],
            MlxDtype::Float32,
        );
        let tie_block_array = cast_tensor(
            &tie_keys,
            &[tie_batch, tie_blocks, tie_dim],
            MlxDtype::Float32,
        );
        let tie_rows = [2, 2];
        let tie_monolithic = score_bits(&tie_queries, &tie_block_array);
        let tie_tiled = tiled_score_bits(&tie_queries, &tie_block_array, &tie_rows);
        assert_same_bits(&tie_tiled, &tie_monolithic);
        let tie_values: Vec<f32> = tie_monolithic.iter().copied().map(f32::from_bits).collect();
        for query in 0..tie_seq as usize {
            let row = query * tie_blocks as usize;
            for block in 1..tie_blocks as usize {
                assert_eq!(
                    tie_values[row].to_bits(),
                    tie_values[row + block].to_bits(),
                    "identical blocks must tie"
                );
            }
            let order = select_blocks(
                &tie_values[row..row + tie_blocks as usize],
                tie_blocks as usize,
                0,
                query,
            )
            .unwrap();
            assert_eq!(
                order,
                (0..tie_blocks as usize).collect::<Vec<_>>(),
                "a tie must keep the lower block first"
            );
        }
        let tie_cfg = QsaConfig::new(4, 1, 64, 2, 4, 8, 8, EPS, BASE).unwrap();
        let tie_tokens =
            tokens_from_score_data(tie_cfg, tie_batch, tie_seq, 0, 0, tie_blocks, &tie_values)
                .unwrap();
        let tiled_values: Vec<f32> = tie_tiled.iter().copied().map(f32::from_bits).collect();
        let tiled_tokens =
            tokens_from_score_data(tie_cfg, tie_batch, tie_seq, 0, 0, tie_blocks, &tiled_values)
                .unwrap();
        assert_eq!(tiled_tokens, tie_tokens);

        let bf = Geom {
            heads: 4,
            dim: 128,
            ratio: 4,
            budget: 8,
            hidden: 8,
        };
        let past_blocks = 129;
        let past_tokens = past_blocks * bf.ratio;
        let indexer = geom_indexer_dtype(&bf, MlxDtype::Bfloat16);
        let past = cast_tensor(
            &wave(past_tokens * bf.dim, 0.19, 0.2, 0.1),
            &[1, past_tokens as i32, bf.dim as i32],
            MlxDtype::Bfloat16,
        );
        let cache = QsaIndexKeyCache::from_serialized(Some(past));
        let hidden = cast_tensor(
            &wave(5 * bf.hidden, 0.23, 0.4, 0.2),
            &[1, 5, bf.hidden as i32],
            MlxDtype::Bfloat16,
        );
        let off = host_selection(&indexer, &hidden, &cache, past_tokens, false, limits);
        let on = host_selection(&indexer, &hidden, &cache, past_tokens, true, limits);
        assert_eq!(on, off);

        let short = Geom {
            heads: 4,
            dim: 64,
            ratio: 4,
            budget: 4,
            hidden: 8,
        };
        let indexer = geom_indexer(&short);
        let hidden = hidden_tensor(&short, 1, 4, 0.02);
        let cache = QsaIndexKeyCache::empty();
        let host = with_tile_limits(2, QSA_TILE_WORKSPACE_BUDGET, || {
            with_tiled_scores(true, || {
                indexer
                    .select_with_mode(&hidden, &cache, 0, project, false)
                    .unwrap()
            })
        });
        let device = with_device_select_multi(true, || indexer.select(&hidden, &cache, 0).unwrap());
        assert!(device.device_tokens().is_some());
        for query in 0..4 {
            assert_eq!(
                host.tokens_for_query(0, query),
                device.tokens_for_query(0, query),
                "device oracle query {query}"
            );
        }
    }

    #[test]
    fn qsa_tiled_preserves_signed_zero_order() {
        let (batch, seq, heads, dim, n_blocks) = (1, 2, 2, 4, 2);
        let queries = cast_tensor(
            &vec![0.0f32; (batch * seq * heads * dim) as usize],
            &[batch, seq, heads, dim],
            MlxDtype::Float32,
        );
        let mut keys = vec![0.0f32; (n_blocks * dim) as usize];
        for value in keys.iter_mut().skip(dim as usize) {
            *value = -0.0;
        }
        let blocks = cast_tensor(&keys, &[batch, n_blocks, dim], MlxDtype::Float32);
        let plan = with_tile_limits(1, QSA_TILE_WORKSPACE_BUDGET, || {
            planned(batch, seq, heads, dim, n_blocks, MlxDtype::Float32)
        })
        .unwrap();
        assert_eq!(plan.query_rows, vec![1, 1]);
        let monolithic = score_bits(&queries, &blocks);
        let tiled = tiled_score_bits(&queries, &blocks, &plan.query_rows);
        assert_same_bits(&tiled, &monolithic);
        let scores: Vec<f32> = monolithic
            .chunks(n_blocks as usize)
            .flat_map(|row| {
                assert_eq!(row.len(), 2);
                assert_eq!(
                    f32::from_bits(row[0]).partial_cmp(&f32::from_bits(row[1])),
                    Some(std::cmp::Ordering::Equal),
                    "signed zeros must tie"
                );
                row.iter().copied().map(f32::from_bits)
            })
            .collect();
        let cfg = QsaConfig::new(heads as usize, 1, dim as usize, 2, 2, 2, 4, EPS, BASE).unwrap();
        let tokens = tokens_from_score_data(cfg, batch, seq, 0, 0, n_blocks, &scores).unwrap();
        for (query, chosen) in tokens[0].iter().enumerate() {
            assert_eq!(
                chosen.first().copied(),
                Some(0),
                "query {query} must break the ±0 tie toward block 0"
            );
        }
        assert_eq!(select_blocks(&[0.0, -0.0], 2, 0, 0).unwrap(), vec![0, 1]);
        assert_eq!(select_blocks(&[-0.0, 0.0], 2, 0, 0).unwrap(), vec![0, 1]);
    }

    #[test]
    fn qsa_tiled_masks_future_nonfinite() {
        let (batch, seq, heads, dim, n_blocks) = (1, 4, 4, 128, 129);
        let queries = cast_tensor(
            &wave((seq * heads * dim) as usize, 0.17, 0.55, 0.04),
            &[batch, seq, heads, dim],
            MlxDtype::Float32,
        );
        let clean_keys = wave((n_blocks * dim) as usize, 0.11, 0.35, -0.03);
        let mut poisoned = clean_keys.clone();
        let dim_u = dim as usize;
        for value in &mut poisoned[(n_blocks as usize - 1) * dim_u..] {
            *value = f32::NAN;
        }
        let clean = cast_tensor(&clean_keys, &[batch, n_blocks, dim], MlxDtype::Float32);
        let blocks = cast_tensor(&poisoned, &[batch, n_blocks, dim], MlxDtype::Float32);
        let rows = [2, 2];
        let monolithic = score_bits(&queries, &clean);
        let poisoned_full = score_bits(&queries, &blocks);
        let poisoned_tiled = tiled_score_bits(&queries, &blocks, &rows);
        let visible = (n_blocks - 1) as usize;
        for query in 0..seq as usize {
            let start = query * n_blocks as usize;
            assert_same_bits(
                &poisoned_tiled[start..start + visible],
                &poisoned_full[start..start + visible],
            );
            assert_same_bits(
                &poisoned_full[start..start + visible],
                &monolithic[start..start + visible],
            );
            for bits in &poisoned_full[start..start + visible] {
                assert!(f32::from_bits(*bits).is_finite(), "visible score poisoned");
            }
            assert!(!f32::from_bits(poisoned_full[start + visible]).is_finite());
            assert!(!f32::from_bits(poisoned_tiled[start + visible]).is_finite());
        }
        let cfg = QsaConfig::new(4, 1, 128, 2, 4, 16, 8, EPS, BASE).unwrap();
        let offset = 512usize;
        let full: Vec<f32> = poisoned_full.iter().copied().map(f32::from_bits).collect();
        let tiled: Vec<f32> = poisoned_tiled.iter().copied().map(f32::from_bits).collect();
        for query in 0..3 {
            let complete = (offset + query + 1) / 4;
            assert!(complete < n_blocks as usize);
            let row = query * n_blocks as usize;
            let host =
                select_blocks(&full[row..row + complete], cfg.block_topk(), 0, query).unwrap();
            let tile =
                select_blocks(&tiled[row..row + complete], cfg.block_topk(), 0, query).unwrap();
            assert_eq!(tile, host, "future nonfinite changed query {query}");
        }
        let last = 3 * n_blocks as usize;
        let seen = (offset + 4) / 4;
        assert_eq!(seen, n_blocks as usize);
        assert!(matches!(
            select_blocks(&full[last..last + seen], cfg.block_topk(), 0, 3),
            Err(QsaError::NonFiniteScore { query: 3, .. })
        ));
        assert!(matches!(
            select_blocks(&tiled[last..last + seen], cfg.block_topk(), 0, 3),
            Err(QsaError::NonFiniteScore { query: 3, .. })
        ));

        let spec = Geom {
            heads: 4,
            dim: 64,
            ratio: 4,
            budget: 16,
            hidden: 8,
        };
        let indexer = geom_indexer(&spec);
        let cache = QsaIndexKeyCache::empty();
        let limits = Some((2, QSA_TILE_WORKSPACE_BUDGET));
        let finite = hidden_tensor(&spec, 1, 5, 0.0);
        let off = host_selection(&indexer, &finite, &cache, 0, false, limits);
        let on = host_selection(&indexer, &finite, &cache, 0, true, limits);
        assert_eq!(on, off);
        let mut data = wave(6 * spec.hidden, 0.41, 0.9, 0.12);
        for value in &mut data[5 * spec.hidden..] {
            *value = f32::NAN;
        }
        let hidden = array_f32(&data, &[1, 6, spec.hidden as i32]);
        for tiled in [false, true] {
            let error = with_tile_limits(2, QSA_TILE_WORKSPACE_BUDGET, || {
                with_tiled_scores(tiled, || {
                    indexer.select_with_mode(&hidden, &cache, 0, project, false)
                })
            });
            match error {
                Err(QsaError::NonFiniteScore { query: 5, .. }) => {}
                Err(error) => panic!("tiled={tiled} reported {error}"),
                Ok(_) => panic!("tiled={tiled} accepted a nonfinite query"),
            }
        }
    }

    #[test]
    fn qsa_tiled_partial_tail_matches() {
        let spec = Geom {
            heads: 4,
            dim: 64,
            ratio: 4,
            budget: 16,
            hidden: 8,
        };
        let indexer = geom_indexer(&spec);
        let limits = Some((2, QSA_TILE_WORKSPACE_BUDGET));
        let first = hidden_tensor(&spec, 1, 5, 0.0);
        let staged = with_tiled_scores(false, || {
            indexer
                .select_with_mode(&first, &QsaIndexKeyCache::empty(), 0, project, false)
                .unwrap()
                .into_next_cache()
        });
        let second = hidden_tensor(&spec, 1, 5, 0.03);
        let off = host_selection(&indexer, &second, &staged, 5, false, limits);
        let on = host_selection(&indexer, &second, &staged, 5, true, limits);
        assert_eq!(on, off);
        let tail = &on[0][4];
        assert!(
            tail.contains(&8) && tail.contains(&9),
            "partial tail missing: {tail:?}"
        );
        assert!(tail.iter().all(|token| *token <= 9));
    }

    #[test]
    fn qsa_tiled_failure_preserves_cache() {
        let spec = Geom {
            heads: 4,
            dim: 64,
            ratio: 4,
            budget: 16,
            hidden: 8,
        };
        let indexer = geom_indexer(&spec);
        let first = hidden_tensor(&spec, 1, 4, 0.0);
        let cache = with_tiled_scores(false, || {
            indexer
                .select_with_mode(&first, &QsaIndexKeyCache::empty(), 0, project, false)
                .unwrap()
                .into_next_cache()
        });
        let before = {
            let keys = cache.keys().unwrap();
            eval(&[keys]);
            keys.data_f32()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        let mut data = wave(8 * spec.hidden, 0.41, 0.9, 0.12);
        for value in &mut data[4 * spec.hidden..] {
            *value = f32::NAN;
        }
        let hidden = array_f32(&data, &[1, 8, spec.hidden as i32]);
        let error = with_tile_limits(2, QSA_TILE_WORKSPACE_BUDGET, || {
            with_tiled_scores(true, || {
                indexer.select_with_mode(&hidden, &cache, 4, project, false)
            })
        });
        match error {
            Err(QsaError::NonFiniteScore { query, .. }) => {
                assert!(
                    query >= 2,
                    "the failure must land in a later tile, query {query}"
                );
            }
            Err(error) => panic!("expected a later-tile nonfinite score, got {error}"),
            Ok(_) => panic!("expected a later-tile nonfinite score"),
        }
        let after = {
            let keys = cache.keys().unwrap();
            eval(&[keys]);
            keys.data_f32()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(before, after);
        assert!(cache.pooled_blocks().is_some());
    }

    #[test]
    fn prefill_device_select_defers_to_tiled_scores_when_monolithic_scores_do_not_fit() {
        let spec = Geom {
            heads: 2,
            dim: 4,
            ratio: 2,
            budget: 4,
            hidden: 8,
        };
        let indexer = geom_indexer(&spec);
        let past_tokens = 8usize;
        let past = array_f32(
            &vec![0.05f32; past_tokens * spec.dim],
            &[1, past_tokens as i32, spec.dim as i32],
        );
        let cache = QsaIndexKeyCache::from_serialized(Some(past));
        let seq = 8usize;
        let hidden = hidden_tensor(&spec, 1, seq, 0.0);
        let limit = 128usize;
        let run = |device_select: bool, tiled: bool| {
            with_tiled_score_runtime(true, || {
                with_tile_limits(2, QSA_TILE_WORKSPACE_BUDGET, || {
                    with_score_workspace_limit(limit, || {
                        with_prefill_device_select(true, || {
                            with_tiled_scores(tiled, || {
                                indexer.select_with_mode(
                                    &hidden,
                                    &cache,
                                    past_tokens,
                                    project,
                                    device_select,
                                )
                            })
                        })
                    })
                })
            })
        };
        match run(true, false) {
            Err(QsaError::InvalidScalar(message)) => {
                assert!(message.contains("score workspace"), "{message}");
            }
            Err(error) => panic!("untiled device selection must reject this chunk, got {error}"),
            Ok(_) => panic!("untiled device selection must reject this chunk"),
        }
        let device_on = run(true, true).expect("tiled scores succeed with device selection on");
        let device_off = run(false, true).expect("tiled host path");
        assert!(
            device_on.device_tokens().is_none(),
            "an over-budget chunk uses the tiled host path"
        );
        for query in 0..seq {
            assert_eq!(
                device_on.tokens_for_query(0, query),
                device_off.tokens_for_query(0, query),
                "query {query}"
            );
        }
        let wide = with_tiled_score_runtime(true, || {
            with_tile_limits(2, QSA_TILE_WORKSPACE_BUDGET, || {
                with_score_workspace_limit(QSA_SCORE_WORKSPACE_LIMIT, || {
                    with_prefill_device_select(true, || {
                        with_tiled_scores(true, || {
                            indexer
                                .select_with_mode(&hidden, &cache, past_tokens, project, true)
                                .unwrap()
                        })
                    })
                })
            })
        });
        let host = with_score_workspace_limit(QSA_SCORE_WORKSPACE_LIMIT, || {
            with_tiled_scores(false, || {
                indexer
                    .select_with_mode(&hidden, &cache, past_tokens, project, false)
                    .unwrap()
            })
        });
        assert!(
            wide.device_tokens().is_some(),
            "device selection runs when the monolithic scores fit"
        );
        for query in 0..seq {
            assert_eq!(
                wide.tokens_for_query(0, query),
                host.tokens_for_query(0, query),
                "fitted device query {query}"
            );
        }
    }

    #[test]
    fn qsa_tiled_peak_workspace_is_bounded() {
        with_isolated_peak_memory(|| {
            let seq = 2048i32;
            let heads = 4i32;
            let dim = 128i32;
            let n_blocks = 8193i32;
            let monolithic = full_score_workspace_bytes(1, seq, heads, n_blocks as usize).unwrap();
            assert!(monolithic > QSA_SCORE_WORKSPACE_LIMIT);
            let plan = planned(1, seq, heads, dim, n_blocks, MlxDtype::Float32).unwrap();
            assert!(plan.query_rows.len() > 1);
            assert!(plan.workspace_bytes <= QSA_TILE_WORKSPACE_BUDGET);
            assert_eq!(plan.query_rows.iter().sum::<i32>(), seq);
            for &rows in &plan.query_rows {
                let bytes = full_score_workspace_bytes(1, rows, heads, n_blocks as usize).unwrap();
                assert!(bytes <= QSA_SCORE_WORKSPACE_LIMIT);
            }
            let spec = Geom {
                heads: heads as usize,
                dim: dim as usize,
                ratio: 4,
                budget: 4,
                hidden: 4,
            };
            let indexer = geom_indexer(&spec);
            let past_tokens = n_blocks as usize * spec.ratio - seq as usize;
            let past = array_f32(
                &vec![0.05f32; past_tokens * spec.dim],
                &[1, past_tokens as i32, spec.dim as i32],
            );
            let cache = QsaIndexKeyCache::from_serialized(Some(past));
            let hidden = hidden_tensor(&spec, 1, seq as usize, 0.0);
            let rejected = with_tiled_scores(false, || {
                indexer.select_with_mode(&hidden, &cache, past_tokens, project, false)
            });
            match rejected {
                Err(QsaError::InvalidScalar(message)) => {
                    assert!(message.contains("score workspace"), "{message}");
                }
                Err(error) => panic!("monolithic path must reject this chunk, got {error}"),
                Ok(_) => panic!("monolithic path must reject this chunk"),
            }
            let previous_limit = mlx_sys::set_cache_limit(0);
            let active_before = mlx_sys::mempressure::device_active_bytes().unwrap_or(0);
            mlx_sys::reset_peak_memory();
            let peak_after_reset = mlx_sys::get_peak_memory() as u64;
            let selected = with_tiled_scores(true, || {
                indexer
                    .select_with_mode(&hidden, &cache, past_tokens, project, false)
                    .unwrap()
            });
            let peak = mlx_sys::get_peak_memory() as u64;
            mlx_sys::set_cache_limit(previous_limit);
            let growth = peak.saturating_sub(peak_after_reset);
            let over_active = peak.saturating_sub(active_before);
            let observed = match (growth > 0, over_active > 0) {
                (true, true) => growth.min(over_active),
                (true, false) => growth,
                (false, true) => over_active,
                (false, false) => 0,
            };
            assert!(observed > 1024 * 1024, "tiled select allocated nothing");
            assert!(
                observed < monolithic as u64,
                "tiled peak growth {observed} reached the monolithic score workspace {monolithic}"
            );
            assert_eq!(selected.tokens.len(), 1);
            assert_eq!(selected.tokens[0].len(), seq as usize);
            assert!(selected.next_cache().keys().is_some());
        });
    }
}
