//! Request-owned QSA block selection for Flash Next (`qwen4_exp`).
//!
//! Shared indexer weights project index queries and raw keys. Each request
//! owns the raw key history. Completed causal blocks are mean-pooled in
//! float32, restored to the input dtype, RMS-normalized, and rotated at the
//! block start. Queries are RMS-normalized and rotated at their absolute
//! positions. The score of a block is the sum over query heads of
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

#![allow(dead_code)]

#[cfg(test)]
use mlx_sys::eval;
use mlx_sys::{
    MlxArray, MlxDtype, MlxQuantizationMode, arange, astype, broadcast_to, concatenate, contiguous,
    divide, matmul, maximum, quantized_matmul_with_mode, reshape, rms_norm, rope, rope_dynamic,
    slice, slice_last_dim, sum_axis, transpose,
};
use thiserror::Error;

use crate::weights::QuantizedWeight;

type Result<T> = std::result::Result<T, QsaError>;

#[derive(Debug, Error)]
pub enum QsaError {
    #[error("qwen4_exp QSA evaluation failed: {0}")]
    Evaluation(String),
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
    pub fn new(
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

    pub fn query_heads(self) -> usize {
        self.query_heads as usize
    }

    pub fn head_dim(self) -> usize {
        self.head_dim as usize
    }

    pub fn rotary_dim(self) -> usize {
        self.rotary_dim as usize
    }

    pub fn compress_ratio(self) -> usize {
        self.compress_ratio as usize
    }

    pub fn token_budget(self) -> usize {
        self.token_budget as usize
    }

    pub fn block_topk(self) -> usize {
        self.block_topk as usize
    }

    pub fn hidden_size(self) -> usize {
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
}

/// Shared index q/k projection and sanitized RMS gains (1 + raw delta).
pub struct QsaIndexerWeights {
    pub qk_proj: QuantizedWeight,
    pub q_norm: MlxArray,
    pub k_norm: MlxArray,
}

/// Request-owned raw index keys `[batch, tokens, head_dim]`.
#[derive(Clone)]
pub struct QsaIndexKeyCache {
    keys: Option<MlxArray>,
}

impl QsaIndexKeyCache {
    pub fn empty() -> Self {
        Self { keys: None }
    }

    /// Crate-private restore path for durable snapshots. The wire decoder
    /// hands back a plain optional tensor; shape/dtype legality against a
    /// specific model's geometry is checked afterward, once the state is
    /// paired with the model it is being rebound to.
    pub(crate) fn from_serialized(keys: Option<MlxArray>) -> Self {
        Self { keys }
    }

    pub fn keys(&self) -> Option<&MlxArray> {
        self.keys.as_ref()
    }

    pub fn token_count(&self) -> Result<usize> {
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
pub struct QsaSelection {
    gather_indices: MlxArray,
    tokens: Vec<Vec<Vec<i32>>>,
    next_cache: QsaIndexKeyCache,
}

impl QsaSelection {
    /// `[batch, queries, token_budget + ratio - 1]` int32 indices, `-1` padded.
    pub fn gather_indices(&self) -> &MlxArray {
        &self.gather_indices
    }

    pub fn tokens_for_query(&self, batch: usize, query: usize) -> &[i32] {
        &self.tokens[batch][query]
    }

    pub fn next_cache(&self) -> &QsaIndexKeyCache {
        &self.next_cache
    }

    pub fn into_next_cache(self) -> QsaIndexKeyCache {
        self.next_cache
    }
}

pub struct QsaIndexer {
    config: QsaConfig,
    qk_proj: QuantizedWeight,
    q_norm: MlxArray,
    k_norm: MlxArray,
}

impl QsaIndexer {
    pub fn new(config: QsaConfig, weights: QsaIndexerWeights) -> Result<Self> {
        validate_projection(
            "index_qk_proj",
            &weights.qk_proj,
            config.proj_out(),
            config.hidden_size,
        )?;
        validate_gain("q_norm", &weights.q_norm, config.head_dim)?;
        validate_gain("k_norm", &weights.k_norm, config.head_dim)?;
        Ok(Self {
            config,
            qk_proj: weights.qk_proj,
            q_norm: weights.q_norm,
            k_norm: weights.k_norm,
        })
    }

    pub fn config(&self) -> QsaConfig {
        self.config
    }

    /// Project the new hidden chunk, append raw keys, and return gather indices
    /// for every new query. `cache` is not modified; adopt [`QsaSelection::next_cache`]
    /// only after the rest of the step succeeds.
    pub fn select(
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
        let cfg = self.config;
        let (batch, seq) = rank3_dims("hidden", hidden, cfg.hidden_size)?;
        let total = position_offset
            .checked_add(seq as usize)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or(QsaError::InvalidScalar("total context length"))?;
        let score_bytes = (batch as usize)
            .checked_mul(seq as usize)
            .and_then(|n| n.checked_mul(cfg.query_heads as usize))
            .and_then(|n| n.checked_mul(total / cfg.compress_ratio()))
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()));
        if score_bytes.is_none_or(|n| n > 256 * 1024 * 1024) {
            return Err(QsaError::InvalidScalar(
                "score workspace; use a smaller prefill chunk",
            ));
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
        let queries = rms_with_gain(&queries, &self.q_norm, cfg.rms_eps);
        let queries = rope_at_offset(
            &queries,
            cfg.rotary_dim,
            cfg.rope_base,
            i32::try_from(position_offset)
                .map_err(|_| QsaError::InvalidScalar("position offset"))?,
        );

        let staged_keys = match cache.keys() {
            Some(past) => contiguous(&concatenate(&[past, &raw_keys], 1, None), None),
            None => contiguous(&raw_keys, None),
        };
        let total = staged_keys.shape()[1];
        let n_complete = total / cfg.compress_ratio;
        let scores = if n_complete > 0 {
            let blocks = pooled_block_keys(&staged_keys, n_complete, cfg, &self.k_norm)?;
            Some(block_scores(&queries, &blocks))
        } else {
            None
        };

        let tokens = select_tokens(
            cfg,
            batch,
            seq,
            position_offset,
            n_complete,
            scores.as_ref(),
        )?;
        let gather_indices = pack_gather_indices(cfg, batch, seq, &tokens)?;
        let mut owned = vec![&staged_keys, &gather_indices];
        if let Some(scores) = &scores {
            owned.push(scores);
        }
        mlx_sys::try_eval(&owned).map_err(QsaError::Evaluation)?;

        Ok(QsaSelection {
            gather_indices,
            tokens,
            next_cache: QsaIndexKeyCache {
                keys: Some(staged_keys),
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

fn rms_with_gain(x: &MlxArray, weight: &MlxArray, eps: f32) -> MlxArray {
    let dtype = x.dtype();
    let x32 = astype(x, MlxDtype::Float32, None);
    let scale = astype(weight, MlxDtype::Float32, None);
    astype(&rms_norm(&x32, Some(&scale), eps, None), dtype, None)
}

fn rope_at_offset(queries: &MlxArray, rotary_dim: i32, rope_base: f32, offset: i32) -> MlxArray {
    let bhsd = transpose(queries, &[0, 2, 1, 3], None);
    let roped = rope(
        &bhsd,
        rotary_dim,
        false,
        Some(rope_base),
        1.0,
        offset,
        None,
        None,
    );
    transpose(&roped, &[0, 2, 1, 3], None)
}

fn pooled_block_keys(
    keys: &MlxArray,
    n_complete: i32,
    cfg: QsaConfig,
    k_norm: &MlxArray,
) -> Result<MlxArray> {
    let shape = keys.shape();
    let (batch, dim) = (shape[0], shape[2]);
    let used = n_complete * cfg.compress_ratio;
    let grouped = slice(keys, &[0, 0, 0], &[batch, used, dim], &[1, 1, 1], None);
    let grouped = reshape(
        &grouped,
        &[batch, n_complete, cfg.compress_ratio, dim],
        None,
    );
    let grouped_f32 = astype(&grouped, MlxDtype::Float32, None);
    let summed = sum_axis(&grouped_f32, 2, false, None);
    let mean = divide(
        &summed,
        &MlxArray::from_f32(cfg.compress_ratio as f32),
        None,
    );
    let pooled = astype(&mean, keys.dtype(), None);
    let pooled = rms_with_gain(&pooled, k_norm, cfg.rms_eps);
    rope_block_starts(&pooled, cfg.rotary_dim, cfg.rope_base, cfg.compress_ratio)
}

fn rope_block_starts(
    blocks: &MlxArray,
    rotary_dim: i32,
    rope_base: f32,
    ratio: i32,
) -> Result<MlxArray> {
    let shape = blocks.shape();
    let (batch, n_blocks, dim) = (shape[0], shape[1], shape[2]);
    let rows = batch
        .checked_mul(n_blocks)
        .ok_or(QsaError::InvalidScalar("block row count"))?;
    let flat = reshape(blocks, &[rows, 1, 1, dim], None);
    let starts = arange(
        0.0,
        f64::from(n_blocks) * f64::from(ratio),
        f64::from(ratio),
        MlxDtype::Int32,
        None,
    );
    let starts = reshape(&starts, &[1, n_blocks], None);
    let starts = broadcast_to(&starts, &[batch, n_blocks], None);
    let starts = reshape(&starts, &[rows], None);
    let roped = rope_dynamic(
        &flat,
        rotary_dim,
        false,
        Some(rope_base),
        1.0,
        &starts,
        None,
        None,
    );
    Ok(reshape(&roped, &[batch, n_blocks, dim], None))
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
    let batch_u = batch as usize;
    let seq_u = seq as usize;
    let n_blocks = n_complete as usize;
    let score_data = if let Some(scores) = scores {
        mlx_sys::try_eval(&[scores]).map_err(QsaError::Evaluation)?;
        scores.data_f32().to_vec()
    } else {
        Vec::new()
    };
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(batch_u)
        .map_err(|_| QsaError::InvalidScalar("token index buffer"))?;
    for b in 0..batch_u {
        let mut queries = Vec::new();
        queries
            .try_reserve_exact(seq_u)
            .map_err(|_| QsaError::InvalidScalar("token index buffer"))?;
        for q in 0..seq_u {
            let abs_pos = position_offset + q;
            let visible = abs_pos + 1;
            let complete = visible / cfg.compress_ratio();
            let mut chosen = Vec::new();
            if complete > 0 {
                let mut order: Vec<usize> = (0..complete).collect();
                order.sort_by(|lhs, rhs| {
                    let left = score_at(&score_data, seq_u, n_blocks, b, q, *lhs);
                    let right = score_at(&score_data, seq_u, n_blocks, b, q, *rhs);
                    right
                        .partial_cmp(&left)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(lhs.cmp(rhs))
                });
                order.truncate(complete.min(cfg.block_topk()));
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

fn score_at(
    scores: &[f32],
    seq: usize,
    n_blocks: usize,
    batch: usize,
    query: usize,
    block: usize,
) -> f32 {
    scores[(batch * seq + query) * n_blocks + block]
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
    mlx_sys::try_eval(&[&indices]).map_err(QsaError::Evaluation)?;
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
    if projection.group_size <= 0 || projection.bits <= 0 {
        return Err(QsaError::InvalidQuantization {
            tensor,
            group_size: projection.group_size,
            bits: projection.bits,
        });
    }
    let scales_shape = scales.shape();
    if scales_shape.len() != 2 || scales_shape[0] != out_dim {
        return Err(shape_error(scales_shape));
    }
    let logical_in = scales_shape[1].checked_mul(projection.group_size);
    if logical_in != Some(in_dim) {
        return Err(shape_error(scales_shape));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

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
}
