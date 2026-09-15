//! Bounded sparse attention for Flash Next (`qwen4_exp`).
//!
//! Query projection is packed as `[head0 query, head0 gate, head1 query,
//! head1 gate, ...]`, each chunk `head_dim`. After Q/K RMS (float32, sanitized)
//! gain `1 + raw delta`) and RoPE on the main rotary width, keys and values
//! are appended to a request-owned cache. Attention gathers only the token
//! rows chosen by [`QsaIndexer`]; it does not build a dense sequence-squared
//! mask. A per-query MLX gather plus unmasked `scaled_dot_product_attention`
//! is a correctness baseline, not a speed claim.
//!
//! Cache updates are staged locally. The caller publishes the returned next
//! state only after the rest of the step succeeds.

#![allow(dead_code)]

use mlx_sys::{
    MlxArray, MlxDtype, astype, concatenate, contiguous, multiply, repeat_axis, reshape, rms_norm,
    rope, scaled_dot_product_attention, slice, split, take, transpose,
};
use thiserror::Error;

use super::qwen4_exp_residual::{
    Qwen4ExpResidualError, sigmoid_projection_dtype, validate_projection,
};
use super::utils::{ProjectionBatchPolicy, qw_with_policy};
use crate::qwen4_exp_qsa::{QsaError, QsaIndexKeyCache, QsaIndexer, QsaSelection};
use crate::weights::QuantizedWeight;

type Result<T> = std::result::Result<T, Qwen4ExpAttentionError>;

fn gated_attention_output(attention: &MlxArray, gate: &MlxArray, dtype: MlxDtype) -> MlxArray {
    // Official QSA rounds sigmoid before the separate attention product.
    let gated = multiply(attention, &sigmoid_projection_dtype(gate), None);
    astype(&gated, dtype, None)
}

#[derive(Debug, Error)]
pub(crate) enum Qwen4ExpAttentionError {
    #[error(transparent)]
    Residual(#[from] Qwen4ExpResidualError),
    #[error(transparent)]
    Qsa(#[from] QsaError),
    #[error(
        "qwen4_exp attention geometry is invalid: hidden={hidden_size}, query_heads={query_heads}, kv_heads={kv_heads}, head_dim={head_dim}, rotary_dim={rotary_dim}"
    )]
    InvalidGeometry {
        hidden_size: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    },
    #[error("qwen4_exp attention {0} must be finite and positive")]
    InvalidScalar(&'static str),
    #[error("qwen4_exp attention {tensor} shape {actual:?} does not match {expected}")]
    TensorShape {
        tensor: &'static str,
        expected: String,
        actual: Vec<i32>,
    },
    #[error("qwen4_exp attention {tensor} dtype {actual:?} is not a floating dtype")]
    NonFloatingDtype {
        tensor: &'static str,
        actual: MlxDtype,
    },
    #[error(
        "qwen4_exp attention position offset {offset} does not match cached token length {cached_tokens}"
    )]
    PositionMismatch { offset: usize, cached_tokens: usize },
    #[error("qwen4_exp attention cache is missing paired K/V tensors")]
    IncompleteCache,
    #[error("qwen4_exp attention indexer {field} {actual} does not match main {expected}")]
    IndexerMismatch {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
}

/// Validated main-attention geometry. Indexer geometry lives on [`QsaIndexer`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen4ExpAttentionConfig {
    hidden_size: i32,
    query_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    groups: i32,
    query_width: i32,
    kv_width: i32,
    q_out: i32,
    rope_base: f32,
    rms_eps: f32,
    scale: f32,
}

impl Qwen4ExpAttentionConfig {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        hidden_size: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_base: f32,
        rms_eps: f32,
    ) -> Result<Self> {
        let invalid = || Qwen4ExpAttentionError::InvalidGeometry {
            hidden_size,
            query_heads,
            kv_heads,
            head_dim,
            rotary_dim,
        };
        if hidden_size == 0
            || query_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || rotary_dim == 0
            || !rotary_dim.is_multiple_of(2)
            || rotary_dim > head_dim
            || !query_heads.is_multiple_of(kv_heads)
        {
            return Err(invalid());
        }
        if !rms_eps.is_finite() || rms_eps <= 0.0 {
            return Err(Qwen4ExpAttentionError::InvalidScalar("rms_eps"));
        }
        if !rope_base.is_finite() || rope_base <= 0.0 {
            return Err(Qwen4ExpAttentionError::InvalidScalar("rope_base"));
        }
        let hidden_size = i32::try_from(hidden_size).map_err(|_| invalid())?;
        let query_heads = i32::try_from(query_heads).map_err(|_| invalid())?;
        let kv_heads = i32::try_from(kv_heads).map_err(|_| invalid())?;
        let head_dim = i32::try_from(head_dim).map_err(|_| invalid())?;
        let rotary_dim = i32::try_from(rotary_dim).map_err(|_| invalid())?;
        let query_width = query_heads.checked_mul(head_dim).ok_or_else(invalid)?;
        let q_out = query_width.checked_mul(2).ok_or_else(invalid)?;
        let kv_width = kv_heads.checked_mul(head_dim).ok_or_else(invalid)?;
        Ok(Self {
            hidden_size,
            query_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            groups: query_heads / kv_heads,
            query_width,
            kv_width,
            q_out,
            rope_base,
            rms_eps,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub(crate) fn hidden_size(self) -> usize {
        self.hidden_size as usize
    }

    pub(crate) fn query_heads(self) -> usize {
        self.query_heads as usize
    }

    pub(crate) fn kv_heads(self) -> usize {
        self.kv_heads as usize
    }

    pub(crate) fn head_dim(self) -> usize {
        self.head_dim as usize
    }

    pub(crate) fn rotary_dim(self) -> usize {
        self.rotary_dim as usize
    }
}

/// Separate Q/K/V/O projections and sanitized Q/K RMS gains (`1 + raw delta`).
pub(crate) struct Qwen4ExpAttentionWeights {
    pub q_proj: QuantizedWeight,
    pub k_proj: QuantizedWeight,
    pub v_proj: QuantizedWeight,
    pub o_proj: QuantizedWeight,
    pub q_norm: MlxArray,
    pub k_norm: MlxArray,
}

/// Request-owned K/V plus the indexer key history. Never mutated in place.
#[derive(Clone)]
pub(crate) struct Qwen4ExpAttentionCache {
    keys: Option<MlxArray>,
    values: Option<MlxArray>,
    index: QsaIndexKeyCache,
}

impl Qwen4ExpAttentionCache {
    pub(crate) fn empty() -> Self {
        Self {
            keys: None,
            values: None,
            index: QsaIndexKeyCache::empty(),
        }
    }

    pub(crate) fn keys(&self) -> Option<&MlxArray> {
        self.keys.as_ref()
    }

    pub(crate) fn values(&self) -> Option<&MlxArray> {
        self.values.as_ref()
    }

    pub(crate) fn index(&self) -> &QsaIndexKeyCache {
        &self.index
    }

    pub(crate) fn token_count(&self) -> Result<usize> {
        match (&self.keys, &self.values) {
            (None, None) => Ok(0),
            (Some(keys), Some(values)) => {
                let key_len = cache_tokens("key cache", keys)?;
                let value_len = cache_tokens("value cache", values)?;
                if key_len != value_len {
                    return Err(Qwen4ExpAttentionError::TensorShape {
                        tensor: "value cache",
                        expected: format!("[batch, {key_len}, kv_heads, head_dim]"),
                        actual: values.shape(),
                    });
                }
                Ok(key_len)
            }
            _ => Err(Qwen4ExpAttentionError::IncompleteCache),
        }
    }

    /// Independent copy for a forked request. Array clones are refcount bumps.
    pub(crate) fn fork(&self) -> Self {
        self.clone()
    }

    /// Crate-private restore path for durable snapshots. The wire format
    /// already keeps `keys`/`values` behind one presence tag, so this
    /// re-checks that pairing defensively rather than trusting the decoder.
    /// Shape/dtype legality against a specific model is
    /// [`Self::validate_against`], checked afterward.
    pub(crate) fn from_serialized(
        keys: Option<MlxArray>,
        values: Option<MlxArray>,
        index: QsaIndexKeyCache,
    ) -> Result<Self> {
        if keys.is_some() != values.is_some() {
            return Err(Qwen4ExpAttentionError::IncompleteCache);
        }
        Ok(Self {
            keys,
            values,
            index,
        })
    }

    /// Validate a decoded cache against the concrete model it is being
    /// rebound to. Called only from request-state restore, before that call
    /// adopts a new owner.
    pub(crate) fn validate_against(
        &self,
        config: Qwen4ExpAttentionConfig,
        index_head_dim: usize,
        dtype: MlxDtype,
    ) -> std::result::Result<(), String> {
        let cached_tokens = self.token_count().map_err(|e| e.to_string())?;
        if let Some(keys) = self.keys() {
            expect_cache_layout(
                "key cache",
                keys,
                keys.shape()[0],
                config.kv_heads() as i32,
                config.head_dim() as i32,
                dtype,
            )
            .map_err(|e| e.to_string())?;
        }
        if let Some(values) = self.values() {
            expect_cache_layout(
                "value cache",
                values,
                values.shape()[0],
                config.kv_heads() as i32,
                config.head_dim() as i32,
                dtype,
            )
            .map_err(|e| e.to_string())?;
        }
        let index_tokens = self.index.token_count().map_err(|e| e.to_string())?;
        if index_tokens != cached_tokens {
            return Err(format!(
                "qwen4_exp QSA cache/index token mismatch: {cached_tokens} vs {index_tokens}"
            ));
        }
        if let Some(index_keys) = self.index.keys() {
            let shape = index_keys.shape();
            if shape.len() != 3 || shape[2] != index_head_dim as i32 {
                return Err(format!(
                    "qwen4_exp QSA index cache shape {shape:?} does not match head_dim {index_head_dim}"
                ));
            }
            if index_keys.dtype() != dtype {
                return Err("qwen4_exp QSA index cache dtype mismatch".to_string());
            }
        }
        Ok(())
    }
}

/// Branch delta and the cache to adopt after a successful step.
pub(crate) struct Qwen4ExpAttentionOutput {
    delta: MlxArray,
    next_state: Qwen4ExpAttentionCache,
}

impl Qwen4ExpAttentionOutput {
    pub(crate) fn delta(&self) -> &MlxArray {
        &self.delta
    }

    pub(crate) fn next_state(&self) -> &Qwen4ExpAttentionCache {
        &self.next_state
    }

    pub(crate) fn into_next_state(self) -> Qwen4ExpAttentionCache {
        self.next_state
    }
}

pub(crate) struct Qwen4ExpAttention {
    config: Qwen4ExpAttentionConfig,
    q_proj: QuantizedWeight,
    k_proj: QuantizedWeight,
    v_proj: QuantizedWeight,
    o_proj: QuantizedWeight,
    q_norm: MlxArray,
    k_norm: MlxArray,
    indexer: QsaIndexer,
}

impl Qwen4ExpAttention {
    pub(crate) fn new(
        config: Qwen4ExpAttentionConfig,
        weights: Qwen4ExpAttentionWeights,
        indexer: QsaIndexer,
    ) -> Result<Self> {
        let indexer_cfg = indexer.config();
        if indexer_cfg.hidden_size() != config.hidden_size() {
            return Err(Qwen4ExpAttentionError::IndexerMismatch {
                field: "hidden_size",
                actual: indexer_cfg.hidden_size(),
                expected: config.hidden_size(),
            });
        }
        if indexer_cfg.rotary_dim() != config.rotary_dim() {
            return Err(Qwen4ExpAttentionError::IndexerMismatch {
                field: "rotary_dim",
                actual: indexer_cfg.rotary_dim(),
                expected: config.rotary_dim(),
            });
        }
        validate_projection("q_proj", &weights.q_proj, config.q_out, config.hidden_size)?;
        validate_projection(
            "k_proj",
            &weights.k_proj,
            config.kv_width,
            config.hidden_size,
        )?;
        validate_projection(
            "v_proj",
            &weights.v_proj,
            config.kv_width,
            config.hidden_size,
        )?;
        validate_projection(
            "o_proj",
            &weights.o_proj,
            config.hidden_size,
            config.query_width,
        )?;
        validate_gain("q_norm", &weights.q_norm, config.head_dim)?;
        validate_gain("k_norm", &weights.k_norm, config.head_dim)?;
        Ok(Self {
            config,
            q_proj: weights.q_proj,
            k_proj: weights.k_proj,
            v_proj: weights.v_proj,
            o_proj: weights.o_proj,
            q_norm: weights.q_norm,
            k_norm: weights.k_norm,
            indexer,
        })
    }

    pub(crate) fn config(&self) -> Qwen4ExpAttentionConfig {
        self.config
    }

    pub(crate) fn indexer(&self) -> &QsaIndexer {
        &self.indexer
    }

    /// Project a hidden chunk, attend over QSA-selected rows, and stage the
    /// next K/V plus indexer cache. `cache` is not modified.
    pub(crate) fn forward(
        &self,
        hidden: &MlxArray,
        cache: &Qwen4ExpAttentionCache,
        position_offset: usize,
        policy: ProjectionBatchPolicy,
    ) -> Result<Qwen4ExpAttentionOutput> {
        let cfg = self.config;
        let (batch, seq) = self.validate_hidden(hidden)?;
        let _total = position_offset
            .checked_add(seq as usize)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or(Qwen4ExpAttentionError::InvalidScalar(
                "total context length",
            ))?;
        let rope_offset = i32::try_from(position_offset)
            .map_err(|_| Qwen4ExpAttentionError::InvalidScalar("position offset"))?;
        self.validate_cache(cache, batch, hidden.dtype(), position_offset)?;

        let selection = match policy {
            ProjectionBatchPolicy::Shared => {
                self.indexer
                    .select(hidden, cache.index(), position_offset)?
            }
            ProjectionBatchPolicy::RowExact => self.indexer.select_with_projection(
                hidden,
                cache.index(),
                position_offset,
                |input, weight| qw_with_policy(input, weight, ProjectionBatchPolicy::RowExact),
            )?,
        };

        let q_packed = qw_with_policy(hidden, &self.q_proj, policy);
        let k_raw = qw_with_policy(hidden, &self.k_proj, policy);
        let v_raw = qw_with_policy(hidden, &self.v_proj, policy);

        let packed = reshape(
            &q_packed,
            &[batch, seq, cfg.query_heads, cfg.head_dim * 2],
            None,
        );
        let last_axis = packed.ndim() as i32 - 1;
        let halves = split(&packed, 2, last_axis, None);
        let [queries, gate] = <[MlxArray; 2]>::try_from(halves)
            .map_err(|_| Qwen4ExpAttentionError::InvalidScalar("query/gate split"))?;

        let queries = rms_with_gain(&queries, &self.q_norm, cfg.rms_eps);
        let queries = rope_bhsd(
            &transpose(&queries, &[0, 2, 1, 3], None),
            cfg.rotary_dim,
            cfg.rope_base,
            rope_offset,
        );

        let keys = reshape(&k_raw, &[batch, seq, cfg.kv_heads, cfg.head_dim], None);
        let keys = rms_with_gain(&keys, &self.k_norm, cfg.rms_eps);
        let keys = rope_bhsd(
            &transpose(&keys, &[0, 2, 1, 3], None),
            cfg.rotary_dim,
            cfg.rope_base,
            rope_offset,
        );
        let keys = transpose(&keys, &[0, 2, 1, 3], None);
        let values = reshape(&v_raw, &[batch, seq, cfg.kv_heads, cfg.head_dim], None);

        let dtype = hidden.dtype();
        let keys = astype(&keys, dtype, None);
        let values = astype(&values, dtype, None);
        let staged_keys = append_cache(cache.keys(), &keys)?;
        let staged_values = append_cache(cache.values(), &values)?;

        let attn = attend_selected(&queries, &staged_keys, &staged_values, &selection, cfg)?;
        let attn = reshape(
            &transpose(&attn, &[0, 2, 1, 3], None),
            &[batch, seq, cfg.query_width],
            None,
        );
        let gate = reshape(&gate, &[batch, seq, cfg.query_width], None);
        let gated = gated_attention_output(&attn, &gate, dtype);
        let delta = qw_with_policy(&gated, &self.o_proj, policy);
        #[cfg(test)]
        {
            for (stage, array) in [
                ("qsa_attention_before_gate", &attn),
                ("qsa_gate", &gate),
                ("qsa_gated", &gated),
                ("qsa_output", &delta),
            ] {
                crate::model::qwen4_exp::profiling::dump(stage, &[array]);
            }
        }

        Ok(Qwen4ExpAttentionOutput {
            delta,
            next_state: Qwen4ExpAttentionCache {
                keys: Some(staged_keys),
                values: Some(staged_values),
                index: selection.into_next_cache(),
            },
        })
    }

    fn validate_hidden(&self, hidden: &MlxArray) -> Result<(i32, i32)> {
        let shape = hidden.shape();
        if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != self.config.hidden_size
        {
            return Err(Qwen4ExpAttentionError::TensorShape {
                tensor: "hidden",
                expected: format!("[batch, seq, {}]", self.config.hidden_size),
                actual: shape,
            });
        }
        ensure_floating("hidden", hidden.dtype())?;
        Ok((shape[0], shape[1]))
    }

    fn validate_cache(
        &self,
        cache: &Qwen4ExpAttentionCache,
        batch: i32,
        dtype: MlxDtype,
        position_offset: usize,
    ) -> Result<()> {
        let cached_tokens = cache.token_count()?;
        if position_offset != cached_tokens {
            return Err(Qwen4ExpAttentionError::PositionMismatch {
                offset: position_offset,
                cached_tokens,
            });
        }
        let index_tokens = cache.index.token_count()?;
        if position_offset != index_tokens {
            return Err(Qwen4ExpAttentionError::PositionMismatch {
                offset: position_offset,
                cached_tokens: index_tokens,
            });
        }
        let cfg = self.config;
        if let Some(keys) = cache.keys() {
            expect_cache_layout("key cache", keys, batch, cfg.kv_heads, cfg.head_dim, dtype)?;
        }
        if let Some(values) = cache.values() {
            expect_cache_layout(
                "value cache",
                values,
                batch,
                cfg.kv_heads,
                cfg.head_dim,
                dtype,
            )?;
        }
        Ok(())
    }
}

fn attend_selected(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    selection: &QsaSelection,
    cfg: Qwen4ExpAttentionConfig,
) -> Result<MlxArray> {
    let q_shape = queries.shape();
    let k_shape = keys.shape();
    let (batch, q_heads, seq, dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let tokens = k_shape[1];
    let mut batch_out = Vec::new();
    batch_out
        .try_reserve_exact(batch as usize)
        .map_err(|_| Qwen4ExpAttentionError::InvalidScalar("attention batch buffer"))?;
    for b in 0..batch {
        let q_b = slice(
            queries,
            &[b, 0, 0, 0],
            &[b + 1, q_heads, seq, dim],
            &[1, 1, 1, 1],
            None,
        );
        let k_b = slice(
            keys,
            &[b, 0, 0, 0],
            &[b + 1, tokens, cfg.kv_heads, dim],
            &[1, 1, 1, 1],
            None,
        );
        let v_b = slice(
            values,
            &[b, 0, 0, 0],
            &[b + 1, tokens, cfg.kv_heads, dim],
            &[1, 1, 1, 1],
            None,
        );
        let mut seq_out = Vec::new();
        seq_out
            .try_reserve_exact(seq as usize)
            .map_err(|_| Qwen4ExpAttentionError::InvalidScalar("attention query buffer"))?;
        for s in 0..seq {
            let chosen = selection.tokens_for_query(b as usize, s as usize);
            if chosen.is_empty() {
                return Err(Qwen4ExpAttentionError::InvalidScalar(
                    "selected token count",
                ));
            }
            let idx = index_array(chosen)?;
            let k_sel = transpose(&take(&k_b, &idx, 1, None), &[0, 2, 1, 3], None);
            let v_sel = transpose(&take(&v_b, &idx, 1, None), &[0, 2, 1, 3], None);
            let k_sel = map_kv_heads(&k_sel, cfg.groups);
            let v_sel = map_kv_heads(&v_sel, cfg.groups);
            let q_s = slice(
                &q_b,
                &[0, 0, s, 0],
                &[1, q_heads, s + 1, dim],
                &[1, 1, 1, 1],
                None,
            );
            seq_out.push(scaled_dot_product_attention(
                &contiguous(&q_s, None),
                &contiguous(&k_sel, None),
                &contiguous(&v_sel, None),
                cfg.scale,
                false,
                None,
            ));
        }
        batch_out.push(concat_owned(&seq_out, 2));
    }
    Ok(concat_owned(&batch_out, 0))
}

fn map_kv_heads(cache_bhsd: &MlxArray, groups: i32) -> MlxArray {
    if groups > 1 {
        repeat_axis(cache_bhsd, groups, 1, None)
    } else {
        cache_bhsd.clone()
    }
}

fn append_cache(past: Option<&MlxArray>, new: &MlxArray) -> Result<MlxArray> {
    let staged = match past {
        Some(past) => concatenate(&[past, new], 1, None),
        None => new.clone(),
    };
    Ok(contiguous(&staged, None))
}

fn rope_bhsd(x: &MlxArray, rotary_dim: i32, rope_base: f32, offset: i32) -> MlxArray {
    rope(
        x,
        rotary_dim,
        false,
        Some(rope_base),
        1.0,
        offset,
        None,
        None,
    )
}

fn rms_with_gain(x: &MlxArray, weight: &MlxArray, eps: f32) -> MlxArray {
    let dtype = x.dtype();
    let x32 = astype(x, MlxDtype::Float32, None);
    let scale = astype(weight, MlxDtype::Float32, None);
    astype(&rms_norm(&x32, Some(&scale), eps, None), dtype, None)
}

fn index_array(tokens: &[i32]) -> Result<MlxArray> {
    let n = i32::try_from(tokens.len())
        .map_err(|_| Qwen4ExpAttentionError::InvalidScalar("selected token count"))?;
    let indices = MlxArray::from_raw_data(
        tokens.as_ptr() as *const u8,
        std::mem::size_of_val(tokens),
        &[n],
        MlxDtype::Int32,
    );
    mlx_sys::try_eval(&[&indices]).map_err(QsaError::Evaluation)?;
    Ok(indices)
}

fn concat_owned(parts: &[MlxArray], axis: i32) -> MlxArray {
    if parts.len() == 1 {
        return parts[0].clone();
    }
    let refs: Vec<&MlxArray> = parts.iter().collect();
    concatenate(&refs, axis, None)
}

fn cache_tokens(tensor: &'static str, array: &MlxArray) -> Result<usize> {
    let shape = array.shape();
    if shape.len() != 4 || shape[1] < 0 {
        return Err(Qwen4ExpAttentionError::TensorShape {
            tensor,
            expected: "[batch, tokens, kv_heads, head_dim]".to_string(),
            actual: shape,
        });
    }
    Ok(shape[1] as usize)
}

fn expect_cache_layout(
    tensor: &'static str,
    array: &MlxArray,
    batch: i32,
    kv_heads: i32,
    head_dim: i32,
    dtype: MlxDtype,
) -> Result<()> {
    let shape = array.shape();
    if shape.len() != 4
        || shape[0] != batch
        || shape[2] != kv_heads
        || shape[3] != head_dim
        || shape[1] < 0
    {
        return Err(Qwen4ExpAttentionError::TensorShape {
            tensor,
            expected: format!("[{batch}, tokens, {kv_heads}, {head_dim}]"),
            actual: shape,
        });
    }
    ensure_floating(tensor, array.dtype())?;
    if array.dtype() != dtype {
        return Err(Qwen4ExpAttentionError::InvalidScalar(
            "cache dtype mismatch",
        ));
    }
    Ok(())
}

fn ensure_floating(tensor: &'static str, dtype: MlxDtype) -> Result<()> {
    if matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::Bfloat16
    ) {
        Ok(())
    } else {
        Err(Qwen4ExpAttentionError::NonFloatingDtype {
            tensor,
            actual: dtype,
        })
    }
}

fn validate_gain(tensor: &'static str, gain: &MlxArray, dim: i32) -> Result<()> {
    let shape = gain.shape();
    if shape != [dim] {
        return Err(Qwen4ExpAttentionError::TensorShape {
            tensor,
            expected: format!("[{dim}]"),
            actual: shape,
        });
    }
    ensure_floating(tensor, gain.dtype())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::qwen4_exp_qsa::{QsaConfig, QsaIndexerWeights};
    use mlx_sys::{contiguous, eval, slice, zeros};
    use serde_json::Value;

    const EPS: f32 = 1e-6;
    const BASE: f32 = 10_000.0;
    const TOL: f32 = 3e-6;

    #[test]
    fn qsa_bf16_gate_matches_official_product_rounding() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/flash_next/qsa_bf16_gate.json"
        ))
        .unwrap();
        let shape = [1, fixture["gate"].as_array().unwrap().len() as i32];
        let attention = astype(
            &array(&fixture["attention"], &shape),
            MlxDtype::Bfloat16,
            None,
        );
        let gate = astype(&array(&fixture["gate"], &shape), MlxDtype::Bfloat16, None);
        let actual = astype(
            &gated_attention_output(&attention, &gate, MlxDtype::Bfloat16),
            MlxDtype::Float32,
            None,
        );
        let expected = array(&fixture["gated"], &shape);
        eval(&[&actual, &expected]);
        assert_eq!(actual.data_f32(), expected.data_f32());
    }

    fn values(value: &Value) -> Vec<f32> {
        match value {
            Value::Array(items) => items.iter().flat_map(values).collect(),
            _ => vec![value.as_f64().unwrap() as f32],
        }
    }

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    fn array(value: &Value, shape: &[i32]) -> MlxArray {
        array_f32(&values(value), shape)
    }

    fn close(actual: &MlxArray, expected: &MlxArray, label: &str) {
        let actual = contiguous(actual, None);
        let expected = contiguous(expected, None);
        eval(&[&actual, &expected]);
        assert_eq!(actual.shape(), expected.shape(), "{label}");
        for (i, (&a, &b)) in actual
            .data_f32()
            .iter()
            .zip(expected.data_f32())
            .enumerate()
        {
            assert!(
                (a - b).abs() <= TOL,
                "{label}[{i}]: {a} != {b} (diff {})",
                (a - b).abs()
            );
        }
    }

    fn oracle() -> (Value, Qwen4ExpAttention) {
        let fixture: Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/flash_next/qsa.json"))
                .unwrap();
        let w = &fixture["weights"];
        let dense = |name: &str, output: i32, input: i32| {
            QuantizedWeight::new(array(&w[name], &[output, input]), None, None)
        };
        let gain = |name: &str, dim: i32| {
            let data: Vec<f32> = values(&w[name]).into_iter().map(|v| v + 1.0).collect();
            array_f32(&data, &[dim])
        };
        let indexer = QsaIndexer::new(
            QsaConfig::new(2, 1, 4, 4, 2, 4, 16, EPS, BASE).unwrap(),
            QsaIndexerWeights {
                qk_proj: dense("indexer.index_qk_proj.weight", 12, 16),
                q_norm: gain("indexer.q_layernorm.weight", 4),
                k_norm: gain("indexer.k_layernorm.weight", 4),
            },
        )
        .unwrap();
        let module = Qwen4ExpAttention::new(
            Qwen4ExpAttentionConfig::new(16, 2, 1, 8, 4, BASE, EPS).unwrap(),
            Qwen4ExpAttentionWeights {
                q_proj: dense("q_proj.weight", 32, 16),
                k_proj: dense("k_proj.weight", 8, 16),
                v_proj: dense("v_proj.weight", 8, 16),
                o_proj: dense("o_proj.weight", 16, 16),
                q_norm: gain("q_norm.weight", 8),
                k_norm: gain("k_norm.weight", 8),
            },
            indexer,
        )
        .unwrap();
        (fixture, module)
    }

    fn run(
        module: &Qwen4ExpAttention,
        hidden: &MlxArray,
        cache: &Qwen4ExpAttentionCache,
        offset: usize,
    ) -> Qwen4ExpAttentionOutput {
        module
            .forward(hidden, cache, offset, ProjectionBatchPolicy::Shared)
            .unwrap()
    }

    fn close_state(
        actual: &Qwen4ExpAttentionCache,
        expected: &Qwen4ExpAttentionCache,
        label: &str,
    ) {
        close(
            actual.keys().unwrap(),
            expected.keys().unwrap(),
            &format!("{label} keys"),
        );
        close(
            actual.values().unwrap(),
            expected.values().unwrap(),
            &format!("{label} values"),
        );
        close(
            actual.index().keys().unwrap(),
            expected.index().keys().unwrap(),
            &format!("{label} index"),
        );
    }

    #[test]
    fn attention_matches_pinned_transformers_output() {
        let (f, module) = oracle();
        let input = array(&f["input"], &[1, 19, 16]);
        let out = run(&module, &input, &Qwen4ExpAttentionCache::empty(), 0);
        close(out.delta(), &array(&f["output"], &[1, 19, 16]), "output");
        assert_eq!(out.next_state().token_count().unwrap(), 19);
        assert!(Qwen4ExpAttentionConfig::new(16, 2, 3, 8, 4, BASE, EPS).is_err());
        assert!(Qwen4ExpAttentionConfig::new(16, 2, 1, 8, 9, BASE, EPS).is_err());
    }

    #[test]
    fn chunk_boundaries_and_token_steps_match_output_and_state() {
        let (f, module) = oracle();
        let input = array(&f["input"], &[1, 19, 16]);
        let whole = run(&module, &input, &Qwen4ExpAttentionCache::empty(), 0);
        close(
            whole.delta(),
            &array(&f["output"], &[1, 19, 16]),
            "full output",
        );

        for boundary in 1..=19 {
            let mut cache = Qwen4ExpAttentionCache::empty();
            let mut pieces = Vec::new();
            for (start, end) in [(0, boundary), (boundary, 19)] {
                if start == end {
                    continue;
                }
                let chunk = slice(
                    &input,
                    &[0, start as i32, 0],
                    &[1, end as i32, 16],
                    &[1, 1, 1],
                    None,
                );
                let step = run(&module, &chunk, &cache, start);
                pieces.push(step.delta().clone());
                cache = step.into_next_state();
            }
            close(
                &concat_owned(&pieces, 1),
                whole.delta(),
                &format!("boundary {boundary} output"),
            );
            close_state(&cache, whole.next_state(), &format!("boundary {boundary}"));
        }

        let mut cache = Qwen4ExpAttentionCache::empty();
        let mut steps = Vec::new();
        for token in 0..19 {
            let row = slice(
                &input,
                &[0, token, 0],
                &[1, token + 1, 16],
                &[1, 1, 1],
                None,
            );
            let step = run(&module, &row, &cache, token as usize);
            steps.push(step.delta().clone());
            cache = step.into_next_state();
        }
        close(&concat_owned(&steps, 1), whole.delta(), "token-step output");
        close_state(&cache, whole.next_state(), "token-step");
    }

    #[test]
    fn request_fork_isolation_and_rejected_calls_preserve_state() {
        let (f, module) = oracle();
        let input = array(&f["input"], &[1, 19, 16]);
        let prefix = slice(&input, &[0, 0, 0], &[1, 8, 16], &[1, 1, 1], None);
        let suffix = slice(&input, &[0, 8, 0], &[1, 19, 16], &[1, 1, 1], None);
        let first = run(&module, &prefix, &Qwen4ExpAttentionCache::empty(), 0);
        let original = first.next_state().fork();
        let snapshot_k = first.next_state().keys().unwrap().clone();
        eval(&[&snapshot_k]);
        let snapshot = snapshot_k.data_f32().to_vec();

        assert!(
            module
                .forward(
                    &zeros(&[1, 2, 15], MlxDtype::Float32, None),
                    first.next_state(),
                    8,
                    ProjectionBatchPolicy::Shared,
                )
                .is_err()
        );
        assert!(
            module
                .forward(
                    &suffix,
                    first.next_state(),
                    7,
                    ProjectionBatchPolicy::Shared,
                )
                .is_err()
        );
        close(
            first.next_state().keys().unwrap(),
            &array_f32(&snapshot, &snapshot_k.shape()),
            "source cache after rejected calls",
        );
        close(
            original.keys().unwrap(),
            &array_f32(&snapshot, &snapshot_k.shape()),
            "forked cache after rejected calls",
        );

        let published = run(&module, &suffix, first.next_state(), 8);
        close(
            first.next_state().keys().unwrap(),
            &array_f32(&snapshot, &snapshot_k.shape()),
            "source cache after a later forward",
        );
        close(
            original.keys().unwrap(),
            &array_f32(&snapshot, &snapshot_k.shape()),
            "forked cache after a later forward",
        );
        let repeated = run(&module, &suffix, &original, 8);
        close(repeated.delta(), published.delta(), "fork output");
        close_state(
            repeated.next_state(),
            published.next_state(),
            "fork next state",
        );
        assert_ne!(
            eval_len(published.next_state().keys().unwrap()),
            snapshot.len()
        );

        let recovered = run(&module, &suffix, first.next_state(), 8);
        close(recovered.delta(), published.delta(), "recovery output");
        close_state(
            recovered.next_state(),
            published.next_state(),
            "recovery state",
        );
    }

    fn eval_len(array: &MlxArray) -> usize {
        let array = contiguous(array, None);
        eval(&[&array]);
        array.data_f32().len()
    }
}
