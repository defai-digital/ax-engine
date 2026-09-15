//! Per-Layer Embedding (PLE) injection for Qwen 3.8 Flash Next (`qwen4_exp`).
//!
//! PLE runs on a single early linear-attention layer and adds a lexical
//! correction on top of the packed hyper-connection residual, before that
//! layer's own read/write. This module owns only the injection math: it
//! receives already-gathered n-gram embeddings `[batch, seq, E]` and the
//! packed residual `[batch, seq, C * H]`, and returns a `[batch, seq, C * H]`
//! delta the caller adds to the residual, plus the next depthwise-convolution
//! state. Hashing, table gather and any file IO are the caller's concern.
//!
//! Per token and stream: a key is projected from the embedding and grouped
//! RMS-normalized; a query is the grouped RMS norm of the residual itself; a
//! value is projected from the embedding (not normalized, not per-stream). A
//! per-stream gate is `sign(score) * sqrt(max(|score|, 1e-6))` of the
//! key/query dot product scaled by `1 / sqrt(H)`, then `sigmoid`. The gated
//! value is grouped RMS-normalized again and passed through a causal
//! depthwise convolution dilated by the n-gram size, `silu`-activated, and
//! added back to the (un-normalized) gated value.
//!
//! Contract: `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d`. Grouped norms run in f32
//! and cast back to the residual dtype (see
//! [`super::qwen4_exp_residual::qwen4_exp_grouped_rms_norm`]); the gate and
//! convolution run in the residual/weight dtype, matching the checkpoint's
//! own arithmetic. The convolution weight is stored `[C * H, K, 1]` (MLX
//! depthwise layout), not the raw HF `[C * H, 1, K]` layout.

use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, concatenate, conv1d, divide, less, maximum, multiply,
    negative, power, reshape, slice, sum_axis, where_cond, zeros,
};
use thiserror::Error;

use super::qwen4_exp_residual::{
    Qwen4ExpResidualError, Qwen4ExpStreamLayout, qwen4_exp_grouped_rms_norm,
    sigmoid_projection_dtype, silu_projection_dtype,
};
use super::utils::{ProjectionBatchPolicy, qw_with_policy};
use crate::weights::QuantizedWeight;

const GATE_FLOOR: f32 = 1e-6;

fn scaled_dot(key: &MlxArray, query: &MlxArray, hidden: usize) -> MlxArray {
    // The product and completed sum each round before the scalar division.
    let product = multiply(key, query, None);
    let dtype = product.dtype();
    let dot = sum_axis(&astype(&product, MlxDtype::Float32, None), 3, true, None);
    let dot = astype(&dot, dtype, None);
    let scale = cached_scalar((hidden as f32).sqrt(), MlxDtype::Float32);
    astype(
        &divide(&astype(&dot, MlxDtype::Float32, None), &scale, None),
        dtype,
        None,
    )
}

#[derive(Debug, Error)]
pub(crate) enum Qwen4ExpPleError {
    #[error(transparent)]
    Residual(#[from] Qwen4ExpResidualError),
    #[error("qwen4_exp ple embedding dim {0} is invalid")]
    InvalidEmbedDim(usize),
    #[error(
        "qwen4_exp ple convolution geometry is invalid: kernel={conv_kernel_size}, dilation={dilation}"
    )]
    InvalidConvGeometry {
        conv_kernel_size: usize,
        dilation: usize,
    },
}

type Result<T> = std::result::Result<T, Qwen4ExpPleError>;
type ResidualResult<T> = std::result::Result<T, Qwen4ExpResidualError>;

/// Checkpoint tensors for one PLE layer. Norm gains are sanitized (`1 +
/// stored delta`), `[C * H]`, the same convention as
/// [`super::qwen4_exp_residual::Qwen4ExpGatedResidualWeights`]. Projections
/// are bias-free `[out, in]` linears. `conv_weight` is `[C * H, K, 1]`.
pub(crate) struct Qwen4ExpPleWeights {
    pub norm_key_gain: MlxArray,
    pub norm_query_gain: MlxArray,
    pub norm_conv_gain: MlxArray,
    pub key_proj: QuantizedWeight,
    pub value_proj: QuantizedWeight,
    pub conv_weight: MlxArray,
}

/// A validated PLE layer. Construction never evaluates a tensor.
pub(crate) struct Qwen4ExpPle {
    layout: Qwen4ExpStreamLayout,
    embed_dim: i32,
    dilation: i32,
    conv_state_len: i32,
    eps: f32,
    norm_key_gain: MlxArray,
    norm_query_gain: MlxArray,
    norm_conv_gain: MlxArray,
    key_proj: QuantizedWeight,
    value_proj: QuantizedWeight,
    conv_weight: MlxArray,
}

impl Qwen4ExpPle {
    pub(crate) fn new(
        stream_count: usize,
        hidden_size: usize,
        embed_dim: usize,
        conv_kernel_size: usize,
        dilation: usize,
        eps: f32,
        weights: Qwen4ExpPleWeights,
    ) -> Result<Self> {
        let layout = Qwen4ExpStreamLayout::new(stream_count, hidden_size)?;
        let embed_dim_i32 = i32::try_from(embed_dim)
            .ok()
            .filter(|value| *value > 0)
            .ok_or(Qwen4ExpPleError::InvalidEmbedDim(embed_dim))?;
        let invalid_conv = || Qwen4ExpPleError::InvalidConvGeometry {
            conv_kernel_size,
            dilation,
        };
        let kernel = i32::try_from(conv_kernel_size)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(invalid_conv)?;
        let dilation_i32 = i32::try_from(dilation)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(invalid_conv)?;
        let conv_state_len = kernel
            .checked_sub(1)
            .and_then(|taps| taps.checked_mul(dilation_i32))
            .ok_or_else(invalid_conv)?;

        validate_eps(eps)?;
        validate_gain(layout, &weights.norm_key_gain, "norm_key_gain")?;
        validate_gain(layout, &weights.norm_query_gain, "norm_query_gain")?;
        validate_gain(layout, &weights.norm_conv_gain, "norm_conv_gain")?;
        let width = layout.packed_width() as i32;
        validate_projection("key_proj", &weights.key_proj, width, embed_dim_i32)?;
        validate_projection(
            "value_proj",
            &weights.value_proj,
            layout.hidden_size() as i32,
            embed_dim_i32,
        )?;
        validate_conv_weight(&weights.conv_weight, width, kernel)?;

        Ok(Self {
            layout,
            embed_dim: embed_dim_i32,
            dilation: dilation_i32,
            conv_state_len,
            eps,
            norm_key_gain: weights.norm_key_gain,
            norm_query_gain: weights.norm_query_gain,
            norm_conv_gain: weights.norm_conv_gain,
            key_proj: weights.key_proj,
            value_proj: weights.value_proj,
            conv_weight: weights.conv_weight,
        })
    }

    /// `embeddings`: `[batch, seq, E]` gathered n-gram features. `residual`:
    /// `[batch, seq, C * H]` packed hyper-connection state (the layer input,
    /// before this PLE delta is added). `history` is the previous call's
    /// convolution state, or `None` to start from zero (a fresh request).
    pub(crate) fn forward(
        &self,
        embeddings: &MlxArray,
        residual: &MlxArray,
        history: Option<&Qwen4ExpPleConvState>,
        policy: ProjectionBatchPolicy,
    ) -> Result<Qwen4ExpPleOutput> {
        let (batch, seq) = self.validate_inputs(embeddings, residual)?;
        let dtype = residual.dtype();
        if let Some(history) = history {
            self.validate_history(history.array(), batch)?;
        }
        let embeddings = astype(embeddings, dtype, None);
        let (streams, hidden) = (
            self.layout.stream_count() as i32,
            self.layout.hidden_size() as i32,
        );

        let key_raw = astype(
            &qw_with_policy(&embeddings, &self.key_proj, policy),
            dtype,
            None,
        );
        let key_normed =
            qwen4_exp_grouped_rms_norm(self.layout, &key_raw, &self.norm_key_gain, self.eps)?;
        let key = reshape(&key_normed, &[batch, seq, streams, hidden], None);

        let value = astype(
            &qw_with_policy(&embeddings, &self.value_proj, policy),
            dtype,
            None,
        );

        let query_normed =
            qwen4_exp_grouped_rms_norm(self.layout, residual, &self.norm_query_gain, self.eps)?;
        let query = reshape(&query_normed, &[batch, seq, streams, hidden], None);

        let score = scaled_dot(&key, &query, self.layout.hidden_size());
        let gate = signed_sqrt_floor(&score);

        let value_bcast = reshape(&value, &[batch, seq, 1, hidden], None);
        let gate_score = sigmoid_projection_dtype(&gate);
        let gated = multiply(&gate_score, &value_bcast, None);
        let gated_flat = reshape(
            &gated,
            &[batch, seq, self.layout.packed_width() as i32],
            None,
        );

        let normed_for_conv =
            qwen4_exp_grouped_rms_norm(self.layout, &gated_flat, &self.norm_conv_gain, self.eps)?;

        let (conv_out, next_state) = self.short_conv(&normed_for_conv, history, batch)?;
        let delta = add(&gated_flat, &astype(&conv_out, dtype, None), None);
        #[cfg(test)]
        {
            for (stage, array) in [
                ("ple_key", &key),
                ("ple_query", &query),
                ("ple_score", &score),
                ("ple_gate", &gate),
                ("ple_gate_score", &gate_score),
                ("ple_value", &value_bcast),
                ("ple_gated", &gated),
                ("ple_delta", &delta),
            ] {
                crate::model::qwen4_exp::profiling::dump(stage, &[array]);
            }
        }

        Ok(Qwen4ExpPleOutput { delta, next_state })
    }

    /// Validate a decoded convolution history against this layer's geometry.
    /// Called only from request-state restore, before the caller adopts a
    /// new owner for the whole snapshot.
    pub(crate) fn validate_state(
        &self,
        state: &Qwen4ExpPleConvState,
        batch: usize,
    ) -> std::result::Result<(), String> {
        let batch = i32::try_from(batch)
            .map_err(|_| "qwen4_exp ple state batch exceeds i32".to_string())?;
        self.validate_history(state.array(), batch)
            .map_err(|e| e.to_string())?;
        if state.array().dtype() != self.conv_weight.dtype() {
            return Err("qwen4_exp PLE cache dtype differs from convolution weights".into());
        }
        Ok(())
    }

    fn validate_inputs(&self, embeddings: &MlxArray, residual: &MlxArray) -> Result<(i32, i32)> {
        let embed_shape = embeddings.shape();
        if embed_shape.len() != 3
            || embed_shape[0] <= 0
            || embed_shape[1] <= 0
            || embed_shape[2] != self.embed_dim
        {
            return Err(Qwen4ExpResidualError::TensorShape {
                tensor: "ple embeddings",
                expected: format!("[batch, seq, {}]", self.embed_dim),
                actual: embed_shape,
            }
            .into());
        }
        ensure_floating("ple embeddings", embeddings.dtype())?;

        let residual_shape = residual.shape();
        let width = self.layout.packed_width() as i32;
        if residual_shape.len() != 3 || residual_shape[2] != width {
            return Err(Qwen4ExpResidualError::TensorShape {
                tensor: "ple residual",
                expected: format!("[batch, seq, {width}]"),
                actual: residual_shape,
            }
            .into());
        }
        ensure_floating("ple residual", residual.dtype())?;

        if embed_shape[0] != residual_shape[0] || embed_shape[1] != residual_shape[1] {
            return Err(Qwen4ExpResidualError::TensorShape {
                tensor: "ple embeddings",
                expected: format!(
                    "[{}, {}, {}]",
                    residual_shape[0], residual_shape[1], self.embed_dim
                ),
                actual: embed_shape,
            }
            .into());
        }
        Ok((residual_shape[0], residual_shape[1]))
    }

    fn validate_history(&self, history: &MlxArray, batch: i32) -> Result<()> {
        let width = self.layout.packed_width() as i32;
        let shape = history.shape();
        if shape != [batch, self.conv_state_len, width] {
            return Err(Qwen4ExpResidualError::TensorShape {
                tensor: "ple conv history",
                expected: format!("[{batch}, {}, {width}]", self.conv_state_len),
                actual: shape,
            }
            .into());
        }
        ensure_floating("ple conv history", history.dtype())
    }

    /// Causal depthwise convolution dilated by the n-gram size, `silu`
    /// activated. Runs in the convolution weight's dtype; the caller casts
    /// the result back to the activation dtype.
    fn short_conv(
        &self,
        x: &MlxArray,
        history: Option<&Qwen4ExpPleConvState>,
        batch: i32,
    ) -> Result<(MlxArray, Qwen4ExpPleConvState)> {
        let width = self.layout.packed_width() as i32;
        let conv_dtype = self.conv_weight.dtype();
        let x_conv = astype(x, conv_dtype, None);

        let state = match history {
            Some(state) => {
                self.validate_history(state.array(), batch)?;
                astype(state.array(), conv_dtype, None)
            }
            None => zeros(&[batch, self.conv_state_len, width], conv_dtype, None),
        };

        let window = concatenate(&[&state, &x_conv], 1, None);
        let total = window.shape()[1];
        let next_state = slice(
            &window,
            &[0, total - self.conv_state_len, 0],
            &[batch, total, width],
            &[1, 1, 1],
            None,
        );
        let conv_out = conv1d(&window, &self.conv_weight, 1, 0, self.dilation, width, None);
        let activated = silu_projection_dtype(&conv_out);
        #[cfg(test)]
        {
            crate::model::qwen4_exp::profiling::dump("ple_conv_pre_activation", &[&conv_out]);
            crate::model::qwen4_exp::profiling::dump("ple_conv_activated", &[&activated]);
            crate::model::qwen4_exp::profiling::dump("ple_conv_window", &[&window]);
            crate::model::qwen4_exp::profiling::dump("ple_conv_weight", &[&self.conv_weight]);
        }
        Ok((
            activated,
            Qwen4ExpPleConvState {
                history: next_state,
            },
        ))
    }
}

/// Output of [`Qwen4ExpPle::forward`]. Owns the delta and the next
/// convolution state; nothing is mutated in place, so a caller only commits
/// `next_state` after every earlier step of the request has succeeded.
pub(crate) struct Qwen4ExpPleOutput {
    delta: MlxArray,
    next_state: Qwen4ExpPleConvState,
}

impl Qwen4ExpPleOutput {
    /// `[batch, seq, C * H]` delta; the caller adds this to the residual.
    pub(crate) fn delta(&self) -> &MlxArray {
        &self.delta
    }

    pub(crate) fn next_state(&self) -> &Qwen4ExpPleConvState {
        &self.next_state
    }
}

/// Per-request PLE convolution history, `[batch, (K - 1) * dilation, C * H]`.
/// `None` in [`Qwen4ExpPle::forward`] means a fresh request (zero history).
#[derive(Clone)]
pub(crate) struct Qwen4ExpPleConvState {
    history: MlxArray,
}

impl Qwen4ExpPleConvState {
    /// Independent copy for a forked request; the fork and the original
    /// advance separately from here (`MlxArray` clones are cheap refcount
    /// bumps, so this does not copy the underlying buffer).
    #[cfg(test)]
    pub(crate) fn fork(&self) -> Self {
        self.clone()
    }

    pub(crate) fn array(&self) -> &MlxArray {
        &self.history
    }

    /// Crate-private restore path for durable snapshots. Shape legality
    /// against a specific model is [`Qwen4ExpPle::validate_state`], checked
    /// afterward, once the state is paired with the layer it came from.
    pub(crate) fn from_serialized(history: MlxArray) -> Self {
        Self { history }
    }
}

fn signed_sqrt_floor(score: &MlxArray) -> MlxArray {
    let dtype = score.dtype();
    let zero = cached_scalar(0.0, dtype);
    let one = cached_scalar(1.0, dtype);
    let neg_one = cached_scalar(-1.0, dtype);
    let floor = cached_scalar(GATE_FLOOR, dtype);
    let half = cached_scalar(0.5, dtype);

    let abs_score = maximum(score, &negative(score, None), None);
    let magnitude = power(&maximum(&abs_score, &floor, None), &half, None);
    let is_positive = less(&zero, score, None);
    let is_negative = less(score, &zero, None);
    let sign = where_cond(
        &is_positive,
        &one,
        &where_cond(&is_negative, &neg_one, &zero, None),
        None,
    );
    multiply(&sign, &magnitude, None)
}

fn ensure_floating(tensor: &'static str, dtype: MlxDtype) -> Result<()> {
    if matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::Bfloat16
    ) {
        Ok(())
    } else {
        Err(Qwen4ExpResidualError::NonFloatingDtype {
            tensor,
            actual: dtype,
        }
        .into())
    }
}

fn validate_eps(eps: f32) -> ResidualResult<()> {
    if eps.is_finite() && eps > 0.0 {
        Ok(())
    } else {
        Err(Qwen4ExpResidualError::InvalidEps(eps))
    }
}

fn validate_gain(
    layout: Qwen4ExpStreamLayout,
    gain: &MlxArray,
    tensor: &'static str,
) -> ResidualResult<()> {
    let shape = gain.shape();
    if shape != [layout.packed_width() as i32] {
        return Err(Qwen4ExpResidualError::TensorShape {
            tensor,
            expected: format!("[{}]", layout.packed_width()),
            actual: shape,
        });
    }
    if !matches!(
        gain.dtype(),
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::Bfloat16
    ) {
        return Err(Qwen4ExpResidualError::NonFloatingDtype {
            tensor,
            actual: gain.dtype(),
        });
    }
    Ok(())
}

fn validate_projection(
    tensor: &'static str,
    projection: &QuantizedWeight,
    out_dim: i32,
    in_dim: i32,
) -> ResidualResult<()> {
    super::qwen4_exp_residual::validate_projection(tensor, projection, out_dim, in_dim)
}

fn validate_conv_weight(weight: &MlxArray, width: i32, kernel: i32) -> ResidualResult<()> {
    let shape = weight.shape();
    if shape != [width, kernel, 1] {
        return Err(Qwen4ExpResidualError::TensorShape {
            tensor: "conv_weight",
            expected: format!("[{width}, {kernel}, 1]"),
            actual: shape,
        });
    }
    if !matches!(
        weight.dtype(),
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::Bfloat16
    ) {
        return Err(Qwen4ExpResidualError::NonFloatingDtype {
            tensor: "conv_weight",
            actual: weight.dtype(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use mlx_sys::eval;

    #[test]
    fn ple_bf16_score_matches_official_accumulation_and_scale() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/flash_next/hc_ple_bf16_rounding.json"
        ))
        .unwrap();
        let array = |name: &str, shape: &[i32]| {
            let values: Vec<f32> = fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            astype(&array_f32(&values, shape), MlxDtype::Bfloat16, None)
        };
        let actual = astype(
            &scaled_dot(
                &array("key", &[1, 2, 4, 10]),
                &array("query", &[1, 2, 4, 10]),
                10,
            ),
            MlxDtype::Float32,
            None,
        );
        let expected = astype(&array("score", &[1, 2, 4, 1]), MlxDtype::Float32, None);
        eval(&[&actual, &expected]);
        assert_eq!(actual.data_f32(), expected.data_f32());
    }

    #[test]
    fn ple_matches_pinned_transformers_for_whole_and_split_prefill() {
        fn values(value: &serde_json::Value) -> Vec<f32> {
            match value {
                serde_json::Value::Array(items) => items.iter().flat_map(values).collect(),
                _ => vec![value.as_f64().unwrap() as f32],
            }
        }
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/flash_next/ple.json"))
                .unwrap();
        let w = &f["weights"];
        let gain = |name: &str| {
            array_f32(
                &values(&w[name]).iter().map(|v| v + 1.0).collect::<Vec<_>>(),
                &[64],
            )
        };
        let module = Qwen4ExpPle::new(
            4,
            16,
            16,
            3,
            3,
            EPS,
            Qwen4ExpPleWeights {
                norm_key_gain: gain("norm_key.weight"),
                norm_query_gain: gain("norm_query.weight"),
                norm_conv_gain: gain("norm_conv.weight"),
                key_proj: dense(&values(&w["key_proj.weight"]), 64, 16),
                value_proj: dense(&values(&w["value_proj.weight"]), 16, 16),
                conv_weight: mlx_sys::transpose(
                    &array_f32(&values(&w["conv1d.weight"]), &[64, 1, 3]),
                    &[0, 2, 1],
                    None,
                ),
            },
        )
        .unwrap();
        let input = array_f32(&values(&f["input"]), &[1, 9, 64]);
        let embedding = array_f32(&values(&f["embedding"]), &[1, 9, 16]);
        let expected = values(&f["output"]);
        for boundary in 1..=9 {
            let mut history = None;
            let mut output = Vec::new();
            for (start, end) in [(0, boundary), (boundary, 9)] {
                if start == end {
                    continue;
                }
                let part = |array: &MlxArray, width: i32| {
                    slice(array, &[0, start, 0], &[1, end, width], &[1, 1, 1], None)
                };
                let result = module
                    .forward(
                        &part(&embedding, 16),
                        &part(&input, 64),
                        history.as_ref(),
                        ProjectionBatchPolicy::Shared,
                    )
                    .unwrap();
                output.extend(eval_f32(result.delta()));
                history = Some(result.next_state().clone());
            }
            assert_eq!(output.len(), expected.len());
            for (index, (&actual, &expected)) in output.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "PLE split {boundary}, index {index}: {actual} != {expected}"
                );
            }
        }
    }

    const EPS: f32 = 1e-6;
    const TOL: f64 = 3e-4;

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

    fn wave(len: usize, freq: f32, amplitude: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + 1.0) * freq).sin() * amplitude + offset)
            .collect()
    }

    fn dense(data: &[f32], out: usize, input: usize) -> QuantizedWeight {
        QuantizedWeight::new(array_f32(data, &[out as i32, input as i32]), None, None)
    }

    struct HostPle {
        streams: usize,
        hidden: usize,
        embed_dim: usize,
        kernel: usize,
        dilation: usize,
        norm_key_gain: Vec<f32>,
        norm_query_gain: Vec<f32>,
        norm_conv_gain: Vec<f32>,
        key_proj: Vec<f32>,
        value_proj: Vec<f32>,
        conv_weight: Vec<f32>,
    }

    fn host_ple(
        streams: usize,
        hidden: usize,
        embed_dim: usize,
        kernel: usize,
        dilation: usize,
    ) -> HostPle {
        let width = streams * hidden;
        HostPle {
            streams,
            hidden,
            embed_dim,
            kernel,
            dilation,
            norm_key_gain: wave(width, 0.31, 0.4, 1.0),
            norm_query_gain: wave(width, 0.47, 0.35, 1.0),
            norm_conv_gain: wave(width, 0.59, 0.3, 1.0),
            key_proj: wave(width * embed_dim, 0.53, 0.25, 0.02),
            value_proj: wave(hidden * embed_dim, 0.71, 0.3, -0.03),
            conv_weight: wave(width * kernel, 0.19, 0.5, 0.05),
        }
    }

    fn weights_from(host: &HostPle) -> Qwen4ExpPleWeights {
        let width = host.streams * host.hidden;
        Qwen4ExpPleWeights {
            norm_key_gain: array_f32(&host.norm_key_gain, &[width as i32]),
            norm_query_gain: array_f32(&host.norm_query_gain, &[width as i32]),
            norm_conv_gain: array_f32(&host.norm_conv_gain, &[width as i32]),
            key_proj: dense(&host.key_proj, width, host.embed_dim),
            value_proj: dense(&host.value_proj, host.hidden, host.embed_dim),
            conv_weight: array_f32(&host.conv_weight, &[width as i32, host.kernel as i32, 1]),
        }
    }

    fn build(host: &HostPle) -> Qwen4ExpPle {
        Qwen4ExpPle::new(
            host.streams,
            host.hidden,
            host.embed_dim,
            host.kernel,
            host.dilation,
            EPS,
            weights_from(host),
        )
        .expect("ple")
    }

    fn project(weight: &[f32], out: usize, input: usize, x: &[f64]) -> Vec<f64> {
        (0..out)
            .map(|row| {
                (0..input)
                    .map(|col| f64::from(weight[row * input + col]) * x[col])
                    .sum()
            })
            .collect()
    }

    fn grouped_norm(gain: &[f32], streams: usize, hidden: usize, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; streams * hidden];
        for stream in 0..streams {
            let group = &x[stream * hidden..(stream + 1) * hidden];
            let mean_square = group.iter().map(|v| v.powi(2)).sum::<f64>() / hidden as f64;
            let inv_rms = 1.0 / (mean_square + f64::from(EPS)).sqrt();
            for (element, value) in group.iter().enumerate() {
                let index = stream * hidden + element;
                out[index] = value * inv_rms * f64::from(gain[index]);
            }
        }
        out
    }

    fn sigmoid64(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    fn silu64(v: f64) -> f64 {
        v * sigmoid64(v)
    }

    fn sign64(v: f64) -> f64 {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    /// `(gated_flat, normed_for_conv)` for one token, independent of history.
    fn host_token_gate(
        host: &HostPle,
        embedding: &[f32],
        residual: &[f32],
    ) -> (Vec<f64>, Vec<f64>) {
        let (streams, hidden) = (host.streams, host.hidden);
        let embed_f64: Vec<f64> = embedding.iter().map(|v| f64::from(*v)).collect();
        let key_raw = project(&host.key_proj, streams * hidden, host.embed_dim, &embed_f64);
        let key_normed = grouped_norm(&host.norm_key_gain, streams, hidden, &key_raw);
        let value = project(&host.value_proj, hidden, host.embed_dim, &embed_f64);
        let residual_f64: Vec<f64> = residual.iter().map(|v| f64::from(*v)).collect();
        let query_normed = grouped_norm(&host.norm_query_gain, streams, hidden, &residual_f64);

        let mut gated_flat = vec![0.0f64; streams * hidden];
        for stream in 0..streams {
            let mut dot = 0.0f64;
            for element in 0..hidden {
                let index = stream * hidden + element;
                dot += key_normed[index] * query_normed[index];
            }
            let score = dot / (hidden as f64).sqrt();
            let magnitude = score.abs().max(1e-6).sqrt();
            let gate = sigmoid64(sign64(score) * magnitude);
            for element in 0..hidden {
                gated_flat[stream * hidden + element] = gate * value[element];
            }
        }
        let normed_for_conv = grouped_norm(&host.norm_conv_gain, streams, hidden, &gated_flat);
        (gated_flat, normed_for_conv)
    }

    /// Full-sequence host reference: zero initial history, `seq` tokens.
    fn host_forward(host: &HostPle, embeddings: &[f32], residual: &[f32], seq: usize) -> Vec<f64> {
        let width = host.streams * host.hidden;
        let history_len = (host.kernel - 1) * host.dilation;
        let mut normed_seq: Vec<Vec<f64>> = Vec::with_capacity(seq);
        let mut gated_seq: Vec<Vec<f64>> = Vec::with_capacity(seq);
        for token in 0..seq {
            let (gated, normed) = host_token_gate(
                host,
                &embeddings[token * host.embed_dim..(token + 1) * host.embed_dim],
                &residual[token * width..(token + 1) * width],
            );
            gated_seq.push(gated);
            normed_seq.push(normed);
        }

        let mut padded: Vec<Vec<f64>> = vec![vec![0.0; width]; history_len];
        padded.extend(normed_seq);

        let mut out = vec![0.0f64; seq * width];
        for (token, gated) in gated_seq.iter().enumerate() {
            for channel in 0..width {
                let mut acc = 0.0f64;
                for k in 0..host.kernel {
                    let pos = token + k * host.dilation;
                    acc += padded[pos][channel]
                        * f64::from(host.conv_weight[channel * host.kernel + k]);
                }
                out[token * width + channel] = gated[channel] + silu64(acc);
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f64], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            let diff = (f64::from(*a) - e).abs();
            assert!(
                diff <= TOL * (1.0 + e.abs()),
                "{label}[{index}]: {a} vs {e} (diff {diff})"
            );
        }
    }

    #[test]
    fn one_shot_forward_matches_scalar_reference() {
        let (batch, seq) = (2usize, 5usize);
        for (streams, hidden, embed_dim, kernel, dilation) in
            [(1usize, 4usize, 6usize, 3usize, 2usize), (4, 4, 6, 4, 3)]
        {
            let host = host_ple(streams, hidden, embed_dim, kernel, dilation);
            let ple = build(&host);
            let width = streams * hidden;

            let embeddings = wave(batch * seq * embed_dim, 0.83, 0.9, 0.1);
            let residual = wave(batch * seq * width, 0.43, 1.0, 0.15);
            let embed_array = array_f32(&embeddings, &[batch as i32, seq as i32, embed_dim as i32]);
            let residual_array = array_f32(&residual, &[batch as i32, seq as i32, width as i32]);

            let out = ple
                .forward(
                    &embed_array,
                    &residual_array,
                    None,
                    ProjectionBatchPolicy::Shared,
                )
                .expect("forward");
            assert_eq!(
                out.delta().shape(),
                [batch as i32, seq as i32, width as i32]
            );
            let delta = eval_f32(out.delta());

            for b in 0..batch {
                let e = &embeddings[b * seq * embed_dim..(b + 1) * seq * embed_dim];
                let r = &residual[b * seq * width..(b + 1) * seq * width];
                let expected = host_forward(&host, e, r, seq);
                assert_close(
                    &delta[b * seq * width..(b + 1) * seq * width],
                    &expected,
                    &format!("C={streams} batch {b}"),
                );
            }
        }
    }

    #[test]
    fn chunked_and_token_step_forward_match_one_shot() {
        let (streams, hidden, embed_dim, kernel, dilation) =
            (4usize, 4usize, 5usize, 4usize, 3usize);
        let host = host_ple(streams, hidden, embed_dim, kernel, dilation);
        let ple = build(&host);
        let width = streams * hidden;
        let seq = 7usize;

        let embeddings = wave(seq * embed_dim, 0.61, 0.7, -0.1);
        let residual = wave(seq * width, 0.37, 0.9, 0.2);
        let full_embed = array_f32(&embeddings, &[1, seq as i32, embed_dim as i32]);
        let full_residual = array_f32(&residual, &[1, seq as i32, width as i32]);
        let one_shot = ple
            .forward(
                &full_embed,
                &full_residual,
                None,
                ProjectionBatchPolicy::Shared,
            )
            .expect("one-shot");
        let one_shot_delta = eval_f32(one_shot.delta());

        // Two chunks, state carried across the split.
        let split = 3usize;
        let chunk_embed = |start: usize, len: usize| {
            array_f32(
                &embeddings[start * embed_dim..(start + len) * embed_dim],
                &[1, len as i32, embed_dim as i32],
            )
        };
        let chunk_residual = |start: usize, len: usize| {
            array_f32(
                &residual[start * width..(start + len) * width],
                &[1, len as i32, width as i32],
            )
        };
        let first = ple
            .forward(
                &chunk_embed(0, split),
                &chunk_residual(0, split),
                None,
                ProjectionBatchPolicy::Shared,
            )
            .expect("chunk 1");
        let second = ple
            .forward(
                &chunk_embed(split, seq - split),
                &chunk_residual(split, seq - split),
                Some(first.next_state()),
                ProjectionBatchPolicy::Shared,
            )
            .expect("chunk 2");
        let mut chunked_delta = eval_f32(first.delta());
        chunked_delta.extend(eval_f32(second.delta()));
        assert_close(
            &chunked_delta,
            &one_shot_delta
                .iter()
                .map(|v| f64::from(*v))
                .collect::<Vec<_>>(),
            "chunked vs one-shot",
        );

        // One token at a time, state carried at every step.
        let mut state: Option<Qwen4ExpPleConvState> = None;
        let mut stepped_delta = Vec::with_capacity(seq * width);
        for token in 0..seq {
            let out = ple
                .forward(
                    &chunk_embed(token, 1),
                    &chunk_residual(token, 1),
                    state.as_ref(),
                    ProjectionBatchPolicy::Shared,
                )
                .expect("token step");
            stepped_delta.extend(eval_f32(out.delta()));
            state = Some(out.next_state().fork());
        }
        assert_close(
            &stepped_delta,
            &one_shot_delta
                .iter()
                .map(|v| f64::from(*v))
                .collect::<Vec<_>>(),
            "token steps vs one-shot",
        );
    }

    #[test]
    fn forked_states_advance_independently() {
        let (streams, hidden, embed_dim, kernel, dilation) =
            (4usize, 4usize, 5usize, 4usize, 3usize);
        let host = host_ple(streams, hidden, embed_dim, kernel, dilation);
        let ple = build(&host);
        let width = streams * hidden;

        let seed_embed = wave(embed_dim, 0.29, 0.6, 0.05);
        let seed_residual = wave(width, 0.71, 0.8, -0.1);
        let seed_out = ple
            .forward(
                &array_f32(&seed_embed, &[1, 1, embed_dim as i32]),
                &array_f32(&seed_residual, &[1, 1, width as i32]),
                None,
                ProjectionBatchPolicy::Shared,
            )
            .expect("seed");
        let baseline_state = seed_out.next_state().fork();
        let fork_a = seed_out.next_state().fork();
        let fork_b = seed_out.next_state().fork();

        let embed_a = wave(embed_dim, 0.13, 0.5, 0.2);
        let residual_a = wave(width, 0.17, 0.4, 0.1);
        let embed_b = wave(embed_dim, 0.91, 0.5, -0.2);
        let residual_b = wave(width, 0.97, 0.4, -0.1);

        let out_a = ple
            .forward(
                &array_f32(&embed_a, &[1, 1, embed_dim as i32]),
                &array_f32(&residual_a, &[1, 1, width as i32]),
                Some(&fork_a),
                ProjectionBatchPolicy::Shared,
            )
            .expect("fork a");
        let out_b = ple
            .forward(
                &array_f32(&embed_b, &[1, 1, embed_dim as i32]),
                &array_f32(&residual_b, &[1, 1, width as i32]),
                Some(&fork_b),
                ProjectionBatchPolicy::Shared,
            )
            .expect("fork b");

        let state_a = eval_f32(out_a.next_state().array());
        let state_b = eval_f32(out_b.next_state().array());
        assert_ne!(
            state_a, state_b,
            "forks must diverge under different inputs"
        );

        // The seed state itself is untouched by either fork's continuation.
        let baseline = eval_f32(baseline_state.array());
        let seed_again = eval_f32(seed_out.next_state().array());
        assert_eq!(
            baseline, seed_again,
            "forking must not mutate the source state"
        );
    }

    #[test]
    fn rejected_invalid_input_leaves_state_untouched() {
        let (streams, hidden, embed_dim, kernel, dilation) =
            (4usize, 4usize, 5usize, 4usize, 3usize);
        let host = host_ple(streams, hidden, embed_dim, kernel, dilation);
        let ple = build(&host);
        let width = streams * hidden;

        let embed = wave(embed_dim, 0.29, 0.6, 0.05);
        let residual = wave(width, 0.71, 0.8, -0.1);
        let out = ple
            .forward(
                &array_f32(&embed, &[1, 1, embed_dim as i32]),
                &array_f32(&residual, &[1, 1, width as i32]),
                None,
                ProjectionBatchPolicy::Shared,
            )
            .expect("valid forward");
        let state = out.next_state().fork();
        let before = eval_f32(state.array());

        let wrong_embed = array_f32(
            &wave(embed_dim + 1, 0.1, 0.1, 0.0),
            &[1, 1, (embed_dim + 1) as i32],
        );
        let good_residual = array_f32(&residual, &[1, 1, width as i32]);
        assert!(matches!(
            ple.forward(
                &wrong_embed,
                &good_residual,
                Some(&state),
                ProjectionBatchPolicy::Shared
            ),
            Err(Qwen4ExpPleError::Residual(
                Qwen4ExpResidualError::TensorShape {
                    tensor: "ple embeddings",
                    ..
                }
            ))
        ));

        let good_embed = array_f32(&embed, &[1, 1, embed_dim as i32]);
        let wrong_residual =
            array_f32(&wave(width + 1, 0.1, 0.1, 0.0), &[1, 1, (width + 1) as i32]);
        assert!(matches!(
            ple.forward(
                &good_embed,
                &wrong_residual,
                Some(&state),
                ProjectionBatchPolicy::Shared
            ),
            Err(Qwen4ExpPleError::Residual(
                Qwen4ExpResidualError::TensorShape {
                    tensor: "ple residual",
                    ..
                }
            ))
        ));

        let bad_history = Qwen4ExpPleConvState {
            history: array_f32(&vec![0.0; width], &[1, 1, width as i32]),
        };
        assert!(matches!(
            ple.forward(
                &good_embed,
                &good_residual,
                Some(&bad_history),
                ProjectionBatchPolicy::Shared
            ),
            Err(Qwen4ExpPleError::Residual(
                Qwen4ExpResidualError::TensorShape {
                    tensor: "ple conv history",
                    ..
                }
            ))
        ));

        assert_eq!(
            eval_f32(state.array()),
            before,
            "state must be unchanged after rejected calls"
        );
    }

    #[test]
    fn construction_rejects_invalid_geometry_and_weights() {
        // Distinct stream/hidden/embed dims so a projection built for the
        // wrong pair of dimensions cannot coincidentally still be valid.
        let host = host_ple(3, 4, 5, 4, 3);
        let (streams, hidden, embed_dim, kernel, dilation) = (
            host.streams,
            host.hidden,
            host.embed_dim,
            host.kernel,
            host.dilation,
        );

        assert!(matches!(
            Qwen4ExpPle::new(
                0,
                hidden,
                embed_dim,
                kernel,
                dilation,
                EPS,
                weights_from(&host)
            ),
            Err(Qwen4ExpPleError::Residual(
                Qwen4ExpResidualError::InvalidGeometry { .. }
            ))
        ));
        assert!(matches!(
            Qwen4ExpPle::new(
                streams,
                hidden,
                0,
                kernel,
                dilation,
                EPS,
                weights_from(&host)
            ),
            Err(Qwen4ExpPleError::InvalidEmbedDim(0))
        ));
        assert!(matches!(
            Qwen4ExpPle::new(
                streams,
                hidden,
                embed_dim,
                0,
                dilation,
                EPS,
                weights_from(&host)
            ),
            Err(Qwen4ExpPleError::InvalidConvGeometry { .. })
        ));
        assert!(matches!(
            Qwen4ExpPle::new(
                streams,
                hidden,
                embed_dim,
                kernel,
                0,
                EPS,
                weights_from(&host)
            ),
            Err(Qwen4ExpPleError::InvalidConvGeometry { .. })
        ));
        assert!(matches!(
            Qwen4ExpPle::new(
                streams,
                hidden,
                embed_dim,
                kernel,
                dilation,
                0.0,
                weights_from(&host)
            ),
            Err(Qwen4ExpPleError::Residual(
                Qwen4ExpResidualError::InvalidEps(_)
            ))
        ));

        let is_shape_error = |result: Result<Qwen4ExpPle>, name: &str| {
            matches!(
                result,
                Err(Qwen4ExpPleError::Residual(Qwen4ExpResidualError::TensorShape { tensor, .. })) if tensor == name
            )
        };
        let build_with = |edit: fn(&mut Qwen4ExpPleWeights)| {
            let mut w = weights_from(&host);
            edit(&mut w);
            Qwen4ExpPle::new(streams, hidden, embed_dim, kernel, dilation, EPS, w)
        };

        assert!(is_shape_error(
            build_with(|w| w.norm_key_gain = array_f32(&[1.0; 4], &[4])),
            "norm_key_gain"
        ));
        assert!(is_shape_error(
            build_with(|w| w.key_proj = dense(&[0.5; 4], 1, 4)),
            "key_proj"
        ));
        // A plausible mistake: the value projection's out dim is `hidden`,
        // not the packed `streams * hidden` width `key_proj` uses.
        assert!(is_shape_error(
            build_with(|w| w.value_proj = dense(&[0.5; 60], 12, 5)),
            "value_proj"
        ));
        assert!(is_shape_error(
            build_with(|w| w.conv_weight = array_f32(&[0.5; 16], &[16, 1, 1])),
            "conv_weight"
        ));
        assert!(matches!(
            build_with(|w| w.key_proj.linear_bias = Some(array_f32(&[0.0; 2], &[2]))),
            Err(Qwen4ExpPleError::Residual(
                Qwen4ExpResidualError::UnexpectedLinearBias { tensor: "key_proj" }
            ))
        ));
    }
}
