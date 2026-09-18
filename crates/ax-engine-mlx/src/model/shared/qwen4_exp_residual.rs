//! Gated residual streams for Qwen 3.8 Flash Next (`qwen4_exp`).
//!
//! The residual is packed as `[batch, seq, C * H]`: `C` parallel streams of
//! width `H`, stream-major. Each attention/MoE branch is bracketed by a read
//! and a write:
//!
//! - read: grouped RMS norm `n` (per stream, over `H`), an elementwise read
//!   gate `sigmoid(up(silu(down(n) / C)))`, and the stream mean of `gate * n`;
//! - write: `residual + branch_output * 2 * sigmoid(inject(n) / C)`, one
//!   scalar gate per stream, with the residual itself carried unchanged.
//!
//! The final mixer performs the read only (no inject projection) and returns
//! the `[batch, seq, H]` trunk output; no extra norm follows it.
//!
//! This is not DeepSeek V4 mHC (`hyper_connection.rs`): there is no Sinkhorn
//! combine matrix and the read gate is elementwise rather than per stream.
//!
//! Contract: `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d`. Dtype follows it: the
//! grouped norm runs in f32 and is cast back to the residual dtype; gates and
//! projections run in the residual dtype.

// Staged: consumed once the qwen4_exp forward is integrated.
#![allow(dead_code)]

use mlx_sys::ops::{cached_scalar, silu};
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, broadcast_to, divide, multiply, reshape, rms_norm, sigmoid,
    sum_axis,
};
use thiserror::Error;

use super::utils::{ProjectionBatchPolicy, qw_with_policy};
use crate::weights::QuantizedWeight;

const WRITE_GATE_SCALE: f32 = 2.0;

pub(crate) fn silu_projection_dtype(input: &MlxArray) -> MlxArray {
    // Official activation rounds once before the separate projection-dtype product.
    astype(
        &silu(&astype(input, MlxDtype::Float32, None), None),
        input.dtype(),
        None,
    )
}

pub(crate) fn sigmoid_projection_dtype(input: &MlxArray) -> MlxArray {
    astype(
        &sigmoid(&astype(input, MlxDtype::Float32, None), None),
        input.dtype(),
        None,
    )
}

fn stream_mean(input: &MlxArray, streams: i32) -> MlxArray {
    let mean = divide(
        &sum_axis(&astype(input, MlxDtype::Float32, None), 2, false, None),
        &cached_scalar(streams as f32, MlxDtype::Float32),
        None,
    );
    astype(&mean, input.dtype(), None)
}

#[derive(Debug, Error)]
pub(crate) enum Qwen4ExpResidualError {
    #[error("qwen4_exp stream layout is invalid: streams={stream_count}, hidden={hidden_size}")]
    InvalidGeometry {
        stream_count: usize,
        hidden_size: usize,
    },
    #[error("qwen4_exp residual low rank {0} is invalid")]
    InvalidLowRank(usize),
    #[error("qwen4_exp residual norm eps {0} must be finite and positive")]
    InvalidEps(f32),
    #[error("qwen4_exp {tensor} shape {actual:?} does not match {expected}")]
    TensorShape {
        tensor: &'static str,
        expected: String,
        actual: Vec<i32>,
    },
    #[error("qwen4_exp {tensor} dtype {actual:?} is not a floating dtype")]
    NonFloatingDtype {
        tensor: &'static str,
        actual: MlxDtype,
    },
    #[error("qwen4_exp {tensor} quantization is invalid: group_size={group_size}, bits={bits}")]
    InvalidQuantization {
        tensor: &'static str,
        group_size: i32,
        bits: i32,
    },
    #[error("qwen4_exp {tensor} must not carry a linear bias")]
    UnexpectedLinearBias { tensor: &'static str },
    #[error("qwen4_exp {role} residual does not support {operation}")]
    RoleMismatch {
        role: &'static str,
        operation: &'static str,
    },
}

type Result<T> = std::result::Result<T, Qwen4ExpResidualError>;

/// Validated `C` streams of width `H`, held as MLX `i32` dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4ExpStreamLayout {
    stream_count: i32,
    hidden_size: i32,
    packed_width: i32,
}

impl Qwen4ExpStreamLayout {
    pub(crate) fn new(stream_count: usize, hidden_size: usize) -> Result<Self> {
        let invalid = || Qwen4ExpResidualError::InvalidGeometry {
            stream_count,
            hidden_size,
        };
        let streams = i32::try_from(stream_count).map_err(|_| invalid())?;
        let hidden = i32::try_from(hidden_size).map_err(|_| invalid())?;
        if streams <= 0 || hidden <= 0 {
            return Err(invalid());
        }
        let packed_width = streams.checked_mul(hidden).ok_or_else(invalid)?;
        Ok(Self {
            stream_count: streams,
            hidden_size: hidden,
            packed_width,
        })
    }

    pub(crate) fn stream_count(self) -> usize {
        self.stream_count as usize
    }

    pub(crate) fn hidden_size(self) -> usize {
        self.hidden_size as usize
    }

    pub(crate) fn packed_width(self) -> usize {
        self.packed_width as usize
    }

    /// `[batch, seq, H]` embedding → `[batch, seq, C * H]`, every stream a
    /// copy of the embedding.
    pub(crate) fn expand(self, embedding: &MlxArray) -> Result<MlxArray> {
        let (batch, seq) = self.rank3_dims("embedding", embedding, self.hidden_size)?;
        ensure_floating("embedding", embedding.dtype())?;
        if self.stream_count == 1 {
            return Ok(embedding.clone());
        }
        let (streams, hidden) = (self.stream_count, self.hidden_size);
        let column = reshape(embedding, &[batch, seq, 1, hidden], None);
        let per_stream = broadcast_to(&column, &[batch, seq, streams, hidden], None);
        Ok(reshape(&per_stream, &[batch, seq, self.packed_width], None))
    }

    fn rank3_dims(self, tensor: &'static str, x: &MlxArray, last: i32) -> Result<(i32, i32)> {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != last {
            return Err(Qwen4ExpResidualError::TensorShape {
                tensor,
                expected: format!("[batch, seq, {last}]"),
                actual: shape,
            });
        }
        Ok((shape[0], shape[1]))
    }

    fn streams_scalar(self, dtype: MlxDtype) -> MlxArray {
        cached_scalar(self.stream_count as f32, dtype)
    }
}

/// Grouped RMS norm over each stream's `H` features of a packed
/// `[batch, seq, C * H]` tensor, with eps inside the root and a sanitized
/// per-stream plain gain `[C * H]`. Runs in f32 and returns the input dtype.
pub(crate) fn qwen4_exp_grouped_rms_norm(
    layout: Qwen4ExpStreamLayout,
    packed: &MlxArray,
    gain: &MlxArray,
    eps: f32,
) -> Result<MlxArray> {
    let (batch, seq) = layout.rank3_dims("grouped norm input", packed, layout.packed_width)?;
    let dtype = packed.dtype();
    ensure_floating("grouped norm input", dtype)?;
    validate_gain(layout, gain)?;
    validate_eps(eps)?;
    let (streams, hidden) = (layout.stream_count, layout.hidden_size);
    let grouped = reshape(
        &astype(packed, MlxDtype::Float32, None),
        &[batch, seq, streams, hidden],
        None,
    );
    let unit = rms_norm(&grouped, None, eps, None);
    let unit = reshape(&unit, &[batch, seq, layout.packed_width], None);
    let gain = astype(gain, MlxDtype::Float32, None);
    Ok(astype(&multiply(&unit, &gain, None), dtype, None))
}

/// Checkpoint tensors of one gated residual. `norm_gain` is the sanitized
/// plain gain (`1 + stored delta`), `[C * H]`. Projections are bias-free
/// `[out, in]` linears; `write_inject` is `None` only for the final mixer.
pub(crate) struct Qwen4ExpGatedResidualWeights {
    pub norm_gain: MlxArray,
    pub read_down: QuantizedWeight,
    pub read_up: QuantizedWeight,
    pub write_inject: Option<QuantizedWeight>,
}

/// A branch gated residual, or the final mixer when it has no inject
/// projection. Construction validates shapes and never evaluates a tensor.
pub(crate) struct Qwen4ExpGatedResidual {
    layout: Qwen4ExpStreamLayout,
    eps: f32,
    norm_gain: MlxArray,
    read_down: QuantizedWeight,
    read_up: QuantizedWeight,
    write_inject: Option<QuantizedWeight>,
}

impl Qwen4ExpGatedResidual {
    pub(crate) fn new(
        layout: Qwen4ExpStreamLayout,
        low_rank: usize,
        eps: f32,
        weights: Qwen4ExpGatedResidualWeights,
    ) -> Result<Self> {
        let rank = i32::try_from(low_rank)
            .ok()
            .filter(|rank| *rank > 0)
            .ok_or(Qwen4ExpResidualError::InvalidLowRank(low_rank))?;
        validate_eps(eps)?;
        validate_gain(layout, &weights.norm_gain)?;
        let width = layout.packed_width;
        validate_projection("read_down", &weights.read_down, rank, width)?;
        validate_projection("read_up", &weights.read_up, width, rank)?;
        if let Some(inject) = &weights.write_inject {
            validate_projection("write_inject", inject, layout.stream_count, width)?;
        }
        Ok(Self {
            layout,
            eps,
            norm_gain: weights.norm_gain,
            read_down: weights.read_down,
            read_up: weights.read_up,
            write_inject: weights.write_inject,
        })
    }

    pub(crate) fn layout(&self) -> Qwen4ExpStreamLayout {
        self.layout
    }

    pub(crate) fn is_final_mixer(&self) -> bool {
        self.write_inject.is_none()
    }

    /// Branch read. The result retains the packed residual and the write gate,
    /// so its [`Qwen4ExpResidualRead::write`] needs no caller-held state.
    pub(crate) fn read(
        &self,
        packed: &MlxArray,
        policy: ProjectionBatchPolicy,
    ) -> Result<Qwen4ExpResidualRead> {
        let Some(inject) = &self.write_inject else {
            return Err(Qwen4ExpResidualError::RoleMismatch {
                role: "final mixer",
                operation: "branch read",
            });
        };
        let normalized = self.normalize(packed)?;
        let shape = packed.shape();
        let (batch, seq) = (shape[0], shape[1]);
        let branch_input = self.read_mix(packed.dtype(), &normalized, batch, seq, policy);
        let logits = qw_with_policy(&normalized, inject, policy);
        let dtype = logits.dtype();
        let scaled = divide(&logits, &self.layout.streams_scalar(dtype), None);
        let write_gate = multiply(
            &sigmoid_projection_dtype(&scaled),
            &cached_scalar(WRITE_GATE_SCALE, dtype),
            None,
        );
        #[cfg(test)]
        {
            crate::model::qwen4_exp::profiling::dump("hc_inject_scaled", &[&scaled]);
            crate::model::qwen4_exp::profiling::dump("hc_write_gate", &[&write_gate]);
        }
        Ok(Qwen4ExpResidualRead {
            layout: self.layout,
            batch,
            seq,
            residual: packed.clone(),
            branch_input,
            write_gate,
        })
    }

    /// Final mixer: `[batch, seq, C * H]` → `[batch, seq, H]`.
    pub(crate) fn collapse(
        &self,
        packed: &MlxArray,
        policy: ProjectionBatchPolicy,
    ) -> Result<MlxArray> {
        if self.write_inject.is_some() {
            return Err(Qwen4ExpResidualError::RoleMismatch {
                role: "branch",
                operation: "final collapse",
            });
        }
        let normalized = self.normalize(packed)?;
        let shape = packed.shape();
        let (batch, seq) = (shape[0], shape[1]);
        Ok(self.read_mix(packed.dtype(), &normalized, batch, seq, policy))
    }

    fn normalize(&self, packed: &MlxArray) -> Result<MlxArray> {
        qwen4_exp_grouped_rms_norm(self.layout, packed, &self.norm_gain, self.eps)
    }

    fn read_mix(
        &self,
        residual_dtype: MlxDtype,
        normalized: &MlxArray,
        batch: i32,
        seq: i32,
        policy: ProjectionBatchPolicy,
    ) -> MlxArray {
        let layout = self.layout;
        let down = qw_with_policy(normalized, &self.read_down, policy);
        let scaled = divide(&down, &layout.streams_scalar(down.dtype()), None);
        let low = silu_projection_dtype(&scaled);
        let up = qw_with_policy(&low, &self.read_up, policy);
        let gate = sigmoid_projection_dtype(&up);
        let stream_shape = [batch, seq, layout.stream_count, layout.hidden_size];
        let gated = multiply(
            &reshape(&gate, &stream_shape, None),
            &reshape(normalized, &stream_shape, None),
            None,
        );
        let mean = stream_mean(&gated, layout.stream_count);
        #[cfg(test)]
        {
            for (stage, array) in [
                ("hc_scaled", &scaled),
                ("hc_normalized", normalized),
                ("hc_low", &low),
                ("hc_up", &up),
                ("hc_gate", &gate),
                ("hc_gated", &gated),
                ("hc_mean", &mean),
            ] {
                crate::model::qwen4_exp::profiling::dump(stage, &[array]);
            }
        }
        astype(&mean, residual_dtype, None)
    }
}

/// Output of [`Qwen4ExpGatedResidual::read`]. Owns handles to the packed
/// residual and the write gate, so writes stay valid after the caller drops
/// its residual handle. Nothing is updated in place.
pub(crate) struct Qwen4ExpResidualRead {
    layout: Qwen4ExpStreamLayout,
    batch: i32,
    seq: i32,
    residual: MlxArray,
    branch_input: MlxArray,
    write_gate: MlxArray,
}

impl Qwen4ExpResidualRead {
    /// `[batch, seq, H]` branch input in the residual dtype.
    pub(crate) fn branch_input(&self) -> &MlxArray {
        &self.branch_input
    }

    /// `[batch, seq, C]` per-stream write gate.
    pub(crate) fn write_gate(&self) -> &MlxArray {
        &self.write_gate
    }

    /// The retained packed residual, the same array the read consumed.
    pub(crate) fn residual(&self) -> &MlxArray {
        &self.residual
    }

    /// `residual + branch_output * write_gate`, broadcast over streams, in
    /// the residual dtype.
    pub(crate) fn write(&self, branch_output: &MlxArray) -> Result<MlxArray> {
        let (batch, seq) = (self.batch, self.seq);
        let (streams, hidden) = (self.layout.stream_count, self.layout.hidden_size);
        let shape = branch_output.shape();
        if shape != [batch, seq, hidden] {
            return Err(Qwen4ExpResidualError::TensorShape {
                tensor: "branch output",
                expected: format!("[{batch}, {seq}, {hidden}]"),
                actual: shape,
            });
        }
        ensure_floating("branch output", branch_output.dtype())?;
        let gate = reshape(&self.write_gate, &[batch, seq, streams, 1], None);
        let branch = reshape(branch_output, &[batch, seq, 1, hidden], None);
        let delta = reshape(
            &multiply(&gate, &branch, None),
            &[batch, seq, self.layout.packed_width],
            None,
        );
        let dtype = self.residual.dtype();
        Ok(add(&self.residual, &astype(&delta, dtype, None), None))
    }
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
        })
    }
}

fn validate_eps(eps: f32) -> Result<()> {
    if eps.is_finite() && eps > 0.0 {
        Ok(())
    } else {
        Err(Qwen4ExpResidualError::InvalidEps(eps))
    }
}

fn validate_gain(layout: Qwen4ExpStreamLayout, gain: &MlxArray) -> Result<()> {
    let shape = gain.shape();
    if shape != [layout.packed_width] {
        return Err(Qwen4ExpResidualError::TensorShape {
            tensor: "norm gain",
            expected: format!("[{}]", layout.packed_width),
            actual: shape,
        });
    }
    ensure_floating("norm gain", gain.dtype())
}

/// Logical `[out, in]` check. Quantized tensors derive `in` from the scales
/// (`[out, in / group_size]`) because the packed weight width depends on mode.
pub(crate) fn validate_projection(
    tensor: &'static str,
    projection: &QuantizedWeight,
    out_dim: i32,
    in_dim: i32,
) -> Result<()> {
    if projection.linear_bias.is_some() {
        return Err(Qwen4ExpResidualError::UnexpectedLinearBias { tensor });
    }
    let shape_error = |actual: Vec<i32>| Qwen4ExpResidualError::TensorShape {
        tensor,
        expected: format!("[{out_dim}, {in_dim}]"),
        actual,
    };
    let shape = projection.weight.shape();
    if shape.len() != 2 || shape[0] != out_dim {
        return Err(shape_error(shape));
    }
    let Some(scales) = &projection.scales else {
        if shape[1] != in_dim || projection.biases.is_some() {
            return Err(shape_error(shape));
        }
        return ensure_floating(tensor, projection.weight.dtype());
    };
    let invalid_quantization = || Qwen4ExpResidualError::InvalidQuantization {
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

    fn rounding_fixture() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/flash_next/hc_ple_bf16_rounding.json"
        ))
        .unwrap()
    }

    fn fixture_array(value: &serde_json::Value, shape: &[i32]) -> MlxArray {
        let values: Vec<f32> = value
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        astype(&array_f32(&values, shape), MlxDtype::Bfloat16, None)
    }

    #[test]
    fn hc_ple_bf16_activations_match_official_rounding() {
        let fixture = rounding_fixture();
        let shape = [fixture["gate"].as_array().unwrap().len() as i32];
        let input = fixture_array(&fixture["gate"], &shape);
        for (name, actual) in [
            ("silu", silu_projection_dtype(&input)),
            ("sigmoid", sigmoid_projection_dtype(&input)),
        ] {
            let expected = astype(
                &fixture_array(&fixture[name], &shape),
                MlxDtype::Float32,
                None,
            );
            let actual = astype(&actual, MlxDtype::Float32, None);
            mlx_sys::eval(&[&actual, &expected]);
            assert_eq!(actual.data_f32(), expected.data_f32(), "{name}");
        }
    }

    #[test]
    fn hc_bf16_mean_matches_official_accumulation() {
        let fixture = rounding_fixture();
        let input = fixture_array(&fixture["streams"], &[1, 3, 4, 16]);
        let actual = astype(&stream_mean(&input, 4), MlxDtype::Float32, None);
        let expected = astype(
            &fixture_array(&fixture["mean"], &[1, 3, 16]),
            MlxDtype::Float32,
            None,
        );
        mlx_sys::eval(&[&actual, &expected]);
        assert_eq!(actual.data_f32(), expected.data_f32());
    }
    use mlx_sys::{MlxQuantizationMode, dequantize, eval, quantize};

    #[test]
    fn quantized_projection_rejects_malformed_storage_before_execution() {
        let input = array_f32(&vec![0.125; 64 * 4], &[4, 64]);
        let arrays = quantize(
            &input,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let valid = QuantizedWeight::new(
            arrays[0].clone(),
            Some(arrays[1].clone()),
            Some(arrays[2].clone()),
        );
        let mut valid = valid;
        valid.group_size = 32;
        valid.bits = 4;
        validate_projection("test", &valid, 4, 64).unwrap();
        let mut broken = valid.clone();
        broken.biases = None;
        assert!(validate_projection("test", &broken, 4, 64).is_err());
        broken = valid.clone();
        broken.mode = "unknown".into();
        assert!(validate_projection("test", &broken, 4, 64).is_err());
        broken = valid.clone();
        broken.weight = array_f32(&[0.0; 32], &[4, 8]);
        assert!(validate_projection("test", &broken, 4, 64).is_err());
        broken = valid.clone();
        broken.biases = Some(array_f32(&[0.0; 4], &[4, 1]));
        assert!(validate_projection("test", &broken, 4, 64).is_err());
        broken = valid;
        broken.bits = 7;
        assert!(validate_projection("test", &broken, 4, 64).is_err());
    }

    #[test]
    fn read_and_write_match_pinned_transformers_oracle() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/flash_next/residual.json"
        ))
        .unwrap();
        fn values(value: &serde_json::Value) -> Vec<f32> {
            match value {
                serde_json::Value::Array(items) => items.iter().flat_map(values).collect(),
                _ => vec![value.as_f64().unwrap() as f32],
            }
        }
        let w = &fixture["weights"];
        let weights = Qwen4ExpGatedResidualWeights {
            norm_gain: array_f32(
                &values(&w["hc_norm.weight"])
                    .iter()
                    .map(|v| v + 1.0)
                    .collect::<Vec<_>>(),
                &[64],
            ),
            read_down: dense(&values(&w["input_mix_weight_down.weight"]), 8, 64),
            read_up: dense(&values(&w["input_mix_weight_up.weight"]), 64, 8),
            write_inject: Some(dense(&values(&w["block_inject_weight.weight"]), 4, 64)),
        };
        let module =
            Qwen4ExpGatedResidual::new(Qwen4ExpStreamLayout::new(4, 16).unwrap(), 8, EPS, weights)
                .unwrap();
        let packed = array_f32(&values(&fixture["input"]), &[1, 3, 64]);
        let read = shared_read(&module, &packed);
        let combined = read
            .write(&array_f32(&values(&fixture["branch"]), &[1, 3, 16]))
            .unwrap();
        for (name, output) in [
            ("mixed", read.branch_input()),
            ("injection", read.write_gate()),
            ("combined", &combined),
        ] {
            let expected = values(&fixture[name]);
            let actual = eval_f32(output);
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "{name}[{index}]: {actual} != {expected}"
                );
            }
        }
    }

    const EPS: f32 = 1e-6;
    const TOL: f64 = 3e-5;

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

    fn shared_read(residual: &Qwen4ExpGatedResidual, packed: &MlxArray) -> Qwen4ExpResidualRead {
        residual
            .read(packed, ProjectionBatchPolicy::Shared)
            .expect("read")
    }

    fn shared_collapse(mixer: &Qwen4ExpGatedResidual, packed: &MlxArray) -> MlxArray {
        mixer
            .collapse(packed, ProjectionBatchPolicy::Shared)
            .expect("collapse")
    }

    /// Host copies of the effective weights used by the scalar reference.
    struct HostWeights {
        streams: usize,
        hidden: usize,
        rank: usize,
        gain: Vec<f32>,
        down: Vec<f32>,
        up: Vec<f32>,
        inject: Option<Vec<f32>>,
    }

    fn host_weights(streams: usize, hidden: usize, rank: usize, inject: bool) -> HostWeights {
        let width = streams * hidden;
        HostWeights {
            streams,
            hidden,
            rank,
            gain: wave(width, 0.37, 0.4, 1.0),
            down: wave(rank * width, 0.53, 0.25, 0.03),
            up: wave(width * rank, 0.71, 0.3, -0.05),
            inject: inject.then(|| wave(streams * width, 0.29, 0.4, 0.1)),
        }
    }

    fn dense_weights(host: &HostWeights) -> Qwen4ExpGatedResidualWeights {
        let width = host.streams * host.hidden;
        Qwen4ExpGatedResidualWeights {
            norm_gain: array_f32(&host.gain, &[width as i32]),
            read_down: dense(&host.down, host.rank, width),
            read_up: dense(&host.up, width, host.rank),
            write_inject: host
                .inject
                .as_ref()
                .map(|inject| dense(inject, host.streams, width)),
        }
    }

    fn try_build(
        host: &HostWeights,
        edit: impl FnOnce(&mut Qwen4ExpGatedResidualWeights),
    ) -> Result<Qwen4ExpGatedResidual> {
        let layout = Qwen4ExpStreamLayout::new(host.streams, host.hidden).expect("layout");
        let mut weights = dense_weights(host);
        edit(&mut weights);
        Qwen4ExpGatedResidual::new(layout, host.rank, EPS, weights)
    }

    fn build(host: &HostWeights) -> Qwen4ExpGatedResidual {
        try_build(host, |_| {}).expect("residual")
    }

    /// Streams get distinct magnitudes so grouped and whole-width RMS differ.
    fn packed_input(tokens: usize, streams: usize, hidden: usize) -> Vec<f32> {
        let mut data = wave(tokens * streams * hidden, 0.43, 1.0, 0.15);
        for (index, value) in data.iter_mut().enumerate() {
            let stream = (index / hidden) % streams;
            *value *= 0.5 + 1.5 * stream as f32;
        }
        data
    }

    fn sigmoid64(value: f64) -> f64 {
        1.0 / (1.0 + (-value).exp())
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

    struct HostRead {
        branch: Vec<f64>,
        write_gate: Option<Vec<f64>>,
    }

    /// Independent f64 evaluation of one packed token.
    fn host_read(host: &HostWeights, token: &[f32]) -> HostRead {
        let (streams, hidden) = (host.streams, host.hidden);
        let width = streams * hidden;
        let count = streams as f64;
        let mut normed = vec![0.0f64; width];
        for stream in 0..streams {
            let group = &token[stream * hidden..(stream + 1) * hidden];
            let mean_square =
                group.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / hidden as f64;
            let inv_rms = 1.0 / (mean_square + f64::from(EPS)).sqrt();
            for (element, value) in group.iter().enumerate() {
                let index = stream * hidden + element;
                normed[index] = f64::from(*value) * inv_rms * f64::from(host.gain[index]);
            }
        }
        let low: Vec<f64> = project(&host.down, host.rank, width, &normed)
            .into_iter()
            .map(|v| {
                let z = v / count;
                z * sigmoid64(z)
            })
            .collect();
        let gate: Vec<f64> = project(&host.up, width, host.rank, &low)
            .into_iter()
            .map(sigmoid64)
            .collect();
        let branch = (0..hidden)
            .map(|element| {
                let indices = (0..streams).map(|stream| stream * hidden + element);
                indices
                    .map(|index| gate[index] * normed[index])
                    .sum::<f64>()
                    / count
            })
            .collect();
        let write_gate = host.inject.as_ref().map(|inject| {
            project(inject, streams, width, &normed)
                .into_iter()
                .map(|v| 2.0 * sigmoid64(v / count))
                .collect()
        });
        HostRead { branch, write_gate }
    }

    /// `x[s * H + e] + write_gate[s] * branch[e]` for one token.
    fn host_write(x: &[f32], write_gate: &[f64], branch: &[f32]) -> Vec<f64> {
        let hidden = branch.len();
        x.iter()
            .enumerate()
            .map(|(index, value)| {
                f64::from(*value) + write_gate[index / hidden] * f64::from(branch[index % hidden])
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f64], tol: f64, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            let diff = (f64::from(*a) - e).abs();
            assert!(
                diff <= tol * (1.0 + e.abs()),
                "{label}[{index}]: {a} vs {e} (diff {diff})"
            );
        }
    }

    #[test]
    fn block_read_and_write_match_scalar_reference() {
        let (batch, seq, hidden, rank) = (2usize, 3usize, 5usize, 3usize);
        for streams in [1usize, 4] {
            let host = host_weights(streams, hidden, rank, true);
            let residual = build(&host);
            let width = streams * hidden;
            let tokens = batch * seq;
            let input = packed_input(tokens, streams, hidden);
            let packed = array_f32(&input, &[batch as i32, seq as i32, width as i32]);
            let branch_output = wave(tokens * hidden, 0.61, 0.7, -0.05);
            let branch_shape = [batch as i32, seq as i32, hidden as i32];

            let read = shared_read(&residual, &packed);
            let written = read
                .write(&array_f32(&branch_output, &branch_shape))
                .expect("write");
            assert_eq!(read.branch_input().shape(), branch_shape);
            assert_eq!(
                read.write_gate().shape(),
                [batch as i32, seq as i32, streams as i32]
            );
            assert_eq!(written.shape(), packed.shape());
            assert_eq!(written.dtype(), MlxDtype::Float32);

            let branch = eval_f32(read.branch_input());
            let gates = eval_f32(read.write_gate());
            let out = eval_f32(&written);
            for token in 0..tokens {
                let x = &input[token * width..(token + 1) * width];
                let expected = host_read(&host, x);
                let label = format!("C={streams} token {token}");
                let hidden_range = token * hidden..(token + 1) * hidden;
                assert_close(
                    &branch[hidden_range.clone()],
                    &expected.branch,
                    TOL,
                    &format!("{label} branch input"),
                );
                let write_gate = expected.write_gate.expect("block write gate");
                assert_close(
                    &gates[token * streams..(token + 1) * streams],
                    &write_gate,
                    TOL,
                    &format!("{label} write gate"),
                );
                assert_close(
                    &out[token * width..(token + 1) * width],
                    &host_write(x, &write_gate, &branch_output[hidden_range]),
                    TOL,
                    &format!("{label} write"),
                );
            }
        }
    }

    #[test]
    fn final_mixer_collapse_matches_scalar_reference() {
        let (batch, seq, hidden, rank) = (1usize, 4usize, 6usize, 4usize);
        for streams in [1usize, 4] {
            let host = host_weights(streams, hidden, rank, false);
            let mixer = build(&host);
            assert!(mixer.is_final_mixer());
            let width = streams * hidden;
            let input = packed_input(batch * seq, streams, hidden);
            let packed = array_f32(&input, &[batch as i32, seq as i32, width as i32]);

            let collapsed = shared_collapse(&mixer, &packed);
            assert_eq!(collapsed.shape(), [batch as i32, seq as i32, hidden as i32]);
            let out = eval_f32(&collapsed);
            for token in 0..batch * seq {
                let expected = host_read(&host, &input[token * width..(token + 1) * width]);
                assert!(expected.write_gate.is_none());
                assert_close(
                    &out[token * hidden..(token + 1) * hidden],
                    &expected.branch,
                    TOL,
                    &format!("C={streams} token {token} collapse"),
                );
            }
        }
    }

    #[test]
    fn expand_copies_embedding_into_every_stream_then_collapses() {
        let (batch, seq, hidden, rank) = (2usize, 2usize, 5usize, 3usize);
        for streams in [1usize, 4] {
            let layout = Qwen4ExpStreamLayout::new(streams, hidden).expect("layout");
            let embedding = wave(batch * seq * hidden, 0.83, 0.9, 0.1);
            let embedding_shape = [batch as i32, seq as i32, hidden as i32];
            let expanded = layout
                .expand(&array_f32(&embedding, &embedding_shape))
                .expect("expand");
            let width = streams * hidden;
            assert_eq!(expanded.shape(), [batch as i32, seq as i32, width as i32]);
            let packed = eval_f32(&expanded);
            for token in 0..batch * seq {
                for index in 0..width {
                    assert_eq!(
                        packed[token * width + index],
                        embedding[token * hidden + index % hidden],
                        "C={streams} token {token} index {index}"
                    );
                }
            }

            let host = host_weights(streams, hidden, rank, false);
            let out = eval_f32(&shared_collapse(&build(&host), &expanded));
            for token in 0..batch * seq {
                let expected = host_read(&host, &packed[token * width..(token + 1) * width]);
                assert_close(
                    &out[token * hidden..(token + 1) * hidden],
                    &expected.branch,
                    TOL,
                    &format!("C={streams} token {token} expand-collapse"),
                );
            }
        }
    }

    #[test]
    fn read_retains_unmodified_input_after_source_handle_drops() {
        let (seq, streams, hidden, rank) = (3usize, 4usize, 5usize, 3usize);
        let width = streams * hidden;
        let host = host_weights(streams, hidden, rank, true);
        let residual = build(&host);
        let input = packed_input(seq, streams, hidden);
        let packed = array_f32(&input, &[1, seq as i32, width as i32]);
        let read = shared_read(&residual, &packed);
        drop(packed);

        let branch_shape = [1, seq as i32, hidden as i32];
        let branch_a = wave(seq * hidden, 0.47, 0.8, 0.2);
        let branch_b = wave(seq * hidden, 0.91, 0.6, -0.3);
        let zero_branch = vec![0.0; seq * hidden];
        let first = read
            .write(&array_f32(&branch_a, &branch_shape))
            .expect("first write");
        let zero = read
            .write(&array_f32(&zero_branch, &branch_shape))
            .expect("zero write");
        let first_values = eval_f32(&first);
        assert_eq!(eval_f32(&zero), input);
        assert_eq!(eval_f32(read.residual()), input);

        let second = read
            .write(&array_f32(&branch_b, &branch_shape))
            .expect("second write");
        let second_values = eval_f32(&second);
        assert_eq!(first.data_f32(), first_values.as_slice());
        assert_eq!(eval_f32(read.residual()), input);

        for (branch, values) in [(&branch_a, &first_values), (&branch_b, &second_values)] {
            for token in 0..seq {
                let x = &input[token * width..(token + 1) * width];
                let write_gate = host_read(&host, x).write_gate.expect("write gate");
                assert_close(
                    &values[token * width..(token + 1) * width],
                    &host_write(
                        x,
                        &write_gate,
                        &branch[token * hidden..(token + 1) * hidden],
                    ),
                    TOL,
                    &format!("retained write token {token}"),
                );
            }
        }
    }

    #[test]
    fn quantized_projections_match_dequantized_reference() {
        let (seq, streams, hidden, rank) = (2usize, 4usize, 8usize, 32usize);
        let (group_size, bits) = (32, 8);
        let width = streams * hidden;
        let mut host = host_weights(streams, hidden, rank, false);
        let quantize_host = |data: &mut Vec<f32>, out: usize, input: usize| {
            let source = array_f32(data, &[out as i32, input as i32]);
            let q = quantize(
                &source,
                Some(group_size),
                Some(bits),
                MlxQuantizationMode::Affine,
                None,
                None,
            );
            // The reference uses the effective (dequantized) weights.
            let effective = dequantize(
                &q[0],
                &q[1],
                Some(&q[2]),
                Some(group_size),
                Some(bits),
                None,
            );
            *data = eval_f32(&effective);
            let mut weight =
                QuantizedWeight::new(q[0].clone(), Some(q[1].clone()), Some(q[2].clone()));
            weight.group_size = group_size;
            weight.bits = bits;
            weight
        };
        host.down = wave(rank * width, 0.53, 0.2, 0.02);
        host.up = wave(width * rank, 0.71, 0.15, -0.02);
        let mut inject = wave(streams * width, 0.29, 0.3, 0.05);
        let read_down = quantize_host(&mut host.down, rank, width);
        let read_up = quantize_host(&mut host.up, width, rank);
        let write_inject = quantize_host(&mut inject, streams, width);
        host.inject = Some(inject);
        let residual = try_build(&host, |weights| {
            weights.read_down = read_down;
            weights.read_up = read_up;
            weights.write_inject = Some(write_inject);
        })
        .expect("quantized residual");

        let input = packed_input(seq, streams, hidden);
        let read = shared_read(
            &residual,
            &array_f32(&input, &[1, seq as i32, width as i32]),
        );
        let branch = eval_f32(read.branch_input());
        let gates = eval_f32(read.write_gate());
        for token in 0..seq {
            let expected = host_read(&host, &input[token * width..(token + 1) * width]);
            assert_close(
                &branch[token * hidden..(token + 1) * hidden],
                &expected.branch,
                2e-4,
                "quantized branch input",
            );
            assert_close(
                &gates[token * streams..(token + 1) * streams],
                &expected.write_gate.expect("write gate"),
                2e-4,
                "quantized write gate",
            );
        }
    }

    #[test]
    fn bfloat16_residual_keeps_its_dtype() {
        let (seq, streams, hidden, rank) = (2usize, 4usize, 5usize, 3usize);
        let width = streams * hidden;
        let host = host_weights(streams, hidden, rank, true);
        let bf16 = |array: &MlxArray| astype(array, MlxDtype::Bfloat16, None);
        let residual = try_build(&host, |weights| {
            weights.norm_gain = bf16(&weights.norm_gain);
            let projections = [&mut weights.read_down, &mut weights.read_up];
            for projection in projections.into_iter().chain(weights.write_inject.as_mut()) {
                projection.weight = bf16(&projection.weight);
            }
        })
        .expect("bf16 residual");

        let input = packed_input(seq, streams, hidden);
        let packed = bf16(&array_f32(&input, &[1, seq as i32, width as i32]));
        let read = shared_read(&residual, &packed);
        let branch_values = wave(seq * hidden, 0.61, 0.7, -0.05);
        let branch_output = bf16(&array_f32(&branch_values, &[1, seq as i32, hidden as i32]));
        let written = read.write(&branch_output).expect("write");
        assert_eq!(read.branch_input().dtype(), MlxDtype::Bfloat16);
        assert_eq!(read.write_gate().dtype(), MlxDtype::Bfloat16);
        assert_eq!(written.dtype(), MlxDtype::Bfloat16);

        // bf16 keeps 7 mantissa bits: this guards gross errors, not rounding.
        let branch = eval_f32(&astype(read.branch_input(), MlxDtype::Float32, None));
        for token in 0..seq {
            let expected = host_read(&host, &input[token * width..(token + 1) * width]);
            let actual = &branch[token * hidden..(token + 1) * hidden];
            for (index, (a, e)) in actual.iter().zip(&expected.branch).enumerate() {
                let diff = (f64::from(*a) - e).abs();
                assert!(diff < 0.1, "bf16 token {token}[{index}]: {a} vs {e}");
            }
        }
    }

    #[test]
    fn construction_and_calls_reject_invalid_shapes_and_roles() {
        assert!(matches!(
            Qwen4ExpStreamLayout::new(0, 4),
            Err(Qwen4ExpResidualError::InvalidGeometry { .. })
        ));
        assert!(matches!(
            Qwen4ExpStreamLayout::new(4, usize::MAX),
            Err(Qwen4ExpResidualError::InvalidGeometry { .. })
        ));

        let (streams, hidden, rank) = (4usize, 3usize, 2usize);
        let width = streams * hidden;
        let host = host_weights(streams, hidden, rank, true);
        let is_shape_error = |result: Result<Qwen4ExpGatedResidual>, name: &str| {
            matches!(
                result,
                Err(Qwen4ExpResidualError::TensorShape { tensor, .. }) if tensor == name
            )
        };

        assert!(is_shape_error(
            try_build(&host, |w| {
                w.norm_gain = array_f32(&host.gain[..hidden], &[hidden as i32]);
            }),
            "norm gain"
        ));
        assert!(is_shape_error(
            try_build(&host, |w| w.read_down =
                dense(&host.down[..width], 1, width)),
            "read_down"
        ));
        assert!(is_shape_error(
            try_build(&host, |w| w.read_up = dense(&host.down, rank, width)),
            "read_up"
        ));
        assert!(is_shape_error(
            try_build(&host, |w| w.write_inject =
                Some(dense(&host.gain, 1, width))),
            "write_inject"
        ));
        assert!(matches!(
            try_build(&host, |w| {
                w.read_down.linear_bias = Some(array_f32(&[0.0; 2], &[2]));
            }),
            Err(Qwen4ExpResidualError::UnexpectedLinearBias {
                tensor: "read_down"
            })
        ));
        // Quantized scales must imply the logical input width.
        assert!(is_shape_error(
            try_build(&host, |w| {
                let bits = [0u8; 8];
                let packed =
                    MlxArray::from_raw_data(bits.as_ptr(), 8, &[rank as i32, 1], MlxDtype::Uint32);
                let scales = array_f32(&[1.0; 2], &[rank as i32, 1]);
                let biases = array_f32(&[0.0; 2], &[rank as i32, 1]);
                w.read_down = QuantizedWeight::new(packed, Some(scales), Some(biases));
                w.read_down.group_size = 32;
                w.read_down.bits = 4;
            }),
            "read_down"
        ));

        let layout = Qwen4ExpStreamLayout::new(streams, hidden).expect("layout");
        assert!(matches!(
            Qwen4ExpGatedResidual::new(layout, 0, EPS, dense_weights(&host)),
            Err(Qwen4ExpResidualError::InvalidLowRank(0))
        ));
        assert!(matches!(
            Qwen4ExpGatedResidual::new(layout, rank, 0.0, dense_weights(&host)),
            Err(Qwen4ExpResidualError::InvalidEps(_))
        ));

        let block = build(&host);
        let packed = array_f32(&packed_input(2, streams, hidden), &[1, 2, width as i32]);
        assert!(matches!(
            block.collapse(&packed, ProjectionBatchPolicy::Shared),
            Err(Qwen4ExpResidualError::RoleMismatch { .. })
        ));
        let narrow = array_f32(&vec![0.5; 2 * hidden], &[1, 2, hidden as i32]);
        assert!(matches!(
            block.read(&narrow, ProjectionBatchPolicy::Shared),
            Err(Qwen4ExpResidualError::TensorShape { .. })
        ));
        let read = shared_read(&block, &packed);
        let wide_branch = array_f32(&vec![0.0; 2 * width], &[1, 2, width as i32]);
        assert!(matches!(
            read.write(&wide_branch),
            Err(Qwen4ExpResidualError::TensorShape {
                tensor: "branch output",
                ..
            })
        ));

        let mixer = try_build(&host, |w| w.write_inject = None).expect("mixer");
        assert!(matches!(
            mixer.read(&packed, ProjectionBatchPolicy::Shared),
            Err(Qwen4ExpResidualError::RoleMismatch { .. })
        ));
        assert!(matches!(
            layout.expand(&packed),
            Err(Qwen4ExpResidualError::TensorShape {
                tensor: "embedding",
                ..
            })
        ));
    }
}
