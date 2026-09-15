//! Flash Next GDN branch with immutable recurrent and convolution state.
//!
//! The portable MLX recurrence establishes a numerical baseline. The Qwen 3.5
//! fused path has different Q/K normalization and output-gate contracts.

use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, concatenate, divide, exp, log1p, maximum, minimum, multiply,
    negative, power, reshape, rms_norm, sigmoid, slice, subtract, sum_axis, take, zeros,
};

use super::qwen4_exp_residual::validate_projection;
use super::utils::{ProjectionBatchPolicy, qw_with_policy};
use crate::linear_attention_ops::{
    linear_attention_conv1d_pre_activation, split_linear_attention_qkv,
};
use crate::model::LinearAttentionConfig;
use crate::weights::QuantizedWeight;

pub(crate) struct Qwen4ExpGdnWeights {
    pub qkv: QuantizedWeight,
    pub gate: QuantizedWeight,
    pub decay: QuantizedWeight,
    pub beta: QuantizedWeight,
    pub output: QuantizedWeight,
    pub conv: MlxArray,
    pub a_log: MlxArray,
    pub dt_bias: MlxArray,
    pub norm_gain: MlxArray,
}

#[derive(Clone)]
pub(crate) struct Qwen4ExpGdnState {
    pub conv: MlxArray,
    /// AX orientation: [batch, value heads, value dimension, key dimension].
    pub recurrent: MlxArray,
}

pub(crate) struct Qwen4ExpGdn {
    config: LinearAttentionConfig,
    hidden: i32,
    eps: f32,
    weights: Qwen4ExpGdnWeights,
}

fn require_float_shape(name: &str, tensor: &MlxArray, shape: &[i32]) -> Result<(), String> {
    if tensor.shape() != shape
        || !matches!(
            tensor.dtype(),
            MlxDtype::Float32 | MlxDtype::Float16 | MlxDtype::Bfloat16
        )
    {
        return Err(format!(
            "qwen4_exp {name}: expected floating {shape:?}, got {:?} {:?}",
            tensor.shape(),
            tensor.dtype()
        ));
    }
    Ok(())
}

fn qwen4_beta(input: &MlxArray) -> MlxArray {
    // The official sigmoid rounds to the projection dtype before the FP32 recurrence.
    let beta = sigmoid(&astype(input, MlxDtype::Float32, None), None);
    astype(&astype(&beta, input.dtype(), None), MlxDtype::Float32, None)
}

fn qwen4_conv1d(
    cfg: &LinearAttentionConfig,
    qkv: &MlxArray,
    weight: &MlxArray,
    state: Option<&MlxArray>,
) -> (MlxArray, MlxArray) {
    let (convolved, state) = linear_attention_conv1d_pre_activation(cfg, qkv, weight, state);
    // Match the official BF16 activation's single rounding after sigmoid and multiply.
    let activated = mlx_sys::ops::silu(&astype(&convolved, MlxDtype::Float32, None), None);
    (astype(&activated, qkv.dtype(), None), state)
}

impl Qwen4ExpGdn {
    pub(crate) fn new(
        config: LinearAttentionConfig,
        hidden: usize,
        eps: f32,
        weights: Qwen4ExpGdnWeights,
    ) -> Result<Self, String> {
        let dimensions = [
            hidden,
            config.num_key_heads,
            config.num_value_heads,
            config.key_head_dim,
            config.value_head_dim,
            config.conv_kernel_dim,
        ];
        if dimensions.iter().any(|&v| v == 0 || v > i32::MAX as usize)
            || !config.num_value_heads.is_multiple_of(config.num_key_heads)
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err("invalid qwen4_exp GDN geometry or epsilon".into());
        }
        let key = config.num_key_heads.checked_mul(config.key_head_dim);
        let value = config.num_value_heads.checked_mul(config.value_head_dim);
        let conv = key
            .and_then(|k| k.checked_mul(2))
            .and_then(|k| value.and_then(|v| k.checked_add(v)));
        if conv.is_none_or(|v| v > i32::MAX as usize) {
            return Err("qwen4_exp GDN width overflow".into());
        }
        let hidden = hidden as i32;
        for (name, weight, output) in [
            ("GDN qkv", &weights.qkv, config.conv_dim() as i32),
            ("GDN gate", &weights.gate, config.value_dim() as i32),
            ("GDN decay", &weights.decay, config.num_value_heads as i32),
            ("GDN beta", &weights.beta, config.num_value_heads as i32),
        ] {
            validate_projection(name, weight, output, hidden).map_err(|e| e.to_string())?;
        }
        validate_projection(
            "GDN output",
            &weights.output,
            hidden,
            config.value_dim() as i32,
        )
        .map_err(|e| e.to_string())?;
        require_float_shape(
            "GDN convolution",
            &weights.conv,
            &[config.conv_dim() as i32, config.conv_kernel_dim as i32, 1],
        )?;
        require_float_shape(
            "GDN norm",
            &weights.norm_gain,
            &[config.value_head_dim as i32],
        )?;
        for (name, tensor) in [
            ("GDN A_log", &weights.a_log),
            ("GDN dt_bias", &weights.dt_bias),
        ] {
            require_float_shape(name, tensor, &[config.num_value_heads as i32])?;
        }
        Ok(Self {
            config,
            hidden,
            eps,
            weights,
        })
    }

    pub(crate) fn validate_state(
        &self,
        state: &Qwen4ExpGdnState,
        batch: usize,
        dtype: MlxDtype,
    ) -> Result<(), String> {
        let batch = i32::try_from(batch).map_err(|_| "qwen4_exp GDN state batch exceeds i32")?;
        if batch <= 0 {
            return Err("qwen4_exp GDN state batch must be positive".into());
        }
        let cfg = &self.config;
        require_float_shape(
            "GDN conv state",
            &state.conv,
            &[batch, cfg.conv_kernel_dim as i32 - 1, cfg.conv_dim() as i32],
        )?;
        require_float_shape(
            "GDN recurrent state",
            &state.recurrent,
            &[
                batch,
                cfg.num_value_heads as i32,
                cfg.value_head_dim as i32,
                cfg.key_head_dim as i32,
            ],
        )?;
        if state.conv.dtype() != dtype || state.recurrent.dtype() != MlxDtype::Float32 {
            return Err("qwen4_exp GDN state dtype mismatch".into());
        }
        Ok(())
    }

    pub(crate) fn forward(
        &self,
        input: &MlxArray,
        state: Option<&Qwen4ExpGdnState>,
        policy: ProjectionBatchPolicy,
    ) -> Result<(MlxArray, Qwen4ExpGdnState), String> {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != self.hidden {
            return Err(format!("invalid qwen4_exp GDN input shape {shape:?}"));
        }
        require_float_shape("GDN input", input, &shape)?;
        let (batch, seq) = (shape[0], shape[1]);
        let cfg = &self.config;
        let (hv, dk, dv) = (
            cfg.num_value_heads as i32,
            cfg.key_head_dim as i32,
            cfg.value_head_dim as i32,
        );
        let recurrent_shape = [batch, hv, dv, dk];
        if let Some(state) = state {
            self.validate_state(state, batch as usize, input.dtype())?;
        }
        let qkv = qw_with_policy(input, &self.weights.qkv, policy);
        let (convolved, conv) = qwen4_conv1d(cfg, &qkv, &self.weights.conv, state.map(|s| &s.conv));
        let split = split_linear_attention_qkv(cfg, &convolved);
        #[cfg(test)]
        {
            crate::model::qwen4_exp::profiling::dump("gdn_query_input", &[&split.q]);
            crate::model::qwen4_exp::profiling::dump("gdn_key_input", &[&split.k]);
            crate::model::qwen4_exp::profiling::dump("gdn_value_input", &[&split.v]);
        }
        let q = divide(
            &qwen4_l2(&split.q),
            &cached_scalar((dk as f32).sqrt(), MlxDtype::Float32),
            None,
        );
        let k = qwen4_l2(&split.k);
        let v = astype(&split.v, MlxDtype::Float32, None);
        let a = add(
            &astype(
                &qw_with_policy(input, &self.weights.decay, policy),
                MlxDtype::Float32,
                None,
            ),
            &astype(&self.weights.dt_bias, MlxDtype::Float32, None),
            None,
        );
        let softplus = add(
            &maximum(&a, &cached_scalar(0.0, MlxDtype::Float32), None),
            &log1p(&exp(&minimum(&a, &negative(&a, None), None), None), None),
            None,
        );
        let log_decay = negative(
            &multiply(
                &exp(&astype(&self.weights.a_log, MlxDtype::Float32, None), None),
                &softplus,
                None,
            ),
            None,
        );
        let decay = exp(&log_decay, None);
        let beta = qwen4_beta(&qw_with_policy(input, &self.weights.beta, policy));
        let repeat = hv / cfg.num_key_heads as i32;
        let head_ids: Vec<i32> = (0..hv).map(|h| h / repeat).collect();
        let ids = MlxArray::from_raw_data(
            head_ids.as_ptr().cast(),
            std::mem::size_of_val(head_ids.as_slice()),
            &[hv],
            MlxDtype::Int32,
        );
        let q = take(&q, &ids, 2, None);
        let k = take(&k, &ids, 2, None);
        let mut recurrent = state.map_or_else(
            || zeros(&recurrent_shape, MlxDtype::Float32, None),
            |s| s.recurrent.clone(),
        );
        #[cfg(test)]
        {
            crate::model::qwen4_exp::profiling::dump("gdn_log_decay", &[&log_decay]);
            crate::model::qwen4_exp::profiling::dump("gdn_beta", &[&beta]);
            crate::model::qwen4_exp::profiling::dump("gdn_initial_state", &[&recurrent]);
        }
        let native = if seq == 1 && super::qwen4_exp_gdn_metal::enabled() {
            super::qwen4_exp_gdn_metal::singleton(&q, &k, &v, &decay, &beta, &recurrent)?
        } else {
            None
        };
        let (output, recurrent) = if let Some(output) = native {
            output
        } else {
            let mut outputs = Vec::with_capacity(seq as usize);
            for token in 0..seq {
                let row = |array: &MlxArray, width: i32| {
                    reshape(
                        &slice(
                            array,
                            &[0, token, 0, 0],
                            &[batch, token + 1, hv, width],
                            &[1, 1, 1, 1],
                            None,
                        ),
                        &[batch, hv, width],
                        None,
                    )
                };
                let scalar_row = |array: &MlxArray| {
                    reshape(
                        &slice(
                            array,
                            &[0, token, 0],
                            &[batch, token + 1, hv],
                            &[1, 1, 1],
                            None,
                        ),
                        &[batch, hv, 1, 1],
                        None,
                    )
                };
                let qr = reshape(&row(&q, dk), &[batch, hv, 1, dk], None);
                let kr = reshape(&row(&k, dk), &[batch, hv, 1, dk], None);
                let vr = reshape(&row(&v, dv), &[batch, hv, dv, 1], None);
                let decayed = multiply(&recurrent, &scalar_row(&decay), None);
                let prediction = sum_axis(&multiply(&decayed, &kr, None), -1, true, None);
                let correction =
                    multiply(&subtract(&vr, &prediction, None), &scalar_row(&beta), None);
                recurrent = add(&decayed, &multiply(&correction, &kr, None), None);
                outputs.push(reshape(
                    &sum_axis(&multiply(&recurrent, &qr, None), -1, false, None),
                    &[batch, 1, hv, dv],
                    None,
                ));
            }
            let refs: Vec<&MlxArray> = outputs.iter().collect();
            (concatenate(&refs, 1, None), recurrent)
        };
        #[cfg(test)]
        {
            crate::model::qwen4_exp::profiling::dump("gdn_core_output", &[&output]);
            crate::model::qwen4_exp::profiling::dump("gdn_final_state", &[&recurrent]);
        }
        let output = astype(&output, input.dtype(), None);
        let normed = rms_norm(
            &astype(&output, MlxDtype::Float32, None),
            None,
            self.eps,
            None,
        );
        let normed = multiply(
            &astype(&normed, input.dtype(), None),
            &self.weights.norm_gain,
            None,
        );
        let gate = reshape(
            &qw_with_policy(input, &self.weights.gate, policy),
            &[batch, seq, hv, dv],
            None,
        );
        let gated = multiply(
            &astype(&normed, MlxDtype::Float32, None),
            &sigmoid(&astype(&gate, MlxDtype::Float32, None), None),
            None,
        );
        let gated = astype(
            &reshape(&gated, &[batch, seq, cfg.value_dim() as i32], None),
            input.dtype(),
            None,
        );
        #[cfg(test)]
        {
            crate::model::qwen4_exp::profiling::dump("gdn_norm_gain", &[&self.weights.norm_gain]);
            crate::model::qwen4_exp::profiling::dump("gdn_gate", &[&gate]);
            crate::model::qwen4_exp::profiling::dump("gdn_gated_output", &[&gated]);
        }
        Ok((
            qw_with_policy(&gated, &self.weights.output, policy),
            Qwen4ExpGdnState { conv, recurrent },
        ))
    }
}

fn qwen4_l2(input: &MlxArray) -> MlxArray {
    let input = astype(input, MlxDtype::Float32, None);
    let squared = sum_axis(&multiply(&input, &input, None), -1, true, None);
    multiply(
        &input,
        &power(
            &add(&squared, &cached_scalar(1e-6, MlxDtype::Float32), None),
            &cached_scalar(-0.5, MlxDtype::Float32),
            None,
        ),
        None,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use mlx_sys::{contiguous, eval, transpose};
    use serde_json::Value;

    fn values(value: &Value) -> Vec<f32> {
        match value {
            Value::Array(items) => items.iter().flat_map(values).collect(),
            _ => vec![value.as_f64().unwrap() as f32],
        }
    }

    fn array(value: &Value, shape: &[i32]) -> MlxArray {
        let data = values(value);
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            shape,
            MlxDtype::Float32,
        )
    }

    fn oracle() -> (Value, Qwen4ExpGdn) {
        let fixture: Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/flash_next/gdn.json"))
                .unwrap();
        let w = &fixture["weights"];
        let dense = |name: &str, output: i32, input: i32| {
            QuantizedWeight::new(array(&w[name], &[output, input]), None, None)
        };
        let weights = Qwen4ExpGdnWeights {
            qkv: dense("in_proj_qkv.weight", 24, 16),
            gate: dense("in_proj_z.weight", 8, 16),
            decay: dense("in_proj_a.weight", 2, 16),
            beta: dense("in_proj_b.weight", 2, 16),
            output: dense("out_proj.weight", 16, 8),
            conv: transpose(&array(&w["conv1d.weight"], &[24, 1, 4]), &[0, 2, 1], None),
            a_log: array(&w["A_log"], &[2]),
            dt_bias: array(&w["dt_bias"], &[2]),
            norm_gain: array(&w["norm.weight"], &[4]),
        };
        let config = LinearAttentionConfig {
            full_attention_interval: 4,
            num_key_heads: 2,
            num_value_heads: 2,
            key_head_dim: 4,
            value_head_dim: 4,
            conv_kernel_dim: 4,
            q_scale: 0.25,
            k_scale: 0.5,
        };
        (
            fixture,
            Qwen4ExpGdn::new(config, 16, 1e-6, weights).unwrap(),
        )
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
            assert!((a - b).abs() < 2e-6, "{label}[{i}]: {a} != {b}");
        }
    }

    #[test]
    fn gdn_bf16_activations_match_official_rounding() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/flash_next/gdn_bf16_activations.json"
        ))
        .unwrap();
        let length = fixture["input"].as_array().unwrap().len() as i32;
        let input = astype(
            &array(&fixture["input"], &[1, length, 1]),
            MlxDtype::Bfloat16,
            None,
        );
        let expected = array(&fixture["beta"], &[1, length, 1]);
        let beta = contiguous(&qwen4_beta(&input), None);
        eval(&[&beta, &expected]);
        assert_eq!(beta.data_f32(), expected.data_f32(), "BF16 beta rounding");
        let config = LinearAttentionConfig {
            full_attention_interval: 4,
            num_key_heads: 1,
            num_value_heads: 1,
            key_head_dim: 1,
            value_head_dim: 1,
            conv_kernel_dim: 4,
            q_scale: 1.0,
            k_scale: 1.0,
        };
        let input = concatenate(&[&input, &input, &input], 2, None);
        let weights = astype(
            &array(
                &serde_json::json!([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                &[3, 4, 1],
            ),
            MlxDtype::Bfloat16,
            None,
        );
        let expected = array(&fixture["silu"], &[1, length, 1]);
        let expected = contiguous(
            &concatenate(&[&expected, &expected, &expected], 2, None),
            None,
        );
        for split in [1, 4, length - 1] {
            let prefix = slice(&input, &[0, 0, 0], &[1, split, 3], &[1, 1, 1], None);
            let suffix = slice(&input, &[0, split, 0], &[1, length, 3], &[1, 1, 1], None);
            let (head, state) = qwen4_conv1d(&config, &prefix, &weights, None);
            let (tail, state) = qwen4_conv1d(&config, &suffix, &weights, Some(&state));
            let output = contiguous(
                &astype(
                    &concatenate(&[&head, &tail], 1, None),
                    MlxDtype::Float32,
                    None,
                ),
                None,
            );
            eval(&[&output, &expected]);
            assert_eq!(
                output.data_f32(),
                expected.data_f32(),
                "BF16 SiLU split {split}"
            );
            close(
                &astype(&state, MlxDtype::Float32, None),
                &astype(
                    &slice(
                        &input,
                        &[0, length - 3, 0],
                        &[1, length, 3],
                        &[1, 1, 1],
                        None,
                    ),
                    MlxDtype::Float32,
                    None,
                ),
                "convolution tail",
            );
        }
    }

    #[test]
    fn gdn_output_and_state_match_pinned_transformers() {
        let (f, module) = oracle();
        let input = array(&f["input"], &[1, 7, 16]);
        let (output, state) = module
            .forward(&input, None, ProjectionBatchPolicy::Shared)
            .unwrap();
        close(&output, &array(&f["output"], &[1, 7, 16]), "output");
        close(
            &state.recurrent,
            &transpose(
                &array(&f["recurrent_state"], &[1, 2, 4, 4]),
                &[0, 1, 3, 2],
                None,
            ),
            "recurrent",
        );
        let source_conv = transpose(&array(&f["conv_state"], &[1, 24, 4]), &[0, 2, 1], None);
        close(
            &state.conv,
            &slice(&source_conv, &[0, 1, 0], &[1, 4, 24], &[1, 1, 1], None),
            "conv",
        );
        close(
            &qwen4_l2(&array(&f["near_zero_qk"], &[1, 1, 2, 4])),
            &array(&f["normalized_qk"], &[1, 1, 2, 4]),
            "near-zero normalization",
        );
    }

    #[test]
    fn gdn_chunk_boundaries_forks_and_rejected_calls_preserve_state() {
        let (f, module) = oracle();
        let input = array(&f["input"], &[1, 7, 16]);
        let (whole, final_state) = module
            .forward(&input, None, ProjectionBatchPolicy::Shared)
            .unwrap();
        for split in 1..7 {
            let prefix = slice(&input, &[0, 0, 0], &[1, split, 16], &[1, 1, 1], None);
            let suffix = slice(&input, &[0, split, 0], &[1, 7, 16], &[1, 1, 1], None);
            let (head, state) = module
                .forward(&prefix, None, ProjectionBatchPolicy::Shared)
                .unwrap();
            let fork = state.clone();
            assert!(
                module
                    .forward(
                        &zeros(&[1, 2, 15], MlxDtype::Float32, None),
                        Some(&state),
                        ProjectionBatchPolicy::Shared
                    )
                    .is_err()
            );
            let (tail, next) = module
                .forward(&suffix, Some(&state), ProjectionBatchPolicy::Shared)
                .unwrap();
            close(&concatenate(&[&head, &tail], 1, None), &whole, "chunked");
            close(&next.recurrent, &final_state.recurrent, "chunk state");
            close(&next.conv, &final_state.conv, "chunk conv");
            let (repeated, repeated_state) = module
                .forward(&suffix, Some(&fork), ProjectionBatchPolicy::Shared)
                .unwrap();
            close(&repeated, &tail, "fork");
            close(&repeated_state.recurrent, &next.recurrent, "fork state");
        }
    }
}
