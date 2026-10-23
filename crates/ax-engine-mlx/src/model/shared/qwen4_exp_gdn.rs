//! Flash Next GDN branch with immutable recurrent and convolution state.
//!
//! The portable MLX recurrence establishes a numerical baseline. The Qwen 3.5
//! fused path has different Q/K normalization and output-gate contracts.

use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    MlxArray, MlxClosure, MlxDtype, add, astype, divide, exp, log1p, maximum, minimum, multiply,
    negative, power, reshape, rms_norm, sigmoid, sum_axis, zeros,
};

use super::qwen4_exp_fused::{self, Stretch};
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

/// Gated-RMSNorm output activation, selected by the manifest `output_gate_type`.
///
/// The reference resolves this from `output_gate_type or hidden_act` and
/// defaults to silu when neither is set. The manifest does not carry a general
/// `hidden_act`, so an absent `output_gate_type` falls back to silu here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GdnGateActivation {
    Sigmoid,
    Silu,
}

impl GdnGateActivation {
    /// Map a manifest `output_gate_type` string to an activation. Absent or
    /// `"silu"` selects silu; `"sigmoid"` selects sigmoid; any other value
    /// fails closed so a future variant cannot silently gate with the wrong
    /// output.
    pub(crate) fn from_name(name: Option<&str>) -> Result<Self, String> {
        match name {
            None | Some("silu") => Ok(Self::Silu),
            Some("sigmoid") => Ok(Self::Sigmoid),
            Some(other) => Err(format!(
                "qwen4_exp output_gate_type must be \"sigmoid\" or \"silu\", got {other:?}"
            )),
        }
    }

    /// Apply the activation in the same FP32 accumulate-then-cast pattern the
    /// gated-RMSNorm already uses, matching the reference `ACT2FN`.
    fn apply(&self, gate: &MlxArray) -> MlxArray {
        let gate = astype(gate, MlxDtype::Float32, None);
        match self {
            Self::Sigmoid => sigmoid(&gate, None),
            Self::Silu => multiply(&gate, &sigmoid(&gate, None), None),
        }
    }
}

pub(crate) struct Qwen4ExpGdn {
    config: LinearAttentionConfig,
    hidden: i32,
    eps: f32,
    activation: GdnGateActivation,
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

fn qwen4_verifier_projection(
    input: &MlxArray,
    weight: &QuantizedWeight,
    policy: ProjectionBatchPolicy,
    verifier_policy: ProjectionBatchPolicy,
) -> MlxArray {
    // MXFP4 batch rounding can alter both conv state and the output residual.
    let qkv_policy = if weight.mlx_quantization_mode() == mlx_sys::MlxQuantizationMode::Mxfp4 {
        verifier_policy
    } else {
        policy
    };
    qw_with_policy(input, weight, qkv_policy)
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

/// Closure body of the beta stretch, with the projection as its only input.
fn beta_closure() -> Option<MlxClosure> {
    qwen4_exp_fused::compile(|inputs| vec![qwen4_beta(&inputs.get(0))])
}

/// Closure body of the post-recurrence stretch: gated RMS norm, output gate and
/// the flatten back to `[batch, seq, value_dim]`.
///
/// Inputs: the recurrence output, the gate projection before its reshape, and
/// `norm_gain`. Only plain `Copy` values are captured, and `batch`/`seq` come
/// from the input shape, so nothing shape-specific has to be passed in.
fn post_closure(
    hv: i32,
    dv: i32,
    value_dim: i32,
    eps: f32,
    activation: GdnGateActivation,
    dtype: MlxDtype,
) -> Option<MlxClosure> {
    qwen4_exp_fused::compile(move |inputs| {
        let (output, gate_projection, norm_gain) = (inputs.get(0), inputs.get(1), inputs.get(2));
        let shape = output.shape();
        let (batch, seq) = (shape[0], shape[1]);
        let output = astype(&output, dtype, None);
        let normed = rms_norm(&astype(&output, MlxDtype::Float32, None), None, eps, None);
        let normed = multiply(&astype(&normed, dtype, None), &norm_gain, None);
        let gate = reshape(&gate_projection, &[batch, seq, hv, dv], None);
        let gated = multiply(
            &astype(&normed, MlxDtype::Float32, None),
            &activation.apply(&gate),
            None,
        );
        vec![astype(
            &reshape(&gated, &[batch, seq, value_dim], None),
            dtype,
            None,
        )]
    })
}

impl Qwen4ExpGdn {
    pub(crate) fn new(
        config: LinearAttentionConfig,
        hidden: usize,
        eps: f32,
        activation: GdnGateActivation,
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
            activation,
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
        verifier_policy: ProjectionBatchPolicy,
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
        let qkv = qwen4_verifier_projection(input, &self.weights.qkv, policy, verifier_policy);
        let (convolved, conv) = qwen4_conv1d(cfg, &qkv, &self.weights.conv, state.map(|s| &s.conv));
        let split = split_linear_attention_qkv(cfg, &convolved);
        #[cfg(test)]
        {
            crate::model::qwen4_exp::profiling::dump("gdn_query_input", &[&split.q]);
            crate::model::qwen4_exp::profiling::dump("gdn_key_input", &[&split.k]);
            crate::model::qwen4_exp::profiling::dump("gdn_value_input", &[&split.v]);
        }
        // One registry key per geometry; the dtype stays out because MLX
        // specialises per dtype behind the closure. Every stretch below falls
        // back to its eager chain independently.
        let key = [hv as i64, dk as i64, dv as i64, 0];
        // The decay chain and the q/k L2 normalisations stay eager: a bisect on
        // a real pack showed that compiling them changes the recurrent state
        // bits, so they must not exist as compiled stretches.
        let q = divide(
            &qwen4_l2(&split.q),
            &cached_scalar((dk as f32).sqrt(), MlxDtype::Float32),
            None,
        );
        let k = qwen4_l2(&split.k);
        let v = astype(&split.v, MlxDtype::Float32, None);
        let a = add(
            &astype(
                &qwen4_verifier_projection(input, &self.weights.decay, policy, verifier_policy),
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
        let beta_projection =
            qwen4_verifier_projection(input, &self.weights.beta, policy, verifier_policy);
        let beta = match qwen4_exp_fused::apply(
            Stretch::GdnBeta,
            seq,
            key,
            beta_closure,
            &[&beta_projection],
        ) {
            Some(beta) => beta,
            None => qwen4_beta(&beta_projection),
        };
        let recurrent = state.map_or_else(
            || zeros(&recurrent_shape, MlxDtype::Float32, None),
            |s| s.recurrent.clone(),
        );
        #[cfg(test)]
        {
            crate::model::qwen4_exp::profiling::dump("gdn_log_decay", &[&log_decay]);
            crate::model::qwen4_exp::profiling::dump("gdn_beta", &[&beta]);
            crate::model::qwen4_exp::profiling::dump("gdn_initial_state", &[&recurrent]);
        }
        let prefill = if crate::flash_next_gdn::enabled() {
            crate::flash_next_gdn::try_mlx_prefill(&q, &k, &v, &decay, &beta, &recurrent)
        } else {
            None
        };
        let (output, recurrent) = if let Some(prefill) = prefill {
            prefill
        } else {
            let native = if seq == 1 && super::qwen4_exp_gdn_metal::enabled() {
                let (q, k) = crate::flash_next_gdn::expand_heads(&q, &k, hv);
                super::qwen4_exp_gdn_metal::singleton(&q, &k, &v, &decay, &beta, &recurrent)?
            } else {
                None
            };
            match native {
                Some(output) => output,
                None => crate::flash_next_gdn::portable_recurrence(
                    &q, &k, &v, &decay, &beta, &recurrent,
                )?,
            }
        };
        #[cfg(test)]
        crate::model::qwen4_exp::profiling::mark("gdn_chunked", &[&output, &recurrent]);
        #[cfg(test)]
        {
            crate::model::qwen4_exp::profiling::dump("gdn_core_output", &[&output]);
            crate::model::qwen4_exp::profiling::dump("gdn_final_state", &[&recurrent]);
        }
        let activation = self.activation;
        let eps = self.eps;
        let value_dim = cfg.value_dim() as i32;
        let output_dtype = input.dtype();
        let activation_code: i64 = match activation {
            GdnGateActivation::Sigmoid => 0,
            GdnGateActivation::Silu => 1,
        };
        // The post-recurrence stretch also keys on the epsilon and the gate
        // activation, because both are baked into the compiled graph.
        // The compiled body bakes in the output dtype, and MLX re-traces the
        // same body per input dtype, so the dtype must select the closure.
        let dtype_code: i64 = match output_dtype {
            MlxDtype::Float32 => 0,
            MlxDtype::Float16 => 1,
            MlxDtype::Bfloat16 => 2,
            _ => 3,
        };
        let post_key = [
            hv as i64,
            dv as i64,
            eps.to_bits() as i64,
            activation_code | (dtype_code << 8),
        ];
        let gate_projection =
            qwen4_verifier_projection(input, &self.weights.gate, policy, verifier_policy);
        let gated = match qwen4_exp_fused::apply(
            Stretch::GdnPost,
            seq,
            post_key,
            || post_closure(hv, dv, value_dim, eps, activation, output_dtype),
            &[&output, &gate_projection, &self.weights.norm_gain],
        ) {
            Some(gated) => gated,
            None => {
                let output = astype(&output, output_dtype, None);
                let normed = rms_norm(&astype(&output, MlxDtype::Float32, None), None, eps, None);
                let normed = multiply(
                    &astype(&normed, output_dtype, None),
                    &self.weights.norm_gain,
                    None,
                );
                let gate = reshape(&gate_projection, &[batch, seq, hv, dv], None);
                let gated = multiply(
                    &astype(&normed, MlxDtype::Float32, None),
                    &activation.apply(&gate),
                    None,
                );
                #[cfg(test)]
                {
                    crate::model::qwen4_exp::profiling::dump(
                        "gdn_norm_gain",
                        &[&self.weights.norm_gain],
                    );
                    crate::model::qwen4_exp::profiling::dump("gdn_gate", &[&gate]);
                }
                astype(
                    &reshape(&gated, &[batch, seq, value_dim], None),
                    output_dtype,
                    None,
                )
            }
        };
        #[cfg(test)]
        crate::model::qwen4_exp::profiling::dump("gdn_gated_output", &[&gated]);
        Ok((
            qwen4_verifier_projection(&gated, &self.weights.output, policy, verifier_policy),
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
    use mlx_sys::{concatenate, contiguous, eval, slice, transpose};
    use serde_json::Value;

    fn synthetic_pack_geometry(dtype: MlxDtype) -> Qwen4ExpGdn {
        synthetic_pack_geometry_with(dtype, GdnGateActivation::Silu)
    }

    fn synthetic_pack_geometry_with(dtype: MlxDtype, activation: GdnGateActivation) -> Qwen4ExpGdn {
        let tensor = |shape: &[i32], phase: usize, scale: f32| {
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let data: Vec<f32> = (0..n)
                .map(|i| (((i + phase) * 37 % 997) as f32 / 997.0 - 0.5) * scale)
                .collect();
            astype(
                &MlxArray::from_raw_data(data.as_ptr().cast(), n * 4, shape, MlxDtype::Float32),
                dtype,
                None,
            )
        };
        let dense = |output, input, phase| {
            QuantizedWeight::new(tensor(&[output, input], phase, 0.04), None, None)
        };
        Qwen4ExpGdn::new(
            LinearAttentionConfig {
                full_attention_interval: 4,
                num_key_heads: 16,
                num_value_heads: 48,
                key_head_dim: 128,
                value_head_dim: 128,
                conv_kernel_dim: 4,
                q_scale: 1.0,
                k_scale: 1.0,
            },
            32,
            1e-6,
            activation,
            Qwen4ExpGdnWeights {
                qkv: dense(10240, 32, 1),
                gate: dense(6144, 32, 7),
                decay: dense(48, 32, 13),
                beta: dense(48, 32, 19),
                output: dense(32, 6144, 23),
                conv: tensor(&[10240, 4, 1], 29, 0.5),
                a_log: astype(&tensor(&[48], 31, 0.4), MlxDtype::Float32, None),
                dt_bias: astype(&tensor(&[48], 37, 0.5), MlxDtype::Float32, None),
                norm_gain: mlx_sys::broadcast_to(
                    &astype(&MlxArray::from_f32(1.0), dtype, None),
                    &[128],
                    None,
                ),
            },
        )
        .unwrap()
    }

    fn bounded(actual: &MlxArray, expected: &MlxArray, tolerance: f64) {
        let actual = contiguous(&astype(actual, MlxDtype::Float32, None), None);
        let expected = contiguous(&astype(expected, MlxDtype::Float32, None), None);
        eval(&[&actual, &expected]);
        assert_eq!(actual.shape(), expected.shape());
        let max_abs =
            actual
                .data_f32()
                .iter()
                .zip(expected.data_f32())
                .fold(0.0_f64, |m, (&a, &b)| {
                    assert!(a.is_finite() && b.is_finite());
                    m.max((f64::from(a) - f64::from(b)).abs())
                });
        assert!(
            max_abs <= tolerance,
            "max_abs={max_abs}, tolerance={tolerance}"
        );
    }

    #[test]
    fn mlx_prefill_full_gdn_preserves_conv_and_singleton_routes() {
        for dtype in [MlxDtype::Float32, MlxDtype::Float16, MlxDtype::Bfloat16] {
            let module = synthetic_pack_geometry(dtype);
            let data: Vec<f32> = (0..2 * 131 * 32)
                .map(|i| ((i * 19 % 991) as f32 / 991.0 - 0.5) * 0.2)
                .collect();
            let input = astype(
                &MlxArray::from_raw_data(
                    data.as_ptr().cast(),
                    data.len() * 4,
                    &[2, 131, 32],
                    MlxDtype::Float32,
                ),
                dtype,
                None,
            );
            let run = |enabled, input: &MlxArray, state| {
                crate::flash_next_gdn::with_mode(enabled, || {
                    module.forward(
                        input,
                        state,
                        ProjectionBatchPolicy::Shared,
                        ProjectionBatchPolicy::RowExact,
                    )
                })
            };
            let (ax, count) = run(false, &input, None);
            assert_eq!(count, 0);
            let (ax_y, ax_state) = ax.unwrap();
            let (mlx, count) = run(true, &input, None);
            assert_eq!(count, 1, "dedicated Flash Next adapter selected");
            let (mlx_y, mlx_state) = mlx.unwrap();
            assert_eq!(mlx_y.dtype(), dtype);
            assert_eq!(mlx_state.recurrent.dtype(), MlxDtype::Float32);
            exact_values(&mlx_state.conv, &ax_state.conv);
            bounded(&mlx_y, &ax_y, 5e-4);
            bounded(&mlx_state.recurrent, &ax_state.recurrent, 5e-5);
            let decode = mlx_sys::slice(&input, &[0, 0, 0], &[2, 1, 32], &[1, 1, 1], None);
            let (off, _) = run(false, &decode, Some(&mlx_state));
            let (on, count) = run(true, &decode, Some(&mlx_state));
            assert_eq!(count, 0, "singleton never selects prefill");
            let (off, off_state) = off.unwrap();
            let (on, on_state) = on.unwrap();
            exact_values(&on, &off);
            exact_values(&on_state.conv, &off_state.conv);
            exact_values(&on_state.recurrent, &off_state.recurrent);
            let short = mlx_sys::slice(&input, &[0, 0, 0], &[2, 4, 32], &[1, 1, 1], None);
            let (off, _) = run(false, &short, Some(&mlx_state));
            let (on, count) = run(true, &short, Some(&mlx_state));
            assert_eq!(count, 0, "short MTP verification never selects prefill");
            let (off, off_state) = off.unwrap();
            let (on, on_state) = on.unwrap();
            exact_values(&on, &off);
            exact_values(&on_state.recurrent, &off_state.recurrent);
        }
    }

    /// Run explicitly on the functional test host. Real component weights with
    /// synthetic hidden activations do not qualify whole-model quality or speed.
    #[test]
    #[ignore = "requires an explicit real Flash Next pack on the target test host"]
    fn campaign_mlx_prefill_real_gdn_component() {
        use sha2::{Digest, Sha256};
        let root = std::path::PathBuf::from(std::env::var_os("AX_FLASH_NEXT_PACK_DIR").unwrap());
        let output =
            std::path::PathBuf::from(std::env::var_os("AX_FLASH_NEXT_GDN_REPORT").unwrap());
        let manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
        let weights = crate::weights::qwen4_exp::load(&root, &manifest).unwrap();
        let crate::weights::qwen4_exp::Qwen4ExpAttentionBranch::Gdn(module) =
            &weights.layers[0].attention
        else {
            panic!("the campaign requires layer-zero GDN");
        };
        let hidden = module.hidden;
        let dtype = module.weights.conv.dtype();
        let data: Vec<f32> = (0..131 * hidden)
            .map(|i| ((i * 19 % 991) as f32 / 991.0 - 0.5) * 0.2)
            .collect();
        let input = astype(
            &MlxArray::from_raw_data(
                data.as_ptr().cast(),
                data.len() * 4,
                &[1, 131, hidden],
                MlxDtype::Float32,
            ),
            dtype,
            None,
        );
        let run = |enabled, input: &MlxArray, state| {
            crate::flash_next_gdn::with_mode(enabled, || {
                module.forward(
                    input,
                    state,
                    ProjectionBatchPolicy::Shared,
                    ProjectionBatchPolicy::RowExact,
                )
            })
        };
        let (ax, ax_dispatches) = run(false, &input, None);
        let (mlx, mlx_dispatches) = run(true, &input, None);
        let (ax_y, ax_state) = ax.unwrap();
        let (mlx_y, mlx_state) = mlx.unwrap();
        let metrics = |a: &MlxArray, b: &MlxArray| {
            assert_eq!(a.shape(), b.shape());
            let a = contiguous(&astype(a, MlxDtype::Float32, None), None);
            let b = contiguous(&astype(b, MlxDtype::Float32, None), None);
            eval(&[&a, &b]);
            let finite = a
                .data_f32()
                .iter()
                .chain(b.data_f32())
                .all(|v| v.is_finite());
            let max_abs = a
                .data_f32()
                .iter()
                .zip(b.data_f32())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .fold(0.0_f64, f64::max);
            let squared_error: f64 = a
                .data_f32()
                .iter()
                .zip(b.data_f32())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum();
            let squared_reference: f64 = b.data_f32().iter().map(|&b| f64::from(b).powi(2)).sum();
            serde_json::json!({"finite":finite,"max_abs":max_abs,"relative_l2":(squared_error / squared_reference.max(1e-30)).sqrt()})
        };
        let output_error = metrics(&mlx_y, &ax_y);
        let state_error = metrics(&mlx_state.recurrent, &ax_state.recurrent);
        let conv_error = metrics(&mlx_state.conv, &ax_state.conv);
        let decode = slice(&input, &[0, 0, 0], &[1, 1, hidden], &[1, 1, 1], None);
        let (off, _) = run(false, &decode, Some(&mlx_state));
        let (on, decode_dispatches) = run(true, &decode, Some(&mlx_state));
        let (off, off_state) = off.unwrap();
        let (on, on_state) = on.unwrap();
        let decode_error = metrics(&on, &off);
        let decode_state_error = metrics(&on_state.recurrent, &off_state.recurrent);
        let tensor_hash = |array: &MlxArray| {
            let array = contiguous(array, None);
            eval(&[&array]);
            let mut digest = Sha256::new();
            if array.dtype() == MlxDtype::Uint32 {
                for value in array.data_u32() {
                    digest.update(value.to_le_bytes());
                }
            } else {
                let array = contiguous(&astype(&array, MlxDtype::Float32, None), None);
                eval(&[&array]);
                for value in array.data_f32() {
                    digest.update(value.to_le_bytes());
                }
            }
            format!("{:x}", digest.finalize())
        };
        let mut tensors = serde_json::Map::new();
        for (name, projection) in [
            ("qkv", &module.weights.qkv),
            ("gate", &module.weights.gate),
            ("decay", &module.weights.decay),
            ("beta", &module.weights.beta),
            ("output", &module.weights.output),
        ] {
            tensors.insert(
                format!("{name}.weight"),
                serde_json::json!(tensor_hash(&projection.weight)),
            );
            if let Some(scales) = &projection.scales {
                tensors.insert(
                    format!("{name}.scales"),
                    serde_json::json!(tensor_hash(scales)),
                );
            }
            if let Some(biases) = &projection.biases {
                tensors.insert(
                    format!("{name}.biases"),
                    serde_json::json!(tensor_hash(biases)),
                );
            }
        }
        for (name, tensor) in [
            ("conv", &module.weights.conv),
            ("a_log", &module.weights.a_log),
            ("dt_bias", &module.weights.dt_bias),
            ("norm_gain", &module.weights.norm_gain),
        ] {
            tensors.insert(name.into(), serde_json::json!(tensor_hash(tensor)));
        }
        // Fixed development bounds, not model-quality or certification thresholds.
        let pass = ax_dispatches == 0
            && mlx_dispatches == 1
            && decode_dispatches == 0
            && output_error["finite"] == true
            && state_error["finite"] == true
            && output_error["max_abs"].as_f64().unwrap() <= 0.0625
            && output_error["relative_l2"].as_f64().unwrap() <= 0.002
            && state_error["max_abs"].as_f64().unwrap() <= 0.002
            && state_error["relative_l2"].as_f64().unwrap() <= 0.002
            && conv_error["max_abs"] == 0.0
            && decode_error["max_abs"] == 0.0
            && decode_state_error["max_abs"] == 0.0;
        let report = serde_json::json!({
            "schema":"ax.flash_next.gdn_component.v1", "qualification":false, "release_ready":false,
            "scope":"Layer-zero real GDN weights with synthetic hidden activations; no whole-model quality or performance claim",
            "pack":root.file_name().unwrap().to_string_lossy(), "layer":0, "seq":131,
            "projection_dtype":format!("{dtype:?}"), "mlx_version":mlx_sys::runtime_version().unwrap(),
            "config_sha256":format!("{:x}", Sha256::digest(std::fs::read(root.join("config.json")).unwrap())),
            "index_sha256":format!("{:x}", Sha256::digest(std::fs::read(root.join("model.safetensors.index.json")).unwrap())),
            "tensor_hash_scope":"Loaded uint32 payloads or exact floating values cast to float32, row-contiguous little-endian",
            "loaded_gdn_sha256":tensors, "input_sha256":tensor_hash(&input),
            "sources_sha256":{
                "trunk":format!("{:x}", Sha256::digest(include_str!("qwen4_exp_gdn.rs"))),
                "recurrence":format!("{:x}", Sha256::digest(include_str!("../../flash_next_gdn.rs"))),
            },
            "dispatches":{"ax":ax_dispatches,"mlx":mlx_dispatches,"decode":decode_dispatches},
            "bounds":{"output_max_abs":0.0625,"state_max_abs":0.002,"relative_l2":0.002,"conv_and_same_state_decode_max_abs":0.0},
            "output_error":output_error,"state_error":state_error,"conv_error":conv_error,
            "same_state_decode_error":decode_error,"same_state_decode_state_error":decode_state_error,"correctness_pass":pass,
        });
        std::fs::write(
            output,
            format!("{}\n", serde_json::to_string_pretty(&report).unwrap()),
        )
        .unwrap();
        assert!(
            pass,
            "real GDN comparison failed; evidence retained: {report}"
        );
    }

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

    fn oracle(activation: GdnGateActivation) -> (Value, Qwen4ExpGdn) {
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
            Qwen4ExpGdn::new(config, 16, 1e-6, activation, weights).unwrap(),
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

    fn exact_values(a: &MlxArray, b: &MlxArray) {
        let a = contiguous(&astype(a, MlxDtype::Float32, None), None);
        let b = contiguous(&astype(b, MlxDtype::Float32, None), None);
        eval(&[&a, &b]);
        assert_eq!(a.shape(), b.shape());
        for (index, (actual, expected)) in a.data_f32().iter().zip(b.data_f32()).enumerate() {
            assert_eq!(actual, expected, "projection value {index}");
        }
    }

    fn mxfp4_projection_rows(hidden: i32, output: i32) -> (MlxArray, Vec<MlxArray>) {
        let raw: Vec<f32> = (0..hidden * output)
            .map(|i| ((i * 37 + 17) % 1009 - 504) as f32 / 1024.0)
            .collect();
        let dense = MlxArray::from_raw_data(
            raw.as_ptr().cast(),
            std::mem::size_of_val(raw.as_slice()),
            &[output, hidden],
            MlxDtype::Float32,
        );
        let quantized = mlx_sys::quantize(
            &dense,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
        let mut weight =
            QuantizedWeight::new(quantized[0].clone(), Some(quantized[1].clone()), None);
        weight.mode = "mxfp4".into();
        weight.bits = 4;
        weight.group_size = 32;
        let input_data: Vec<f32> = (0..2 * hidden)
            .map(|i| ((i * 19 + 7) % 997 - 498) as f32 / 256.0)
            .collect();
        let input = astype(
            &MlxArray::from_raw_data(
                input_data.as_ptr().cast(),
                std::mem::size_of_val(input_data.as_slice()),
                &[1, 2, hidden],
                MlxDtype::Float32,
            ),
            MlxDtype::Bfloat16,
            None,
        );
        let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        let shared = ProjectionBatchPolicy::Shared;
        let verifier =
            qwen4_verifier_projection(&input, &weight, shared, ProjectionBatchPolicy::RowExact);
        let direct: Vec<_> = (0..2)
            .map(|i| {
                let row = astype(
                    &MlxArray::from_raw_data(
                        input_data[(i * hidden) as usize..((i + 1) * hidden) as usize]
                            .as_ptr()
                            .cast(),
                        hidden as usize * std::mem::size_of::<f32>(),
                        &[1, 1, hidden],
                        MlxDtype::Float32,
                    ),
                    MlxDtype::Bfloat16,
                    None,
                );
                qw_with_policy(&row, &weight, shared)
            })
            .collect();
        let expected = concatenate(&[&direct[0], &direct[1]], 1, None);
        exact_values(&verifier, &expected);
        let batch = astype(
            &qw_with_policy(&input, &weight, shared),
            MlxDtype::Float32,
            None,
        );
        let single = astype(&expected, MlxDtype::Float32, None);
        eval(&[&batch, &single]);
        let changed = batch
            .data_f32()
            .iter()
            .zip(single.data_f32())
            .filter(|(a, b)| a != b)
            .count();
        eprintln!("MXFP4 BF16 {output}x{hidden} Shared batch/singleton changed values: {changed}");
        exact_values(
            &qwen4_verifier_projection(&input, &weight, shared, shared),
            &qw_with_policy(&input, &weight, shared),
        );
        (verifier, direct)
    }

    #[test]
    fn mxfp4_verifier_output_matches_singletons() {
        mxfp4_projection_rows(6144, 2560);
    }

    #[test]
    fn mxfp4_verifier_qkv_conv_matches_singletons() {
        let output = 10240;
        let (verifier, direct) = mxfp4_projection_rows(2560, output);
        let config = LinearAttentionConfig {
            full_attention_interval: 4,
            num_key_heads: 16,
            num_value_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
            q_scale: 1.0,
            k_scale: 1.0,
        };
        let conv_data: Vec<f32> = (0..output * 4).map(|i| (i % 7 - 3) as f32 / 8.0).collect();
        let conv = astype(
            &MlxArray::from_raw_data(
                conv_data.as_ptr().cast(),
                std::mem::size_of_val(conv_data.as_slice()),
                &[output, 4, 1],
                MlxDtype::Float32,
            ),
            MlxDtype::Bfloat16,
            None,
        );
        let (batch_output, batch_state) = qwen4_conv1d(&config, &verifier, &conv, None);
        let (first, state) = qwen4_conv1d(&config, &direct[0], &conv, None);
        let (second, state) = qwen4_conv1d(&config, &direct[1], &conv, Some(&state));
        exact_values(&batch_state, &state);
        exact_values(&batch_output, &concatenate(&[&first, &second], 1, None));
    }

    #[test]
    fn verifier_projection_preserves_dense_and_affine_policy() {
        let (_, module) = oracle(GdnGateActivation::Sigmoid);
        let dense = QuantizedWeight::new(
            concatenate(
                &[&module.weights.qkv.weight, &module.weights.qkv.weight],
                1,
                None,
            ),
            None,
            None,
        );
        let raw: Vec<f32> = (0..64).map(|i| (i - 16) as f32 / 32.0).collect();
        let input = MlxArray::from_raw_data(
            raw.as_ptr().cast(),
            std::mem::size_of_val(raw.as_slice()),
            &[1, 2, 32],
            MlxDtype::Float32,
        );
        let packed = mlx_sys::quantize(
            &dense.weight,
            Some(32),
            Some(8),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
            None,
        );
        let mut affine = QuantizedWeight::new(
            packed[0].clone(),
            Some(packed[1].clone()),
            Some(packed[2].clone()),
        );
        affine.bits = 8;
        affine.group_size = 32;
        let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        for weight in [&dense, &affine] {
            let actual = qwen4_verifier_projection(
                &input,
                weight,
                ProjectionBatchPolicy::Shared,
                ProjectionBatchPolicy::RowExact,
            );
            let expected = qw_with_policy(&input, weight, ProjectionBatchPolicy::Shared);
            eval(&[&actual, &expected]);
            assert_eq!(actual.data_f32(), expected.data_f32());
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
        let (f, module) = oracle(GdnGateActivation::Sigmoid);
        let input = array(&f["input"], &[1, 7, 16]);
        let (output, state) = module
            .forward(
                &input,
                None,
                ProjectionBatchPolicy::Shared,
                ProjectionBatchPolicy::Shared,
            )
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
    fn gate_activation_matches_reference_math() {
        let gate = array(&serde_json::json!([2.0]), &[1]);
        let sig = contiguous(&GdnGateActivation::Sigmoid.apply(&gate), None);
        let silu = contiguous(&GdnGateActivation::Silu.apply(&gate), None);
        eval(&[&sig, &silu]);
        assert!((sig.data_f32()[0] - 0.880_797).abs() < 1e-6);
        assert!((silu.data_f32()[0] - 1.761_594).abs() < 1e-6);
    }

    #[test]
    fn gate_activation_from_name_resolves_and_rejects() {
        assert_eq!(
            GdnGateActivation::from_name(None),
            Ok(GdnGateActivation::Silu)
        );
        assert_eq!(
            GdnGateActivation::from_name(Some("silu")),
            Ok(GdnGateActivation::Silu)
        );
        assert_eq!(
            GdnGateActivation::from_name(Some("sigmoid")),
            Ok(GdnGateActivation::Sigmoid)
        );
        assert!(GdnGateActivation::from_name(Some("gelu")).is_err());
        assert!(GdnGateActivation::from_name(Some("")).is_err());
    }

    #[test]
    fn gdn_silu_gate_differs_from_sigmoid_gate() {
        let (f, sigmoid_module) = oracle(GdnGateActivation::Sigmoid);
        let (_, silu_module) = oracle(GdnGateActivation::Silu);
        let input = array(&f["input"], &[1, 7, 16]);
        let forward = |module: &Qwen4ExpGdn| {
            module
                .forward(
                    &input,
                    None,
                    ProjectionBatchPolicy::Shared,
                    ProjectionBatchPolicy::Shared,
                )
                .unwrap()
                .0
        };
        let sigmoid_output = contiguous(
            &astype(&forward(&sigmoid_module), MlxDtype::Float32, None),
            None,
        );
        let silu_output = contiguous(
            &astype(&forward(&silu_module), MlxDtype::Float32, None),
            None,
        );
        eval(&[&sigmoid_output, &silu_output]);
        let differs = sigmoid_output
            .data_f32()
            .iter()
            .zip(silu_output.data_f32())
            .any(|(a, b)| (a - b).abs() > 2e-6);
        assert!(differs, "silu gate must differ from sigmoid gate");
    }

    #[test]
    fn gdn_chunk_boundaries_forks_and_rejected_calls_preserve_state() {
        let (f, module) = oracle(GdnGateActivation::Sigmoid);
        let input = array(&f["input"], &[1, 7, 16]);
        let (whole, final_state) = module
            .forward(
                &input,
                None,
                ProjectionBatchPolicy::Shared,
                ProjectionBatchPolicy::Shared,
            )
            .unwrap();
        for split in 1..7 {
            let prefix = slice(&input, &[0, 0, 0], &[1, split, 16], &[1, 1, 1], None);
            let suffix = slice(&input, &[0, split, 0], &[1, 7, 16], &[1, 1, 1], None);
            let (head, state) = module
                .forward(
                    &prefix,
                    None,
                    ProjectionBatchPolicy::Shared,
                    ProjectionBatchPolicy::Shared,
                )
                .unwrap();
            let fork = state.clone();
            assert!(
                module
                    .forward(
                        &zeros(&[1, 2, 15], MlxDtype::Float32, None),
                        Some(&state),
                        ProjectionBatchPolicy::Shared,
                        ProjectionBatchPolicy::Shared,
                    )
                    .is_err()
            );
            let (tail, next) = module
                .forward(
                    &suffix,
                    Some(&state),
                    ProjectionBatchPolicy::Shared,
                    ProjectionBatchPolicy::Shared,
                )
                .unwrap();
            close(&concatenate(&[&head, &tail], 1, None), &whole, "chunked");
            close(&next.recurrent, &final_state.recurrent, "chunk state");
            close(&next.conv, &final_state.conv, "chunk conv");
            let (repeated, repeated_state) = module
                .forward(
                    &suffix,
                    Some(&fork),
                    ProjectionBatchPolicy::Shared,
                    ProjectionBatchPolicy::Shared,
                )
                .unwrap();
            close(&repeated, &tail, "fork");
            close(&repeated_state.recurrent, &next.recurrent, "fork state");
        }
    }

    /// Small deterministic xorshift64 PRNG, so differential inputs are
    /// reproducible without pulling in an external crate.
    struct XorShift64(u64);

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Uniform in `+-amplitude`.
        fn next_f32(&mut self, amplitude: f32) -> f32 {
            let unit = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
            (unit * 2.0 - 1.0) * amplitude
        }
    }

    /// Distinct nonzero PRNG seed per test case.
    fn case_seed(tag: u64) -> u64 {
        tag.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1
    }

    /// `[1, rows, hidden]` of deterministic pseudo-random values uniform in
    /// `+-amplitude`, cast to `dtype`.
    fn deterministic_input(
        hidden: i32,
        rows: i32,
        seed: u64,
        amplitude: f32,
        dtype: MlxDtype,
    ) -> MlxArray {
        let mut rng = XorShift64::new(seed);
        let data: Vec<f32> = (0..rows * hidden)
            .map(|_| rng.next_f32(amplitude))
            .collect();
        astype(
            &MlxArray::from_raw_data(
                data.as_ptr().cast(),
                data.len() * 4,
                &[1, rows, hidden],
                MlxDtype::Float32,
            ),
            dtype,
            None,
        )
    }

    /// Widen to f32 (lossless for every dtype used here) and compare raw bits:
    /// the compiled stretches must match the eager chain bit for bit.
    fn widened_bits(array: &MlxArray) -> Vec<u32> {
        let widened = contiguous(&astype(array, MlxDtype::Float32, None), None);
        eval(&[&widened]);
        widened.data_f32().iter().map(|v| v.to_bits()).collect()
    }

    fn assert_bit_identical(actual: &MlxArray, expected: &MlxArray, label: &str) {
        assert_eq!(actual.shape(), expected.shape(), "{label} shape");
        assert_eq!(widened_bits(actual), widened_bits(expected), "{label} bits");
    }

    /// Forward on the portable recurrence, so the differentials below exercise
    /// exactly the beta and post stretches on top of one fixed core.
    fn run_forward(
        module: &Qwen4ExpGdn,
        input: &MlxArray,
        state: Option<&Qwen4ExpGdnState>,
    ) -> (MlxArray, Qwen4ExpGdnState) {
        crate::flash_next_gdn::with_mode(false, || {
            module.forward(
                input,
                state,
                ProjectionBatchPolicy::Shared,
                ProjectionBatchPolicy::Shared,
            )
        })
        .0
        .unwrap()
    }

    fn prng_dense(
        rng: &mut XorShift64,
        output: usize,
        input: usize,
        amplitude: f32,
        dtype: MlxDtype,
    ) -> QuantizedWeight {
        let data: Vec<f32> = (0..output * input)
            .map(|_| rng.next_f32(amplitude))
            .collect();
        QuantizedWeight::new(
            astype(
                &MlxArray::from_raw_data(
                    data.as_ptr().cast(),
                    data.len() * 4,
                    &[output as i32, input as i32],
                    MlxDtype::Float32,
                ),
                dtype,
                None,
            ),
            None,
            None,
        )
    }

    fn prng_vector(rng: &mut XorShift64, len: usize, amplitude: f32) -> MlxArray {
        let data: Vec<f32> = (0..len).map(|_| rng.next_f32(amplitude)).collect();
        astype(
            &MlxArray::from_raw_data(
                data.as_ptr().cast(),
                data.len() * 4,
                &[len as i32],
                MlxDtype::Float32,
            ),
            MlxDtype::Float32,
            None,
        )
    }

    /// The real pack geometry: hidden 2560, 16 key heads / 48 value heads of
    /// 128 dims, conv kernel 4.
    fn real_geometry_module(dtype: MlxDtype) -> Qwen4ExpGdn {
        let mut rng = XorShift64::new(case_seed(42));
        let conv_data: Vec<f32> = (0..10240 * 4).map(|_| rng.next_f32(0.5)).collect();
        let conv = astype(
            &MlxArray::from_raw_data(
                conv_data.as_ptr().cast(),
                conv_data.len() * 4,
                &[10240, 4, 1],
                MlxDtype::Float32,
            ),
            dtype,
            None,
        );
        Qwen4ExpGdn::new(
            LinearAttentionConfig {
                full_attention_interval: 4,
                num_key_heads: 16,
                num_value_heads: 48,
                key_head_dim: 128,
                value_head_dim: 128,
                conv_kernel_dim: 4,
                q_scale: 1.0,
                k_scale: 1.0,
            },
            2560,
            1e-6,
            GdnGateActivation::Silu,
            Qwen4ExpGdnWeights {
                qkv: prng_dense(&mut rng, 10240, 2560, 0.02, dtype),
                gate: prng_dense(&mut rng, 6144, 2560, 0.02, dtype),
                decay: prng_dense(&mut rng, 48, 2560, 0.02, dtype),
                beta: prng_dense(&mut rng, 48, 2560, 0.02, dtype),
                output: prng_dense(&mut rng, 2560, 6144, 0.02, dtype),
                conv,
                a_log: prng_vector(&mut rng, 48, 0.4),
                dt_bias: prng_vector(&mut rng, 48, 0.4),
                norm_gain: mlx_sys::broadcast_to(
                    &astype(&MlxArray::from_f32(1.0), dtype, None),
                    &[128],
                    None,
                ),
            },
        )
        .unwrap()
    }

    /// The design-review differential: for every dtype, gate activation, decode
    /// row count and state mode, the fused beta and post stretches must really
    /// run (counter deltas prove it) and must reproduce the eager bits exactly.
    #[test]
    fn fused_gdn_stretches_match_eager_bits() {
        for dtype in [MlxDtype::Float32, MlxDtype::Bfloat16, MlxDtype::Float16] {
            for activation in [GdnGateActivation::Silu, GdnGateActivation::Sigmoid] {
                let module = synthetic_pack_geometry_with(dtype, activation);
                // Primed state from an eager 3-row call on a different input.
                let prime = deterministic_input(32, 3, case_seed(7), 1.5, dtype);
                let primed = {
                    let _eager = qwen4_exp_fused::force_for_test(Some(false));
                    run_forward(&module, &prime, None).1
                };
                for rows in [1_i32, 2, 3, 4, 8] {
                    for use_state in [false, true] {
                        let input = deterministic_input(
                            32,
                            rows,
                            case_seed(1000 + rows as u64 * 2 + u64::from(use_state)),
                            1.5,
                            dtype,
                        );
                        let state = use_state.then_some(&primed);
                        let (eager_output, eager_state) = {
                            let _eager = qwen4_exp_fused::force_for_test(Some(false));
                            run_forward(&module, &input, state)
                        };
                        let (beta_before, post_before) = (
                            qwen4_exp_fused::applied_for_test(Stretch::GdnBeta),
                            qwen4_exp_fused::applied_for_test(Stretch::GdnPost),
                        );
                        let (fused_output, fused_state) = {
                            qwen4_exp_fused::reset_registry_for_test();
                            let _fused = qwen4_exp_fused::force_for_test(Some(true));
                            run_forward(&module, &input, state)
                        };
                        assert!(
                            qwen4_exp_fused::applied_for_test(Stretch::GdnBeta) > beta_before,
                            "beta stretch must apply while fused (rows={rows}, state={use_state}, {dtype:?})",
                        );
                        assert!(
                            qwen4_exp_fused::applied_for_test(Stretch::GdnPost) > post_before,
                            "post stretch must apply while fused (rows={rows}, state={use_state}, {dtype:?})",
                        );
                        assert_bit_identical(&fused_output, &eager_output, "output");
                        assert_bit_identical(&fused_state.conv, &eager_state.conv, "conv state");
                        assert_bit_identical(
                            &fused_state.recurrent,
                            &eager_state.recurrent,
                            "recurrent state",
                        );
                    }
                }
            }
        }
    }

    /// The registry is keyed by geometry, so one thread serves every dtype from
    /// the same slot. The compiled post body bakes in its output dtype, so the
    /// dtype must be part of the key: this walks dtypes WITHOUT clearing the
    /// registry between them, in an order that would reuse a stale closure.
    #[test]
    fn fused_gdn_registry_reuse_across_dtypes_stays_bit_identical() {
        qwen4_exp_fused::reset_registry_for_test();
        for dtype in [
            MlxDtype::Float32,
            MlxDtype::Bfloat16,
            MlxDtype::Float16,
            MlxDtype::Bfloat16,
            MlxDtype::Float32,
        ] {
            let module = synthetic_pack_geometry_with(dtype, GdnGateActivation::Silu);
            let input = deterministic_input(32, 2, case_seed(4242), 1.5, dtype);
            let (eager_output, eager_state) = {
                let _eager = qwen4_exp_fused::force_for_test(Some(false));
                run_forward(&module, &input, None)
            };
            let post_before = qwen4_exp_fused::applied_for_test(Stretch::GdnPost);
            let (fused_output, fused_state) = {
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                run_forward(&module, &input, None)
            };
            assert!(
                qwen4_exp_fused::applied_for_test(Stretch::GdnPost) > post_before,
                "post stretch must apply ({dtype:?})"
            );
            assert_eq!(
                fused_output.dtype(),
                eager_output.dtype(),
                "{dtype:?} dtype"
            );
            assert_bit_identical(&fused_output, &eager_output, "output");
            assert_bit_identical(&fused_state.conv, &eager_state.conv, "conv state");
            assert_bit_identical(
                &fused_state.recurrent,
                &eager_state.recurrent,
                "recurrent state",
            );
        }
    }

    /// Prefill-sized inputs must skip the stretches entirely: eager bits, no
    /// applies, and no tombstone (the row gate is a skip, not a failure).
    #[test]
    fn fused_gdn_stretches_skip_prefill_sized_inputs() {
        let module = synthetic_pack_geometry_with(MlxDtype::Bfloat16, GdnGateActivation::Silu);
        for rows in [qwen4_exp_fused::FUSE_MAX_ROWS + 1, 16] {
            let input = deterministic_input(
                32,
                rows,
                case_seed(500 + rows as u64),
                1.5,
                MlxDtype::Bfloat16,
            );
            let (eager_output, eager_state) = {
                let _eager = qwen4_exp_fused::force_for_test(Some(false));
                run_forward(&module, &input, None)
            };
            let (beta_before, post_before) = (
                qwen4_exp_fused::applied_for_test(Stretch::GdnBeta),
                qwen4_exp_fused::applied_for_test(Stretch::GdnPost),
            );
            let (fused_output, fused_state) = {
                qwen4_exp_fused::reset_registry_for_test();
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                run_forward(&module, &input, None)
            };
            assert_eq!(
                qwen4_exp_fused::applied_for_test(Stretch::GdnBeta),
                beta_before,
                "beta must not apply at rows={rows}",
            );
            assert_eq!(
                qwen4_exp_fused::applied_for_test(Stretch::GdnPost),
                post_before,
                "post must not apply at rows={rows}",
            );
            assert!(
                !qwen4_exp_fused::tombstoned_for_test(Stretch::GdnBeta)
                    && !qwen4_exp_fused::tombstoned_for_test(Stretch::GdnPost),
                "the row gate must skip, not tombstone",
            );
            assert_bit_identical(&fused_output, &eager_output, "output");
            assert_bit_identical(&fused_state.conv, &eager_state.conv, "conv state");
            assert_bit_identical(
                &fused_state.recurrent,
                &eager_state.recurrent,
                "recurrent state at rows={rows}",
            );
        }
    }

    /// A failed apply must tombstone only its own stretch and fall back to the
    /// eager chain, which reproduces the eager bits exactly.
    #[test]
    fn fused_gdn_stretch_failure_falls_back_to_eager_bits() {
        let module = synthetic_pack_geometry_with(MlxDtype::Bfloat16, GdnGateActivation::Silu);
        let input = deterministic_input(32, 2, case_seed(300), 1.5, MlxDtype::Bfloat16);
        let (eager_output, eager_state) = {
            let _eager = qwen4_exp_fused::force_for_test(Some(false));
            run_forward(&module, &input, None)
        };
        for stretch in [Stretch::GdnBeta, Stretch::GdnPost] {
            let other = if stretch == Stretch::GdnBeta {
                Stretch::GdnPost
            } else {
                Stretch::GdnBeta
            };
            let other_before = qwen4_exp_fused::applied_for_test(other);
            let (fused_output, fused_state) = {
                qwen4_exp_fused::reset_registry_for_test();
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                qwen4_exp_fused::fail_next_apply_for_test(stretch);
                run_forward(&module, &input, None)
            };
            assert!(
                qwen4_exp_fused::tombstoned_for_test(stretch),
                "{stretch:?} must tombstone after a failed apply",
            );
            assert!(
                qwen4_exp_fused::applied_for_test(other) > other_before,
                "{other:?} must still apply when only {stretch:?} fails",
            );
            assert_bit_identical(&fused_output, &eager_output, "output");
            assert_bit_identical(&fused_state.conv, &eager_state.conv, "conv state");
            assert_bit_identical(
                &fused_state.recurrent,
                &eager_state.recurrent,
                "recurrent state after {stretch:?} fallback",
            );
        }
    }

    /// Same differential on the production geometry, where the projections are
    /// 2560 wide and every head dimension is the real 128: the compiled
    /// stretches must stay bit-identical where it actually ships.
    #[test]
    fn fused_gdn_matches_eager_at_real_geometry() {
        for dtype in [MlxDtype::Bfloat16, MlxDtype::Float32] {
            // Weights are built once per dtype and reused across all cases.
            let module = real_geometry_module(dtype);
            qwen4_exp_fused::reset_registry_for_test();
            for seed in 0_u64..4 {
                for rows in 1_i32..=8 {
                    let input = deterministic_input(
                        2560,
                        rows,
                        case_seed(9000 + seed * 16 + rows as u64),
                        1.0,
                        dtype,
                    );
                    let (eager_output, eager_state) = {
                        let _eager = qwen4_exp_fused::force_for_test(Some(false));
                        run_forward(&module, &input, None)
                    };
                    let (beta_before, post_before) = (
                        qwen4_exp_fused::applied_for_test(Stretch::GdnBeta),
                        qwen4_exp_fused::applied_for_test(Stretch::GdnPost),
                    );
                    let (fused_output, fused_state) = {
                        let _fused = qwen4_exp_fused::force_for_test(Some(true));
                        run_forward(&module, &input, None)
                    };
                    assert!(
                        qwen4_exp_fused::applied_for_test(Stretch::GdnBeta) > beta_before,
                        "beta must apply at rows={rows}, seed={seed}, {dtype:?}",
                    );
                    assert!(
                        qwen4_exp_fused::applied_for_test(Stretch::GdnPost) > post_before,
                        "post must apply at rows={rows}, seed={seed}, {dtype:?}",
                    );
                    assert_bit_identical(&fused_output, &eager_output, "output");
                    assert_bit_identical(&fused_state.conv, &eager_state.conv, "conv state");
                    assert_bit_identical(
                        &fused_state.recurrent,
                        &eager_state.recurrent,
                        "recurrent state at real geometry",
                    );
                }
            }
        }
    }
}
