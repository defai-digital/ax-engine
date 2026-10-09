use std::ffi::CString;
use std::sync::LazyLock;

use crate::array::{MlxArray, null_ffi_array};
use crate::error::{ensure_error_handler, panic_on_status, status_to_result};
use crate::ffi;
use crate::stream::{MlxStream, default_gpu_raw};

macro_rules! checked_ffi {
    ($operation:literal, $call:expr) => {{
        ensure_error_handler();
        let rc = $call;
        panic_on_status($operation, rc);
    }};
}

/// Attention mask accepted by MLX fast SDPA.
pub enum ScaledDotProductAttentionMask<'a> {
    None,
    Causal,
    Array(&'a MlxArray),
}

// Pre-allocated CStrings for the two SDPA mask modes.  These are on the
// decode hot path (one allocation per attention layer per token), so caching
// them as process-global statics eliminates a heap alloc+free round-trip.
static MASK_CAUSAL: LazyLock<CString> = LazyLock::new(|| CString::new("causal").unwrap());
static MASK_EMPTY: LazyLock<CString> = LazyLock::new(|| CString::new("").unwrap());

/// Unmasked MLX gated-delta recurrence with an explicit initial state.
///
/// This binding accepts only float32 tensors: MLX 0.32.3 casts gamma and beta
/// to the query dtype internally, which would otherwise round FP32 decay.
/// Q/K are `[B,T,Hk,Dk]`, V is `[B,T,Hv,Dv]`, gamma/beta are `[B,T,Hv]`,
/// and state is `[B,Hv,Dv,Dk]`. Returns output and final FP32 state lazily.
/// Unsupported Metal shapes may use MLX's graph fallback; performance callers
/// must apply their own shape gate. Errors during later eval remain eval errors.
pub fn try_gated_delta_update(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    gamma: &MlxArray,
    beta: &MlxArray,
    initial_state: &MlxArray,
    s: Option<&MlxStream>,
) -> Result<(MlxArray, MlxArray), String> {
    let qs = q.shape();
    let vs = v.shape();
    if qs.len() != 4 || vs.len() != 4 || qs.iter().chain(&vs).any(|&d| d <= 0) {
        return Err("gated_delta_update requires positive rank-four Q/K/V shapes".into());
    }
    let [batch, seq, hk, dk] = [qs[0], qs[1], qs[2], qs[3]];
    let [hv, dv] = [vs[2], vs[3]];
    if k.shape() != qs
        || vs[..2] != qs[..2]
        || hv % hk != 0
        || gamma.shape() != [batch, seq, hv]
        || beta.shape() != [batch, seq, hv]
        || initial_state.shape() != [batch, hv, dv, dk]
    {
        return Err("gated_delta_update has incompatible tensor shapes".into());
    }
    if [q, k, v, gamma, beta, initial_state]
        .iter()
        .any(|a| a.dtype() != crate::MlxDtype::Float32)
    {
        return Err("gated_delta_update binding requires float32 tensors".into());
    }
    crate::op_count::bump();
    ensure_error_handler();
    let mut output = MlxArray::empty();
    let mut final_state = MlxArray::empty();
    let rc = unsafe {
        ffi::ax_mlx_gated_delta_update(
            &mut output.inner,
            &mut final_state.inner,
            q.inner,
            k.inner,
            v.inner,
            gamma.inner,
            beta.inner,
            initial_state.inner,
            s.map(|s| s.inner).unwrap_or_else(default_gpu_raw),
        )
    };
    status_to_result("ax_mlx_gated_delta_update", rc)?;
    Ok((output, final_state))
}

/// RMS layer normalization.
pub fn rms_norm(
    x: &MlxArray,
    weight: Option<&MlxArray>,
    eps: f32,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let weight_raw = weight.map(|w| w.inner).unwrap_or_else(null_ffi_array);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_fast_rms_norm",
            ffi::mlx_fast_rms_norm(&mut res.inner, x.inner, weight_raw, eps, stream)
        );
        res
    }
}

/// Rotary position embedding.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    x: &MlxArray,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
    freqs: Option<&MlxArray>,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let base_opt = ffi::mlx_optional_float_ {
            has_value: base.is_some(),
            value: base.unwrap_or(10000.0),
        };
        let freqs_raw = freqs.map(|f| f.inner).unwrap_or_else(null_ffi_array);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_fast_rope",
            ffi::mlx_fast_rope(
                &mut res.inner,
                x.inner,
                dims,
                traditional,
                base_opt,
                scale,
                offset,
                freqs_raw,
                stream,
            )
        );
        res
    }
}

/// Rotary position embedding with a dynamic (array-valued) offset.
///
/// Unlike [`rope`] which bakes `offset` as a scalar constant, this variant
/// passes the offset as an `MlxArray` node in the computation graph.  This
/// is required inside `mx.compile`-traced closures where the RoPE position
/// changes across calls without changing graph structure.
#[allow(clippy::too_many_arguments)]
pub fn rope_dynamic(
    x: &MlxArray,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: &MlxArray,
    freqs: Option<&MlxArray>,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let base_opt = ffi::mlx_optional_float_ {
            has_value: base.is_some(),
            value: base.unwrap_or(10000.0),
        };
        let freqs_raw = freqs.map(|f| f.inner).unwrap_or_else(null_ffi_array);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_fast_rope_dynamic",
            ffi::mlx_fast_rope_dynamic(
                &mut res.inner,
                x.inner,
                dims,
                traditional,
                base_opt,
                scale,
                offset.inner,
                freqs_raw,
                stream,
            )
        );
        res
    }
}

/// Scaled dot-product attention (flash attention).
///
/// `causal`: when true applies a causal (lower-triangular) mask; required for
/// prefill (seq > 1). During single-token decode no mask is needed.
pub fn scaled_dot_product_attention(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    causal: bool,
    s: Option<&MlxStream>,
) -> MlxArray {
    let mask = if causal {
        ScaledDotProductAttentionMask::Causal
    } else {
        ScaledDotProductAttentionMask::None
    };
    scaled_dot_product_attention_with_mask(queries, keys, values, scale, mask, s)
}

/// Scaled dot-product attention with an explicit MLX mask.
pub fn scaled_dot_product_attention_with_mask(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    mask: ScaledDotProductAttentionMask<'_>,
    s: Option<&MlxStream>,
) -> MlxArray {
    scaled_dot_product_attention_with_mask_and_sinks(queries, keys, values, scale, mask, None, s)
}

/// Scaled dot-product attention with an explicit MLX mask and optional
/// per-query-head attention sinks (`[n_q_heads]`). The sink logit joins the
/// softmax denominator but contributes no value — matching mlx-lm's
/// `scaled_dot_product_attention(..., sinks=)` used by GPT-OSS.
///
/// `force_fused` stays false. mlx-c 0.7.0 exposes the flag, and a true value
/// rejects shapes that have no fused kernel instead of using MLX's fallback.
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_with_mask_and_sinks(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    mask: ScaledDotProductAttentionMask<'_>,
    sinks: Option<&MlxArray>,
    s: Option<&MlxStream>,
) -> MlxArray {
    try_scaled_dot_product_attention(queries, keys, values, scale, mask, sinks, false, s)
        .unwrap_or_else(|message| panic!("{message}"))
}

/// Same attention as [`scaled_dot_product_attention_with_mask_and_sinks`],
/// with mlx-c's `force_fused` flag. `false` keeps MLX's kernel selection.
/// `true` requires a fused kernel and returns the MLX rejection otherwise.
#[allow(clippy::too_many_arguments)]
pub fn try_scaled_dot_product_attention(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    mask: ScaledDotProductAttentionMask<'_>,
    sinks: Option<&MlxArray>,
    force_fused: bool,
    s: Option<&MlxStream>,
) -> Result<MlxArray, String> {
    crate::op_count::bump();
    ensure_error_handler();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mask_mode = match mask {
            ScaledDotProductAttentionMask::Causal => &*MASK_CAUSAL,
            ScaledDotProductAttentionMask::None | ScaledDotProductAttentionMask::Array(_) => {
                &*MASK_EMPTY
            }
        };
        let null_arr = null_ffi_array();
        let mask_arr = match mask {
            ScaledDotProductAttentionMask::Array(mask) => mask.inner,
            ScaledDotProductAttentionMask::None | ScaledDotProductAttentionMask::Causal => null_arr,
        };
        let sinks_arr = sinks.map(|sinks| sinks.inner).unwrap_or(null_arr);
        let mut res = MlxArray::empty();
        let rc = ffi::mlx_fast_scaled_dot_product_attention(
            &mut res.inner,
            queries.inner,
            keys.inner,
            values.inner,
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            sinks_arr,
            force_fused,
            stream,
        );
        status_to_result("mlx_fast_scaled_dot_product_attention", rc)?;
        Ok(res)
    }
}

#[cfg(test)]
mod gated_delta_tests {
    use super::*;
    use crate::{MlxDtype, eval, zeros};

    fn array(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    #[test]
    fn gated_delta_binding_matches_independent_recurrence() {
        let q_data = [
            0.1, 0.2, -0.3, 0.4, 0.2, -0.1, 0.4, 0.3, -0.1, 0.4, 0.2, 0.3,
        ];
        let k_data = [0.2, -0.3, 0.1, 0.2, 0.4, 0.3, 0.2, -0.1, 0.1, 0.2, 0.3, 0.4];
        let v_data = [0.3, 0.4, -0.2, 0.1, 0.2, -0.3];
        let gamma = [0.6, 0.7, 0.8];
        let beta = [0.5, 0.4, 0.3];
        let initial = [0.1, -0.2, 0.2, 0.3, 0.4, 0.2, -0.1, 0.1];
        let q = array(&q_data, &[1, 3, 1, 4]);
        let k = array(&k_data, &[1, 3, 1, 4]);
        let v = array(&v_data, &[1, 3, 1, 2]);
        let g = array(&gamma, &[1, 3, 1]);
        let b = array(&beta, &[1, 3, 1]);
        let s = array(&initial, &[1, 1, 2, 4]);
        let (out, state) =
            try_gated_delta_update(&q, &k, &v, &g, &b, &s, None).expect("valid recurrence");
        eval(&[&out, &state]);
        let mut expected_state = initial;
        let mut expected = Vec::new();
        for t in 0..3 {
            for row in 0..2 {
                let values = &mut expected_state[row * 4..(row + 1) * 4];
                for x in values.iter_mut() {
                    *x *= gamma[t];
                }
                let kv: f32 = values
                    .iter()
                    .zip(&k_data[t * 4..(t + 1) * 4])
                    .map(|(s, k)| s * k)
                    .sum();
                let delta = (v_data[t * 2 + row] - kv) * beta[t];
                for (s, k) in values.iter_mut().zip(&k_data[t * 4..(t + 1) * 4]) {
                    *s += k * delta;
                }
                expected.push(
                    values
                        .iter()
                        .zip(&q_data[t * 4..(t + 1) * 4])
                        .map(|(s, q)| s * q)
                        .sum::<f32>(),
                );
            }
        }
        assert_eq!(out.shape(), vec![1, 3, 1, 2]);
        assert_eq!(state.shape(), vec![1, 1, 2, 4]);
        for (actual, expected) in out
            .data_f32()
            .iter()
            .zip(&expected)
            .chain(state.data_f32().iter().zip(&expected_state))
        {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn gated_delta_binding_rejects_invalid_shapes_and_gate_precision() {
        let q = zeros(&[1, 32, 16, 128], MlxDtype::Float32, None);
        let v = zeros(&[1, 32, 32, 128], MlxDtype::Float32, None);
        let g = zeros(&[1, 32, 32], MlxDtype::Float32, None);
        let s = zeros(&[1, 32, 128, 128], MlxDtype::Float32, None);
        let rank = zeros(&[1], MlxDtype::Float32, None);
        let zero = zeros(&[1, 32, 0, 128], MlxDtype::Float32, None);
        let wrong_heads = zeros(&[1, 32, 17, 128], MlxDtype::Float32, None);
        let low_precision = zeros(&[1, 32, 32], MlxDtype::Bfloat16, None);
        for (q, k, v, g, b, s) in [
            (&rank, &q, &v, &g, &g, &s),
            (&zero, &zero, &v, &g, &g, &s),
            (&wrong_heads, &wrong_heads, &v, &g, &g, &s),
            (&q, &q, &v, &low_precision, &g, &s),
            (&q, &q, &v, &g, &g, &rank),
            (&q, &rank, &v, &g, &g, &s),
        ] {
            assert!(try_gated_delta_update(q, k, v, g, b, s, None).is_err());
        }
    }
}

#[cfg(test)]
mod sdpa_binding_tests {
    use super::*;
    use crate::{MlxDtype, eval};

    fn attention_inputs(heads: i32, seq: i32, dim: i32) -> (MlxArray, MlxArray, MlxArray) {
        let shape = [1, heads, seq, dim];
        let n = (heads * seq * dim) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) / 11.0).collect();
        let array = |phase: f32| {
            let shifted: Vec<f32> = data.iter().map(|value| value + phase).collect();
            MlxArray::from_raw_data(
                shifted.as_ptr().cast(),
                shifted.len() * 4,
                &shape,
                MlxDtype::Float32,
            )
        };
        (array(0.0), array(0.1), array(-0.2))
    }

    fn max_abs(a: &MlxArray, b: &MlxArray) -> f32 {
        a.data_f32()
            .iter()
            .zip(b.data_f32())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn force_fused_rejects_unsupported_head_dim_without_changing_fallback() {
        let (q, k, v) = attention_inputs(1, 2, 8);
        let rejected = try_scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0,
            ScaledDotProductAttentionMask::None,
            None,
            true,
            None,
        )
        .expect_err("head dim 8 has no fused kernel");
        assert!(
            rejected.contains("force_fused"),
            "unexpected rejection: {rejected}"
        );
        let fallback = scaled_dot_product_attention(&q, &k, &v, 1.0, false, None);
        eval(&[&fallback]);
        assert!(fallback.data_f32().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn force_fused_matches_default_selection_for_vector_kernel_shape() {
        let (q, k, v) = attention_inputs(1, 1, 64);
        let run = |force_fused| {
            try_scaled_dot_product_attention(
                &q,
                &k,
                &v,
                0.125,
                ScaledDotProductAttentionMask::None,
                None,
                force_fused,
                None,
            )
            .expect("head dim 64 sequence 1 is a fused vector shape")
        };
        let forced = run(true);
        let selected = run(false);
        eval(&[&forced, &selected]);
        assert!(forced.data_f32().iter().all(|value| value.is_finite()));
        assert!(max_abs(&forced, &selected) <= 1e-5);
    }
}
