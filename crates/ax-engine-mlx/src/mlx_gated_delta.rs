//! Opt-in MLX 0.32.3 GDN prefill adapter. No decode or verifier promotion.

use std::sync::OnceLock;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, astype,
    try_gated_delta_update,
};

static GATES_KERNEL: OnceLock<Result<MlxMetalKernel, String>> = OnceLock::new();

/// Release-specific Metal dispatch guard. Avoid MLX's per-token graph fallback.
pub fn mlx_gated_delta_prefill_supported(q: &[i32], v: &[i32], dtype: MlxDtype) -> bool {
    if q.len() != 4 || v.len() != 4 || q[0] <= 0 || v[..2] != q[..2] {
        return false;
    }
    (32..=2048).contains(&q[1])
        && q[3] == 128
        && v[3] == 128
        && matches!(
            (q[2], v[2]),
            (24, 24) | (32, 32) | (16, 32) | (16, 48) | (16, 16) | (16, 64)
        )
        && matches!(
            dtype,
            MlxDtype::Float32 | MlxDtype::Float16 | MlxDtype::Bfloat16
        )
}

/// Attempt the experimental prefill path without consulting environment flags.
///
/// Returns `None` for unsupported shapes/dtypes or graph-construction failure.
/// This allows deterministic A/B probes without changing process-cached flags.
/// The probe admits T=32..2048; production dispatch starts at T=128.
/// No tensor is evaluated here. Later evaluation failures propagate normally.
#[allow(clippy::too_many_arguments)]
pub fn try_mlx_gated_delta_prefill(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
) -> Option<(MlxArray, MlxArray)> {
    let qs = q.shape();
    let vs = v.shape();
    if !mlx_gated_delta_prefill_supported(&qs, &vs, q.dtype()) {
        return None;
    }
    let gates_shape = vec![qs[0], qs[1], vs[2]];
    if k.shape() != qs
        || a_raw.shape() != gates_shape
        || b_raw.shape() != gates_shape
        || a_log.shape() != [vs[2]]
        || dt_bias.shape() != [vs[2]]
        || state.shape() != [qs[0], vs[2], vs[3], qs[3]]
        || [k, v, a_raw, b_raw].iter().any(|a| a.dtype() != q.dtype())
        || [a_log, dt_bias, state]
            .iter()
            .any(|a| a.dtype() != MlxDtype::Float32)
    {
        return None;
    }
    let count = qs[0].checked_mul(qs[1])?.checked_mul(vs[2])?;
    let gates_kernel = GATES_KERNEL
        .get_or_init(|| {
            MlxMetalKernel::try_new(
                "ax_mlx_gdn_prefill_gates_v1",
                &["a_log", "a_raw", "dt_bias", "b_raw"],
                &["gamma", "beta"],
                GATES_SOURCE,
                "",
                true,
            )
        })
        .as_ref()
        .ok()?;
    let outputs = gates_kernel
        .try_apply_with_template(
            &[a_log, a_raw, dt_bias, b_raw],
            &[
                KernelOutputSpec {
                    shape: gates_shape.clone(),
                    dtype: MlxDtype::Float32,
                },
                KernelOutputSpec {
                    shape: gates_shape,
                    dtype: MlxDtype::Float32,
                },
            ],
            &[
                KernelTemplateArg::Dtype {
                    name: "InT",
                    dtype: q.dtype(),
                },
                KernelTemplateArg::Int {
                    name: "Hv",
                    value: vs[2],
                },
                KernelTemplateArg::Int {
                    name: "Count",
                    value: count,
                },
            ],
            (count, 1, 1),
            (256, 1, 1),
            None,
        )
        .ok()?;
    let mut outputs = outputs.into_iter();
    let gamma = outputs.next()?;
    let beta = outputs.next()?;
    // The published API rounds gates to the query dtype. Promote Q/K/V so
    // FP32 decay and state survive; retain activation rounding only at output.
    let q32 = astype(q, MlxDtype::Float32, None);
    let k32 = astype(k, MlxDtype::Float32, None);
    let v32 = astype(v, MlxDtype::Float32, None);
    let (y, state_out) =
        try_gated_delta_update(&q32, &k32, &v32, &gamma, &beta, state, None).ok()?;
    Some((astype(&y, q.dtype(), None), state_out))
}

// Same gate arithmetic as AX's existing recurrent kernels, evaluated once per
// token/head. In particular beta rounds through InT while decay stays FP32.
const GATES_SOURCE: &str = r#"
    const uint i = thread_position_in_grid.x;
    if (i >= Count) return;
    const uint h = i % Hv;
    const float a = static_cast<float>(a_raw[i]) + dt_bias[h];
    const float sp = a > 20.0f ? a : log1p(exp(a));
    gamma[i] = exp(-exp(a_log[h]) * sp);
    const float b = static_cast<float>(b_raw[i]);
    beta[i] = static_cast<float>(static_cast<InT>(1.0f / (1.0f + exp(-b))));
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_attention_ops::{gated_delta_kernel, gated_delta_kernel_impl};
    use mlx_sys::{contiguous, eval, slice};

    fn input(shape: &[i32], phase: usize, scale: f32, dtype: MlxDtype) -> MlxArray {
        let n = shape.iter().map(|&d| d as usize).product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i + phase) % 29) as f32 / 29.0 * scale - scale / 2.0)
            .collect();
        astype(
            &MlxArray::from_raw_data(data.as_ptr().cast(), n * 4, shape, MlxDtype::Float32),
            dtype,
            None,
        )
    }

    fn close(actual: &MlxArray, expected: &MlxArray, tolerance: f32) {
        assert_eq!(actual.shape(), expected.shape());
        let a = contiguous(&astype(actual, MlxDtype::Float32, None), None);
        let e = contiguous(&astype(expected, MlxDtype::Float32, None), None);
        eval(&[&a, &e]);
        let error = a
            .data_f32()
            .iter()
            .zip(e.data_f32())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            a.data_f32()
                .iter()
                .chain(e.data_f32())
                .all(|v| v.is_finite())
        );
        assert!(error <= tolerance, "max_abs={error}, tolerance={tolerance}");
    }

    #[test]
    fn dispatch_guard_excludes_decode_verifier_and_unsupported_heads() {
        for (hk, hv) in [(24, 24), (32, 32), (16, 32), (16, 48), (16, 16), (16, 64)] {
            for dtype in [MlxDtype::Float32, MlxDtype::Float16, MlxDtype::Bfloat16] {
                assert!(mlx_gated_delta_prefill_supported(
                    &[2, 32, hk, 128],
                    &[2, 32, hv, 128],
                    dtype
                ));
            }
        }
        for seq in [0, 1, 2, 4, 16, 31, 2049] {
            assert!(!mlx_gated_delta_prefill_supported(
                &[1, seq, 16, 128],
                &[1, seq, 32, 128],
                MlxDtype::Float32
            ));
        }
        for (q, v, dtype) in [
            (vec![1, 32, 8, 128], vec![1, 32, 16, 128], MlxDtype::Float32),
            (vec![1, 32, 16, 64], vec![1, 32, 32, 128], MlxDtype::Float32),
            (vec![1, 32, 16, 128], vec![1, 32, 32, 64], MlxDtype::Float32),
            (
                vec![1, 32, 16, 128],
                vec![2, 32, 32, 128],
                MlxDtype::Float32,
            ),
            (vec![1, 32, 16, 128], vec![1, 32, 32, 128], MlxDtype::Uint32),
            (vec![], vec![], MlxDtype::Float32),
        ] {
            assert!(!mlx_gated_delta_prefill_supported(&q, &v, dtype));
        }
    }

    #[test]
    fn prefill_matches_ax_with_strided_inputs_and_nonzero_continuation() {
        // Use distinct batches and GQA heads. The 73-token parent creates
        // strided batch views; two 32-token calls test the chunk boundary.
        for dtype in [MlxDtype::Float32, MlxDtype::Float16, MlxDtype::Bfloat16] {
            let q = input(&[2, 73, 16, 128], 1, 0.12, dtype);
            let k = input(&[2, 73, 16, 128], 7, 0.10, dtype);
            let v = input(&[2, 73, 32, 128], 13, 0.20, dtype);
            let a = input(&[2, 73, 32], 2, 0.80, dtype);
            let b = input(&[2, 73, 32], 17, 0.70, dtype);
            let a_log = input(&[32], 4, 0.4, MlxDtype::Float32);
            let bias = input(&[32], 9, 0.5, MlxDtype::Float32);
            let state = input(&[2, 32, 128, 128], 3, 0.04, MlxDtype::Float32);
            let rows = |x: &MlxArray, start, end| {
                let mut lo = vec![0; x.shape().len()];
                let mut hi = x.shape();
                lo[1] = start;
                hi[1] = end;
                slice(x, &lo, &hi, &vec![1; hi.len()], None)
            };
            let run = |start, end, state: &MlxArray, upstream: bool| {
                let (q, k, v, a, b) = (
                    rows(&q, start, end),
                    rows(&k, start, end),
                    rows(&v, start, end),
                    rows(&a, start, end),
                    rows(&b, start, end),
                );
                if upstream {
                    try_mlx_gated_delta_prefill(&q, &k, &v, &a_log, &a, &bias, &b, state)
                        .expect("supported prefill")
                } else {
                    gated_delta_kernel_impl(&q, &k, &v, &a_log, &a, &bias, &b, state)
                }
            };
            let (y, first) = run(1, 33, &state, true);
            let (ax_y, ax_first) = run(1, 33, &state, false);
            assert_eq!(y.dtype(), dtype);
            assert_eq!(first.dtype(), MlxDtype::Float32);
            let tolerance = if dtype == MlxDtype::Bfloat16 {
                0.0005
            } else {
                0.00005
            };
            close(&y, &ax_y, tolerance);
            close(&first, &ax_first, 0.00005);
            let (y2, second) = run(33, 65, &first, true);
            let (ax_y2, ax_second) = run(33, 65, &ax_first, false);
            close(&y2, &ax_y2, tolerance);
            close(&second, &ax_second, 0.00005);
            let (full, full_state) = run(1, 65, &state, true);
            close(&rows(&full, 32, 64), &y2, tolerance);
            close(&full_state, &second, 0.00005);
            let malformed = mlx_sys::zeros(&[32], dtype, None);
            assert!(
                try_mlx_gated_delta_prefill(&q, &k, &v, &a_log, &malformed, &bias, &b, &state)
                    .is_none()
            );
        }
    }

    #[test]
    fn process_flag_selects_prefill_and_preserves_short_window_fallback() {
        for seq in [1, 4, 32, 128] {
            let q = input(&[1, seq, 16, 128], 1, 0.12, MlxDtype::Float32);
            let k = input(&[1, seq, 16, 128], 7, 0.10, MlxDtype::Float32);
            let v = input(&[1, seq, 32, 128], 13, 0.20, MlxDtype::Float32);
            let a = input(&[1, seq, 32], 2, 0.80, MlxDtype::Float32);
            let b = input(&[1, seq, 32], 17, 0.70, MlxDtype::Float32);
            let a_log = input(&[32], 4, 0.4, MlxDtype::Float32);
            let bias = input(&[32], 9, 0.5, MlxDtype::Float32);
            let state = input(&[1, 32, 128, 128], 3, 0.04, MlxDtype::Float32);
            let (y, s) = gated_delta_kernel(&q, &k, &v, &a_log, &a, &bias, &b, &state);
            let expected = if seq >= 128 && crate::fastpath::qwen_gated_delta_prefill_mlx_enabled()
            {
                try_mlx_gated_delta_prefill(&q, &k, &v, &a_log, &a, &bias, &b, &state)
                    .expect("selected prefill")
            } else {
                gated_delta_kernel_impl(&q, &k, &v, &a_log, &a, &bias, &b, &state)
            };
            close(&y, &expected.0, 0.0);
            close(&s, &expected.1, 0.0);
        }
    }

    #[test]
    fn supported_head_pairs_and_extreme_gates_match_ax() {
        for (hk, hv) in [(24, 24), (32, 32), (16, 32), (16, 48), (16, 16), (16, 64)] {
            let q = input(&[1, 33, hk, 128], 1, 0.12, MlxDtype::Float32);
            let k = input(&[1, 33, hk, 128], 7, 0.10, MlxDtype::Float32);
            let v = input(&[1, 33, hv, 128], 13, 0.20, MlxDtype::Float32);
            let b = input(&[1, 33, hv], 17, 0.70, MlxDtype::Float32);
            let state = input(&[1, hv, 128, 128], 3, 0.04, MlxDtype::Float32);
            let bias = mlx_sys::zeros(&[hv], MlxDtype::Float32, None);
            // Underflowing and near-one decay, across a partial final chunk.
            let rates: Vec<f32> = (0..hv)
                .map(|h| if h % 2 == 0 { 5.0 } else { -5.0 })
                .collect();
            let gates: Vec<f32> = (0..33 * hv)
                .map(|i| if i % 3 == 0 { 30.0 } else { -30.0 })
                .collect();
            let a_log = MlxArray::from_raw_data(
                rates.as_ptr().cast(),
                rates.len() * 4,
                &[hv],
                MlxDtype::Float32,
            );
            let a = MlxArray::from_raw_data(
                gates.as_ptr().cast(),
                gates.len() * 4,
                &[1, 33, hv],
                MlxDtype::Float32,
            );
            let (y, s) = try_mlx_gated_delta_prefill(&q, &k, &v, &a_log, &a, &bias, &b, &state)
                .expect("supported heads");
            let (ax_y, ax_s) = gated_delta_kernel_impl(&q, &k, &v, &a_log, &a, &bias, &b, &state);
            close(&y, &ax_y, 0.00005);
            close(&s, &ax_s, 0.00005);
        }
    }
}
