//! Dedicated Flash Next recurrence probes and default-off MLX prefill dispatch.
//! Inputs already carry Flash Next normalization, query scaling and FP32 gates.
//! Public for diagnostic binaries; not a stable external API.

use mlx_sys::{
    MlxArray, MlxDtype, add, concatenate, multiply, reshape, slice, subtract, sum_axis, take,
    try_gated_delta_update,
};

fn validate(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    decay: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
) -> Result<(), String> {
    let qs = q.shape();
    let vs = v.shape();
    if qs.len() != 4
        || vs.len() != 4
        || qs.iter().chain(&vs).any(|&d| d <= 0)
        || k.shape() != qs
        || vs[..2] != qs[..2]
        || vs[2] % qs[2] != 0
        || decay.shape() != [qs[0], qs[1], vs[2]]
        || beta.shape() != decay.shape()
        || state.shape() != [qs[0], vs[2], vs[3], qs[3]]
        || [q, k, v, decay, beta, state]
            .iter()
            .any(|a| a.dtype() != MlxDtype::Float32)
    {
        return Err("incompatible Flash Next FP32 recurrence tensors".into());
    }
    Ok(())
}

/// Expand contiguous GQA groups exactly as the dedicated Flash Next baseline.
pub(crate) fn expand_heads(q: &MlxArray, k: &MlxArray, hv: i32) -> (MlxArray, MlxArray) {
    let repeat = hv / q.shape()[2];
    let head_ids: Vec<i32> = (0..hv).map(|h| h / repeat).collect();
    let ids = MlxArray::from_raw_data(
        head_ids.as_ptr().cast(),
        std::mem::size_of_val(head_ids.as_slice()),
        &[hv],
        MlxDtype::Int32,
    );
    (take(q, &ids, 2, None), take(k, &ids, 2, None))
}

/// Original sequential AX recurrence, with immutable explicit initial state.
pub fn portable_recurrence(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    decay: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
) -> Result<(MlxArray, MlxArray), String> {
    validate(q, k, v, decay, beta, state)?;
    let qs = q.shape();
    let vs = v.shape();
    let (batch, seq, hv, dk, dv) = (qs[0], qs[1], vs[2], qs[3], vs[3]);
    let (q, k) = expand_heads(q, k, hv);
    let mut recurrent = state.clone();
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
        let vr = reshape(&row(v, dv), &[batch, hv, dv, 1], None);
        let decayed = multiply(&recurrent, &scalar_row(decay), None);
        let prediction = sum_axis(&multiply(&decayed, &kr, None), -1, true, None);
        let correction = multiply(&subtract(&vr, &prediction, None), &scalar_row(beta), None);
        recurrent = add(&decayed, &multiply(&correction, &kr, None), None);
        outputs.push(reshape(
            &sum_axis(&multiply(&recurrent, &qr, None), -1, false, None),
            &[batch, 1, hv, dv],
            None,
        ));
    }
    let refs: Vec<&MlxArray> = outputs.iter().collect();
    Ok((concatenate(&refs, 1, None), recurrent))
}

/// Attempt MLX's published chunk recurrence without evaluating any tensor.
/// Unsupported shapes and graph-construction errors retain the AX route.
/// Gate tensors are consumed unchanged; Qwen 3.5 gate arithmetic is not used.
pub fn try_mlx_prefill(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    decay: &MlxArray,
    beta: &MlxArray,
    state: &MlxArray,
) -> Option<(MlxArray, MlxArray)> {
    let qs = q.shape();
    if !crate::mlx_gated_delta::mlx_gated_delta_prefill_supported(&qs, &v.shape(), q.dtype())
        || !crate::fastpath::qwen_gated_delta_prefill_mlx_seq_eligible(qs[1])
        || validate(q, k, v, decay, beta, state).is_err()
    {
        return None;
    }
    let result = try_gated_delta_update(q, k, v, decay, beta, state, None).ok()?;
    #[cfg(test)]
    DISPATCHES.with(|count| count.set(count.get() + 1));
    Some(result)
}

pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_MODE.with(std::cell::Cell::get) {
        return value;
    }
    crate::fastpath::flash_next_gdn_prefill_mlx_enabled()
}

#[cfg(test)]
thread_local! {
    static TEST_MODE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static DISPATCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn with_mode<T>(enabled: bool, action: impl FnOnce() -> T) -> (T, u64) {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_MODE.with(|mode| mode.set(self.0));
        }
    }
    let restore = Restore(TEST_MODE.with(|mode| mode.replace(Some(enabled))));
    let before = DISPATCHES.with(std::cell::Cell::get);
    let result = action();
    let after = DISPATCHES.with(std::cell::Cell::get);
    drop(restore);
    (result, after - before)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;
    use mlx_sys::{astype, contiguous, eval};

    fn input(shape: &[i32], phase: i32, scale: f32) -> MlxArray {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let data: Vec<f32> = (0..n)
            .map(|i| (((i as i32 + phase) * 37 % 997) as f32 / 997.0 - 0.5) * scale)
            .collect();
        MlxArray::from_raw_data(data.as_ptr().cast(), n * 4, shape, MlxDtype::Float32)
    }

    fn close(a: &MlxArray, b: &MlxArray, tolerance: f64) {
        assert_eq!(a.shape(), b.shape());
        let a = contiguous(a, None);
        let b = contiguous(b, None);
        eval(&[&a, &b]);
        let error = a
            .data_f32()
            .iter()
            .zip(b.data_f32())
            .fold(0.0_f64, |m, (&a, &b)| {
                assert!(a.is_finite() && b.is_finite());
                m.max((f64::from(a) - f64::from(b)).abs())
            });
        assert!(error <= tolerance, "max_abs={error}, tolerance={tolerance}");
    }

    #[test]
    fn portable_gqa_matches_independent_host_recurrence() {
        let q = input(&[2, 3, 2, 4], 1, 0.2);
        let k = input(&[2, 3, 2, 4], 5, 0.2);
        let v = input(&[2, 3, 6, 4], 9, 0.5);
        let decay = mlx_sys::broadcast_to(&MlxArray::from_f32(0.8), &[2, 3, 6], None);
        let beta = mlx_sys::broadcast_to(&MlxArray::from_f32(0.6), &[2, 3, 6], None);
        let state = input(&[2, 6, 4, 4], 13, 0.04);
        eval(&[&q, &k, &v, &state]);
        let initial = state.data_f32().to_vec();
        let mut expected = initial.clone();
        let mut outputs = vec![0.0_f32; 2 * 3 * 6 * 4];
        for b in 0..2 {
            for t in 0..3 {
                for h in 0..6 {
                    for d in 0..4 {
                        let s = (b * 6 + h) * 16 + d * 4;
                        let qk = ((b * 3 + t) * 2 + h / 3) * 4;
                        let value = ((b * 3 + t) * 6 + h) * 4 + d;
                        let prediction: f32 = (0..4)
                            .map(|j| expected[s + j] * 0.8 * k.data_f32()[qk + j])
                            .sum();
                        let correction = (v.data_f32()[value] - prediction) * 0.6;
                        for j in 0..4 {
                            expected[s + j] =
                                expected[s + j] * 0.8 + correction * k.data_f32()[qk + j];
                            outputs[value] += expected[s + j] * q.data_f32()[qk + j];
                        }
                    }
                }
            }
        }
        let (out, final_state) =
            portable_recurrence(&q, &k, &v, &decay, &beta, &state).expect("portable");
        eval(&[&out, &final_state]);
        for (a, b) in out.data_f32().iter().zip(outputs) {
            assert!((*a - b).abs() < 1e-7);
        }
        for (a, b) in final_state.data_f32().iter().zip(expected) {
            assert!((*a - b).abs() < 1e-7);
        }
        assert_eq!(state.data_f32(), initial);
        assert!(try_mlx_prefill(&q, &k, &v, &decay, &beta, &state).is_none());
    }

    #[test]
    fn mlx_prefill_preserves_strided_batches_gates_and_continuation() {
        let q = input(&[2, 271, 16, 128], 1, 0.03);
        let k = input(&[2, 271, 16, 128], 7, 0.12);
        let v = input(&[2, 271, 48, 128], 13, 0.3);
        let decay = mlx_sys::add(
            &input(&[2, 271, 48], 4, 0.4),
            &MlxArray::from_f32(0.75),
            None,
        );
        let beta = astype(
            &astype(
                &mlx_sys::add(
                    &input(&[2, 271, 48], 8, 0.4),
                    &MlxArray::from_f32(0.5),
                    None,
                ),
                MlxDtype::Bfloat16,
                None,
            ),
            MlxDtype::Float32,
            None,
        );
        let state = input(&[2, 48, 128, 128], 3, 0.04);
        let initial = contiguous(&state, None);
        eval(&[&initial]);
        let initial_values = initial.data_f32().to_vec();
        let rows = |x: &MlxArray, start, end| {
            let mut lo = vec![0; x.shape().len()];
            let mut hi = x.shape();
            lo[1] = start;
            hi[1] = end;
            slice(x, &lo, &hi, &vec![1; hi.len()], None)
        };
        let run = |start, end, state: &MlxArray, mlx: bool| {
            let (q, k, v, d, b) = (
                rows(&q, start, end),
                rows(&k, start, end),
                rows(&v, start, end),
                rows(&decay, start, end),
                rows(&beta, start, end),
            );
            if mlx {
                try_mlx_prefill(&q, &k, &v, &d, &b, state).expect("supported prefill")
            } else {
                portable_recurrence(&q, &k, &v, &d, &b, state).expect("portable")
            }
        };
        let (out, first) = run(1, 132, &state, true);
        let (ax_out, ax_first) = run(1, 132, &state, false);
        close(&out, &ax_out, 5e-5);
        close(&first, &ax_first, 5e-5);
        let (out2, second) = run(132, 263, &first, true);
        let (ax_out2, ax_second) = run(132, 263, &ax_first, false);
        close(&out2, &ax_out2, 5e-5);
        close(&second, &ax_second, 5e-5);
        let (full, full_state) = run(1, 263, &state, true);
        close(&rows(&full, 131, 262), &out2, 5e-5);
        close(&full_state, &second, 5e-5);
        assert_eq!(initial.data_f32(), initial_values);
        // The experiment cannot select the short verification or singleton paths.
        for seq in [1, 2, 4, 32, 127] {
            let (q, k, v, d, b) = (
                rows(&q, 0, seq),
                rows(&k, 0, seq),
                rows(&v, 0, seq),
                rows(&decay, 0, seq),
                rows(&beta, 0, seq),
            );
            assert!(try_mlx_prefill(&q, &k, &v, &d, &b, &state).is_none());
        }
        let bad_gate = mlx_sys::zeros(&[2, 271, 48], MlxDtype::Bfloat16, None);
        assert!(try_mlx_prefill(&q, &k, &v, &bad_gate, &beta, &state).is_none());
        assert!(portable_recurrence(&q, &k, &v, &bad_gate, &beta, &state).is_err());
    }

    #[test]
    fn mlx_prefill_extreme_decay_remains_finite_with_disclosed_clamp() {
        let q = input(&[1, 131, 16, 128], 1, 0.03);
        let k = input(&[1, 131, 16, 128], 7, 0.12);
        let v = input(&[1, 131, 48, 128], 13, 0.3);
        let state = input(&[1, 48, 128, 128], 3, 0.04);
        for gamma in [0.0, 1e-12, 1.0] {
            let decay = mlx_sys::broadcast_to(&MlxArray::from_f32(gamma), &[1, 131, 48], None);
            let beta = mlx_sys::zeros(&[1, 131, 48], MlxDtype::Float32, None);
            let (out, final_state) =
                try_mlx_prefill(&q, &k, &v, &decay, &beta, &state).expect("supported");
            let (ax_out, ax_state) =
                portable_recurrence(&q, &k, &v, &decay, &beta, &state).expect("portable");
            close(&out, &ax_out, 5e-5);
            close(&final_state, &ax_state, 5e-5);
        }
    }
}
