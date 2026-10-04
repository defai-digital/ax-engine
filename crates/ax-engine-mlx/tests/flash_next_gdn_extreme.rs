//! Extreme gate controls for the dedicated, FP32 Flash Next prefill experiment.

use ax_engine_mlx::flash_next_gdn::{portable_recurrence, try_mlx_prefill};
use mlx_sys::{MlxArray, MlxDtype, contiguous, eval};

fn array(shape: &[i32], phase: usize, scale: f32) -> MlxArray {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let values: Vec<f32> = (0..n)
        .map(|i| (((i + phase) * 37 % 997) as f32 / 997.0 - 0.5) * scale)
        .collect();
    MlxArray::from_raw_data(values.as_ptr().cast(), n * 4, shape, MlxDtype::Float32)
}

fn compare(actual: &MlxArray, expected: &MlxArray) {
    assert_eq!(actual.shape(), expected.shape());
    let actual = contiguous(actual, None);
    let expected = contiguous(expected, None);
    eval(&[&actual, &expected]);
    let error = actual
        .data_f32()
        .iter()
        .zip(expected.data_f32())
        .fold(0.0_f64, |m, (&a, &b)| {
            assert!(a.is_finite() && b.is_finite());
            m.max((f64::from(a) - f64::from(b)).abs())
        });
    assert!(error <= 5e-5, "extreme gate max_abs={error}");
}

#[test]
#[allow(clippy::expect_used)]
fn nonzero_beta_with_reset_and_carry_decay_preserves_partial_chunks() {
    let q = array(&[2, 131, 16, 128], 1, 0.03);
    let k = array(&[2, 131, 16, 128], 7, 0.12);
    let v = array(&[2, 131, 48, 128], 13, 0.30);
    let state = array(&[2, 48, 128, 128], 19, 0.04);
    for mode in 0..3 {
        let gates: Vec<f32> = (0..2 * 131 * 48)
            .map(|i| match mode {
                0 => 0.0,
                1 => 1.0,
                _ => [0.0, 1e-12, 0.5, 1.0][i % 4],
            })
            .collect();
        let decay = MlxArray::from_raw_data(
            gates.as_ptr().cast(),
            gates.len() * 4,
            &[2, 131, 48],
            MlxDtype::Float32,
        );
        let beta = mlx_sys::broadcast_to(&MlxArray::from_f32(1.0), &[2, 131, 48], None);
        let (y, s) =
            try_mlx_prefill(&q, &k, &v, &decay, &beta, &state).expect("supported Flash Next shape");
        let (ax_y, ax_s) =
            portable_recurrence(&q, &k, &v, &decay, &beta, &state).expect("portable reference");
        compare(&y, &ax_y);
        compare(&s, &ax_s);
    }
}
