//! Regression coverage for the shim's process-static dual GPU streams.
//! MLX 0.32 registers a stream's Metal command encoder only on the thread that
//! created it, so a stream pair created once and reused from another thread
//! fails at eval with "There is no Stream(gpu, N) in current thread". MLX
//! treats the first thread that touches it as the main thread, so the test
//! thread warms MLX first, as the server's startup thread does, and the worker
//! threads below create and use the streams.

use mlx_sys::{MlxArray, ops, try_eval};

fn run_dual_stream_qmm() -> Result<(), String> {
    let n = 128 * 128;
    let w: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) / 13.0).collect();
    let w = ops::reshape(&MlxArray::from_f32_slice(&w), &[128, 128], None);
    let q = ops::quantize(
        &w,
        Some(64),
        Some(4),
        ops::MlxQuantizationMode::Affine,
        None,
        None,
    );
    let x: Vec<f32> = (0..4 * 128).map(|i| ((i % 7) as f32 - 3.0) / 7.0).collect();
    let x = ops::reshape(&MlxArray::from_f32_slice(&x), &[4, 128], None);
    let (gate, up) =
        ops::dual_stream_affine_qmm(&x, &q[0], &q[1], &q[2], &q[0], &q[1], &q[2], 64, 4, None)
            .ok_or_else(|| "dual_stream_affine_qmm refused valid inputs".to_string())?;
    try_eval(&[&gate, &up])
}

#[test]
fn dual_stream_qmm_evaluates_on_a_second_thread() {
    let one = MlxArray::from_f32_slice(&[1.0]);
    try_eval(&[&ops::add(&one, &one, None)]).expect("warm MLX on the test thread");
    std::thread::spawn(run_dual_stream_qmm)
        .join()
        .expect("first thread panicked")
        .expect("first thread eval");
    std::thread::spawn(run_dual_stream_qmm)
        .join()
        .expect("second thread panicked")
        .expect("second thread eval");
}
