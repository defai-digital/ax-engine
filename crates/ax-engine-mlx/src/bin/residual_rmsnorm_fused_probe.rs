//! Phase B.1 microbench — residual-add + RMSNorm fused kernel.
//!
//! Compares a custom `MlxMetalKernel` that performs
//! `residual = x + attn; out = rms_norm(residual) * w` in one Metal dispatch
//! against the two-op MLX sequence (`mlx_sys::ops::add` followed by
//! `mlx_sys::fast::rms_norm`). MLX has no built-in fused
//! residual-rmsnorm op, so this is exactly the gap a sidecar kernel can
//! plausibly close on the Gemma 4 E2B `post_attn_residual_norm`
//! sub-stage (one of the two main residual+norm sites per layer; ~46%
//! of decode wall is post_attn aggregate).
//!
//! The earlier `rmsnorm-fused-probe` established that MLX's
//! `fast::rms_norm` alone is ~0.6% faster than the same kernel written
//! by hand. The new question for Phase B.1: does eliminating one full
//! traversal of `hidden_size` data (the intermediate `residual` write+read
//! between MLX `add` and `rms_norm`) and one Metal command-buffer submit
//! recover that gap and then some?
//!
//! Run:
//!   cargo run --release --bin residual-rmsnorm-fused-probe
//!
//! Output mirrors the rmsnorm probe — bit-exact equivalence check,
//! wall-clock comparison, verdict bucket. PASS gate is +3% vs the
//! two-op MLX path (matches the rmsnorm probe's ADR 0017 reference).

use std::time::Instant;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, eval,
    fast::rms_norm as mlx_rms_norm, ops::add as mlx_add,
};

// Gemma 4 E2B decode shape: batch=1, hidden_size=2048 (rounded for the
// probe). Same constants as the rmsnorm probe so the two artifacts diff
// cleanly.
const BATCH: usize = 1;
const DIM: usize = 2048;
const ITERS: usize = 1000;
const EPS: f32 = 1.0e-5;
const EPS_E9: i32 = (EPS * 1.0e9) as i32;
const THREADS_PER_GROUP: i32 = 256;
const MAX_ABS_DIFF: f32 = 1.0e-3;

// One pass over each row:
//   1. Compute residual[i] = x[i] + attn[i] and accumulate sum of squares.
//   2. Reduce sum of squares across simdgroups.
//   3. Normalize and scale: out[i] = residual[i] * rsqrt(mean_sq + eps) * w[i].
// The kernel writes both `residual` and `out` outputs; the production wiring
// will use the residual output as the next layer's input and the normed
// output as the FFN input, exactly matching the existing
// `let hidden = add(...); let normed = rms_norm(hidden, w, eps);` pattern.
const KERNEL_SOURCE: &str = r#"
    const uint tid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.x;
    if (row >= ROWS) {
        return;
    }

    threadgroup float partial[32];

    float thread_sq = 0.0f;
    for (uint i = tid; i < D; i += THREADS) {
        const uint k = row * D + i;
        const float v = (float)x[k] + (float)attn[k];
        residual[k] = v;
        thread_sq += v * v;
    }

    const float sg_sum = simd_sum(thread_sq);
    const uint sg_idx = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    if (lane == 0) {
        partial[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_simd = simdgroups_per_threadgroup;
    if (sg_idx == 0) {
        const float p = (lane < n_simd) ? partial[lane] : 0.0f;
        const float total = simd_sum(p);
        if (lane == 0) {
            partial[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean_sq = partial[0] / float(D);
    const float eps = float(EPS_E9) * 1.0e-9f;
    const float denom = rsqrt(mean_sq + eps);
    for (uint i = tid; i < D; i += THREADS) {
        const uint k = row * D + i;
        const float v = residual[k];
        out[k] = v * denom * (float)w[i];
    }
"#;

fn xorshift_seed(seed: &mut u64) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 32) as u32
}

fn build_random_input(seed: u64, mean: f32, scale: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity(DIM);
    let mut state: u64 = seed;
    for _ in 0..DIM {
        let raw = xorshift_seed(&mut state);
        let f = (raw as f32 / u32::MAX as f32) * scale + mean;
        data.push(f);
    }
    data
}

fn run_custom_kernel(
    kernel: &MlxMetalKernel,
    x: &MlxArray,
    attn: &MlxArray,
    w: &MlxArray,
) -> (MlxArray, MlxArray) {
    let outputs = kernel.apply_with_template(
        &[x, attn, w],
        &[
            KernelOutputSpec {
                shape: vec![BATCH as i32, DIM as i32],
                dtype: MlxDtype::Float32,
            },
            KernelOutputSpec {
                shape: vec![BATCH as i32, DIM as i32],
                dtype: MlxDtype::Float32,
            },
        ],
        &[
            KernelTemplateArg::Int {
                name: "ROWS",
                value: BATCH as i32,
            },
            KernelTemplateArg::Int {
                name: "D",
                value: DIM as i32,
            },
            KernelTemplateArg::Int {
                name: "THREADS",
                value: THREADS_PER_GROUP,
            },
            KernelTemplateArg::Int {
                name: "EPS_E9",
                value: EPS_E9,
            },
        ],
        (BATCH as i32 * THREADS_PER_GROUP, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
        None,
    );
    let mut iter = outputs.into_iter();
    let residual = iter.next().expect("kernel must produce residual output");
    let normed = iter.next().expect("kernel must produce normed output");
    (residual, normed)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn time_loop<F: FnMut()>(label: &str, iters: usize, mut f: F) -> f64 {
    f();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_iter_us = elapsed_ms * 1000.0 / iters as f64;
    println!("  {label}: {elapsed_ms:.1} ms total, {per_iter_us:.2} us/iter");
    per_iter_us
}

fn main() {
    println!(
        "residual+RMSNorm fused probe (batch={BATCH}, dim={DIM}, iters={ITERS}, eps={EPS:.0e})"
    );
    println!();

    let x_data = build_random_input(0xbf58_476d_1ce4_e5b9, 0.0, 0.5);
    let attn_data = build_random_input(0x94d0_49bb_1331_11eb, 0.0, 0.3);
    let w_data = build_random_input(0xbb67_ae85_84ca_a73b, 0.9, 0.2);

    let x = MlxArray::from_raw_data(
        x_data.as_ptr().cast(),
        std::mem::size_of_val(x_data.as_slice()),
        &[BATCH as i32, DIM as i32],
        MlxDtype::Float32,
    );
    let attn = MlxArray::from_raw_data(
        attn_data.as_ptr().cast(),
        std::mem::size_of_val(attn_data.as_slice()),
        &[BATCH as i32, DIM as i32],
        MlxDtype::Float32,
    );
    let w = MlxArray::from_raw_data(
        w_data.as_ptr().cast(),
        std::mem::size_of_val(w_data.as_slice()),
        &[DIM as i32],
        MlxDtype::Float32,
    );

    let kernel = MlxMetalKernel::new(
        "ax_residual_rmsnorm_fused_probe",
        &["x", "attn", "w"],
        &["residual", "out"],
        KERNEL_SOURCE,
        "",
        true,
    );

    // 1. Correctness against the two-op MLX sequence.
    println!("=== Numerical equivalence ===");
    let mlx_residual = mlx_add(&x, &attn, None);
    let mlx_out = mlx_rms_norm(&mlx_residual, Some(&w), EPS, None);
    eval(&[&mlx_residual, &mlx_out]);

    let (custom_residual, custom_out) = run_custom_kernel(&kernel, &x, &attn, &w);
    eval(&[&custom_residual, &custom_out]);

    let mlx_residual_values = mlx_residual.data_f32();
    let mlx_out_values = mlx_out.data_f32();
    let custom_residual_values = custom_residual.data_f32();
    let custom_out_values = custom_out.data_f32();

    let residual_diff = max_abs_diff(mlx_residual_values, custom_residual_values);
    let out_diff = max_abs_diff(mlx_out_values, custom_out_values);
    println!("  residual max abs diff: {residual_diff:.6}");
    println!("  out max abs diff:      {out_diff:.6}");
    println!("  tolerance:             {MAX_ABS_DIFF:.6}");
    if residual_diff > MAX_ABS_DIFF || out_diff > MAX_ABS_DIFF {
        println!("  status: FAIL (custom kernel disagrees with MLX two-op sequence)");
        std::process::exit(1);
    }
    println!("  status: ok");
    println!();

    // 2. Wall-clock comparison.
    println!("=== Wall-clock (iters={ITERS}) ===");
    let mlx_us = time_loop("mlx_native add + rms_norm", ITERS, || {
        let r = mlx_add(&x, &attn, None);
        let y = mlx_rms_norm(&r, Some(&w), EPS, None);
        eval(&[&r, &y]);
    });
    let custom_us = time_loop("custom fused MlxMetalKernel", ITERS, || {
        let (r, y) = run_custom_kernel(&kernel, &x, &attn, &w);
        eval(&[&r, &y]);
    });

    // 3. Verdict.
    println!();
    println!("=== Verdict ===");
    let delta_pct = (mlx_us - custom_us) / mlx_us * 100.0;
    println!(
        "  custom is {delta_pct:+.1}% vs MLX two-op sequence ({mlx_us:.2} -> {custom_us:.2} us/iter)"
    );
    if delta_pct >= 3.0 {
        println!("  PASS: fused kernel beats two-op sequence by >=3% (ADR 0017 gate)");
    } else if delta_pct >= 0.0 {
        println!("  MARGINAL: fused kernel ties or marginally beats sequence (<3%, below gate)");
    } else {
        println!("  REJECT: fused kernel is slower than two-op sequence; do not wire");
    }
}
