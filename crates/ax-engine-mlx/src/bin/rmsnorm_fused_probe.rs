//! Track K REQ-K2 microbenchmark.
//!
//! Compares a custom `MlxMetalKernel`-registered RMSNorm against MLX's
//! built-in `mlx_sys::fast::rms_norm` on identical inputs. Validates the
//! integration path (whether `mlx_fast_metal_kernel` can be a useful
//! production vehicle) and produces wall-clock evidence for ADR 0017.
//!
//! Run:
//!   cargo run --release --bin rmsnorm-fused-probe
//!
//! Output is a small JSON-ish summary suitable for pasting into
//! .internal/benchmark/track-k-*.md as the REQ-K2 evidence.

use std::time::Instant;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, eval,
    fast::rms_norm as mlx_rms_norm,
};

// Realistic decode-stage activation shape for Gemma 4 e2b (hidden_size=2048,
// one token per decode step). Batch=1, dim=2048.
const BATCH: usize = 1;
const DIM: usize = 2048;
const ITERS: usize = 1000;
const EPS: f32 = 1.0e-5;
// Encoded as int*1e9 so the kernel can reconstruct it via template arg.
const EPS_E9: i32 = (EPS * 1.0e9) as i32;
// Threads per threadgroup. 256 is a common Apple Silicon sweet spot.
const THREADS_PER_GROUP: i32 = 256;
// Numerical tolerance for the equivalence check. 1e-3 in f32 is loose enough
// to absorb the difference between reduction tree shapes while still catching
// algorithmic errors.
const MAX_ABS_DIFF: f32 = 1.0e-3;

const KERNEL_SOURCE: &str = r#"
    const uint tid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.x;
    if (row >= ROWS) {
        return;
    }

    threadgroup float partial[32];

    // Phase 1: per-thread sum of squares over a strided slice of the row.
    float thread_sq = 0.0f;
    for (uint i = tid; i < D; i += THREADS) {
        const float v = (float)x[row * D + i];
        thread_sq += v * v;
    }

    // Phase 2: simdgroup-level reduction.
    const float sg_sum = simd_sum(thread_sq);
    const uint sg_idx = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    if (lane == 0) {
        partial[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: first simdgroup aggregates the partials into partial[0].
    const uint n_simd = simdgroups_per_threadgroup;
    if (sg_idx == 0) {
        const float p = (lane < n_simd) ? partial[lane] : 0.0f;
        const float total = simd_sum(p);
        if (lane == 0) {
            partial[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: normalize and scale.
    const float mean_sq = partial[0] / float(D);
    const float eps = float(EPS_E9) * 1.0e-9f;
    const float denom = rsqrt(mean_sq + eps);
    for (uint i = tid; i < D; i += THREADS) {
        const float v = (float)x[row * D + i];
        out[row * D + i] = v * denom * (float)w[i];
    }
"#;

fn build_input_data() -> Vec<f32> {
    // Deterministic pseudo-random activations. The exact distribution doesn't
    // matter as long as both kernels see the same bytes.
    let mut data = Vec::with_capacity(BATCH * DIM);
    let mut state: u64 = 0x9e3779b97f4a7c15;
    for _ in 0..(BATCH * DIM) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let raw = (state >> 32) as u32;
        let f = (raw as f32 / u32::MAX as f32) - 0.5;
        data.push(f);
    }
    data
}

fn build_weight_data() -> Vec<f32> {
    // RMSNorm weight close to 1.0 (typical post-training distribution).
    let mut data = Vec::with_capacity(DIM);
    let mut state: u64 = 0xbf58476d1ce4e5b9;
    for _ in 0..DIM {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let raw = (state >> 32) as u32;
        let f = (raw as f32 / u32::MAX as f32) * 0.2 + 0.9;
        data.push(f);
    }
    data
}

fn run_custom_kernel(kernel: &MlxMetalKernel, x: &MlxArray, w: &MlxArray) -> MlxArray {
    let outputs = kernel.apply_with_template(
        &[x, w],
        &[KernelOutputSpec {
            shape: vec![BATCH as i32, DIM as i32],
            dtype: MlxDtype::Float32,
        }],
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
        // Grid: ROWS threadgroups; each threadgroup is THREADS threads wide.
        (BATCH as i32 * THREADS_PER_GROUP, 1, 1),
        (THREADS_PER_GROUP, 1, 1),
        None,
    );
    outputs
        .into_iter()
        .next()
        .expect("kernel must produce one output")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn time_loop<F: FnMut()>(label: &str, iters: usize, mut f: F) -> f64 {
    // Warm up the path once.
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
    println!("RMSNorm fused-kernel probe (batch={BATCH}, dim={DIM}, iters={ITERS}, eps={EPS:.0e})");
    println!();

    let x_data = build_input_data();
    let w_data = build_weight_data();
    let x = MlxArray::from_raw_data(
        x_data.as_ptr().cast(),
        std::mem::size_of_val(x_data.as_slice()),
        &[BATCH as i32, DIM as i32],
        MlxDtype::Float32,
    );
    let w = MlxArray::from_raw_data(
        w_data.as_ptr().cast(),
        std::mem::size_of_val(w_data.as_slice()),
        &[DIM as i32],
        MlxDtype::Float32,
    );

    // Build the custom kernel once.
    let kernel = MlxMetalKernel::new(
        "ax_rmsnorm_fused_probe",
        &["x", "w"],
        &["out"],
        KERNEL_SOURCE,
        "", // no extra header
        true,
    );

    // 1. Correctness check on a single application.
    println!("=== Numerical equivalence ===");
    let mlx_out = mlx_rms_norm(&x, Some(&w), EPS, None);
    eval(&[&mlx_out]);
    let custom_out = run_custom_kernel(&kernel, &x, &w);
    eval(&[&custom_out]);

    let mlx_values = mlx_out.data_f32();
    let custom_values = custom_out.data_f32();
    let diff = max_abs_diff(mlx_values, custom_values);
    println!("  max abs diff: {diff:.6}");
    println!("  tolerance:    {MAX_ABS_DIFF:.6}");
    if diff > MAX_ABS_DIFF {
        println!("  status: FAIL (custom kernel disagrees with MLX native)");
        println!();
        println!(
            "  sample MLX:    {:?}",
            &mlx_values[..8.min(mlx_values.len())]
        );
        println!(
            "  sample custom: {:?}",
            &custom_values[..8.min(custom_values.len())]
        );
        std::process::exit(1);
    }
    println!("  status: ok");
    println!();

    // 2. Wall-clock comparison.
    println!("=== Wall-clock (iters={ITERS}) ===");
    let mlx_us = time_loop("mlx_native rms_norm", ITERS, || {
        let y = mlx_rms_norm(&x, Some(&w), EPS, None);
        eval(&[&y]);
    });
    let custom_us = time_loop("custom MlxMetalKernel", ITERS, || {
        let y = run_custom_kernel(&kernel, &x, &w);
        eval(&[&y]);
    });

    // 3. Verdict.
    println!();
    println!("=== Verdict ===");
    let delta_pct = (mlx_us - custom_us) / mlx_us * 100.0;
    println!("  custom is {delta_pct:+.1}% vs MLX native ({mlx_us:.2} -> {custom_us:.2} us/iter)");
    if delta_pct >= 3.0 {
        println!("  PASS: custom kernel beats MLX by >=3% (ADR 0017 gate)");
    } else if delta_pct >= 0.0 {
        println!(
            "  MARGINAL: custom kernel ties or marginally beats MLX (<3%, below ADR 0017 gate)"
        );
    } else {
        println!("  REJECT: custom kernel is slower than MLX native; do not wire");
    }
}
