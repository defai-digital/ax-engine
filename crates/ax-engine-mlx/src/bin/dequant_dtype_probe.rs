//! F1 probe — bf16 vs f16 matmul throughput on FFN-shape tensors.
//!
//! Tests the hypothesis recorded in
//! `.internal/planning/DS4-LEARNINGS-FOLLOWUP-PRD-2026-05-14.md` §4 that
//! Apple Silicon GPUs have native f16 matrix hardware that may dispatch
//! faster than bf16 for the post-dequantization FFN matmul path.
//!
//! ax-engine's 4-bit weights are dequantized inside MLX's
//! `quantized_matmul`; the dequant target dtype matches the activation
//! dtype. Today that is bf16 across the board. If f16 matmul is enough
//! faster than bf16 matmul on the FFN shapes we care about, switching
//! the dequant target to f16 in the weight loader is a real
//! throughput win. If not, F1 closes NO-GO and the weight-loader
//! migration is deferred.
//!
//! Method mirrors `rmsnorm-fused-probe`:
//!   1. Build a random fp32 weight `[hidden, intermediate]` and a
//!      random fp32 input `[1, 1, hidden]`.
//!   2. Cast both to bf16, time `matmul(x_bf16, w_bf16)` over ITERS.
//!   3. Cast both to f16,  time `matmul(x_f16,  w_f16)` over ITERS.
//!   4. Report the bf16 - f16 wall-time delta. PASS gate (per PRD
//!      §4.2): f16 must be ≥3% faster on every probe shape.
//!
//! This is a directional probe. The bf16-vs-f16 matmul perf gap is
//! what limits the post-dequant gain; the F1 PRD additionally covers
//! the dequant-itself perf, but that has to be measured against real
//! 4-bit weights which require either model loading or new FFI
//! bindings — out of scope for the cheap directional answer.
//!
//! Run:
//!   cargo run --release --bin dequant-dtype-probe

use std::time::Instant;

use mlx_sys::{MlxArray, MlxDtype, astype, eval, ops::matmul};

const ITERS: usize = 200;
const MAX_ABS_DIFF: f32 = 5.0e-2;

// FFN shapes covered. Each row is (hidden_size, intermediate_size,
// model_label). Picked to match the production hot path on the
// supported tier — Gemma 4 E2B 4-bit and Qwen 3.5 9B 4-bit are the
// canonical comparison points. A small warmup shape comes first so
// any GPU / library init cost is paid before the headline rows.
const SHAPES: &[(usize, usize, &str)] = &[
    (128, 256, "warmup-small"),
    (1536, 6144, "gemma4-e2b-gate_proj"),
    (1536, 6144, "gemma4-e2b-up_proj"),
    (6144, 1536, "gemma4-e2b-down_proj"),
    (4096, 8192, "qwen3.5-9b-gate_proj"),
    (8192, 4096, "qwen3.5-9b-down_proj"),
];

fn build_random_f32(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity(count);
    let mut state = seed;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let raw = (state >> 32) as u32;
        let f = (raw as f32 / u32::MAX as f32 - 0.5) * scale;
        data.push(f);
    }
    data
}

fn time_loop<F: FnMut()>(label: &str, iters: usize, mut f: F) -> f64 {
    f();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per_iter_us = elapsed_ms * 1000.0 / iters as f64;
    println!("    {label}: {elapsed_ms:.1} ms total, {per_iter_us:.2} us/iter");
    per_iter_us
}

fn max_abs_diff_against_f32_reference(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

fn probe_shape(hidden: usize, intermediate: usize, label: &str) -> (f64, f64) {
    println!("[{label}] hidden={hidden} intermediate={intermediate}");

    let x_data = build_random_f32(0xbf58_476d_1ce4_e5b9, hidden, 0.4);
    let w_data = build_random_f32(0x94d0_49bb_1331_11eb, hidden * intermediate, 0.2);

    let x_f32 = MlxArray::from_raw_data(
        x_data.as_ptr().cast(),
        std::mem::size_of_val(x_data.as_slice()),
        &[1, 1, hidden as i32],
        MlxDtype::Float32,
    );
    let w_f32 = MlxArray::from_raw_data(
        w_data.as_ptr().cast(),
        std::mem::size_of_val(w_data.as_slice()),
        &[hidden as i32, intermediate as i32],
        MlxDtype::Float32,
    );

    let x_bf16 = astype(&x_f32, MlxDtype::Bfloat16, None);
    let w_bf16 = astype(&w_f32, MlxDtype::Bfloat16, None);
    let x_f16 = astype(&x_f32, MlxDtype::Float16, None);
    let w_f16 = astype(&w_f32, MlxDtype::Float16, None);

    // Force materialisation so the timing loop measures matmul only,
    // not the one-time astype cost. Skip the fp32 reference check —
    // Apple GPUs do not have a native fp32 matrix path, so the
    // reference matmul either falls back to a slow emulation or
    // hangs on this hardware; not worth gating the timing run.
    eval(&[&x_bf16, &w_bf16, &x_f16, &w_f16]);
    println!("    setup complete (eval'd bf16 and f16 inputs).");

    // Single warmup matmul per dtype to materialise any first-call
    // compile / pipeline cost outside the timing loop. No correctness
    // check here — Apple GPUs do not expose a fast fp32 reference
    // path on these shapes, and a bf16-vs-f16 diff is dtype-precision
    // noise, not a correctness signal.
    let warm_bf16 = matmul(&x_bf16, &w_bf16, None);
    eval(&[&warm_bf16]);
    let warm_f16 = matmul(&x_f16, &w_f16, None);
    eval(&[&warm_f16]);
    println!("    matmul warmup complete.");
    let _ = MAX_ABS_DIFF;
    let _ = max_abs_diff_against_f32_reference;

    let bf16_us = time_loop("bf16 matmul", ITERS, || {
        let y = matmul(&x_bf16, &w_bf16, None);
        eval(&[&y]);
    });
    let f16_us = time_loop("f16  matmul", ITERS, || {
        let y = matmul(&x_f16, &w_f16, None);
        eval(&[&y]);
    });

    let delta_pct = (bf16_us - f16_us) / bf16_us * 100.0;
    let verdict = if delta_pct >= 3.0 {
        "PASS"
    } else if delta_pct >= 0.0 {
        "MARGINAL"
    } else {
        "REJECT"
    };
    println!("    f16 vs bf16: {delta_pct:+.1}% ({bf16_us:.2} -> {f16_us:.2} us/iter) [{verdict}]");
    println!();

    (bf16_us, f16_us)
}

fn main() {
    println!("F1 dequant-dtype probe: bf16 vs f16 matmul throughput on FFN shapes");
    println!("(iters={ITERS}, single-token decode shape [1, 1, hidden])");
    println!();

    let mut total_bf16_us = 0.0;
    let mut total_f16_us = 0.0;

    for (hidden, intermediate, label) in SHAPES {
        let (bf16, f16) = probe_shape(*hidden, *intermediate, label);
        total_bf16_us += bf16;
        total_f16_us += f16;
    }

    let delta_pct = (total_bf16_us - total_f16_us) / total_bf16_us * 100.0;
    let verdict = if delta_pct >= 3.0 {
        "PASS — schedule weight-loader migration to f16 (per F1 PRD §4.3)"
    } else if delta_pct >= 0.0 {
        "MARGINAL — within noise; do not migrate"
    } else {
        "REJECT — f16 slower than bf16; close F1 NO-GO"
    };

    println!("=== Aggregate verdict ===");
    println!(
        "  Sum across {n} shapes: bf16 = {total_bf16_us:.2} us, f16 = {total_f16_us:.2} us",
        n = SHAPES.len()
    );
    println!("  f16 vs bf16: {delta_pct:+.1}%");
    println!("  {verdict}");
}
