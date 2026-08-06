//! AXQuant kernel-latency probe — decode/prefill affine-qmm timings.
//!
//! Sweeps the exact `mlx_sys::ops::quantized_matmul` dispatch the engine's
//! weight path executes, per (bits, group size, hidden size), plus the bf16
//! matmul reference, and emits one machine-readable JSON document on stdout.
//! AXQuant ingests it with `axquant benchmark-kernels --from-ax-engine` and
//! binds it into a host-scoped `axquant.kernel-latency.v1` table that
//! `axquant plan --latency-table` consumes (runtime co-design loop).
//!
//! Human-readable progress goes to stderr; stdout carries exactly one JSON
//! line so subprocess consumers never have to strip log noise.
//!
//! Run:
//!   cargo run -p ax-engine-microbench --release --bin axquant-kernel-latency-probe

use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxDtype, MlxQuantizationMode, astype, eval,
    ops::{matmul, quantize, quantized_matmul},
};

const HIDDEN_SIZES: &[usize] = &[2048, 4096];
const BITS: &[i32] = &[2, 3, 4, 6, 8];
const GROUP_SIZES: &[i32] = &[32, 64, 128];
const PREFILL_ROWS: usize = 512;
const ITERS: usize = 20;
const WARMUP: usize = 5;

fn build_random_f32(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity(count);
    let mut state = seed;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let raw = (state >> 32) as u32;
        data.push((raw as f32 / u32::MAX as f32 - 0.5) * scale);
    }
    data
}

fn array_from_f32(data: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data),
        shape,
        MlxDtype::Float32,
    )
}

/// Median and relative interquartile dispersion over timed iterations.
fn timed_median_us<F: FnMut() -> MlxArray>(mut op: F) -> (f64, f64) {
    for _ in 0..WARMUP {
        let out = op();
        eval(&[&out]);
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let out = op();
        eval(&[&out]);
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = samples[samples.len() / 2];
    let q1 = samples[samples.len() / 4];
    let q3 = samples[(samples.len() * 3) / 4];
    let dispersion = if median > 0.0 {
        ((q3 - q1) / median).max(0.0)
    } else {
        0.0
    };
    (median, dispersion)
}

fn entry_json(
    method: &str,
    bits: i32,
    group_size: Option<i32>,
    hidden_size: usize,
    decode_us: f64,
    prefill_us: f64,
    dispersion: f64,
) -> serde_json::Value {
    serde_json::json!({
        "method": method,
        "bits": bits,
        "group_size": group_size,
        "hidden_size": hidden_size,
        "decode_median_us": decode_us,
        "prefill_median_us": prefill_us,
        "dispersion": dispersion,
        "iterations": ITERS,
    })
}

fn main() {
    let mut entries: Vec<serde_json::Value> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    for &hidden in HIDDEN_SIZES {
        eprintln!("[hidden={hidden}] building inputs");
        let w_data = build_random_f32(0x94d0_49bb_1331_11eb, hidden * hidden, 0.2);
        let decode_data = build_random_f32(0xbf58_476d_1ce4_e5b9, hidden, 0.4);
        let prefill_data = build_random_f32(0x2545_f491_4f6c_dd1d, PREFILL_ROWS * hidden, 0.4);

        let w_bf16 = astype(
            &array_from_f32(&w_data, &[hidden as i32, hidden as i32]),
            MlxDtype::Bfloat16,
            None,
        );
        let decode_x = astype(
            &array_from_f32(&decode_data, &[1, hidden as i32]),
            MlxDtype::Bfloat16,
            None,
        );
        let prefill_x = astype(
            &array_from_f32(&prefill_data, &[PREFILL_ROWS as i32, hidden as i32]),
            MlxDtype::Bfloat16,
            None,
        );
        eval(&[&w_bf16, &decode_x, &prefill_x]);

        let (decode_us, decode_disp) = timed_median_us(|| matmul(&decode_x, &w_bf16, None));
        let (prefill_us, _) = timed_median_us(|| matmul(&prefill_x, &w_bf16, None));
        eprintln!("  bf16: decode {decode_us:.1} us, prefill {prefill_us:.1} us");
        entries.push(entry_json(
            "bf16",
            16,
            None,
            hidden,
            decode_us,
            prefill_us,
            decode_disp,
        ));

        for &bits in BITS {
            for &group_size in GROUP_SIZES {
                if hidden % (group_size as usize) != 0 {
                    warnings.push(format!(
                        "skipped bits={bits} group={group_size} hidden={hidden}: \
                         hidden not divisible by group"
                    ));
                    continue;
                }
                let parts = quantize(
                    &w_bf16,
                    Some(group_size),
                    Some(bits),
                    MlxQuantizationMode::Affine,
                    None,
                    None,
                );
                let (Some(packed), Some(scales), Some(biases)) =
                    (parts.first(), parts.get(1), parts.get(2))
                else {
                    warnings.push(format!(
                        "skipped bits={bits} group={group_size} hidden={hidden}: \
                         quantize returned {} parts",
                        parts.len()
                    ));
                    continue;
                };
                eval(&[packed, scales, biases]);

                let (decode_us, decode_disp) = timed_median_us(|| {
                    quantized_matmul(
                        &decode_x,
                        packed,
                        scales,
                        Some(biases),
                        true,
                        Some(group_size),
                        Some(bits),
                        None,
                    )
                });
                let (prefill_us, _) = timed_median_us(|| {
                    quantized_matmul(
                        &prefill_x,
                        packed,
                        scales,
                        Some(biases),
                        true,
                        Some(group_size),
                        Some(bits),
                        None,
                    )
                });
                eprintln!(
                    "  {bits}b g{group_size}: decode {decode_us:.1} us, \
                     prefill {prefill_us:.1} us"
                );
                entries.push(entry_json(
                    "affine",
                    bits,
                    Some(group_size),
                    hidden,
                    decode_us,
                    prefill_us,
                    decode_disp,
                ));
            }
        }
    }

    let document = serde_json::json!({
        "schema_version": "ax-engine.kernel-latency-raw.v1",
        "ax_engine_version": env!("CARGO_PKG_VERSION"),
        "prefill_rows": PREFILL_ROWS,
        "warmup_iterations": WARMUP,
        "entries": entries,
        "warnings": warnings,
    });
    println!("{document}");
}
