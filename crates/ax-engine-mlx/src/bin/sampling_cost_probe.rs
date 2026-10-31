//! Per-token sampling cost probe for the pinned MLX wheel.
//!
//! Times what a decode step pays to pick its token from one logits row, for
//! the Qwen 3.8 vocabulary (248,320 ids):
//!
//! - `greedy`: `argmax` plus the one-id readback.
//! - `topk_<k>`: the GPU candidate selection `sampling.rs::gpu_top_candidates`
//!   performs for top-k sampling (temperature scale, `argpartition`, a
//!   full-vocab `softmax`, gather, eval) plus the candidate readback. The CPU
//!   draw over at most `k` candidates is not included.
//! - `topp_512`: the same selection with the 512 candidates top-p-only
//!   sampling transfers.
//! - `full_readback`: the full-row f32 readback the exact CPU path pays.
//!
//! The op sequence mirrors `gpu_top_candidates`; keep the two in step. The
//! result decides whether a fused top-k draw kernel is worth building: compare
//! the `topk_*` minus `greedy` gap with the host's decode step time.
//!
//! ```text
//! cargo run --release -p ax-engine-mlx --bin sampling_cost_probe
//! ```
//!
//! `AX_SAMPLING_PROBE_WARMUP` (default 5) and `AX_SAMPLING_PROBE_ITERS`
//! (default 50) select the repetitions. Times are upper medians in
//! microseconds of wall time, each ending in a host readback (a sync).

#[path = "common/mod.rs"]
mod common;

use std::process::ExitCode;
use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxDtype, argmax, argpartition_axis, astype, eval, eval_first_u32,
    gpu_device_architecture, multiply, reshape, runtime_version, slice, softmax, take_along_axis,
};

use common::{env_usize, median_upper};

const VOCAB: i32 = 248_320;
const TEMPERATURE: f32 = 0.7;
const TOP_K: [i32; 3] = [20, 40, 64];
const TOP_P_CANDIDATES: i32 = 512;

/// Deterministic logits with an LLM-like spread: a few strong ids over a
/// low, noisy floor. Values only shape the partition; timing is what counts.
fn synthetic_logits(vocab: i32) -> Vec<f32> {
    let mut state = 0x2545_f491_u32;
    (0..vocab)
        .map(|id| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / (1u32 << 24) as f32;
            if id % 9973 == 0 {
                18.0 + noise
            } else {
                noise * 6.0 - 3.0
            }
        })
        .collect()
}

/// `gpu_top_candidates` from `sampling.rs`, with the readback a sampler does.
fn top_candidates(logits: &MlxArray, count: i32) -> (Vec<u32>, Vec<f32>) {
    let inv_temp = MlxArray::from_f32(1.0 / TEMPERATURE);
    let scaled = multiply(logits, &inv_temp, None);
    let part = argpartition_axis(&scaled, -count, -1, None);
    let top_indices = slice(&part, &[0, VOCAB - count], &[1, VOCAB], &[1, 1], None);
    let top_probs = take_along_axis(&softmax(&scaled, -1, None), &top_indices, -1, None);
    let top_indices = astype(&top_indices, MlxDtype::Uint32, None);
    eval(&[&top_indices, &top_probs]);
    (
        top_indices.data_u32().to_vec(),
        top_probs.data_f32().to_vec(),
    )
}

fn time_us(warmup: usize, iters: usize, mut run: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        run();
    }
    let samples = (0..iters)
        .map(|_| {
            let started = Instant::now();
            run();
            started.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    median_upper(samples)
}

fn run() -> Result<(), String> {
    let warmup = env_usize("AX_SAMPLING_PROBE_WARMUP", 5)?;
    let iters = env_usize("AX_SAMPLING_PROBE_ITERS", 50)?.max(1);
    println!(
        "mlx {} on {}, vocab {VOCAB}, warmup {warmup}, iters {iters}",
        runtime_version()?,
        gpu_device_architecture().unwrap_or_else(|_| "unknown GPU".to_string())
    );
    let row = synthetic_logits(VOCAB);
    for dtype in [MlxDtype::Bfloat16, MlxDtype::Float32] {
        let logits = astype(
            &reshape(&MlxArray::from_f32_slice(&row), &[1, VOCAB], None),
            dtype,
            None,
        );
        eval(&[&logits]);
        let greedy = time_us(warmup, iters, || {
            std::hint::black_box(eval_first_u32(&argmax(&logits, None)));
        });
        println!("{dtype:?} greedy        {greedy:9.1} us");
        for count in TOP_K.into_iter().chain([TOP_P_CANDIDATES]) {
            let label = if count == TOP_P_CANDIDATES {
                format!("topp_{count}")
            } else {
                format!("topk_{count}")
            };
            let (ids, _) = top_candidates(&logits, count);
            if ids.len() != count as usize {
                return Err(format!(
                    "{label}: expected {count} candidates, got {}",
                    ids.len()
                ));
            }
            let cost = time_us(warmup, iters, || {
                std::hint::black_box(top_candidates(&logits, count));
            });
            println!(
                "{dtype:?} {label:13} {cost:9.1} us  (+{:.1} us over greedy)",
                cost - greedy
            );
        }
        let full = time_us(warmup, iters, || {
            let row = astype(&logits, MlxDtype::Float32, None);
            eval(&[&row]);
            std::hint::black_box(row.data_f32().to_vec());
        });
        println!("{dtype:?} full_readback {full:9.1} us");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sampling_cost_probe: {error}");
            ExitCode::FAILURE
        }
    }
}
