//! Phase 3.0 ceiling probe — does batched decode amortize the weight read?
//!
//! Decode is bandwidth/host-graph-encoding bound: batch=1 reads all active
//! weights and encodes the whole graph per token. A batched forward reads the
//! weights and encodes the graph ONCE for B tokens, so aggregate tok/s should
//! grow toward B× until some resource (bandwidth, compute) saturates. This
//! probe validates that premise on a dense full-attention model — the only
//! family the current `decode_batched_forward` supports — before any work to
//! extend batched decode to MoE / linear-attention.
//!
//! Method: prefill one short prompt (batch=1), seed a `BatchedDecodeSession`
//! with N clones of it, step the cohort and report aggregate tok/s at
//! B = 1, 2, 4, 8. Aggregate = B tokens per `step`, so tok/s = B / step_wall.
//!
//! Run (dense model only — Llama/Qwen dense, NOT MoE/linear-attn):
//!   cargo run -p ax-engine-microbench --release --bin batched-decode-ceiling-probe -- <dense_model_dir>

use std::env;
use std::path::Path;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    batched_decode_session::BatchedDecodeSession,
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill},
    kv_cache::MlxKVCache,
    model::ModelConfig,
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::load_weights,
};
use mlx_sys::clear_cache;

const BATCHES: [usize; 4] = [1, 2, 4, 8];
const STEPS: usize = 64;
const WARMUP: usize = 8;

fn main() {
    let model_dir = env::args()
        .nth(1)
        .expect("Usage: batched-decode-ceiling-probe <dense_model_dir> [prefill_len]");
    // Prefill length controls the KV size per row, i.e. how much of the step is
    // the per-request attention/KV path (which does NOT amortize across batch)
    // vs the weight-bound matmuls (which do). Sweep it to attribute the gap.
    let prefill_len: u32 = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let artifacts =
        NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).expect("load weights");

    // One batch=1 prefill; every slot is seeded from this same cache.
    let prompt: Vec<u32> = (1..=prefill_len).collect();
    let mut rng = Xorshift64::new(0);
    let mut prefill_cache = MlxKVCache::new(cfg.layer_count);
    let first_token = chunked_prefill(
        &cfg,
        &weights,
        &prompt,
        &mut prefill_cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), &prompt),
        &mut rng,
    );

    println!("model: {model_dir}  prefill_len={prefill_len}");
    println!("batch  agg_tok_s  per_req_tok_s  step_us  scaling_vs_b1");
    let mut b1_tok_s = 0.0f64;
    for &batch in &BATCHES {
        if batch > 64 {
            continue;
        }
        let mut session = BatchedDecodeSession::new(cfg.layer_count, batch);
        for slot in 0..batch {
            session.add(slot as u64, &prefill_cache, first_token);
        }
        for _ in 0..WARMUP {
            let _ = session.step(&cfg, &weights);
        }
        let started = Instant::now();
        for _ in 0..STEPS {
            let out = session.step(&cfg, &weights);
            debug_assert_eq!(out.len(), batch);
        }
        let wall_s = started.elapsed().as_secs_f64();
        let step_us = wall_s * 1e6 / STEPS as f64;
        let agg_tok_s = (batch * STEPS) as f64 / wall_s;
        let per_req = agg_tok_s / batch as f64;
        if batch == 1 {
            b1_tok_s = agg_tok_s;
        }
        println!(
            "{batch:>5}  {agg_tok_s:>9.1}  {per_req:>13.1}  {step_us:>7.0}  {:>5.2}x",
            agg_tok_s / b1_tok_s
        );
        // Per-stage breakdown when AX_MLX_BATCHED_PROFILE=1 (barriers on, so
        // the tok/s above is invalid in that mode — read the stage split, not
        // the rate). Accumulated over all layers × STEPS; report µs/step.
        let stage_us = ax_engine_mlx::model::take_batched_decode_profile();
        if stage_us.iter().any(|&u| u > 0) {
            let per_step: Vec<f64> = stage_us.iter().map(|&u| u as f64 / STEPS as f64).collect();
            let names = ax_engine_mlx::model::BATCHED_DECODE_PROFILE_STAGES;
            let parts: Vec<String> = names
                .iter()
                .zip(&per_step)
                .map(|(n, us)| format!("{n}={us:.0}"))
                .collect();
            println!("        stages µs/step: {}", parts.join("  "));
        }
        clear_cache();
    }
    println!();
    println!(
        "interpretation: agg_tok_s scaling toward Nx proves batched decode amortizes the \
         weight read / graph encoding; per_req_tok_s staying flat means no per-request penalty."
    );
}
