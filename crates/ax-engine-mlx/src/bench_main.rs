//! Prefill + decode rate benchmark for the MLX runner.
//!
//! Measures both standard greedy decode and n-gram speculative decode so the
//! acceptance rate and speedup are visible side-by-side.
//!
//! Usage: cargo run --release --bin mlx-bench -- <model_dir>

use std::env;
use std::path::Path;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill, decode_step},
    kv_cache::MlxKVCache,
    model::ModelConfig,
    speculative::{DEFAULT_DRAFT_LEN, NgramTable, speculative_decode_step},
    weights::load_weights,
};
use mlx_sys::clear_cache;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 0 { (v[n/2 - 1] + v[n/2]) / 2.0 } else { v[n/2] }
}

fn main() {
    let model_dir = env::args().nth(1)
        .expect("Usage: mlx-bench <model_dir>");

    println!("Loading model from {model_dir}...");
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .expect("Failed to load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts)
        .expect("Failed to load weights");
    let weights = std::sync::Arc::new(weights);

    println!("JIT warm-up...");
    {
        let mut cache = MlxKVCache::new(cfg.layer_count);
        decode_step(&cfg, &weights, 0, &mut cache);
        clear_cache();
    }
    println!("Warm-up done.\n");

    const PREFILL_LEN: usize = 512;
    const DECODE_STEPS: usize = 100;
    const RUNS: usize = 5;

    // ── Prefill benchmark ─────────────────────────────────────────────────────
    let prompt: Vec<u32> = (1..=(PREFILL_LEN as u32)).collect();
    let mut prefill_ms: Vec<f64> = Vec::new();
    for run in 0..RUNS {
        let mut cache = MlxKVCache::new(cfg.layer_count);
        let t0 = Instant::now();
        let _first = chunked_prefill(&cfg, &weights, &prompt, &mut cache, DEFAULT_PREFILL_CHUNK);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        clear_cache();
        if run > 0 { prefill_ms.push(ms); }
        println!("  prefill run {run}: {ms:.1} ms  ({:.0} tok/s)", PREFILL_LEN as f64 / (ms / 1000.0));
    }
    let prefill_med_ms = median(prefill_ms);
    let prefill_tps = PREFILL_LEN as f64 / (prefill_med_ms / 1000.0);
    println!();

    // ── Standard (greedy) decode benchmark ────────────────────────────────────
    let short_prompt: Vec<u32> = (1..=32).collect();
    let mut decode_step_ms: Vec<f64> = Vec::new();
    for run in 0..RUNS {
        let mut cache = MlxKVCache::new(cfg.layer_count);
        let mut tok = chunked_prefill(&cfg, &weights, &short_prompt, &mut cache, DEFAULT_PREFILL_CHUNK);
        let mut step_times = Vec::with_capacity(DECODE_STEPS);
        for _ in 0..DECODE_STEPS {
            let t0 = Instant::now();
            tok = decode_step(&cfg, &weights, tok, &mut cache);
            step_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let _ = tok;
        clear_cache();
        let steady: Vec<f64> = step_times[5..].to_vec();
        let med = median(steady.clone());
        if run > 0 { decode_step_ms.extend(steady); }
        println!("  greedy decode run {run}: {med:.2} ms/tok  ({:.1} tok/s)", 1000.0 / med);
    }
    let decode_med_ms = median(decode_step_ms);
    let decode_tps = 1000.0 / decode_med_ms;
    println!();

    // ── Speculative decode benchmark ──────────────────────────────────────────
    // Use a repeating prompt so the n-gram table can build up quickly and we
    // see a realistic acceptance rate.
    let spec_prompt: Vec<u32> = (1..=64).cycle().take(128).map(|x: u32| x).collect();
    let mut spec_tps_runs: Vec<f64> = Vec::new();
    let mut total_accepted = 0usize;
    let mut total_steps = 0usize;

    for run in 0..RUNS {
        let mut cache = MlxKVCache::new(cfg.layer_count);
        let mut ngram = NgramTable::new();
        ngram.feed(&spec_prompt);
        let mut tok = chunked_prefill(&cfg, &weights, &spec_prompt, &mut cache, DEFAULT_PREFILL_CHUNK);

        let t0 = Instant::now();
        let mut tokens_generated = 0usize;
        let mut accepted_run = 0usize;
        let mut steps_run = 0usize;

        // Run until we have generated at least DECODE_STEPS effective tokens.
        while tokens_generated < DECODE_STEPS {
            let emitted = speculative_decode_step(
                &cfg, &weights, &mut cache, &mut ngram, tok, DEFAULT_DRAFT_LEN,
            );
            // emitted[0] is the current step's output; emitted[1..] are bonus.
            // Accepted draft tokens = emitted.len() - 1 (the last is correction/bonus).
            let accepted_this = emitted.len().saturating_sub(1);
            accepted_run += accepted_this;
            tokens_generated += emitted.len();
            steps_run += 1;
            tok = *emitted.last().unwrap();
        }

        let elapsed_s = t0.elapsed().as_secs_f64();
        let tps = tokens_generated as f64 / elapsed_s;
        clear_cache();

        if run > 0 {
            spec_tps_runs.push(tps);
            total_accepted += accepted_run;
            total_steps += steps_run;
        }
        let accept_rate = if steps_run > 0 {
            accepted_run as f64 / (steps_run * DEFAULT_DRAFT_LEN) as f64
        } else { 0.0 };
        println!(
            "  spec decode run {run}: {tps:.1} tok/s  (accept={:.0}%, {tokens_generated} tok in {steps_run} steps)",
            accept_rate * 100.0,
        );
    }
    let spec_med_tps = median(spec_tps_runs);
    let overall_accept = if total_steps > 0 {
        total_accepted as f64 / (total_steps * DEFAULT_DRAFT_LEN) as f64
    } else { 0.0 };

    println!();
    println!("=== Results ===");
    println!("  Prefill ({PREFILL_LEN} tokens):    {prefill_tps:.1} tok/s  ({prefill_med_ms:.0} ms)");
    println!("  Greedy decode:             {decode_tps:.1} tok/s  ({decode_med_ms:.2} ms/tok)");
    println!(
        "  Spec decode (draft={DEFAULT_DRAFT_LEN}):  {spec_med_tps:.1} tok/s  (accept={:.0}%, speedup={:.2}x)",
        overall_accept * 100.0,
        spec_med_tps / decode_tps,
    );
}
