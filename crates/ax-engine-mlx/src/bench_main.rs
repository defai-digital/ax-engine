//! Prefill + decode rate benchmark for the MLX runner.
//!
//! Measures both direct decode and n-gram acceleration so the
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
    ngram_accel::{DEFAULT_DRAFT_LEN, NgramTable, ngram_accel_decode_step},
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::load_weights,
};
use mlx_sys::clear_cache;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn main() {
    let model_dir = env::args().nth(1).expect("Usage: mlx-bench <model_dir>");

    println!("Loading model from {model_dir}...");
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .expect("Failed to load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).expect("Failed to load weights");
    let weights = std::sync::Arc::new(weights);

    println!("JIT warm-up...");
    {
        let mut cache = MlxKVCache::new(cfg.layer_count);
        let mut rng = Xorshift64::new(0);
        decode_step(
            &cfg,
            &weights,
            0,
            &mut cache,
            MlxSamplingParams::greedy(),
            &mut rng,
        );
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
        let mut rng = Xorshift64::new(run as u64);
        let t0 = Instant::now();
        let _first = chunked_prefill(
            &cfg,
            &weights,
            &prompt,
            &mut cache,
            DEFAULT_PREFILL_CHUNK,
            MlxSamplingRequest::new(MlxSamplingParams::greedy(), &prompt),
            &mut rng,
        );
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        clear_cache();
        if run > 0 {
            prefill_ms.push(ms);
        }
        println!(
            "  prefill run {run}: {ms:.1} ms  ({:.0} tok/s)",
            PREFILL_LEN as f64 / (ms / 1000.0)
        );
    }
    let prefill_med_ms = median(prefill_ms);
    let prefill_tps = PREFILL_LEN as f64 / (prefill_med_ms / 1000.0);
    println!();

    // ── Direct decode benchmark ────────────────────────────────────
    let short_prompt: Vec<u32> = (1..=32).collect();
    let mut decode_step_ms: Vec<f64> = Vec::new();
    for run in 0..RUNS {
        let mut cache = MlxKVCache::new(cfg.layer_count);
        let mut rng = Xorshift64::new(0);
        let mut tok = chunked_prefill(
            &cfg,
            &weights,
            &short_prompt,
            &mut cache,
            DEFAULT_PREFILL_CHUNK,
            MlxSamplingRequest::new(MlxSamplingParams::greedy(), &short_prompt),
            &mut rng,
        );
        let mut step_times = Vec::with_capacity(DECODE_STEPS);
        for _ in 0..DECODE_STEPS {
            let t0 = Instant::now();
            tok = decode_step(
                &cfg,
                &weights,
                tok,
                &mut cache,
                MlxSamplingParams::greedy(),
                &mut rng,
            );
            step_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let _ = tok;
        clear_cache();
        let steady: Vec<f64> = step_times[5..].to_vec();
        let med = median(steady.clone());
        if run > 0 {
            decode_step_ms.extend(steady);
        }
        println!(
            "  direct decode run {run}: {med:.2} ms/tok  ({:.1} tok/s)",
            1000.0 / med
        );
    }
    let decode_med_ms = median(decode_step_ms);
    let decode_tps = 1000.0 / decode_med_ms;
    println!();

    // ── N-gram acceleration benchmark ──────────────────────────────────────────
    // Use a repeating prompt so the n-gram table can build up quickly and we
    // see a realistic acceptance rate.
    let ngram_prompt: Vec<u32> = (1..=64).cycle().take(128).map(|x: u32| x).collect();
    let mut ngram_tps_runs: Vec<f64> = Vec::new();
    let mut total_accepted = 0usize;
    // Track actual draft attempts (steps where ngram predicted; single-decode
    // fallbacks attempt 0 drafts and must not inflate the denominator).
    let mut total_draft_attempts = 0usize;
    let mut total_tokens = 0usize;
    let mut total_steps = 0usize;

    for run in 0..RUNS {
        let mut cache = MlxKVCache::new(cfg.layer_count);
        let mut ngram = NgramTable::new();
        let mut rng = Xorshift64::new(0);
        ngram.feed(&ngram_prompt);
        let mut tok = chunked_prefill(
            &cfg,
            &weights,
            &ngram_prompt,
            &mut cache,
            DEFAULT_PREFILL_CHUNK,
            MlxSamplingRequest::new(MlxSamplingParams::greedy(), &ngram_prompt),
            &mut rng,
        );

        let t0 = Instant::now();
        let mut tokens_generated = 0usize;
        let mut accepted_run = 0usize;
        let mut draft_attempts_run = 0usize;
        let mut steps_run = 0usize;

        // Run until we have generated at least DECODE_STEPS effective tokens.
        while tokens_generated < DECODE_STEPS {
            let draft = ngram.predict(DEFAULT_DRAFT_LEN);
            let emitted = ngram_accel_decode_step(
                &cfg,
                &weights,
                &mut cache,
                &mut ngram,
                tok,
                &draft,
                MlxSamplingParams::greedy(),
                &mut rng,
            );
            // emitted[0]       — output token for this step
            // emitted[1..n-1]  — bonus tokens (accepted drafts already in KV)
            // emitted[last]    — seed for next model run (correction or bonus)
            //
            // Accepted draft count = emitted.len() - 1 when an n-gram acceleration pass
            // ran.  Single-decode fallback returns len=1, accepted=0, and also
            // attempted 0 drafts — exclude it from the acceptance denominator.
            let attempted_drafts = if emitted.len() > 1 {
                DEFAULT_DRAFT_LEN
            } else {
                0
            };
            let accepted_this = emitted.len().saturating_sub(1);
            accepted_run += accepted_this;
            draft_attempts_run += attempted_drafts;
            tokens_generated += emitted.len();
            steps_run += 1;
            tok = *emitted.last().unwrap();
        }

        let elapsed_s = t0.elapsed().as_secs_f64();
        let tps = tokens_generated as f64 / elapsed_s;
        // Effective tokens per model invocation: how many output tokens each
        // GPU forward pass produces on average (>1.0 means n-gram acceleration paid off).
        let effective_tpm = tokens_generated as f64 / steps_run as f64;
        clear_cache();

        if run > 0 {
            ngram_tps_runs.push(tps);
            total_accepted += accepted_run;
            total_draft_attempts += draft_attempts_run;
            total_tokens += tokens_generated;
            total_steps += steps_run;
        }
        let accept_rate = if draft_attempts_run > 0 {
            accepted_run as f64 / draft_attempts_run as f64
        } else {
            0.0
        };
        println!(
            "  ngram run {run}: {tps:.1} tok/s  \
             accept={:.0}%  effective={effective_tpm:.2} tok/model-run  \
             ({tokens_generated} tok in {steps_run} steps)",
            accept_rate * 100.0,
        );
    }
    let ngram_med_tps = median(ngram_tps_runs);
    let overall_accept = if total_draft_attempts > 0 {
        total_accepted as f64 / total_draft_attempts as f64
    } else {
        0.0
    };
    let overall_effective_tpm = if total_steps > 0 {
        total_tokens as f64 / total_steps as f64
    } else {
        0.0
    };

    println!();
    println!("=== Results ===");
    println!(
        "  Prefill ({PREFILL_LEN} tokens):    {prefill_tps:.1} tok/s  ({prefill_med_ms:.0} ms)"
    );
    println!("  Direct decode:             {decode_tps:.1} tok/s  ({decode_med_ms:.2} ms/tok)");
    println!(
        "  N-gram acceleration (draft={DEFAULT_DRAFT_LEN}):  {ngram_med_tps:.1} tok/s  \
         accept={:.0}%  effective={overall_effective_tpm:.2} tok/model-run  speedup={:.2}x",
        overall_accept * 100.0,
        ngram_med_tps / decode_tps,
    );
}
