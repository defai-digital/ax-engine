//! Cost of one target verify forward as a function of the number of rows.
//!
//! Step 0 of the lattice-tree port: a draft tree only pays off if verifying N
//! nodes in one forward costs little more than verifying the 8-row DFlash2
//! chain. This probe prefills a prompt, then times `forward_all_positions` plus
//! argmax and eval on a throwaway cache clone for each row count, interleaving
//! the row counts so drift on a shared machine hits every arm equally.
//!
//! The rows are verified as a linear chain. Above the fused Gated-DeltaNet
//! verify limit (8 rows) the linear layers leave the fused kernel, so the times
//! at more than 8 rows are an upper bound on what a tree kernel would pay.
//!
//! Usage:
//!   cargo run --release --bin verify_rows_cost_probe -- <model_dir>
//!
//! Env:
//!   AX_ROWS         comma-separated row counts (default "1,2,4,8,12,16,24,32")
//!   AX_ROWS_REPEATS timed passes per row count (default 20)
//!   AX_PROMPT_LEN   synthetic prompt length (default 512)

#[path = "common/mod.rs"]
mod common;

use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill_with_final_hidden},
    kv_cache::MlxKVCache,
    model::{ModelConfig, forward_all_positions},
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::load_weights,
};
use mlx_sys::{argmax, clear_cache, eval};

use common::{env_usize, median, optional_env, parse_positive_usize};

fn run() -> Result<(), String> {
    let model_dir = env::args()
        .nth(1)
        .ok_or_else(|| "usage: verify_rows_cost_probe <model_dir>".to_string())?;
    let rows: Vec<usize> = optional_env("AX_ROWS")?
        .unwrap_or_else(|| "1,2,4,8,12,16,24,32".to_string())
        .split(',')
        .map(|value| parse_positive_usize("AX_ROWS", value.trim()))
        .collect::<Result<_, _>>()?;
    let repeats = env_usize("AX_ROWS_REPEATS", 20)?;
    let prompt_len = env_usize("AX_PROMPT_LEN", 512)?;

    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load model artifacts: {error}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights =
        load_weights(&artifacts).map_err(|error| format!("failed to load weights: {error}"))?;

    let prompt: Vec<u32> = (1..=prompt_len as u32).collect();
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let _ = chunked_prefill_with_final_hidden(
        &cfg,
        &weights,
        &prompt,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), &prompt),
        &mut rng,
    );
    let token_offset = prompt.len();

    let once = |n: usize| -> f64 {
        let ids: Vec<u32> = (0..n as u32).map(|i| 1000 + i).collect();
        let mut clone = cache.clone();
        let start = Instant::now();
        let logits = forward_all_positions(&cfg, &weights, &ids, &mut clone, token_offset);
        let predicted = argmax(&logits, None);
        eval(&[&predicted]);
        let ms = start.elapsed().as_secs_f64() * 1e3;
        drop(clone);
        clear_cache();
        ms
    };

    for &n in &rows {
        for _ in 0..3 {
            once(n); // warm-up: kernel compilation and allocation
        }
    }
    let mut times: Vec<Vec<f64>> = vec![Vec::with_capacity(repeats); rows.len()];
    for _ in 0..repeats {
        for (slot, &n) in rows.iter().enumerate() {
            times[slot].push(once(n));
        }
    }

    println!("prompt_tokens={prompt_len} repeats={repeats}");
    println!(
        "{:>5} {:>10} {:>10} {:>10}",
        "rows", "median_ms", "min_ms", "ms/row"
    );
    for (slot, &n) in rows.iter().enumerate() {
        let min = times[slot].iter().copied().fold(f64::INFINITY, f64::min);
        let mid = median(times[slot].clone());
        println!("{n:>5} {mid:>10.2} {min:>10.2} {:>10.2}", mid / n as f64);
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}
