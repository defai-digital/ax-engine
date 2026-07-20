//! Greedy token-dump parity probe.
//!
//! Runs the production decode path (chunked prefill → direct pipeline) with
//! greedy sampling over caller-provided prompt token ids and prints the
//! generated token ids, one line, comma-separated. Compare the output against
//! an external reference (e.g. `mlx_lm` `generate_step` with `sampler=None`)
//! on the same ids to get a token-exact end-to-end decode check without the
//! server stack.
//!
//! Usage:
//!   cargo run --release --bin greedy_token_dump_probe -- <model_dir> <id,id,...> [steps]

use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    generate::{
        DEFAULT_PREFILL_CHUNK, advance_direct_pipeline_with_timings, chunked_prefill,
        start_direct_pipeline,
    },
    kv_cache::MlxKVCache,
    model::ModelConfig,
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::load_weights,
};

fn parse_token_ids(spec: &str) -> Result<Vec<u32>, String> {
    let ids = spec
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.trim().is_empty())
        .map(|token| {
            token
                .trim()
                .parse::<u32>()
                .map_err(|_| format!("invalid token id {token:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        return Err("token id list must not be empty".to_string());
    }
    Ok(ids)
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let model_dir = args.next().ok_or_else(|| {
        "usage: greedy_token_dump_probe <model_dir> <id,id,...> [steps]".to_string()
    })?;
    let ids = parse_token_ids(&args.next().ok_or_else(|| {
        "usage: greedy_token_dump_probe <model_dir> <id,id,...> [steps]".to_string()
    })?)?;
    let steps = args
        .next()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("steps must be a positive integer, got {value:?}"))
        })
        .transpose()?
        .unwrap_or(64);
    if steps == 0 {
        return Err("steps must be greater than zero".to_string());
    }
    if let Some(unexpected) = args.next() {
        return Err(format!("unexpected argument: {unexpected}"));
    }

    eprintln!("loading {model_dir}");
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load model artifacts: {error}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights =
        load_weights(&artifacts).map_err(|error| format!("failed to load weights: {error}"))?;

    // Optional prefill timing reps (for windowed-view A/B): each rep prefills
    // into a fresh cache; the last rep's cache is used for decode.
    let prefill_reps = env::var("AX_PROBE_PREFILL_REPS")
        .ok()
        .map(|value| {
            value.parse::<usize>().map_err(|_| {
                format!("AX_PROBE_PREFILL_REPS must be a positive integer, got {value:?}")
            })
        })
        .transpose()?
        .unwrap_or(1);
    if prefill_reps == 0 {
        return Err("AX_PROBE_PREFILL_REPS must be greater than zero".to_string());
    }
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut bootstrap_tok = 0u32;
    for rep in 0..prefill_reps {
        if rep > 0 {
            cache = MlxKVCache::new(cfg.layer_count);
        }
        let mut rng = Xorshift64::new(0);
        let started = Instant::now();
        bootstrap_tok = chunked_prefill(
            &cfg,
            &weights,
            &ids,
            &mut cache,
            DEFAULT_PREFILL_CHUNK,
            MlxSamplingRequest::new(MlxSamplingParams::greedy(), &ids),
            &mut rng,
        );
        eprintln!(
            "prefill rep {rep}: {} tokens in {:.1} ms",
            ids.len(),
            started.elapsed().as_secs_f64() * 1e3
        );
    }

    let mut generated = vec![bootstrap_tok];
    let mut pending = start_direct_pipeline(&cfg, &weights, bootstrap_tok, &mut cache);
    while generated.len() < steps {
        let advanced = advance_direct_pipeline_with_timings(&cfg, &weights, &pending, &mut cache);
        generated.push(advanced.token);
        pending = advanced.next_pending;
    }

    let rendered: Vec<String> = generated.iter().map(|t| t.to_string()).collect();
    println!("{}", rendered.join(","));
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
