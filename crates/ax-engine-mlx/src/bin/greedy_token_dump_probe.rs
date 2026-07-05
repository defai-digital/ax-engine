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
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    generate::{
        DEFAULT_PREFILL_CHUNK, advance_direct_pipeline_with_timings_and_turboquant_context,
        chunked_prefill, start_direct_pipeline,
    },
    kv_cache::MlxKVCache,
    model::ModelConfig,
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::load_weights,
};

fn main() {
    let mut args = env::args().skip(1);
    let model_dir = args
        .next()
        .expect("usage: greedy_token_dump_probe <model_dir> <id,id,...> [steps]");
    let ids: Vec<u32> = args
        .next()
        .expect("usage: greedy_token_dump_probe <model_dir> <id,id,...> [steps]")
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter_map(|t| t.trim().parse::<u32>().ok())
        .collect();
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    assert!(!ids.is_empty(), "empty token ids");

    eprintln!("loading {model_dir}");
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .expect("failed to load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).expect("failed to load weights");

    // Optional prefill timing reps (for windowed-view A/B): each rep prefills
    // into a fresh cache; the last rep's cache is used for decode.
    let prefill_reps: usize = env::var("AX_PROBE_PREFILL_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
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
        let advanced = advance_direct_pipeline_with_timings_and_turboquant_context(
            &cfg, &weights, &pending, &mut cache, None,
        );
        generated.push(advanced.token);
        pending = advanced.next_pending;
    }

    let rendered: Vec<String> = generated.iter().map(|t| t.to_string()).collect();
    println!("{}", rendered.join(","));
}
