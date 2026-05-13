//! Direct-Rust embedding bench — measures `EngineSession::embed` latency
//! without any Python/PyO3 boundary. Used to quantify how much of the
//! ax-engine-py single-call gap on small models (e.g. 0.6B) comes from
//! the Python binding layer vs the runtime/kernel itself.
//!
//! Usage:
//!   cargo run -p ax-engine-bench --example embed_rust_bench --release -- \
//!     --model-dir .internal/models/qwen3-embedding-0.6b-8bit \
//!     [--seq 10] [--trials 20] [--warmup 5]
//!
//! Run on the same model dir as `scripts/bench_embedding_models.py` to get
//! a paired measurement. The output is a single median ms/sentence; compare
//! it to the Python in-process number from the bench JSON to estimate the
//! PyO3 + wrapper overhead per call.

use std::path::PathBuf;
use std::time::Instant;

use ax_engine_core::{CacheGroupId, EmbeddingPooling};
use ax_engine_sdk::{
    EngineSession, EngineSessionConfig, PreviewBackendRequest, PreviewSessionConfigRequest,
};

struct CliArgs {
    model_dir: PathBuf,
    /// Single-sentence mode: one `embed()` call per trial at this seq length.
    seq: usize,
    /// Batched mode: one `embed_batch()` call per trial. When set, the bench
    /// builds N sequences with the given comma-separated seq lengths
    /// (matching `scripts/bench_embedding_models.py`'s 10-sentence corpus
    /// by default).
    batch: Option<Vec<usize>>,
    trials: usize,
    warmup: usize,
}

fn parse_args() -> CliArgs {
    let mut args = std::env::args().skip(1);
    let mut model_dir: Option<PathBuf> = None;
    let mut seq: usize = 10;
    let mut batch: Option<Vec<usize>> = None;
    let mut trials: usize = 20;
    let mut warmup: usize = 5;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => {
                model_dir = args.next().map(PathBuf::from);
            }
            "--seq" => {
                seq = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--seq <usize>");
            }
            "--batch" => {
                let spec = args.next().expect("--batch <a,b,c,...>");
                let lens: Vec<usize> = spec
                    .split(',')
                    .map(|s| s.trim().parse().expect("--batch list of usize"))
                    .collect();
                batch = Some(lens);
            }
            "--trials" => {
                trials = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--trials <usize>");
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--warmup <usize>");
            }
            other => {
                eprintln!("unexpected argument: {other}");
                std::process::exit(2);
            }
        }
    }
    CliArgs {
        model_dir: model_dir.expect("--model-dir <path> is required"),
        seq,
        batch,
        trials,
        warmup,
    }
}

fn main() {
    let args = parse_args();
    eprintln!(
        "[embed-rust-bench] model={} {}",
        args.model_dir.display(),
        match &args.batch {
            Some(lens) =>
                format!("mode=batch n={} lens={:?} warmup={} trials={}", lens.len(), lens, args.warmup, args.trials),
            None =>
                format!("mode=single seq={} warmup={} trials={}", args.seq, args.warmup, args.trials),
        }
    );

    let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
        cache_group_id: CacheGroupId(0),
        block_size_tokens: 16,
        total_blocks: 1024,
        deterministic: true,
        max_batch_tokens: 2048,
        mlx_runtime_artifacts_dir: None,
        backend_request: PreviewBackendRequest::shipping_mlx(),
        mlx_model_artifacts_dir: Some(args.model_dir.clone()),
        mlx_disable_ngram_acceleration: false,
        mlx_kv_compression: ax_engine_sdk::MlxKvCompressionConfig::disabled(),
    })
    .expect("config");
    let session = EngineSession::new(config).expect("session");

    // Total token count is needed for tok/s — same accounting as the Python
    // bench script (sum of all sequence lengths in the batch / wall time).
    let (per_trial_total_tokens, batch_len, single_seq) = match &args.batch {
        Some(lens) => (lens.iter().sum::<usize>(), lens.len(), 0usize),
        None => (args.seq, 1, args.seq),
    };

    // Synthetic token ids — real callers would tokenize first; here we want
    // to measure embed()/embed_batch() itself, not tokenization.
    let single_input: Vec<u32> = (0..single_seq as u32).collect();
    let batch_input: Vec<Vec<u32>> = args
        .batch
        .as_ref()
        .map(|lens| {
            lens.iter()
                .map(|&l| (0..l as u32).collect::<Vec<u32>>())
                .collect()
        })
        .unwrap_or_default();

    eprintln!("[embed-rust-bench] warmup × {}", args.warmup);
    for _ in 0..args.warmup {
        match &args.batch {
            Some(_) => {
                let _ = session
                    .embed_batch(&batch_input, EmbeddingPooling::Last, true)
                    .expect("embed_batch");
            }
            None => {
                let _ = session
                    .embed(&single_input, EmbeddingPooling::Last, true)
                    .expect("embed");
            }
        }
    }

    let mut ms_samples: Vec<f64> = Vec::with_capacity(args.trials);
    for i in 0..args.trials {
        let t0 = Instant::now();
        match &args.batch {
            Some(_) => {
                let _ = session
                    .embed_batch(&batch_input, EmbeddingPooling::Last, true)
                    .expect("embed_batch");
            }
            None => {
                let _ = session
                    .embed(&single_input, EmbeddingPooling::Last, true)
                    .expect("embed");
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        ms_samples.push(ms);
        eprintln!("  trial {:>2}: {:.3} ms", i + 1, ms);
    }

    ms_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ms_samples.len();
    let median = ms_samples[n / 2];
    let min = ms_samples[0];
    let max = ms_samples[n - 1];
    let mean: f64 = ms_samples.iter().sum::<f64>() / n as f64;
    let ms_per_sentence = median / (batch_len as f64);
    let tps = (per_trial_total_tokens as f64) / (median / 1000.0);

    println!();
    println!(
        "per-call ms  min={:.3}  median={:.3}  max={:.3}  mean={:.3}",
        min, median, max, mean
    );
    println!(
        "ms/sentence  {:.3}    tok/s {:.1}",
        ms_per_sentence, tps
    );
}
