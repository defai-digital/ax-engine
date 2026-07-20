//! First-token logit-dump probe.
//!
//! Loads a model, runs a prefill forward over the given prompt token ids, and
//! prints the top-K log-softmax of the LAST-position logits — i.e. the
//! distribution for the first generated token. This lets us compare AX's
//! first-token distribution against an external reference (e.g. `mlx_lm`) for
//! the *same* token ids, decisively separating a real numerical bug from benign
//! quant drift that only shows up after several greedy steps.
//!
//! Compares by token id (decode externally), so no tokenizer is needed here.
//!
//! Usage:
//!   cargo run --release --bin logit_dump_probe -- <model_dir> <id,id,...> [topk] [--dump=PATH]
//!
//! `--dump=PATH` additionally writes the FULL last-position logit vector to
//! `PATH` as raw little-endian f32 (one f32 per vocab entry, no header) for
//! offline comparison against an external reference.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    kv_cache::MlxKVCache,
    model::{ModelConfig, forward},
    weights::{ModelWeights, load_weights},
};
use mlx_sys::eval;

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
    // Separate `--dump=PATH` from the positional args so the existing
    // <model_dir> <ids> [topk] ordering is preserved.
    let mut dump_path: Option<String> = None;
    let positional: Vec<String> = env::args()
        .skip(1)
        .filter(|a| {
            if let Some(p) = a.strip_prefix("--dump=") {
                dump_path = Some(p.to_string());
                false
            } else {
                true
            }
        })
        .collect();

    let model_dir = positional.first().cloned().ok_or_else(|| {
        "usage: logit_dump_probe <model_dir> <id,id,...> [topk] [--dump=PATH]".to_string()
    })?;
    let ids = parse_token_ids(positional.get(1).ok_or_else(|| {
        "usage: logit_dump_probe <model_dir> <id,id,...> [topk] [--dump=PATH]".to_string()
    })?)?;
    let topk = positional
        .get(2)
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("topk must be a positive integer, got {value:?}"))
        })
        .transpose()?
        .unwrap_or(10);
    if topk == 0 {
        return Err("topk must be greater than zero".to_string());
    }
    if let Some(unexpected) = positional.get(3) {
        return Err(format!("unexpected argument: {unexpected}"));
    }
    if dump_path.as_deref() == Some("") {
        return Err("--dump requires a non-empty path".to_string());
    }

    println!("Loading model from {model_dir} ...");
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load model artifacts: {error}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights: Arc<ModelWeights> = Arc::new(
        load_weights(&artifacts).map_err(|error| format!("failed to load weights: {error}"))?,
    );

    // Prefill the whole prompt; `forward` returns the last token's logits [vocab].
    // `AX_PROBE_CHUNKED=1` routes the prefix through production chunked prefill
    // (default chunk size) and only the last token through `forward`, matching
    // the numeric path a served request takes instead of one giant forward.
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let chunked = std::env::var("AX_PROBE_CHUNKED").is_ok_and(|v| v == "1");
    let logits = if chunked && ids.len() > 1 {
        use ax_engine_mlx::generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill};
        use ax_engine_mlx::sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64};
        let prefix = &ids[..ids.len() - 1];
        let mut rng = Xorshift64::new(0);
        let _ = chunked_prefill(
            &cfg,
            &weights,
            prefix,
            &mut cache,
            DEFAULT_PREFILL_CHUNK,
            MlxSamplingRequest::new(MlxSamplingParams::greedy(), prefix),
            &mut rng,
        );
        forward(
            &cfg,
            &weights,
            &ids[ids.len() - 1..],
            &mut cache,
            prefix.len(),
        )
    } else {
        forward(&cfg, &weights, &ids, &mut cache, 0)
    };
    eval(&[&logits]);
    let lg = logits.data_f32();
    if lg.is_empty() || lg.iter().any(|value| !value.is_finite()) {
        return Err("model produced empty or non-finite logits".to_string());
    }

    // log-softmax: logp[i] = logit[i] - logsumexp(logits)
    let maxv = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for &v in lg {
        sum += ((v - maxv) as f64).exp();
    }
    let lse = maxv as f64 + sum.ln();

    let mut idx: Vec<usize> = (0..lg.len()).collect();
    idx.sort_unstable_by(|&a, &b| lg[b].total_cmp(&lg[a]));

    println!(
        "=== AX first-token top-{topk} (logprob)  [prompt = {} ids: {:?}] ===",
        ids.len(),
        ids
    );
    for &i in idx.iter().take(topk) {
        let logp = lg[i] as f64 - lse;
        println!("  {logp:8.4}  id={i:>6}  logit={:8.4}", lg[i]);
    }

    if let Some(path) = dump_path {
        let f = File::create(&path)
            .map_err(|error| format!("failed to create dump file {path:?}: {error}"))?;
        let mut w = BufWriter::new(f);
        for &v in lg {
            w.write_all(&v.to_le_bytes())
                .map_err(|error| format!("failed to write dump file {path:?}: {error}"))?;
        }
        w.flush()
            .map_err(|error| format!("failed to flush dump file {path:?}: {error}"))?;
        println!("Wrote {} f32 logits to {path}", lg.len());
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
