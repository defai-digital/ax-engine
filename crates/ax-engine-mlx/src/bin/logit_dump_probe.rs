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
//!   cargo run --release --bin logit_dump_probe -- <model_dir> <id,id,...> [topk]

use std::env;
use std::path::Path;
use std::sync::Arc;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    kv_cache::MlxKVCache,
    model::{ModelConfig, forward},
    weights::{ModelWeights, load_weights},
};
use mlx_sys::eval;

fn main() {
    let model_dir = env::args()
        .nth(1)
        .expect("usage: logit_dump_probe <model_dir> <id,id,...> [topk]");
    let ids: Vec<u32> = env::args()
        .nth(2)
        .expect("usage: logit_dump_probe <model_dir> <id,id,...> [topk]")
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter_map(|t| t.trim().parse::<u32>().ok())
        .collect();
    let topk: usize = env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    assert!(!ids.is_empty(), "empty token ids");

    println!("Loading model from {model_dir} ...");
    let artifacts =
        NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights: Arc<ModelWeights> = Arc::new(load_weights(&artifacts).expect("load weights"));

    // Prefill the whole prompt; `forward` returns the last token's logits [vocab].
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let logits = forward(&cfg, &weights, &ids, &mut cache, 0);
    eval(&[&logits]);
    let lg = logits.data_f32();

    // log-softmax: logp[i] = logit[i] - logsumexp(logits)
    let maxv = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for &v in lg {
        sum += ((v - maxv) as f64).exp();
    }
    let lse = maxv as f64 + sum.ln();

    let mut idx: Vec<usize> = (0..lg.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        lg[b]
            .partial_cmp(&lg[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "=== AX first-token top-{topk} (logprob)  [prompt = {} ids: {:?}] ===",
        ids.len(),
        ids
    );
    for &i in idx.iter().take(topk) {
        let logp = lg[i] as f64 - lse;
        println!("  {logp:8.4}  id={i:>6}  logit={:8.4}", lg[i]);
    }
}
