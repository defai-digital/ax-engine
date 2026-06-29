//! EmbeddingGemma per-layer depth probe.
//!
//! Runs the Gemma3 encoder layer by layer, printing the mean-pooled hidden
//! state after each layer to stdout. A companion Python script
//! (`scripts/probe_embeddinggemma_depth.py`) runs the same prompt through
//! `mlx-embeddings` and computes per-layer cosine similarity to pinpoint
//! the first layer where AX drifts from the reference.
//!
//! Output format (text, one line per checkpoint):
//!   embed <hidden_size>
//!   layer <idx> <hidden_size>
//!   ...
//!   final <hidden_size>
//!
//! Each data line contains space-separated f32 values for batch row 0.
//!
//! Usage:
//!   embed_gemma_depth_probe <model_dir> < ids.txt > depths.txt
//!
//! stdin: whitespace-separated token-id lines (one sequence per line).
//! Only the first sequence in the batch is emitted for simplicity.

use std::io::{BufRead, Write};
use std::path::Path;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    model::{ModelConfig, forward_for_embedding_gemma3_depth_probe},
    weights::{ModelWeights, load_weights},
};
use mlx_sys::{MlxDtype, astype, eval};

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .expect("usage: embed_gemma_depth_probe <model_dir>");
    mlx_sys::enable_compile();
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("load artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights: ModelWeights = load_weights(&artifacts).expect("load weights");
    eprintln!(
        "embed_gemma_depth_probe: family={} layers={} hidden={}",
        cfg.model_family, cfg.layer_count, cfg.hidden_size,
    );

    let stdin = std::io::stdin();
    let batch: Vec<Vec<u32>> = stdin
        .lock()
        .lines()
        .map(|line| {
            line.expect("read stdin")
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect::<Vec<u32>>()
        })
        .filter(|ids| !ids.is_empty())
        .collect();
    if batch.is_empty() {
        return;
    }

    let (checkpoints, _lens) = forward_for_embedding_gemma3_depth_probe(&cfg, &weights, &batch);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let num_layers = checkpoints.len().saturating_sub(1);

    for (i, cp) in checkpoints.iter().enumerate() {
        let f32_cp = astype(cp, MlxDtype::Float32, None);
        eval(&[&f32_cp]);
        // Extract batch row 0 from [B, H] → first row [H].
        let hidden_size = f32_cp.shape()[1] as i32;
        let row = mlx_sys::slice(&f32_cp, &[0, 0], &[1, hidden_size], &[1, 1], None);
        eval(&[&row]);
        let v = row.data_f32().to_vec();

        let label = if i < num_layers {
            format!("layer {i}")
        } else {
            "final".to_string()
        };
        let vals: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
        writeln!(out, "{label} {}", vals.join(" ")).expect("write stdout");
    }
}
