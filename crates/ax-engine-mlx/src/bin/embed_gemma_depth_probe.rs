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
use std::process::ExitCode;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    model::{ModelConfig, forward_for_embedding_gemma3_depth_probe},
    weights::{ModelWeights, load_weights},
};
use mlx_sys::{MlxDtype, astype, eval};

fn read_batch() -> Result<Vec<Vec<u32>>, String> {
    let stdin = std::io::stdin();
    let mut batch = Vec::new();
    for (line_index, line) in stdin.lock().lines().enumerate() {
        let line = line.map_err(|error| format!("failed to read stdin: {error}"))?;
        let ids = line
            .split_whitespace()
            .map(|token| {
                token.parse::<u32>().map_err(|_| {
                    format!(
                        "stdin line {} contains invalid token id {token:?}",
                        line_index + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !ids.is_empty() {
            batch.push(ids);
        }
    }
    if batch.is_empty() {
        return Err("stdin must contain at least one token-id sequence".to_string());
    }
    Ok(batch)
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .ok_or_else(|| "usage: embed_gemma_depth_probe <model_dir>".to_string())?;
    if let Some(unexpected) = args.next() {
        return Err(format!("unexpected argument: {unexpected}"));
    }
    mlx_sys::enable_compile();
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load model artifacts: {error}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights: ModelWeights =
        load_weights(&artifacts).map_err(|error| format!("failed to load weights: {error}"))?;
    eprintln!(
        "embed_gemma_depth_probe: family={} layers={} hidden={}",
        cfg.model_family, cfg.layer_count, cfg.hidden_size,
    );

    let batch = read_batch()?;

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
        writeln!(out, "{label} {}", vals.join(" "))
            .map_err(|error| format!("failed to write stdout: {error}"))?;
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
