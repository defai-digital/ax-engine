//! EmbeddingGemma correctness probe.
//!
//! Reads whitespace-separated token-id lines from stdin, runs them as one
//! production EmbeddingGemma batch (`forward_for_embedding_batch` → bidirectional
//! Gemma3 encoder), mean-pools each real sequence, applies the Dense head,
//! L2-normalizes, and prints one space-separated f32 embedding per line to
//! stdout. A Python harness compares these against mlx-embeddings reference
//! vectors (cosine ≈ 1.0).
//!
//! Usage:  embed_gemma_verify <model_dir>  < ids.txt  > embs.txt

use std::io::{BufRead, Write};
use std::path::Path;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    model::{ModelConfig, forward_for_embedding_batch},
    weights::{ModelWeights, QuantizedWeight, load_weights},
};
use mlx_sys::{
    MlxArray, MlxDtype, astype, eval, matmul,
    ops::{slice, sum_axis},
    quantized_matmul, transpose,
};

fn qmm(x: &MlxArray, w: &QuantizedWeight) -> MlxArray {
    if let Some(scales) = &w.scales {
        quantized_matmul(
            x,
            &w.weight,
            scales,
            w.biases.as_ref(),
            true,
            Some(w.group_size),
            Some(w.bits),
            None,
        )
    } else {
        matmul(x, &transpose(&w.weight, &[1, 0], None), None)
    }
}

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .expect("usage: embed_gemma_verify <model_dir>");
    mlx_sys::enable_compile();
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("load artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights: ModelWeights = load_weights(&artifacts).expect("load weights");
    eprintln!(
        "embed_gemma_verify: family={} layers={} hidden={} dense0={} dense1={}",
        cfg.model_family,
        cfg.layer_count,
        cfg.hidden_size,
        weights.embedding_dense_0.is_some(),
        weights.embedding_dense_1.is_some(),
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

    let (hidden, lens) = forward_for_embedding_batch(&cfg, &weights, &batch, None);
    let hidden_size = cfg.hidden_size as i32;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for (row, &len) in lens.iter().enumerate() {
        let row_hidden = slice(
            &hidden,
            &[row as i32, 0, 0],
            &[row as i32 + 1, len as i32, hidden_size],
            &[1, 1, 1],
            None,
        );
        let l = len as f32;
        let summed = sum_axis(&row_hidden, 1, false, None);
        let len_arr = MlxArray::from_raw_data(
            &l as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1_i32, 1_i32],
            MlxDtype::Float32,
        );
        let len_bf16 = astype(&len_arr, hidden.dtype(), None);
        let pooled = mlx_sys::divide(&summed, &len_bf16, None);
        // Dense head: dense.1(dense.0(pooled)).
        let pooled = match (&weights.embedding_dense_0, &weights.embedding_dense_1) {
            (Some(d0), Some(d1)) => qmm(&qmm(&pooled, d0), d1),
            _ => pooled,
        };
        let pooled_f32 = astype(&pooled, MlxDtype::Float32, None);
        eval(&[&pooled_f32]);
        let mut v = pooled_f32.data_f32().to_vec();
        // L2 normalize.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= norm;
        }
        let line: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
        writeln!(out, "{}", line.join(" ")).expect("write stdout");
    }
}
