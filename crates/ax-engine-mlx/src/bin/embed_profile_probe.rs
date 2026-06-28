//! Batched-embedding per-stage profile probe.
//!
//! Drives the production imperative batched-embedding forward
//! (`forward_for_embedding_batch` → `layer_forward_dense_embed`) with
//! `AX_MLX_EMBED_PROFILE=1` and prints the per-stage wall-time breakdown
//! collected by `take_embed_profile_snapshot()`.
//!
//! This localises where the batched embedding path spends time — the README's
//! Qwen3-Embedding-0.6B-8bit batch=8 long-chunk regression (−25%/−38% vs
//! `mlx-lm`). Each stage timer forces an `eval()` barrier, so absolute numbers
//! are inflated relative to the pipelined production path; read the **ratios**
//! (compile on/off is throughput-neutral here, so the imperative breakdown is
//! representative). See `docs/EMBEDDINGS.md`.
//!
//! Usage:
//!   cargo run --release --bin embed_profile_probe -- <model_dir> [batch] [seq] [calls]
//!
//! Defaults: batch=8, seq=256, calls=5 (after 2 warmups). The README regression
//! cell is `batch=8 seq=256`; pass `1 256` / `8 64` to contrast.

use std::path::Path;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    diagnostics::take_embed_profile_snapshot,
    model::{ModelConfig, forward_for_embedding_batch},
    weights::load_weights,
};
use mlx_sys::{enable_compile, eval};

fn arg_usize(idx: usize, default: usize) -> usize {
    std::env::args()
        .nth(idx)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // Set before the first `embed_profile_enabled()` read (it caches via OnceLock).
    unsafe { std::env::set_var("AX_MLX_EMBED_PROFILE", "1") };

    let model_dir = std::env::args()
        .nth(1)
        .expect("usage: embed_profile_probe <model_dir> [batch] [seq] [calls]");
    let batch = arg_usize(2, 8);
    let seq = arg_usize(3, 256);
    let calls = arg_usize(4, 5);
    let warmups = 2usize;

    enable_compile();
    let artifacts =
        NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).expect("load weights");

    // Synthetic deterministic token ids; compute time does not depend on values.
    // Keep ids well inside the vocab range.
    let batch_token_ids: Vec<Vec<u32>> = (0..batch)
        .map(|i| {
            (0..seq)
                .map(|j| ((i * seq + j + 1) % 30000) as u32)
                .collect()
        })
        .collect();
    // Last-token pooling positions (the bench default).
    let target_positions: Vec<usize> = batch_token_ids.iter().map(|ids| ids.len() - 1).collect();

    eprintln!(
        "embed_profile_probe: model_family={} layers={} hidden={} heads={}/{} head_dim={} | batch={batch} seq={seq} calls={calls} (+{warmups} warmup)",
        cfg.model_family,
        cfg.layer_count,
        cfg.hidden_size,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
    );

    for _ in 0..warmups {
        let (out, _) =
            forward_for_embedding_batch(&cfg, &weights, &batch_token_ids, Some(&target_positions));
        eval(&[&out]);
    }
    // Discard the warmup accumulation; measure only the steady-state calls.
    let _ = take_embed_profile_snapshot();

    let wall_started = Instant::now();
    for _ in 0..calls {
        let (out, _) =
            forward_for_embedding_batch(&cfg, &weights, &batch_token_ids, Some(&target_positions));
        eval(&[&out]);
    }
    let wall_us = wall_started.elapsed().as_micros() as f64;

    let s = take_embed_profile_snapshot();
    let calls_f = s.calls.max(1) as f64;
    let layers_f = s.layers.max(1) as f64;

    // Per-layer stages are accumulated across every layer of every call; per-call
    // stages (embed_tokens, final_norm_pool) are charged once per call.
    let per_layer: &[(&str, u32)] = &[
        ("attn_norm", s.attn_norm_wall_us),
        ("qkv_proj", s.qkv_proj_wall_us),
        ("value_prep", s.value_prep_wall_us),
        ("qk_norm_rope", s.qk_norm_rope_wall_us),
        ("sdpa", s.sdpa_wall_us),
        ("attn_out_proj", s.attn_out_proj_wall_us),
        ("ffn_norm", s.ffn_norm_wall_us),
        ("ffn", s.ffn_wall_us),
    ];
    let per_call: &[(&str, u32)] = &[
        ("embed_tokens", s.embed_tokens_wall_us),
        ("final_norm_pool", s.final_norm_pool_wall_us),
    ];

    let profiled_total_us: f64 = (per_layer.iter().map(|(_, v)| *v as u64).sum::<u64>()
        + per_call.iter().map(|(_, v)| *v as u64).sum::<u64>())
        as f64;

    eprintln!();
    eprintln!(
        "=== embed per-stage breakdown (calls={} layers={} tokens={}) ===",
        s.calls, s.layers, s.tokens
    );
    eprintln!(
        "wall (un-instrumented region excluded): {:.2} ms/call | profiled-sum: {:.2} ms/call (eval barriers inflate this)",
        wall_us / calls_f / 1000.0,
        profiled_total_us / calls_f / 1000.0,
    );
    eprintln!(
        "NOTE: per-stage eval barriers serialize the graph; absolute ms are inflated. Use the %% column.\n"
    );
    eprintln!(
        "{:<16} {:>12} {:>14} {:>9}",
        "stage", "ms/call", "us/layer-call", "% of sum"
    );
    eprintln!("{}", "-".repeat(54));
    for (name, v) in per_layer {
        let total = *v as f64;
        let ms_per_call = total / calls_f / 1000.0;
        let us_per_layer_call = total / layers_f;
        let pct = if profiled_total_us > 0.0 {
            total / profiled_total_us * 100.0
        } else {
            0.0
        };
        eprintln!("{name:<16} {ms_per_call:>12.3} {us_per_layer_call:>14.1} {pct:>8.1}%");
    }
    eprintln!("{}", "-".repeat(54));
    for (name, v) in per_call {
        let total = *v as f64;
        let ms_per_call = total / calls_f / 1000.0;
        let pct = if profiled_total_us > 0.0 {
            total / profiled_total_us * 100.0
        } else {
            0.0
        };
        eprintln!("{name:<16} {ms_per_call:>12.3} {:>14} {pct:>8.1}%", "-");
    }
}
