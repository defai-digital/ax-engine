//! Phase 0 — batched-decode weight-read amortization probe.
//!
//! ## The question this answers
//!
//! The MLX runner decodes strictly batch=1: a step's N concurrent requests are
//! run as N separate forward passes, so every weight matrix is read from DRAM
//! N times per step (see `runner/mod.rs` `for item in items { run_item(item) }`
//! and the `kv_cache.rs` `k_shape[0] == 1` assert). All prior decode profiling
//! concluded single-stream decode is bandwidth-bound at the roofline — which is
//! exactly the regime where *batching* should be nearly free: a quantized matmul
//! at low batch reads O(weight_bytes) from DRAM and does O(batch * weight)
//! compute, so stacking rows costs no extra weight traffic until compute
//! saturates.
//!
//! Before rewriting the crown-jewel KV cache / attention to support batched
//! decode, this probe proves (or refutes) the amortization hypothesis at the
//! kernel level, on REAL model weights, with zero changes to the model graph.
//!
//! ## What it measures
//!
//! It sums the wall time of every dense quantized-projection matmul in a decode
//! step (per layer: q/k/v or packed-qkv, o_proj, gate/up or packed gate_up,
//! down_proj; plus lm_head once) at batch B in {1,2,4,8,16,32}. These matmuls
//! are ~all the weight bytes a decode step reads; SDPA, RoPE and norms are small
//! and also batch-amortizing, so this total is a faithful proxy for the decode
//! weight-read cost. Each weight is fed a correctly-sized `[B, in_dim]` input
//! (data-independent cost), materialized with a single `eval()` barrier per
//! measured iteration (2 warmup + 5 measured, median — repo convention).
//!
//! ## The headline number
//!
//! `amortization_multiple(B) = B * time(1) / time(B)`:
//! - `== B` → perfectly bandwidth-bound: batching gives full B× aggregate
//!   throughput. GO on batched decode.
//! - `== 1` → compute-bound already at B=1: batching buys nothing. NO-GO.
//!
//! The crossover B (where the multiple stops climbing) is the practical max
//! useful decode batch on this hardware for this model+quant.
//!
//! ## Usage
//!
//!   cargo run --release --bin decode_batch_amortization_probe -- <model_dir>
//!
//! `<model_dir>` must be a dense MLX model artifact directory (e.g. a downloaded
//! Qwen3-4B-4bit). MoE expert weights (`gate_exps`/`down_exps`, 3-D gather_qmm)
//! are intentionally skipped — they need a separate gather_qmm amortization
//! probe. Env `AX_BATCHES` overrides the batch sweep (comma-separated).

use std::env;
use std::path::Path;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    model::ModelConfig,
    weights::{ModelWeights, QuantizedWeight, load_weights},
};
use mlx_sys::{MlxArray, MlxDtype, clear_cache, eval, quantized_matmul, zeros};

/// Replicates `model::shared::utils::qw` (pub(crate), unreachable from a bin):
/// affine quantized_matmul with transpose=true, group_size/bits from the weight.
fn qmm(x: &MlxArray, w: &QuantizedWeight) -> Option<MlxArray> {
    let scales = w.scales.as_ref()?;
    Some(quantized_matmul(
        x,
        &w.weight,
        scales,
        w.biases.as_ref(),
        true,
        Some(w.group_size),
        Some(w.bits),
        None,
    ))
}

/// Unpacked input dim for a transpose=true quantized weight of packed shape
/// `[out_dim, packed_in]`: `packed_in * 32 / bits`. The multiply must precede
/// the divide — for non-power-of-two bit widths (e.g. 6-bit) `32 / bits`
/// truncates (32/6→5) and undercounts the input dim (960*5=4800 vs the true
/// 960*32/6=5120).
fn input_dim(w: &QuantizedWeight) -> Option<i64> {
    let shape = w.weight.shape();
    let packed_in = *shape.get(1)? as i64;
    Some(packed_in * 32 / w.bits as i64)
}

fn elem_count(a: &MlxArray) -> i64 {
    a.shape().iter().map(|&d| d as i64).product()
}

/// Bytes read for one quantized weight: packed uint32 weight (4B/elem) + bf16
/// scales/biases (2B/elem). Constant across batch — that's the whole point.
fn weight_bytes(w: &QuantizedWeight) -> i64 {
    let mut b = elem_count(&w.weight) * 4;
    if let Some(s) = &w.scales {
        b += elem_count(s) * 2;
    }
    if let Some(z) = &w.biases {
        b += elem_count(z) * 2;
    }
    b
}

/// Every dense projection weight in the model, in decode-execution order.
fn decode_projection_weights(weights: &ModelWeights) -> Vec<&QuantizedWeight> {
    let mut out = Vec::new();
    for layer in &weights.layers {
        for w in [
            layer.q_proj.as_ref(),
            layer.k_proj.as_ref(),
            layer.v_proj.as_ref(),
            layer.qkv_packed.as_ref(),
            layer.o_proj.as_ref(),
            layer.gate_proj.as_ref(),
            layer.up_proj.as_ref(),
            layer.gate_up_packed.as_ref(),
            layer.down_proj.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            out.push(w);
        }
    }
    out.push(&weights.lm_head);
    out
}

/// Median of the closure's wall time over `measured` runs after `warmup` runs.
fn median_step_secs<F: Fn() -> f64>(warmup: usize, measured: usize, step: F) -> f64 {
    for _ in 0..warmup {
        step();
    }
    let mut samples: Vec<f64> = (0..measured).map(|_| step()).collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let model_dir = env::args().nth(1).unwrap_or_else(|| {
        eprintln!(
            "usage: decode_batch_amortization_probe <model_dir>\n\
             (a dense MLX model artifact directory, e.g. Qwen3-4B-4bit)"
        );
        std::process::exit(2);
    });

    let batches: Vec<i64> = env::var("AX_BATCHES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<i64>().ok())
                .collect()
        })
        .filter(|v: &Vec<i64>| !v.is_empty())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32]);

    let artifacts =
        NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).expect("load weights");

    let projections = decode_projection_weights(&weights);
    let total_weight_bytes: i64 = projections.iter().map(|w| weight_bytes(w)).sum();

    // Precompute per-weight (input_dim, unique bf16 input template) so the timed
    // loop only pays for the matmuls, not input allocation. Inputs of the same
    // (batch, input_dim) are shared.
    let n_matmuls = projections.len();
    let n_moe_layers = weights
        .layers
        .iter()
        .filter(|l| l.down_exps.is_some() || l.gate_up_exps_packed.is_some())
        .count();

    println!("# decode batched-matmul amortization probe");
    println!("model_dir            {model_dir}");
    println!("model_family         {}", cfg.model_family);
    println!("layers               {}", weights.layers.len());
    println!("dense matmuls/step   {n_matmuls}");
    if n_moe_layers > 0 {
        println!(
            "NOTE                 {n_moe_layers} MoE layers present; expert (gather_qmm) \
             weights are NOT included in this probe (dense projections only)."
        );
    }
    println!(
        "weight bytes/step    {:.3} GiB (constant across batch)",
        total_weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!();
    println!(
        "{:>5}  {:>11}  {:>13}  {:>13}  {:>13}  {:>10}",
        "batch", "ms/step", "GB/s", "agg tok/s", "amort_mult", "eff_vs_B"
    );

    // Warm the MLX buffer pool once; steady-state decode reuses buffers, so we
    // do NOT clear_cache per iteration (that would measure a cold pool, not the
    // production steady state).
    clear_cache();

    let mut baseline_secs = 0.0_f64;
    for (i, &b) in batches.iter().enumerate() {
        // Pre-allocate one input per weight OUTSIDE the timed region so the
        // measurement is matmul + eval only, not input allocation.
        let inputs: Vec<MlxArray> = projections
            .iter()
            .map(|w| {
                let in_dim = input_dim(w).expect("packed weight has rank-2 shape") as i32;
                zeros(&[b as i32, in_dim], MlxDtype::Bfloat16, None)
            })
            .collect();
        eval(&inputs.iter().collect::<Vec<_>>());

        let step = || {
            let started = Instant::now();
            let mut outputs: Vec<MlxArray> = Vec::with_capacity(n_matmuls);
            for (w, x) in projections.iter().zip(inputs.iter()) {
                if let Some(out) = qmm(x, w) {
                    outputs.push(out);
                }
            }
            let refs: Vec<&MlxArray> = outputs.iter().collect();
            eval(&refs);
            started.elapsed().as_secs_f64()
        };

        let secs = median_step_secs(3, 7, step);
        if i == 0 {
            baseline_secs = secs;
        }
        let gbps = total_weight_bytes as f64 / secs / 1.0e9;
        let agg_tok_s = b as f64 / secs;
        // amortization multiple: aggregate throughput vs the B=1 per-token rate.
        let amort_mult = b as f64 * baseline_secs / secs;
        // efficiency vs ideal linear scaling (1.0 == perfectly bandwidth-bound).
        let eff = amort_mult / b as f64;

        println!(
            "{:>5}  {:>11.3}  {:>13.1}  {:>13.1}  {:>12.2}x  {:>9.1}%",
            b,
            secs * 1.0e3,
            gbps,
            agg_tok_s,
            amort_mult,
            eff * 100.0
        );
    }

    println!();
    println!(
        "# amort_mult == B → fully bandwidth-bound (batching wins B×). \
         == 1 → compute-bound (batching buys nothing)."
    );
    println!(
        "# eff_vs_B is amort_mult/B: the fraction of ideal linear throughput \
         scaling retained at that batch."
    );
}
