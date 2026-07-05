//! ADR-037 P2 — MoE `gather_qmm` expert-overlap amortization probe.
//!
//! ## The question this answers
//!
//! Phase 0 proved dense decode matmuls amortize ~1.9×@B2 / ~2.4×@B4 because
//! every batch row reads the same weight bytes. 256-expert top-8 MoE breaks
//! that assumption: rows route to (mostly) different experts, so batched
//! expert reads may not amortize at all — or may still win because each
//! per-expert matmul is tiny (top-8 of 256 × 512-dim intermediates) and the
//! B=1 gather is dispatch/latency-bound rather than bandwidth-bound. This is
//! the single unknown that gates the 35B in the batched-decode PRD: if the
//! aggregate speedup at B=4 is under ~1.3×, the 35B drops out of scope.
//!
//! ## What it measures
//!
//! For B in {1,2,4,8}: the wall time of one full MoE decode step — for every
//! MoE layer, `gather_qmm(gate_up)` → SwiGLU split → `gather_qmm(down)` —
//! with per-row expert indices REPLAYED from real routing traces captured
//! via `AX_MLX_MOE_ROUTER_TRACE` (one trace file per simulated request, so
//! batch rows carry genuinely independent routing, not synthetic overlap).
//! Expert sets vary across measured steps exactly as they varied in the
//! source generations. Dense projections, router, shared expert, and
//! attention are excluded — Phase 0 already covers their amortization.
//!
//! ## Usage
//!
//!   cargo run --release --bin moe_gather_amortization_probe -- \
//!       <model_dir> <trace_file_1> [trace_file_2 ...]
//!
//! Capture traces first (one process per prompt so files are per-request):
//!   AX_MLX_MOE_ROUTER_TRACE=/tmp/trace_0.txt python3 -c '<generate>'
//!
//! Trace lines are `<seq>;<i0>,<i1>,...`; only `seq == 1` (single-token
//! decode) lines are used, chunked into steps of `n_moe_layers` consecutive
//! calls (a decode forward emits exactly one router call per MoE layer, in
//! layer order). `AX_BATCHES` overrides the batch sweep.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    model::ModelConfig,
    weights::{ModelWeights, QuantizedWeight, load_weights},
};
use mlx_sys::{MlxArray, MlxDtype, clear_cache, eval, gather_qmm, multiply, sigmoid, slice, zeros};

/// One decode step of one traced request: per-MoE-layer top-k expert ids.
type TraceStep = Vec<Vec<u32>>;

fn parse_trace(path: &str, n_moe_layers: usize, top_k: usize) -> Vec<TraceStep> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read trace {path}: {e}");
        std::process::exit(2);
    });
    // Keep only single-token decode router calls, in order.
    let calls: Vec<Vec<u32>> = text
        .lines()
        .filter_map(|line| {
            let (seq, rest) = line.split_once(';')?;
            if seq.trim() != "1" {
                return None;
            }
            let ids: Vec<u32> = rest
                .split(',')
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect();
            (ids.len() == top_k).then_some(ids)
        })
        .collect();
    calls
        .chunks_exact(n_moe_layers)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Expert projections of one MoE layer: either a packed `[gate; up]` weight
/// (Gemma-style) or separate gate/up weights (Qwen3.5-MoE checkpoints).
struct MoeLayer<'w> {
    gate_up_packed: Option<&'w QuantizedWeight>,
    gate: Option<&'w QuantizedWeight>,
    up: Option<&'w QuantizedWeight>,
    down: &'w QuantizedWeight,
}

fn moe_layers(weights: &ModelWeights) -> Vec<MoeLayer<'_>> {
    weights
        .layers
        .iter()
        .filter_map(|l| {
            let down = l.down_exps.as_ref()?;
            if l.gate_up_exps_packed.is_none() && (l.gate_exps.is_none() || l.up_exps.is_none()) {
                return None;
            }
            Some(MoeLayer {
                gate_up_packed: l.gate_up_exps_packed.as_ref(),
                gate: l.gate_exps.as_ref(),
                up: l.up_exps.as_ref(),
                down,
            })
        })
        .collect()
}

/// Bytes for one expert's rows in a packed 3-D quantized weight
/// `[n_experts, out, packed_in]` (+ scales/biases).
fn per_expert_bytes(w: &QuantizedWeight) -> f64 {
    let count = |a: &MlxArray, unit: i64| -> f64 {
        let s = a.shape();
        (s.iter().skip(1).map(|&d| d as i64).product::<i64>() * unit) as f64
    };
    let mut b = count(&w.weight, 4);
    if let Some(s) = &w.scales {
        b += count(s, 2);
    }
    if let Some(z) = &w.biases {
        b += count(z, 2);
    }
    b
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "usage: moe_gather_amortization_probe <model_dir> <trace_1> [trace_2 ...]\n\
             (traces from AX_MLX_MOE_ROUTER_TRACE, one file per request/prompt)"
        );
        std::process::exit(2);
    }
    let model_dir = &args[0];
    let trace_paths = &args[1..];

    let batches: Vec<usize> = env::var("AX_BATCHES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);

    let artifacts =
        NativeModelArtifacts::from_dir(Path::new(model_dir)).expect("load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).expect("load weights");
    let layers = moe_layers(&weights);
    let n_moe = layers.len();
    let top_k = cfg.moe_experts_per_token;
    let hidden = cfg.hidden_size as i32;
    assert!(n_moe > 0, "model has no packed MoE expert layers");

    let traces: Vec<Vec<TraceStep>> = trace_paths
        .iter()
        .map(|p| parse_trace(p, n_moe, top_k))
        .collect();
    for (path, t) in trace_paths.iter().zip(&traces) {
        assert!(
            t.len() >= 8,
            "trace {path} has only {} complete decode steps (need >= 8)",
            t.len()
        );
    }
    let max_b = *batches.iter().max().expect("non-empty batches");
    assert!(
        traces.len() >= max_b,
        "need at least {max_b} trace files for the largest batch, got {}",
        traces.len()
    );
    let common_steps = traces.iter().map(Vec::len).min().expect("non-empty");
    // Skip the first two steps (prompt-tail transients) and cycle the rest.
    let usable: Vec<usize> = (2..common_steps).collect();
    assert!(usable.len() >= 4, "need >= 4 usable common steps");

    let expert_bytes: f64 = layers
        .iter()
        .map(|l| {
            let gate_up: f64 = match l.gate_up_packed {
                Some(w) => per_expert_bytes(w),
                None => {
                    per_expert_bytes(l.gate.expect("split gate"))
                        + per_expert_bytes(l.up.expect("split up"))
                }
            };
            gate_up + per_expert_bytes(l.down)
        })
        .sum::<f64>()
        / n_moe as f64;

    println!("# MoE gather_qmm expert-overlap amortization probe (ADR-037 P2)");
    println!("model_dir            {model_dir}");
    println!("model_family         {}", cfg.model_family);
    println!(
        "moe layers           {n_moe}   experts {}   top_k {top_k}   traces {}   steps/trace {}",
        cfg.moe_expert_count,
        traces.len(),
        common_steps
    );
    println!(
        "per-expert bytes     {:.2} MiB (gate_up + down, mean across layers)",
        expert_bytes / (1024.0 * 1024.0)
    );
    println!();
    println!(
        "{:>5}  {:>11}  {:>13}  {:>12}  {:>16}  {:>14}",
        "batch", "ms/step", "agg tok/s", "amort_mult", "distinct/layer", "eff GB/s"
    );

    clear_cache();

    // One gather with transpose=true: x @ w[idx].T for each selected expert.
    let gq = |x: &MlxArray, w: &QuantizedWeight, idx: &MlxArray| -> MlxArray {
        gather_qmm(
            x,
            &w.weight,
            w.scales.as_ref().expect("expert weight scales"),
            w.biases.as_ref(),
            idx,
            true,
            Some(w.group_size),
            Some(w.bits),
            false,
            None,
        )
    };

    let mut baseline = 0.0_f64;
    for (bi, &b) in batches.iter().enumerate() {
        // Pre-build per-step index arrays [B, 1, top_k] u32 outside timing,
        // and the distinct-expert stat while we're at it.
        let mut step_indices: Vec<Vec<MlxArray>> = Vec::with_capacity(usable.len());
        let mut distinct_sum = 0usize;
        for &s in &usable {
            let mut per_layer = Vec::with_capacity(n_moe);
            for l in 0..n_moe {
                let mut flat: Vec<u32> = Vec::with_capacity(b * top_k);
                let mut distinct: HashSet<u32> = HashSet::with_capacity(b * top_k);
                for row_trace in traces.iter().take(b) {
                    let ids = &row_trace[s][l];
                    flat.extend_from_slice(ids);
                    distinct.extend(ids.iter().copied());
                }
                distinct_sum += distinct.len();
                per_layer.push(MlxArray::from_raw_data(
                    flat.as_ptr().cast(),
                    flat.len() * 4,
                    &[b as i32, 1, top_k as i32],
                    MlxDtype::Uint32,
                ));
            }
            step_indices.push(per_layer);
        }
        let mean_distinct = distinct_sum as f64 / (usable.len() * n_moe) as f64;
        let x = zeros(&[b as i32, 1, 1, 1, hidden], MlxDtype::Bfloat16, None);
        eval(
            &step_indices
                .iter()
                .flatten()
                .chain(std::iter::once(&x))
                .collect::<Vec<_>>(),
        );

        let run_pass = |indices_for_step: &[MlxArray]| -> f64 {
            let started = Instant::now();
            let mut outputs: Vec<MlxArray> = Vec::with_capacity(n_moe);
            for (layer, idx) in layers.iter().zip(indices_for_step) {
                let act = if let Some(packed) = layer.gate_up_packed {
                    let gu = gq(&x, packed, idx);
                    let shape = gu.shape();
                    let (r, c) = (shape[3], shape[4]);
                    let inter = c / 2;
                    let ones = [1i32, 1, 1, 1, 1];
                    let gate = slice(
                        &gu,
                        &[0, 0, 0, 0, 0],
                        &[b as i32, 1, 1, r, inter],
                        &ones,
                        None,
                    );
                    let up = slice(
                        &gu,
                        &[0, 0, 0, 0, inter],
                        &[b as i32, 1, 1, r, c],
                        &ones,
                        None,
                    );
                    multiply(&multiply(&gate, &sigmoid(&gate, None), None), &up, None)
                } else {
                    let gate = gq(&x, layer.gate.expect("split gate"), idx);
                    let up = gq(&x, layer.up.expect("split up"), idx);
                    multiply(&multiply(&gate, &sigmoid(&gate, None), None), &up, None)
                };
                outputs.push(gq(&act, layer.down, idx));
            }
            eval(&outputs.iter().collect::<Vec<_>>());
            started.elapsed().as_secs_f64()
        };

        // One iteration = mean over every usable step (expert sets vary per
        // step exactly as they did in the source generations).
        let iteration = || {
            let mut total = 0.0;
            for per_layer in &step_indices {
                total += run_pass(per_layer);
            }
            total / step_indices.len() as f64
        };
        for _ in 0..2 {
            iteration();
        }
        let secs = median((0..5).map(|_| iteration()).collect());
        if bi == 0 {
            baseline = secs;
        }
        let agg_tok_s = b as f64 / secs;
        let amort = b as f64 * baseline / secs;
        // Effective bandwidth from DISTINCT expert bytes actually addressed.
        let gbps = mean_distinct * expert_bytes * n_moe as f64 / secs / 1.0e9;
        println!(
            "{:>5}  {:>11.3}  {:>13.1}  {:>12.2}  {:>16.2}  {:>14.1}",
            b,
            secs * 1e3,
            agg_tok_s,
            amort,
            mean_distinct,
            gbps
        );
    }
}
