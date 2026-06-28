//! Barrier-free batched-embedding throughput + diagnostics probe.
//!
//! Measures REAL (un-instrumented, single-eval) throughput of the production
//! imperative batched-embedding forward (`forward_for_embedding_batch`) and
//! breaks down where the wall time goes, to localize the Qwen3-Embedding-0.6B-8bit
//! batch>1 regression (README #embedding-throughput).
//!
//! Reports per workload (b8 s256 / b8 s64 / b1 s256):
//!   - throughput (tok/s)
//!   - build(FFI lazy-graph) vs eval(GPU) split — localizes CPU vs GPU
//!   - MLX op count per forward (`op_count` bump per FFI dispatch)
//!   - layer-0 quantized-weight dtype/layout dump (vs mlx-lm's loaded tensors)
//!
//! Diagnostic toggles:
//!   AX_PROBE_CACHE=0       run with the MLX buffer cache disabled
//!   AX_PROBE_NO_COMPILE=1  skip enable_compile()
//!
//! Do NOT set `AX_MLX_EMBED_PROFILE` here — that forces eval barriers and
//! inflates wall time. This probe is the barrier-free counterpart to
//! `embed_profile_probe`.
//!
//! Usage:
//!   cargo run --release --bin embed_ab_probe -- <model_dir> [trials]

use std::path::Path;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    model::{ModelConfig, forward_for_embedding_batch},
    weights::{ModelWeights, load_weights},
};
use mlx_sys::{
    enable_compile, eval, max_recommended_working_set_size, op_count_snapshot, op_count_take,
    set_cache_limit,
};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn build_batch(batch: usize, seq: usize) -> (Vec<Vec<u32>>, Vec<usize>) {
    let ids: Vec<Vec<u32>> = (0..batch)
        .map(|i| {
            (0..seq)
                .map(|j| ((i * seq + j + 1) % 30000) as u32)
                .collect()
        })
        .collect();
    let positions = ids.iter().map(|r| r.len() - 1).collect();
    (ids, positions)
}

fn time_once(cfg: &ModelConfig, weights: &ModelWeights, batch: &[Vec<u32>], pos: &[usize]) -> f64 {
    let started = Instant::now();
    let (out, _) = forward_for_embedding_batch(cfg, weights, batch, Some(pos));
    eval(&[&out]);
    started.elapsed().as_secs_f64()
}

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .expect("usage: embed_ab_probe <model_dir> [trials]");
    let trials: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let warmups = 3usize;

    if std::env::var("AX_PROBE_NO_COMPILE").is_err() {
        enable_compile();
    } else {
        eprintln!("(enable_compile skipped)");
    }
    // `set_cache_limit` returns the PREVIOUS limit, so the first call reveals the
    // mlx-c process default (the standalone probe never runs the MlxRunner
    // constructor that sets a limit). AX_PROBE_CACHE=0 runs with it disabled.
    let wired = max_recommended_working_set_size();
    let high = wired.max(1 << 30);
    let mlx_c_default = set_cache_limit(high);
    let applied = if std::env::var("AX_PROBE_CACHE").as_deref() == Ok("0") {
        set_cache_limit(0);
        0
    } else {
        high
    };
    eprintln!(
        "mlx cache limit: mlx-c default was {mlx_c_default} bytes; running with {applied} bytes (wired_cap={wired})"
    );

    let artifacts =
        NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("load model artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).expect("load weights");

    eprintln!(
        "embed_ab_probe: family={} layers={} hidden={} heads={}/{} | trials={trials} (+{warmups} warmup)",
        cfg.model_family, cfg.layer_count, cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads,
    );

    // Layer-0 quantized-weight dtype/layout (compare against mlx-lm's loaded
    // tensors; also shows whether qkv/gate_up are packed via AX_MLX_PACK_*).
    {
        let l0 = &weights.layers[0];
        let dump = |name: &str, qw: Option<&ax_engine_mlx::weights::QuantizedWeight>| {
            if let Some(q) = qw {
                eprintln!(
                    "  {name:<16} w={:?}{:?} scales={:?} biases={:?} gs={} bits={}",
                    q.weight.dtype(),
                    q.weight.shape(),
                    q.scales.as_ref().map(|s| s.dtype()),
                    q.biases.as_ref().map(|b| b.dtype()),
                    q.group_size,
                    q.bits,
                );
            } else {
                eprintln!("  {name:<16} <none>");
            }
        };
        eprintln!("layer-0 weight layout:");
        dump("q_proj", l0.q_proj.as_ref());
        dump("qkv_packed", l0.qkv_packed.as_ref());
        dump("gate_proj", l0.gate_proj.as_ref());
        dump("gate_up_packed", l0.gate_up_packed.as_ref());
        dump("down_proj", l0.down_proj.as_ref());
    }

    // Build-vs-eval split + op count (b8 s256): localize CPU(FFI build) vs
    // GPU(eval), and confirm op count is comparable to mlx-lm (~23/layer).
    {
        let (ids, pos) = build_batch(8, 256);
        for _ in 0..warmups {
            time_once(&cfg, &weights, &ids, &pos);
        }
        let mut builds = Vec::new();
        let mut evals = Vec::new();
        for _ in 0..trials {
            let t0 = Instant::now();
            let (o, _) = forward_for_embedding_batch(&cfg, &weights, &ids, Some(&pos));
            let t1 = Instant::now();
            eval(&[&o]);
            let t2 = Instant::now();
            builds.push((t1 - t0).as_secs_f64() * 1000.0);
            evals.push((t2 - t1).as_secs_f64() * 1000.0);
        }
        let prev = op_count_snapshot();
        let (o, _) = forward_for_embedding_batch(&cfg, &weights, &ids, Some(&pos));
        eval(&[&o]);
        let ops = op_count_take(prev);
        eprintln!(
            "build-vs-eval (b8 s256): build(FFI) {:.2} ms | eval(GPU) {:.2} ms | mlx ops/forward={ops} ({} layers -> {:.1}/layer)",
            median(builds),
            median(evals),
            cfg.layer_count,
            ops as f64 / cfg.layer_count as f64,
        );
    }

    eprintln!("\n{:<12} {:>13}", "workload", "tok/s");
    eprintln!("{}", "-".repeat(26));
    for (batch, seq) in [(8usize, 256usize), (8, 64), (1, 256)] {
        let (ids, pos) = build_batch(batch, seq);
        let tokens = (batch * seq) as f64;
        for _ in 0..warmups {
            time_once(&cfg, &weights, &ids, &pos);
        }
        let mut ts = Vec::with_capacity(trials);
        for _ in 0..trials {
            ts.push(time_once(&cfg, &weights, &ids, &pos));
        }
        eprintln!("b{batch} s{seq:<8} {:>13.0}", tokens / median(ts));
    }
}
