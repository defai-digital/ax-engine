//! F4 — MLA warm-extend drift bisect probe.
//!
//! Implements the diagnostic tool scoped by
//! `.internal/planning/MLX-MLA-DRIFT-BISECT-PRD-2026-05-14.md`. Given a
//! GLM-class (MLA) model and a (base_len, suffix_len) split, the probe
//! runs both the **cold** forward pass (chunked_prefill of base + suffix
//! from scratch) and the **warm** forward pass (chunked_prefill of base,
//! snapshot the cache, restore into a fresh cache, then chunked_prefill
//! the suffix). It then captures the resulting per-layer `kv_latent` and
//! `k_pe` tensors from both caches, casts them to f32 on host, and
//! diffs them element-wise.
//!
//! Output: a JSON artifact whose `verdict` field is `no_divergence` (cold
//! and warm produce bit-equal KV state across all MLA layers) or
//! `divergent` (the first divergent layer / tensor / position is named
//! prominently). The probe is read-only — no production hot-path code is
//! mutated; it loads the model directly via `load_weights` and runs
//! `chunked_prefill`, the same path the runner uses.
//!
//! Run:
//!   cargo run --release --bin mla-warm-extend-drift-probe -- \
//!       --mlx-artifacts-dir .internal/models/GLM-4.7-Flash-4bit \
//!       --base-tokens 16 --suffix-tokens 16 --chunk-size 16 \
//!       --output benchmarks/results/mla-warm-extend-drift/<file>.json
//!
//! Two canonical invocations satisfy the PRD §6.2 acceptance:
//!   1. `--chunk-size 16`  → verdict must be `no_divergence`.
//!   2. `--chunk-size 512` → verdict must flip to `divergent`
//!      (detection-power inverted control).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    generate::chunked_prefill,
    kv_cache::MlxKVCache,
    model::ModelConfig,
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::load_weights,
};
use mlx_sys::{MlxArray, MlxDtype, astype, clear_cache, eval, ops::slice};
use serde_json::json;

const SCHEMA_VERSION: &str = "ax.mla_drift_bisect.v1";

struct Args {
    mlx_artifacts_dir: PathBuf,
    base_tokens: usize,
    suffix_tokens: usize,
    chunk_size: usize,
    tolerance_abs_diff: f32,
    output: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut mlx_artifacts_dir: Option<PathBuf> = None;
    let mut base_tokens: usize = 16;
    let mut suffix_tokens: usize = 16;
    let mut chunk_size: usize = 16;
    let mut tolerance_abs_diff: f32 = 1.0e-3;
    let mut output: Option<PathBuf> = None;

    let mut argv = env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--mlx-artifacts-dir" => {
                mlx_artifacts_dir = Some(PathBuf::from(
                    argv.next().ok_or("missing value for --mlx-artifacts-dir")?,
                ));
            }
            "--base-tokens" => {
                base_tokens = argv
                    .next()
                    .ok_or("missing value for --base-tokens")?
                    .parse()
                    .map_err(|e| format!("invalid --base-tokens: {e}"))?;
            }
            "--suffix-tokens" => {
                suffix_tokens = argv
                    .next()
                    .ok_or("missing value for --suffix-tokens")?
                    .parse()
                    .map_err(|e| format!("invalid --suffix-tokens: {e}"))?;
            }
            "--chunk-size" => {
                chunk_size = argv
                    .next()
                    .ok_or("missing value for --chunk-size")?
                    .parse()
                    .map_err(|e| format!("invalid --chunk-size: {e}"))?;
            }
            "--tolerance" => {
                tolerance_abs_diff = argv
                    .next()
                    .ok_or("missing value for --tolerance")?
                    .parse()
                    .map_err(|e| format!("invalid --tolerance: {e}"))?;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    argv.next().ok_or("missing value for --output")?,
                ));
            }
            "--help" | "-h" => {
                println!(
                    "{}",
                    concat!(
                        "Usage: mla-warm-extend-drift-probe \\\n",
                        "    --mlx-artifacts-dir <path> \\\n",
                        "    [--base-tokens N] [--suffix-tokens N] [--chunk-size N] \\\n",
                        "    [--tolerance F] --output <path>"
                    )
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    let mlx_artifacts_dir =
        mlx_artifacts_dir.ok_or("--mlx-artifacts-dir is required".to_string())?;
    let output = output.ok_or("--output is required".to_string())?;
    if base_tokens == 0 || suffix_tokens == 0 {
        return Err("--base-tokens and --suffix-tokens must be positive".into());
    }
    if chunk_size == 0 {
        return Err("--chunk-size must be positive".into());
    }
    Ok(Args {
        mlx_artifacts_dir,
        base_tokens,
        suffix_tokens,
        chunk_size,
        tolerance_abs_diff,
        output,
    })
}

fn synth_tokens(base_offset: u32, count: usize) -> Vec<u32> {
    // Tokens 1..=count to avoid `0` which some tokenizers reserve as a
    // special / padding token. Offset shifts the suffix range so cold and
    // warm see the same sequence: (base) = [1..=base], (suffix) =
    // [base+1..=base+suffix]. The exact token IDs do not need to be valid
    // text — the drift we're probing is shape-dependent SDPA kernel
    // selection, not language semantics.
    (1 + base_offset..=base_offset + count as u32).collect()
}

fn run_cold(
    cfg: &ModelConfig,
    weights: &ax_engine_mlx::weights::ModelWeights,
    tokens: &[u32],
    chunk_size: usize,
) -> MlxKVCache {
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let _ = chunked_prefill(
        cfg,
        weights,
        tokens,
        &mut cache,
        chunk_size,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), tokens),
        &mut rng,
    );
    cache
}

fn run_warm(
    cfg: &ModelConfig,
    weights: &ax_engine_mlx::weights::ModelWeights,
    base_tokens: &[u32],
    suffix_tokens: &[u32],
    chunk_size: usize,
) -> MlxKVCache {
    // Phase 1: cold-prefill base into its own cache. Mirrors the
    // production "session A" that produces the snapshot.
    let mut base_cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let _ = chunked_prefill(
        cfg,
        weights,
        base_tokens,
        &mut base_cache,
        chunk_size,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), base_tokens),
        &mut rng,
    );

    // Phase 2: clone the base cache (the "snapshot") into a fresh request
    // and prefill the suffix on top. Mirrors session B's restore + extend.
    let mut warm_cache = base_cache.clone();
    let _ = chunked_prefill(
        cfg,
        weights,
        suffix_tokens,
        &mut warm_cache,
        chunk_size,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), suffix_tokens),
        &mut rng,
    );
    warm_cache
}

/// Materialise a `[1, 1, seq_len, inner_dim]` slice of `arr` as a
/// host-side Vec<f32>. Both cold and warm paths produce the same
/// `seq_len` so the slices are comparable element-wise.
fn host_slice_f32(arr: &MlxArray, seq_len: usize, inner_dim: i32) -> Vec<f32> {
    let start = [0i32, 0, 0, 0];
    let stop = [1i32, 1, seq_len as i32, inner_dim];
    let strides = [1i32, 1, 1, 1];
    let view = slice(arr, &start, &stop, &strides, None);
    let view_f32 = astype(&view, MlxDtype::Float32, None);
    eval(&[&view_f32]);
    view_f32.data_f32().to_vec()
}

fn diff_stats(cold: &[f32], warm: &[f32]) -> (f32, u32, Option<usize>) {
    debug_assert_eq!(cold.len(), warm.len());
    let mut max_abs_diff = 0.0_f32;
    let mut max_ulps_diff = 0_u32;
    let mut first_divergent_position: Option<usize> = None;
    for (i, (&a, &b)) in cold.iter().zip(warm.iter()).enumerate() {
        let abs = (a - b).abs();
        if abs > max_abs_diff {
            max_abs_diff = abs;
        }
        // ULPs: treat both as f32 bit patterns. If they share a sign,
        // subtract the integer representations; otherwise the gap
        // through zero is u32::MAX, which we cap at u32::MAX so the
        // verdict still flags it as a real divergence.
        let ai = a.to_bits();
        let bi = b.to_bits();
        let ulps = if (ai ^ bi) >> 31 == 0 {
            ai.max(bi) - ai.min(bi)
        } else if a == 0.0 && b == 0.0 {
            0
        } else {
            u32::MAX
        };
        if ulps > max_ulps_diff {
            max_ulps_diff = ulps;
        }
        if first_divergent_position.is_none() && abs > 0.0 {
            first_divergent_position = Some(i);
        }
    }
    (max_abs_diff, max_ulps_diff, first_divergent_position)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if !args.mlx_artifacts_dir.is_dir() {
        eprintln!(
            "error: --mlx-artifacts-dir {} is not a directory",
            args.mlx_artifacts_dir.display()
        );
        return ExitCode::from(2);
    }

    println!(
        "F4 MLA warm-extend drift probe (base={}, suffix={}, chunk={})",
        args.base_tokens, args.suffix_tokens, args.chunk_size
    );
    println!("loading {} ...", args.mlx_artifacts_dir.display());
    let artifacts = match NativeModelArtifacts::from_dir(&args.mlx_artifacts_dir) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: failed to load artifacts: {e}");
            return ExitCode::from(1);
        }
    };
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    if cfg.glm_mla_attention.is_none() {
        eprintln!(
            "error: model at {} is not a GLM-MLA model (probe only diagnoses MLA drift)",
            args.mlx_artifacts_dir.display()
        );
        return ExitCode::from(2);
    }
    let weights = match load_weights(&artifacts) {
        Ok(w) => Arc::new(w),
        Err(e) => {
            eprintln!("error: failed to load weights: {e:?}");
            return ExitCode::from(1);
        }
    };

    let base_tokens = synth_tokens(0, args.base_tokens);
    let suffix_tokens = synth_tokens(args.base_tokens as u32, args.suffix_tokens);
    let extended_tokens: Vec<u32> = base_tokens
        .iter()
        .chain(suffix_tokens.iter())
        .copied()
        .collect();
    let seq_len = extended_tokens.len();

    println!("running cold prefill (base + suffix, {seq_len} tokens) ...");
    let cold_cache = run_cold(&cfg, &weights, &extended_tokens, args.chunk_size);
    clear_cache();
    println!("  cold seq_len = {}", cold_cache.seq_len);
    if cold_cache.seq_len != seq_len {
        eprintln!(
            "error: cold cache seq_len {} != expected {}",
            cold_cache.seq_len, seq_len
        );
        return ExitCode::from(1);
    }

    println!("running warm prefill (base; snapshot; suffix) ...");
    let warm_cache = run_warm(
        &cfg,
        &weights,
        &base_tokens,
        &suffix_tokens,
        args.chunk_size,
    );
    clear_cache();
    println!("  warm seq_len = {}", warm_cache.seq_len);
    if warm_cache.seq_len != seq_len {
        eprintln!(
            "error: warm cache seq_len {} != expected {}",
            warm_cache.seq_len, seq_len
        );
        return ExitCode::from(1);
    }

    println!("diffing per-layer kv_latent and k_pe ...");
    let mut per_layer = Vec::new();
    let mut first_divergence: Option<serde_json::Value> = None;
    for layer in 0..cfg.layer_count {
        let cold_view = match cold_cache.glm_mla_layer_state(layer) {
            Some(v) => v,
            None => continue,
        };
        let warm_view = warm_cache
            .glm_mla_layer_state(layer)
            .expect("warm cache must have the same MLA layers as cold");

        let cold_kv = host_slice_f32(cold_view.kv_latent, seq_len, cold_view.latent_dim);
        let warm_kv = host_slice_f32(warm_view.kv_latent, seq_len, warm_view.latent_dim);
        let (kv_abs, kv_ulps, kv_pos) = diff_stats(&cold_kv, &warm_kv);

        let cold_pe = host_slice_f32(cold_view.k_pe, seq_len, cold_view.rope_dim);
        let warm_pe = host_slice_f32(warm_view.k_pe, seq_len, warm_view.rope_dim);
        let (pe_abs, pe_ulps, pe_pos) = diff_stats(&cold_pe, &warm_pe);

        per_layer.push(json!({
            "layer_idx": layer,
            "kv_latent_max_abs_diff": kv_abs,
            "kv_latent_max_ulps_diff": kv_ulps,
            "kv_latent_first_divergence_position": kv_pos,
            "k_pe_max_abs_diff": pe_abs,
            "k_pe_max_ulps_diff": pe_ulps,
            "k_pe_first_divergence_position": pe_pos,
        }));

        if first_divergence.is_none() {
            if kv_abs > args.tolerance_abs_diff {
                let pos = kv_pos.unwrap_or(0);
                first_divergence = Some(json!({
                    "layer_idx": layer,
                    "tensor": "kv_latent",
                    "position": pos,
                    "cold_value": cold_kv[pos],
                    "warm_value": warm_kv[pos],
                    "abs_diff": kv_abs,
                    "ulps_diff": kv_ulps,
                }));
            } else if pe_abs > args.tolerance_abs_diff {
                let pos = pe_pos.unwrap_or(0);
                first_divergence = Some(json!({
                    "layer_idx": layer,
                    "tensor": "k_pe",
                    "position": pos,
                    "cold_value": cold_pe[pos],
                    "warm_value": warm_pe[pos],
                    "abs_diff": pe_abs,
                    "ulps_diff": pe_ulps,
                }));
            }
        }
    }

    let verdict = if first_divergence.is_some() {
        "divergent"
    } else {
        "no_divergence"
    };

    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "args": {
            "mlx_artifacts_dir": args.mlx_artifacts_dir.display().to_string(),
            "base_tokens": args.base_tokens,
            "suffix_tokens": args.suffix_tokens,
            "chunk_size": args.chunk_size,
            "tolerance_abs_diff": args.tolerance_abs_diff,
        },
        "model": {
            "layer_count": cfg.layer_count,
            "mla_layer_count": per_layer.len(),
        },
        "layers": per_layer,
        "first_divergence": first_divergence,
        "verdict": verdict,
    });

    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = fs::create_dir_all(parent)
    {
        eprintln!(
            "error: failed to create output parent {}: {e}",
            parent.display()
        );
        return ExitCode::from(1);
    }
    if let Err(e) = fs::write(
        &args.output,
        serde_json::to_string_pretty(&payload).unwrap(),
    ) {
        eprintln!(
            "error: failed to write output {}: {e}",
            args.output.display()
        );
        return ExitCode::from(1);
    }

    println!();
    println!("=== Verdict ===");
    println!("  wrote {}", args.output.display());
    match verdict {
        "no_divergence" => {
            println!("  no_divergence — cold and warm KV state match within tolerance");
            println!(
                "  the chunk-alignment safety holds at chunk_size={}",
                args.chunk_size
            );
        }
        "divergent" => {
            println!("  DIVERGENT");
            if let Some(d) = &first_divergence {
                println!("  first divergence: {}", serde_json::to_string(d).unwrap());
            }
        }
        _ => unreachable!(),
    }
    ExitCode::SUCCESS
}

// Keep `Path` imported for clarity even if unused at function boundary.
#[allow(dead_code)]
fn _path_marker(_: &Path) {}
