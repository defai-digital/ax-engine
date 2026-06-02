# TurboQuant Implementation Status — 2026-06-01

Status audit of the TurboQuant KV cache compression system. Not a completion
report: one production gate remains open. This document records the current
state of the codec, Metal kernels, runtime integration, and model coverage.

## Summary

TurboQuant K8V4 is feature-complete at the codec and kernel layer. The system
is wired into the runner, KV cache, and standard model family. The sole
remaining production gate is `long_context_benchmark_artifact`, which is
explicitly held `false` in the `mlx_shadow_fused_kernel` readiness preset.

## Production Readiness Gate (`turboquant.rs:17-73`)

Five gates govern production launch. Current state of `mlx_shadow_fused_kernel()`:

| Gate | Status |
|---|---|
| `fused_decode_kernel` | met |
| `runtime_kv_storage` | met |
| `runner_route_metadata` | met |
| `public_switch_and_docs` | met |
| `long_context_benchmark_artifact` | **not met — launch blocker** |

`production_ready` resolves to `false`. `compression_production_ready` in
route metadata telemetry reflects this at runtime. The gate is intentional:
no code change will satisfy it, only a benchmark run that produces a checked-in
long-context artifact.

## Codec Layer (`turboquant.rs`, 5 509 lines)

Reference CPU codec is complete and stable:

- K8V4 (8-bit keys, 4-bit values) — production preset
- K4V4, K3V4Research — research presets, gated behind `ResearchLoose` quality profile
- Hadamard rotation pre-conditioning for key quantization
- Per-group min/scale encoding for 4-bit values
- `TurboQuantDecodeQualityGate` with three tiers:
  - `StrictDebug` — max_diff 0.02, mean_diff 0.01, cosine 0.999
  - `ReferenceK8V4` — max_diff 0.04, mean_diff 0.02, cosine 0.998
  - `ResearchLoose` — max_diff 0.08, mean_diff 0.04, cosine 0.995
- 40+ unit tests covering encoding, decoding, layout, quality gates, and
  attention computation

## Metal Kernel Layer (`turboquant_metal.rs`, 1 145 lines)

Three fused cold-decode kernel variants for different workload shapes:

| Variant | Use case |
|---|---|
| Parallel-over-dims | General; default entry point |
| Head-serial | Optimised for 128-dim heads |
| Two-stage (score + weighted sum) | Separable score/value reduction |

Head dim constants in `turboquant.rs:11-13`:

| Constant | Value | Models |
|---|---|---|
| `TURBOQUANT_INITIAL_FUSED_DECODE_HEAD_DIM` | 128 | Qwen3, standard dense |
| `TURBOQUANT_EXTENDED_FUSED_DECODE_HEAD_DIM` | 256 | Extended GQA heads |
| `TURBOQUANT_GEMMA4_FULL_ATTENTION_FUSED_DECODE_HEAD_DIM` | 512 | Gemma 4 full-attention layers |

GQA support tested for 256-dim and 512-dim heads. Kernel output validated
against CPU reference in 5 unit tests.

## Runtime Decode Paths (`runner.rs:167-169`)

Three mutually exclusive decode paths are tracked in route metadata telemetry:

| Path | Code | Status |
|---|---|---|
| `FULL_PRECISION_SHADOW` | 1 | Active default; compression accounting only |
| `CPU_ORACLE_COMPRESSED_DECODE` | 3 | CPU reference oracle; fallback during fused-experimental |
| `FUSED_COMPRESSED_DECODE` | 2 | Metal GPU fused decode; activates on `fused_decode_metal_successes > 0` |

Shadow mode is stable and collects runtime storage telemetry (layers, token
layers, bytes, written slots, eligible layers, candidate token layers, hot
token layers). Fused-experimental path has full fallback reason tracking:
`NONE`, `CPU_ORACLE_UNAVAILABLE`, `RUNNER_NOT_INTEGRATED`.

## Server CLI Flags (`args.rs:22-30`)

Both modes are exposed under `--experimental-mlx-kv-compression`:

- `turboquant-shadow` → `KvCompressionMode::TurboQuantShadow`
- `turboquant-fused-experimental` → `KvCompressionMode::TurboQuantFusedExperimental`

Additional tuning flags:
- `--experimental-mlx-kv-compression-hot-window-tokens` (default: `KvCompressionConfig::DEFAULT_HOT_WINDOW_TOKENS`)
- `--experimental-mlx-kv-compression-min-context-tokens` (default: `KvCompressionConfig::DEFAULT_MIN_CONTEXT_TOKENS`)

## Model Family Coverage

| Family file | TurboQuant decode | Notes |
|---|---|---|
| `standard.rs` | fully wired | decode attention path in `shared/attention.rs` |
| `llama4.rs` | not wired | `_turboquant_context` param present but unused |
| `glm4_moe_lite.rs` | not wired | same — param present, path not implemented |
| `deepseek_v3.rs` | not wired | same — param present, path not implemented |
| `mixtral.rs` | unclear | not audited in this pass |
| `mistral3.rs` | unclear | not audited in this pass |
| `qwen3_linear.rs` | unclear | not audited in this pass |

The underscore prefix on `_turboquant_context` in the MoE family files is
a deliberate Rust convention for accepted-but-unused parameters. These models
fall back to full-precision decode regardless of the compression mode flag.

## Candidate Status States (`turboquant_context.rs`)

Per-layer decode candidate has four states:

| Status | Meaning |
|---|---|
| `Ready` | Compressed decode will be attempted |
| `PrefillOnly` | Layer is in prefill phase; no compressed decode |
| `SlidingWindowLayer` | Sliding-window attention layer; excluded |
| `MissingRuntimeStorage` | Shadow storage not yet populated for this layer |

`MissingRuntimeStorage` is the expected state for the first few tokens after
session start, before the shadow storage write-back catches up.

## What Remains

1. **Long-context benchmark artifact** — run the fused-experimental path on a
   ≥32k-token context, produce and check in the quality artifact. This is the
   only step that clears the production gate.

2. **MoE model wiring** — LLaMA 4, GLM 4 MoE Lite, DeepSeek V3, and possibly
   others have the context parameter threaded through but the decode path
   unimplemented. These require separate work to assess feasibility given their
   attention architecture differences.

3. **Non-standard family audit** — `mixtral.rs`, `mistral3.rs`, `qwen3_linear.rs`
   were not audited in this pass.

## 2026-06-02 PRD Completion Update

The D3 standalone fused-kernel microbenchmark gate now has checked-in
head_dim=128 evidence:

- `benchmarks/results/turboquant/quality-runs/20260602T-prd-d3-microbench/microbench-head128-cold8192.json`
- Validation:
  `python3 scripts/check_turboquant_microbench_artifact.py --min-cold-tokens 8192 --min-speedup-vs-dim 1.5 benchmarks/results/turboquant/quality-runs/20260602T-prd-d3-microbench/microbench-head128-cold8192.json`
- Result: `two_stage_scores` median 3912 us vs `dim_parallel` median 12381 us
  at 8192 cold tokens, a 3.16x speedup with
  `min_cosine_similarity=0.9999998211860657`.

The PRD is still not complete. The current completion report is checked in at
`benchmarks/results/turboquant/quality-runs/20260602T-prd-d3-microbench/prd-completion-after-d3.json`
and keeps the following blockers open:

- missing D1+D2 model quality evidence for Qwen families;
- missing D4 short decode speedup evidence (1 new token, 1024 cold, ≥2x);
- no local model manifest can exercise the current fused K8/V4 promotion gate;
- no passing long-context fused-path performance promotion artifact.
