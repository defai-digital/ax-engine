# Lever: dual-stream gate/up qmm (M5 Max GPU concurrency)

## Residual

Profile pure Gemma 13.8k: **`post_attn_ffn_gate_up` ~3.26s** — two sequential
affine 8-bit qmms (gate then up). Host-FFI collapse alone (`DUAL_AFFINE_QMM`)
measured **1.002×** (noise). Compile / custom Metal / async_eval co-submit all
rejected or ≤0.7% pure.

Need ≤**0.96** pure under cache_eval for thr 1.15 physics (~10% of gate_up or
~4% of total wall).

## mlxcel

`gemma4.rs` ~917–920 multi-token bits=8 still sequential:
`gate_proj.forward` then `up_proj.forward` then GeGLU. mlxcel does **not** dual-
stream these matmuls; residual is the same dual-qmm wall AX sees.

AX opportunity on **M5 Max** (large GPU): independent matmuls on two Metal
command streams may overlap if not bandwidth-saturated.

## Change

`AX_MLX_DUAL_STREAM_GATE_UP=1` engages `ax_mlx_dual_affine_qmm` with two
process-static `mx::new_stream(gpu)` — gate on stream0, up on stream1. Metal
GEGLU stays on the default/production path and inherits MLX cross-stream deps.
Default OFF.

Also re-A/B `AX_MLX_DUAL_AFFINE_QMM=1` alone (same stream) as control.

## Pure A/B (mbp-m5, cache_eval keep_base)

| variant | env |
|---------|-----|
| base | both OFF |
| dual_qmm | `DUAL_AFFINE_QMM=1` |
| dual_stream | `DUAL_STREAM_GATE_UP=1` |

Keep if median cold ≤ **0.96×** base.

## Success

Pure ≤0.96 → cool multi-process S1; thr≥1.15 + gap/TTFT → full S0–S3 flip.

## Result (mbp-m5 pure cache_eval, 2026-07-25, 3-rep)

| variant | median cold ms | ratio |
|---------|---------------:|------:|
| base | 8245 | 1.000 |
| dual_qmm (same stream) | 8268 | **1.003** |
| dual_stream (2 GPU streams) | 9460 | **1.147** |

Decision **keep_base** / reject both. Dual-stream is ~15% worse (stream
sync / command-queue overhead on Metal; matmuls likely bandwidth-bound so
no concurrency win). Defaults OFF. Text first-token parity OK (` The`).
