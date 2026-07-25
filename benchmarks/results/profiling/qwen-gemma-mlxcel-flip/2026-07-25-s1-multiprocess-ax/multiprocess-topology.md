# Lever: multi-process AX topology (mlxcel S1 thr residual)

## Residual

mlxcel deep-review §S1 + flip status: **mlxcel runs one process per model**.
Gemma prefill and Qwen decode submit independently; Metal time-shares. AX
product default is **single-process multi-model** with exclusive arbiter
(dual-hold max=2 fails gap 160–220 ms).

Cool exclusive S1 tip thr **1.036×** (gap PASS). Pure host/GEMM residuals
exhausted for ≥11% pure cut.

Probe `2026-07-25-ax-multiprocess-s1-probe` (two AX servers, 48GB each):

| metric | median |
|--------|-------:|
| thr tok/s | **19.42** (~1.08× vs mlxcel 17.97) |
| gap p95 | **48.2 ms** (≤50) |

Topology is the residual that recovers gap SLO while lifting thr vs exclusive.
Still short of 1.15× in the probe — formal dual-target S0–S3 measures the
official thr metric and S2/S3 lifecycle under multi-process AX.

## mlxcel source

`benchmarks/manifests/targets/mlxcel-v0.4.2-qwen-gemma-m5max.json`:
`managed_processes: true`, distinct ports per model.

## AX change

1. `ProcessSupervisor.command_for` AX branch (`--model-id` / `--mlx` / …).
2. Target `ax-qwen-gemma-m5max-multiprocess.json` (one process per model, 48GB
   each, ports 31421/31422).
3. Cool ≥3-rep dual-target campaign vs mlxcel multi-process.

## Success

Decision `flip` under locked gates on ≥3-rep S0–S3. Else record not_yet with
numbers; keep single-process as default product path.

## Formal S1 result (mbp-m5, 2026-07-25) — **not_yet**

| metric | AX multi-proc | mlxcel multi-proc | ratio | gate |
|--------|--------------:|------------------:|------:|------|
| thr tok/s | **19.12** | 18.01 | **1.062** | FAIL (≥1.15) |
| gap p95 ms | **46.1** | 35.0 | **1.32** | FAIL (≤0.90) |
| TTFT p95 ms | 9339 | 9937 | **0.940** | FAIL (≤0.90) |

Absolute gap ≤50 ms passes; thr/gap-ratio/TTFT fail. Topology lifts thr vs exclusive
(~18.6→19.1) but does **not** reach flip. Keep single-process product default.
