# TurboQuant D3 PRD Microbench Evidence

This directory records standalone fused cold-decode kernel evidence for
`PRD-2026-06-01-turboquant-codec-kernel-improvements.md`.

## Artifact

- `microbench-head128-cold8192.json`
- Schema: `ax.turboquant_fused_decode_microbench.v1`
- Shape: `head_dim=128`, `cold_tokens=8192`, `hot_tokens=128`
- Variants: `dim_parallel`, `two_stage_scores`
- Repetitions: 5, warmup: 1

## Command

```text
cargo run -p ax-engine-microbench --release --bin turboquant-microbench -- \
  --cold-tokens 8192 \
  --hot-tokens 128 \
  --head-dim 128 \
  --n-query-heads 1 \
  --n-kv-heads 1 \
  --variants dim_parallel,two_stage_scores \
  --repetitions 5 \
  --warmup 1 \
  --output benchmarks/results/turboquant/quality-runs/20260602T-prd-d3-microbench/microbench-head128-cold8192.json
```

## Validation

```text
python3 scripts/check_turboquant_microbench_artifact.py \
  --min-cold-tokens 8192 \
  --min-speedup-vs-dim 1.5 \
  benchmarks/results/turboquant/quality-runs/20260602T-prd-d3-microbench/microbench-head128-cold8192.json
```

Result: `ok`.

Observed medians:

| Variant | Median us |
|---|---:|
| dim_parallel | 12381 |
| two_stage_scores | 3912 |

`two_stage_scores` is 3.16x faster than `dim_parallel` for this shape.

## PRD Completion Report

`prd-completion-after-d3.json` records the aggregate PRD status after this D3
evidence was added. The PRD is still incomplete because D1+D2 family quality,
D4 short decode speedup, and production promotion evidence remain open.
