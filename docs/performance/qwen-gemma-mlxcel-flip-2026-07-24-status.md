# Qwen/Gemma mlxcel flip status — 2026-07-24

**Decision: `not_yet`** (locked gates not all met)

Primary host: AKMBPM5MAXx (Apple M5 Max). Gates:
`benchmarks/manifests/qwen_gemma_flip_gates.v1.json` (thresholds **not** relaxed).

Campaign: `benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-24-full-s0s3/`

## Median ratios (3 fresh-process reps)

| Scenario | thr | TTFT | gap | Result |
| --- | ---: | ---: | ---: | --- |
| S0 | **1.109×** (need ≥1.15) | **0.731× PASS** | **0.852× PASS** (9.4 ms) | thr only fail |
| S1 | 0.288× | 3.54× | 23.7× (811 ms) | fail |
| S2 | 1.071× | 0.769× PASS | 1.88× | thr+gap fail |
| S3 | 0.835× | 4.04× | 1.41× | fail |

## S0 thr ceiling (structural)

- Pure `decode-trace`: **107.0–107.5 tok/s**
- AX OpenAI SSE e2e: **~105 tok/s**
- mlxcel e2e: **~94.6 tok/s**
- Max thr ratio ≈ pure/mlx = **1.14 < 1.15** even with TTFT→0
- Host-sleep 4 ms fully absorbed → GPU/BW bound; hybrid n-gram multi-token **hurts** thr

## Path fixes landed this cycle

- Greedy OpenAI `repetition_penalty=1.0` (direct pipeline)
- Stream backlog + engine step burst + lightweight progress
- Short-prompt warmup + first-token TTFT bootstrap
- Sibling prefill quantum 1→16 (64 exploded S1 gap under arbiter)
- Linear n-gram: no permanent LinearNoDraft disable

## Next levers (still open)

1. **S0 thr:** Metal/BW composite kernels to push pure past ~110 tok/s, or thr-positive multi-token for hybrid linear attention
2. **S1:** wall-time adaptive quantum (measure chunk wall ≤50 ms) + fairer arbiter interleave; multi-process isolation is mlxcel's free lunch
3. **S3:** arbiter hold profiling + optional server-mode batched-decode drift product decision (P4)
