# Path B fairness residual — 2026-07-26 (mbp-m5 / M5 Max)

**Decision: `not_yet`.** Locked gates unchanged. No S0–S3 flip claim.

Gates: `benchmarks/manifests/qwen_gemma_flip_gates.v1.json`  
(thr ≥1.15×, TTFT ≤0.90×, gap ≤0.90× and ≤50 ms, zero errors).

## Ship residual (fail-closed default OFF)

1. **`AX_MLX_PIPELINE_EVAL_GRANULARITY`** — blocking prefill layer barriers
   (`off` / `layer` / `block:N` / `sublayer`). Unlike async-only
   `AX_MLX_PIPELINE_GRANULARITY` (pure reject 1.04–1.07×), this inserts
   `eval` completion barriers so Metal can interleave a sibling process.
   Unit-tested parse + fire predicates; default OFF.
2. **Single-stream burst yields to queued admission** — generation worker
   stops a 64-step single-stream burst when `queued_commands > 0` so a new
   `StartStream` is not HOL-blocked for the rest of the burst. Single-process
   fairness; multi-process topology is unaffected.
3. **Target `process_qos_clamp`** — optional macOS `taskpolicy -c` wrap
   (`utility` / `background` / `maintenance`) for multi-process concurrent-tax
   probes. Fail-closed; unit-tested.

## Formal cool 3-rep S1 (same tip binary)

### Baseline recheck (cache_eval only)

Artifact: `2026-07-26-s1-mp-cache-eval-baseline-recheck/`

| metric | AX median | mlxcel median | ratio | gate |
|--------|----------:|--------------:|------:|------|
| thr tok/s | 20.42 | 18.35 | **1.113** | FAIL (≥1.15) |
| gap p95 ms | 38.61 | 37.56 | **1.028** | FAIL (≤0.90); abs PASS |
| TTFT p95 | — | — | **0.895** | PASS |

Tip reconfirm: thr still ~1.11× (historical Jul-25 best 1.109×). Absolute AX thr
~20.4 is the multi-process + cache_eval thr ceiling on this binary.

### Layer-eval (blocking barriers on Gemma)

Artifact: `2026-07-26-s1-mp-cache-eval-layer-eval/`

| metric | AX median | mlxcel median | ratio | gate |
|--------|----------:|--------------:|------:|------|
| thr tok/s | 20.40 | 18.38 | **1.110** | FAIL (≥1.15) |
| gap p95 ms | 35.09 | 35.09 | **1.000** | FAIL (≤0.90); abs PASS |
| TTFT p95 | — | — | **0.898** | PASS |

vs same-day baseline: layer-eval **cuts gap p95 38.6→35.1 ms** (ratio 1.028→1.000)
with thr unchanged. Fairness residual is real; thr residual is not closed.

## Formal cool 3-rep S1 (async pipeline hints)

Artifact: `2026-07-26-s1-mp-cache-eval-async-pipeline/`
(`AX_MLX_PIPELINE_GRANULARITY=layer` async_eval only; no blocking eval)

| metric | AX median | mlxcel median | ratio | gate |
|--------|----------:|--------------:|------:|------|
| thr tok/s | 20.58 | 18.34 | **1.122** | FAIL (≥1.15) |
| gap p95 ms | 38.74 | 34.79 | **1.113** | FAIL (≤0.90); abs PASS |
| TTFT p95 | — | — | **0.888** | PASS |

Best thr ratio this session (+~0.8% abs thr vs same-day baseline 20.42). Gap
unchanged vs baseline. Still short of 1.15.

## Smoke ladder (1-rep; absolute thr is the thr claim)

| config | thr ratio | ax thr | ax gap | note |
|--------|----------:|-------:|-------:|------|
| Qwen `utility` QoS | 1.124 | 20.42 | 38.2 | abs thr ≈ base |
| Qwen `background` QoS | 1.134 | 20.35 | 37.5 | abs thr ≈ base |
| layer + Qwen utility | 1.008 | 18.24 | 39.0 | thr regress |
| sublayer | 1.019 | 18.55 | 35.8 | thr regress |
| `MLX_METAL_FAST_SYNCH=1` | 1.118 | 20.40 | 38.0 | abs thr ≈ base |
| async block:2 only | 1.136 | 20.59 | 38.1 | ≈ async layer thr |
| async layer + block:4 eval | 1.077 | 19.63 | 38.4 | thr regress |
| async layer + block:2 eval | 1.102 | 20.15 | 37.7 | thr regress vs async |

**Reject thrash for thr:** process QoS, FAST_SYNCH, and async+blocking composites
do not clear thr≥1.15. Blocking layer-eval remains a **gap-fairness** probe
(default OFF). Async pipeline hints give a small thr lift (formal 1.122) but
leave gap and thr gates open.

## Physics (unchanged core)

- thr ≥1.15 still needs ~2.5% more on the best thr stack (async 1.122) **and**
  gap ≤0.90 (layer-eval gets gap to 1.000, not 0.90).
- No measured stack clears thr **and** gap simultaneously.
- Path A (steel-class dual-gate_up ≤~0.96 pure under cache_eval) remains the
  thr unlock; host-FFI / custom Metal / pack / dual-stream still reject.
- Full S0–S3 **not** run: thr physics still cannot clear 1.15.

## Product posture

- Gates file thresholds **unchanged**.
- Product default remains single-process exclusive multi-model.
- Multi-process + `CACHE_ONLY_CHUNK_EVAL=1` remains measurement topology.
- All new thr/fairness opt-ins default **OFF**.
