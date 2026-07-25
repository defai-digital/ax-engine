# Lever: dual-hold with tiny adaptive prefill quanta (S1 thr residual)

## Residual (not pure GPU)

Latest cool S1 on mbp-m5 (`2026-07-24-s1-post-qkrope-reject`):

| side | thr tok/s | thr ratio | gap p95 | TTFT p95 |
|------|----------:|----------:|--------:|---------:|
| mlxcel multi-proc | 17.97 | — | 34.8 ms | 9964 ms |
| AX exclusive (max=1) | 18.48 | **1.028×** | 9.0 ms | 8299 ms |

Need thr ≥ **1.15×** → abs thr ≳ **20.7**. Gap/TTFT already pass under exclusive.

Physics (status-continued + deep review §S1):

- AX pure Gemma is already ~14% **faster** than mlxcel pure Gemma.
- Exclusive pure-sum / serialized interleave thr ceiling ~18.5–19.7 — **not** thr≥21.
- mlxcel S1 thr comes from **multi-process dual-stream** (Qwen process keeps decoding while Gemma process prefills; Metal time-shares).
- Prior AX dual-hold (`max=2`, exclusive OFF, adaptive start 64 / SLO 28–40 ms) measured gap p95 **160–220 ms** and thr regression under Metal contention (`2026-07-24-s1-dualhold-slo28`).

## Hypothesis (new measurement, not re-run of rejected dual-hold)

Dual-hold gap may be driven by **quantum wall under contention**, not dual-hold itself. Re-test with:

- `AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT=2`
- `AX_SERVER_LONG_PREFILL_EXCLUSIVE=0`
- `AX_SERVER_ADAPTIVE_PREFILL_LATENCY_TOKENS=4` (start tiny; adaptive may grow to max 256)
- sibling burst 1 or 4

Target: S1 thr ≥1.15× **and** gap p95 ≤50 ms and ≤0.9× mlxcel.

## Success metric

Cool ≥3-rep dual-target S1: thr ratio ≥1.15, gap abs ≤50 ms, gap ratio ≤0.90, zero errors.
Else reject; restore exclusive defaults; gates unchanged; flip remains **not_yet**.

## Result (mbp-m5, 2026-07-25) — **REJECT**

Artifact:
`benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-25-s1-dualhold-q4/flip-decision.json`

| side | thr median | gap p95 median | thr ratio | gap ratio |
|------|----------:|---------------:|----------:|----------:|
| mlxcel multi-proc | 17.97 | 35.4 ms | — | — |
| AX dualhold q4 | **18.94** | **165.8 ms** | **1.054×** | **4.68×** |

Failed required gates: `median_throughput_ratio` (1.054 < 1.15),
`median_stream_gap_p95_ratio` (4.68 > 0.90), `absolute_stream_gap_p95` (166 ms > 50).

Tiny adaptive start (4) does **not** fix dual-hold gap under Metal contention —
same ~160–220 ms envelope as prior dual-hold SLO28. Thr only +~1.5% vs exclusive
1.028× (still far from 1.15×). **Keep exclusive defaults** (max_concurrent=1,
long-prefill exclusive ON). Do not promote dualhold-q4 target.

## Pure stack note

Cache-only chunk eval pure median **0.968** (need ≤0.925) — insufficient alone for thr headroom
(see `cache-only-chunk-eval.md`). Host-side pure fuses for gate_up/down/sdpa/qkv/o_proj/rope
already rejected this session.
