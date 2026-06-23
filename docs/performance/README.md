# Performance And Benchmark Docs

This page is the navigation and policy entry point for AX Engine performance
docs. Use it to decide which result table, benchmark method, or investigation
note applies to the claim you are making.

## Start Here

| Need | Read |
|---|---|
| Current public result tables and interpretation | [`../PERFORMANCE.md`](../PERFORMANCE.md) |
| Benchmark methodology, command contracts, artifact schemas, and reproduction | [`../BENCHMARKS.md`](../BENCHMARKS.md) |
| Benchmark-system design and workload-contract details | [`../BENCH-DESIGN.md`](../BENCH-DESIGN.md) |
| Online serving prompt-mix, concurrency, latency, throughput, and SLO evidence | [`../SERVING-BENCHMARKS.md`](../SERVING-BENCHMARKS.md) |
| User-facing performance caveats and common misunderstandings | [`../FAQ.md`](../FAQ.md) |

## Public Claim Boundaries

AX Engine's public performance claims are scoped to benchmark artifacts that
preserve route identity, model artifacts, prompt suite, sampler settings,
repository provenance, and enough reproduction metadata to rerun or reject the
claim.

| Area | Public claim | Status |
|---|---|---|
| 6-bit MTP matrix | Current benchmark design covers only `qwen3.6-27b-6bit`, `qwen3.6-35b-a3b`, `gemma-4-12b`, `gemma-4-26b`, `gemma-4-31b`, and `glm-4.7-flash` prepared by `ax-engine download-mtp` | Current design |
| MTP+n-gram | Removed from the MTP benchmark matrix; historical rows are diagnostic only and should not be promoted as current MTP evidence | Out of scope |
| N-gram acceleration | Up to 3.1x `mlx_lm` decode throughput on high-hit benchmark rows without a second draft model | Workload-dependent |

## Evidence Rules

- Repo-owned MLX throughput claims require AX route identity and a matching
  upstream `mlx_lm.benchmark` baseline unless the documented peer cannot load
  the model family.
- Delegated `mlx_lm_delegated` and `llama_cpp` rows are route-compatibility
  evidence, not AX-owned MLX token/KV throughput.
- MTP publication rows use the six 6-bit `download-mtp` targets only. Same-model
  AX direct rows may be used as denominators for MTP acceleration, not as a
  cross-model speed leaderboard.
- TTFT, prefill, decode, accept rate, sampler settings, prompt suite, cooldown,
  repetition count, model snapshot, and sidecar or assistant provenance must stay
  attached to the artifact used for a public claim.

## Result Surfaces

| Surface | Scope | Notes |
|---|---|---|
| [`../PERFORMANCE.md`](../PERFORMANCE.md) | Current public performance report | Result tables, latest artifact summaries, interpretation, MTP mode, and reproduction notes |
| [`../LONG-CONTEXT.md`](../LONG-CONTEXT.md) | Long-context and prefix-reuse behavior | Separate from short/mid-prompt README result claims |
| [`../NGRAM-ACCELERATION.md`](../NGRAM-ACCELERATION.md) | N-gram acceleration claim taxonomy | Use for n-gram correctness, promotion gates, and required artifact fields |
| [`../MTP-DRAFT-GATE-THROUGHPUT.md`](../MTP-DRAFT-GATE-THROUGHPUT.md) | MTP draft confidence gate tuning | Diagnostic tuning report; not the current six-model MTP publication matrix |

## Methodology Surfaces

| Surface | Scope | Notes |
|---|---|---|
| [`../BENCHMARKS.md`](../BENCHMARKS.md) | How to run and classify benchmarks | Main command reference for model-inference, MTP, serving, delegated, readiness, and community runs |
| [`../BENCH-DESIGN.md`](../BENCH-DESIGN.md) | Benchmark CLI and artifact design | Workload contracts, manifest schema, gates, and fail-closed principles |
| [`../SERVING-BENCHMARKS.md`](../SERVING-BENCHMARKS.md) | Online serving benchmark contract | Prompt mix, concurrency, request rate, latency percentiles, throughput, and SLO goodput |

## Investigation Notes

These reports explain specific historical optimization or diagnosis work. Treat
them as context unless a current result table links to a fresh artifact.

| Report | Use it for |
|---|---|
| [`../PERFORMANCE-DECODE-GAP.md`](../PERFORMANCE-DECODE-GAP.md) | Direct-decode gap analysis against `mlx_lm.benchmark` |
| [`../PERFORMANCE-MOE-BANDWIDTH-GAP.md`](../PERFORMANCE-MOE-BANDWIDTH-GAP.md) | Qwen3-Coder-Next MoE bandwidth-utilization diagnosis |
| [`../PERFORMANCE-MOE-FUSED-DOWNPROJ.md`](../PERFORMANCE-MOE-FUSED-DOWNPROJ.md) | Failed fused-downprojection optimization report and lessons |
