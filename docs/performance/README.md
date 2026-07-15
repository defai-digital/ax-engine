# Performance And Benchmark Docs

Navigation and policy entry point for AX Engine performance docs. Use it to
decide which result table, method, or investigation note applies to a claim.

## Start Here

| Need | Read |
| --- | --- |
| Current public result tables and charts | [Performance Results](../PERFORMANCE-RESULTS.md) |
| Interpretation and claim context | [Performance](../PERFORMANCE.md) |
| MTP packages, 4-bit vs 6-bit lanes | [MTP Docs](../mtp/README.md) |
| Methodology, commands, artifact schemas | [Benchmarks](../BENCHMARKS.md) |
| Workload-contract CLI design | [Benchmark Design](../BENCH-DESIGN.md) |
| Online serving latency / throughput / SLO | [Serving Benchmarks](../SERVING-BENCHMARKS.md) |
| User-facing caveats | [FAQ](../FAQ.md) |

## Public Claim Boundaries

Public performance claims must be backed by artifacts that preserve route
identity, model artifacts, prompt suite, sampler settings, repository
provenance, and enough metadata to rerun or reject the claim.

| Area | Public claim | Status |
| --- | --- | --- |
| MTP matrix | Recommend 6-bit `download-mtp` targets for practical use; keep labeled 4-bit rows for peer-aligned comparison | Current design |
| MTP+n-gram | Removed from the MTP matrix; historical only | Out of scope |
| N-gram acceleration | Up to ~3.1× `mlx_lm` decode on high-hit rows without a second draft model | Workload-dependent |

## Evidence Rules

- Repo-owned MLX throughput claims require AX route identity and a matching
  upstream `mlx_lm.benchmark` baseline unless the documented peer cannot load
  the model family.
- Delegated `mlx_lm_delegated` and `llama_cpp` rows are route-compatibility
  evidence, not AX-owned MLX token/KV throughput.
- MTP publication rows use the 6-bit `download-mtp` targets as the practical
  lane. Labeled 4-bit rows may be used for peer comparison only. Same-model AX
  direct rows may be denominators for MTP acceleration, not a cross-model
  speed leaderboard.
- TTFT, prefill, decode, accept rate, sampler settings, prompt suite, cooldown,
  repetition count, model snapshot, and sidecar or assistant provenance must
  stay attached to the artifact used for a public claim.
- Prefill/TTFT peer rows require the same resolved `libmlx` on both sides and
  the GEMM / quantized-matmul admission check (see
  [Performance Results](../PERFORMANCE-RESULTS.md) and
  [Benchmarks](../BENCHMARKS.md)).

## Result Surfaces

| Surface | Scope |
| --- | --- |
| [Performance Results](../PERFORMANCE-RESULTS.md) | Full public tables and charts (MTP, direct, embeddings) |
| [Performance](../PERFORMANCE.md) | Interpretation, long-context notes, MTP mode policy |
| [MTP Docs](../mtp/README.md) | `download-mtp`, 6-bit vs 4-bit lanes, MTP validation |
| [Long Context](../LONG-CONTEXT.md) | Prefix reuse and long-context boundaries (not short/mid README claims) |
| [N-gram Acceleration](../NGRAM-ACCELERATION.md) | N-gram claim taxonomy and promotion gates |
| [MTP draft gate throughput](../mtp/draft-gate-throughput.md) | Diagnostic tuning; not the six-model MTP matrix |

## Methodology Surfaces

| Surface | Scope |
| --- | --- |
| [Benchmarks](../BENCHMARKS.md) | Commands and evidence classification |
| [Benchmark Design](../BENCH-DESIGN.md) | Workload contracts, manifests, fail-closed gates |
| [Serving Benchmarks](../SERVING-BENCHMARKS.md) | Online prompt mix, concurrency, SLO goodput |

## Investigation Notes

Historical or diagnostic context unless a current results page links a fresh
artifact.

| Report | Use it for |
| --- | --- |
| [decode-gap.md](decode-gap.md) | Direct-decode gap analysis against `mlx_lm.benchmark` |
| [moe-bandwidth-gap.md](moe-bandwidth-gap.md) | Qwen3-Coder-Next MoE bandwidth diagnosis |
| [moe-fused-downproj.md](moe-fused-downproj.md) | Failed fused-downprojection report and lessons |
