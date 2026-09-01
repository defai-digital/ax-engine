# Performance And Benchmark Docs

Navigation and policy entry point for AX Engine performance docs. Use it to
decide which result table, method, or investigation note applies to a claim.

## Start Here

| Need | Read |
| --- | --- |
| Current public result tables and charts | [Performance Results](../PERFORMANCE-RESULTS.md) |
| Current MLX runtime admission | [MLX 0.32.2 admission](mlx-0.32.2-admission-2026-08-29.md) |
| Root README headline numbers | [README Performance](../../README.md#performance) |
| AX vs peer MLX single-client serving (2026-08-06) | [AX vs peer MLX serving (Qwen 3.6)](ax-vs-peer-mlx-serving-qwen36-2026-08-06.md) |
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
| [Performance Results](../PERFORMANCE-RESULTS.md) | Full public tables and charts (S1, serving, MTP, direct, embeddings) |
| [AX vs peer MLX serving (Qwen 3.6)](ax-vs-peer-mlx-serving-qwen36-2026-08-06.md) | Single-client OpenAI streaming peer (2026-08-06, M5 Max) |
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

## Decode Dispatch Efficiency Program (Phases 0–3)

Outcome of the dispatch-bound decode optimization program (ADR-003). All numbers
Apple M3 Max, MLX 0.32.0. "Parity" = greedy decode-trace FNV checksum.

| Outcome | What | Evidence |
| --- | --- | --- |
| **Shipped, default-on for eligible families** | `AX_MLX_AUTO_BUFFER_CAPS` — raise MLX per-buffer caps so `gather_qmm` stops splitting MoE command buffers. Coder-Next **1.25×** decode, parity clean. On M5+ (macOS 26.2+) the raise is `qwen3_next` only; other families keep MLX defaults (NA policy). Unlimited-OCR, Gemma, and `qwen3_5` stay excluded on every host: M5/MLX 0.32 A/Bs found the raised pair slower (OCR −26% e2e; Gemma −32% p2048 prefill; 35B-A3B server prefill −20…−47% with a one-way per-request degradation and flat decode). `AX_MLX_NAX_ATTENTION=0` kills the allowlisted full-attn native-causal SDPA policy on those same hosts (Gemma stays excluded). | [gather-qmm-async-serialization.md](gather-qmm-async-serialization.md) |
| **Shipped, default-on** | `AX_MLX_BATCHED_SHARED_PROJ` — strict M5 dense probe at **B=8**: RowExact **102.6** → Shared **328.9 aggregate tok/s** (**3.20×**, 5/5 paired wins); Shared scales **4.01×** from B=1 and full-cohort hashes match. Microbenchmark evidence, not end-to-end serving or MoE. | [batched-decode-ceiling.md](batched-decode-ceiling.md) |
| **Shipped** | Runner split (5 slices, −4.2k lines) + decode-skeleton I1/I2 (typed direct-pipeline state, centralized barrier/readback). | internal spec |
| **Opt-in, uncertified** | Batched decode for the Qwen3-Next hybrid (MoE + linear attention). Correct forward (B=1 token-exact), amortizes **1.73/2.64/3.84×** at B=2/4/8, but B>1 drifts from per-row (batched-MoE `gather_qmm` reductions) → fails bit-exact greedy parity, stays behind `AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED`. | [batched-hybrid-moe-linear-decode.md](batched-hybrid-moe-linear-decode.md) |
| **Closed by decision** | Phase 2 decode-skeleton unification — banked I1/I2 + split; the `DecodeRoute` trait fold (I4–I7) is parked: both drivers dissolved (batched extension shipped without it; overlap fix is upstream-gated). | internal spec §8 |
| **Upstream-gated** | Per-layer graph compile / zero-bubble overlap — decode is host-graph-encoding-bound, and the only lever is blocked on MLX GatherQMM shapeless-compile. | [ax-decode-overlap-residual.md](ax-decode-overlap-residual.md), [mlx-upstream-issue-gather-qmm-buffer-accounting.md](mlx-upstream-issue-gather-qmm-buffer-accounting.md) |
| **Built + A/B'd, default-on is a product SLO call** | **Fair chunked-prefill interleave** — `max_prefill_tokens_per_request_per_step` caps a lone prefill so it can't inflate an active decode cohort's step (scheduler tests pin the single-step bound + the multi-step temporal interleave). Serving A/B (Qwen3-4B-4bit, two load points): decode-step tail **p99 −62%** overloaded, **−98%** at realistic non-saturated load; TTFT p50 −19%/−62%, throughput neutral-to-+4%, TTFT-tail cost only under overload (vanishes at realistic load). Overload-artifact objection refuted. Strongly supports default-on for decode-smoothness SLOs; final flip is a product call wanting 1–2-model confirmation. **B-bucket primitive** — batch-size snap helper (`AX_MLX_DECODE_BATCH_BUCKETS`, default off) built + tested; cohort padding deferred until the batched forward is compiled per shape (eager-path padding is a likely regression). | [fair-prefill-serving-ab.md](fair-prefill-serving-ab.md); scheduler tests |
| **Needs product sign-off** | Phase 3 remainder: paged KV as capacity. | — |

## Investigation Notes

Historical or diagnostic context unless a current results page links a fresh
artifact.

| Report | Use it for |
| --- | --- |
| [phase0-overlap-bandwidth-mtp.md](phase0-overlap-bandwidth-mtp.md) | Phase 0 measurement: host/GPU overlap, MoE bandwidth utilization, MTP clone cost |
| [decode-gap.md](decode-gap.md) | Direct-decode gap analysis against `mlx_lm.benchmark` |
| [gather-qmm-async-serialization.md](gather-qmm-async-serialization.md) | `gather_qmm` buffer-accounting root cause + the shipped auto-buffer-caps fix |
| [ax-decode-overlap-residual.md](ax-decode-overlap-residual.md) | Why the decode overlap residual is host-graph-encoding-bound, not a sync bug |
| [batched-decode-ceiling.md](batched-decode-ceiling.md) | Current M5 dense batched-decode certification + historical Shared projection investigation |
| [batched-hybrid-moe-linear-decode.md](batched-hybrid-moe-linear-decode.md) | Phase 3.7 hybrid (MoE + linear) batched decode: capability, amortization, drift |
| [fair-prefill-serving-ab.md](fair-prefill-serving-ab.md) | Fair chunked-prefill interleave serving A/B (decode-step tail vs TTFT trade) |
| [moe-bandwidth-gap.md](moe-bandwidth-gap.md) | Qwen3-Coder-Next MoE bandwidth diagnosis |
| [moe-fused-downproj.md](moe-fused-downproj.md) | Failed fused-downprojection report and lessons |
| [mlx-upstream-issue-gather-qmm-buffer-accounting.md](mlx-upstream-issue-gather-qmm-buffer-accounting.md) | Upstream MLX report draft (GatherQMM buffer accounting) |
