# Performance And Benchmark Docs

Navigation and policy entry point for AX Engine performance docs. Use it to
decide which result table, method, or investigation note applies to a claim.

## Start Here

| Need | Read |
| --- | --- |
| Historical dense 27B peer: retired Qwen 3.8 27B AXQ 6-bit vs MTPLX and OMLX | [Historical dense 27B evidence](../PERFORMANCE-RESULTS.md#qwen-38-27b-axq-historical-6-bit-selector) |
| Tiel / Cyber-Tiel vs MTPLX on M5 Max, M4 Pro, M2 Ultra and M3 Ultra | [Four-machine results, latency and memory](tiel-vs-mtplx-2026-09-20.md) |
| Current public result tables and charts | [Performance Results](../PERFORMANCE-RESULTS.md) |
| Current MLX runtime admission | [MLX 0.32.4 admission](mlx-0.32.4-admission-2026-10-16.md) |
| Flash Next sparse attention: the prefill row-flag scan as its own parallel kernel, and short rows decided on the host; block-key time at 3,256 blocks from 1.3 ms to 0.4 ms (M2 Ultra) and 0.6 ms (M4 Pro), bit-identical; M5 Max first text not yet measured | [Change, kernel timing, tests and what is left](flash-next-qsa-row-flags-2026-11-27.md) |
| Qwen 3.8 27B DFlash2 on M5 Max: 8 against 16 draft-tree rows at 64 to 256 output tokens; 16 rows 1 to 6% faster on seven prompts but different greedy text, so no budget rule and no default change | [Measurements, reply differences and decision](qwen38-27b-dflash2-tree-rows-2026-11-26.md) |
| Flash Next on M5 Max: the routed expert projections of a prefill in 64-row tiles, bit-identical, first text 3 to 9% earlier from 127 to 13,023 tokens and ahead of oMLX at eleven of twelve lengths | [Kernel, measurements against oMLX by prompt length, numerics, switch and what is left](flash-next-moe-gather-tiles-2026-11-26.md) |
| Flash Next on M5 Max: sparse attention on the GPU tensor units from 4,096 tokens of history, cost independent of the prompt length, first text 7% faster at 13,000 tokens, closer to exact attention than the paths it replaces | [Kernel, measurements, numerics against float64, switch and what is left](flash-next-qsa-tensor-unit-2026-11-26.md) |
| Text requests: one shared tokenizer instead of a copy per request, first text 10 to 12 ms earlier at every prompt length | [Cause, change, measurements on M5 Max and M4 Pro](server-tokenizer-shared-2026-11-26.md) |
| First request after the server sat idle: wired weights kept GPU-resident, 1.5 to 0.74 s for a 1,492-token Flash Next prompt on M5 Max, level with oMLX | [Cause, switch, measurements on M5 Max and M4 Pro, limits](idle-residency-refresh-2026-11-26.md) |
| Qwen 3.8 27B: greedy prompts without the block-boundary split and MLX's gated-delta recurrence on M5 and later, direct first text ahead of oMLX (0.17 against 0.22 s at 127 tokens, 1.50 against 1.51 s at 1,492), tile rule for M3/M4 | [Measurements on M5 Max and M4 Pro, logit-level comparison of the recurrence, replay cost and what is left](qwen38-27b-prefill-schedule-2026-11-25.md) |
| Qwen 3.8 on M5 Max: DFlash2 draft-tree verify, MX-format NAX rows, Flash Next decode and prefill kernels | [Paired results against the leading peers, root causes, switches and limits](qwen38-m5-tree-verify-and-flash-next-2026-11-25.md) |
| Flash Next on M5 Max: sparse attention past 2,051 tokens as one masked dense call, first text 20 to 39% faster from 3,000 to 9,000 tokens, rounding-level numerics change | [Measurements against oMLX by prompt length, numerics, switch and what is left](flash-next-qsa-masked-dense-2026-11-25.md) |
| Flash Next on M5 Max: the compiled elementwise stretches at prefill width, bit-identical, first text level with oMLX at 1,492 tokens | [Measurements, closure and trace handling, switch and what is left](flash-next-prefill-fused-stretches-2026-11-25.md) |
| Flash Next on M5 Max: prefill schedule without dead prefix captures, 2,048-token chunks on long prompts, first text by prompt length against oMLX | [Measurements, the capture that was not shipped, and what is left](flash-next-prefill-schedule-2026-11-25.md) |
| M5+ NAX verify kernels with MLX 914732419 (0.32.5): QMM and SDPA hand-offs, DFlash2 window cost | [Measurements, knobs and build recipe](mlx-nax-914732419-2026-11-02.md) |
| Opt-in MLX GDN prefill experiment | [Kernel comparison and limits](mlx-gdn-prefill-2026-10-03.md) |
| Flash Next prefill and decode against the pinned MLX-VLM reference | [Findings, changes, kill switches and measurements](flash-next-prefill-decode-2026-10-05.md) |
| Qwen 3.8 27B MXFP4 on pre-M3 GPUs: fp16 activations, dense-GEMM prefill, exact-MTP lm_head window | [Host-adaptive policies; M2 Ultra, M4 Pro and M5 Max measurements](qwen38-27b-pre-m3-prefill-2026-10-10.md) |
| Why Mac mini M4 Pro and M5 Max decode differed on the retired 6-bit 27B pack | [Decode bandwidth utilization](decode-bandwidth-utilization.md) |
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
| MTP matrix | Qwen 3.8 27B and Flash Next use pinned MXFP4 snapshots. Qwen 3.6 and Gemma assistant packs that still use `download-mtp` keep the 6-bit lane; labeled 4-bit rows stay for peer-aligned comparison | Current design |
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
| **Shipped, default-on for eligible families** | `AX_MLX_AUTO_BUFFER_CAPS` — raise MLX per-buffer caps so `gather_qmm` stops splitting MoE command buffers. Coder-Next **1.25×** decode, parity clean. On M5+ (macOS 26.2+) the both-caps raise is `qwen3_next` only, `qwen4_exp` (Flash Next) raises the byte cap alone (58.0 to 60.2 tok/s decode on M5 Max), and other families keep MLX defaults (NA policy). Unlimited-OCR, Gemma, and `qwen3_5` stay excluded on every host: M5/MLX 0.32 A/Bs found the raised pair slower (OCR −26% e2e; Gemma −32% p2048 prefill; 35B-A3B server prefill −20…−47% with a one-way per-request degradation and flat decode). `AX_MLX_NAX_ATTENTION=0` kills the allowlisted full-attn native-causal SDPA policy on those same hosts (Gemma stays excluded). | [gather-qmm-async-serialization.md](gather-qmm-async-serialization.md) |
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
