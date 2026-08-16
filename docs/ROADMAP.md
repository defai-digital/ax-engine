# Roadmap

AX Engine v7 is the current serving-oriented runtime line. It carries forward
the earlier serving baseline work, including:

- disk-backed L2 prefix cache
- MLA warm-extend correctness (default-on snapshot restore; kill switch
  `AX_DISABLE_MLA_PREFIX_RESTORE`)
- per-request MLA dual-path prefill chunks for GLM-4.7-Flash (large cold /
  block-aligned warm-extend) without losing warm-extend snapshot equivalence

Active plan for remaining KV weak surfaces (MLA cold throughput recovery,
fair multi-prefill progress, FA physical block-pool scaffold):
[`docs/designs/kv-weak-surfaces-2026-07-14.md`](designs/kv-weak-surfaces-2026-07-14.md).

The root README summarizes the current product shape; this page carries active
serving direction and evidence gates.

## Serving Runtime Tracks

The next optimization tracks are:

| Track | Direction |
|---|---|
| KV cache memory layout | Paged or block-aligned KV storage, better per-layer locality, fewer KV copies or transposes, and cache reuse between speculative draft and target verification paths |
| Apple unified memory advantage | Zero-copy weight mapping, memory-mapped quantized weights, direct Metal buffer reuse, fewer temporary tensor materializations, and persistent request buffers to improve cold start, TTFT, and memory pressure |
| MoE expert locality optimization | Expert-weight cache scheduling, token grouping by expert, lower dispatch overhead, likely-expert prefetching, router/dispatch fusion, and top-k routing memory-pattern tuning |
| Speculative decoding software tuning | Adaptive n-gram length, dynamic draft windows, acceptance-rate prediction, fallback thresholds, prompt-pattern-aware speculation, and better cache sharing between draft and verify paths |
| Kernel fusion and quantization path | Fused RMSNorm/matmul, attention projection fusion, fused dequant/matmul, group-wise quantization kernels, Apple AMX/Metal mixed paths, and prepacked weight layouts |

## v7 Serving Hardening Gates

A major version is a product claim, not a counter. The v7.0.0 release review
on 2026-08-15 kept exact-SHA CI as a publication gate and explicitly descoped
the remaining items below from binary publication. They remain claim-promotion
gates: v7.0.0 does not promote the documented experimental, candidate, or
non-production surfaces until their evidence lands.

| Area | Status | Gate |
|---|---|---|
| CI | Required at publish | `main` green, including the scripts/bench smoke gates |
| Multi-model | Ongoing claim gate | A CI or QA gate loads two real MLX models via `POST /v1/model/load` and serves both concurrently (today all co-residency tests use mock or delegated backends) |
| Multi-model | Ongoing claim gate | Unload/idle eviction either frees parked weights (TTL / cap / sweeper) or the soft-park contract is promoted to a documented guarantee with memory-preflight awareness |
| Multi-model | Ongoing claim gate | Soak/SLO matrix from `designs/multimodel-execution-priorities-2026-07-23.md` "Next phases": p95/p99 stream gaps, load/unload churn, memory-pressure behavior |
| Multimodal | Ongoing claim gate | Numeric parity fixtures and at least one published benchmark for a P0 VLM family (Qwen3-VL or Qwen 3.6), beyond the existing Gemma 4 unified coverage |
| Multimodal | Ongoing claim gate | Multimodal prefix reuse (`AX_MLX_MULTIMODAL_PREFIX_REUSE`) promoted via fixtures or descoped; serving docs match actual wiring |
| AXQuant | Ongoing claim gate | Per-layer KV-cache quantization engages on the batched serving path (Gate 2 writeback wired into the runner) or is documented as single-sequence-only |
| AXQuant | Ongoing claim gate | KV-cache quantization has runtime telemetry and a generation-quality gate on a real artifact (current evidence is synthetic-tensor error bounds only) |
| AXQuant | Ongoing claim gate | AXQuant metadata integrity is verified at model load, or doctor-only verification is recorded as the explicit contract; MTP sidecar gains provenance parity with the vision sidecar |
| AXQuant | Ongoing claim gate | The pinned Qwen 3.6 27B AXQ 6-bit flagship candidate clears the [checkpoint certification record](model-certifications/qwen3.6-27b-axq.md); architecture-prior evidence alone cannot promote it to a default |
| Performance | Ongoing claim gate | The cross-family MTP prefill regression and the steady-state eval-wall drift are resolved or explicitly accepted in `PERFORMANCE-RESULTS.md` |
| Positioning | Ongoing claim gate | The `docs/SERVER.md` "not yet a production server surface" caveat can be removed honestly |

## Evidence Gates

Roadmap items become public support claims only after the matching evidence is
checked in and labeled with the right runtime path.

| Claim area | Expected evidence |
|---|---|
| Repo-owned MLX throughput | MLX inference-stack artifacts with matching `mlx_lm.benchmark` rows, prompt-token provenance, AX decode-policy labels, and route identity |
| Long-context behavior | Separate long-context artifacts for cold prefill, decode at depth, startup, concurrency, and prefix reuse |
| Serving behavior | `ax.serving_benchmark.v1` artifacts with TTFT, TPOT, E2E latency, queue delay, throughput, category summaries, and SLO goodput |
| Delegated compatibility | Explicit `mlx_lm_delegated` or `llama_cpp` route-contract artifacts, not AX-owned MLX throughput claims |

See [`BENCHMARKS.md`](BENCHMARKS.md),
[`PERFORMANCE-RESULTS.md`](PERFORMANCE-RESULTS.md),
[`PERFORMANCE.md`](PERFORMANCE.md), and
[`SERVING-BENCHMARKS.md`](SERVING-BENCHMARKS.md) for the benchmark contracts.
