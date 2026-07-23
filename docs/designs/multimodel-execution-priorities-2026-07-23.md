# Multi-model execution priorities

Date: 2026-07-23
Status: implemented and profiled
Validation host: `AKMBPM5MAXx` (Apple GPU `applegpu_g17s`, 128 GB unified memory)

## Decision

The feedback was directionally correct: AX should compete on one-process
Qwen/Gemma serving, not on model-count breadth, and it should establish
repeatable workload evidence before rewriting kernels.

The proposed sequence needed four corrections:

1. A new monolithic global scheduler was not the first requirement. The
   existing per-model owner-thread contract remains valuable. The immediate
   correctness gap was global lifecycle/admission, and the immediate latency
   gap was the duration of each process-wide Metal turn.
2. A universal paged-KV rewrite would combine incompatible rollback contracts.
   Qwen recurrent state, Gemma sliding windows, full-attention KV, and shared
   layers need different physical policies.
3. Tensor-batched decode could not be called production-ready merely because
   it was faster. Numerical certification rejected representative Qwen and
   Gemma tensor batches. The safe production step is row-exact coalescing:
   independent batch-1 graphs and caches, submitted under one MLX completion
   barrier. Both rejection artifacts were regenerated on the required M5 Max
   host:
   [Qwen tensor batch](../../benchmarks/results/profiling/batched-decode/2026-07-23-qwen35-9b-tensor-batch-m5max.json),
   [Gemma tensor batch](../../benchmarks/results/profiling/batched-decode/2026-07-23-gemma4-12b-tensor-batch-m5max.json).
4. MTP batching is useful only where route engagement, output identity, and
   end-to-end serving performance all pass. Generic “MTP supported” telemetry
   is not evidence that the requested route ran.

The resulting priority order is:

| Phase | Priority | Completion criterion |
|---|---|---|
| P0 | Model-scoped lifecycle and admission | A target model can drain, unload, load, or replace without closing sibling request admission |
| P1 | Long-prefill sibling latency isolation | No second-scale streaming gap while a sibling performs an 8K prefill |
| P2 | Qwen/Gemma production decode coalescing | Exact across the full certification matrix, route-proven, and faster than per-item sequential decode |
| P3 | Trustworthy memory accounting and hybrid KV | Physical bytes are not double-counted; topology and rollback are explicit; experimental pools remain gated by evidence |
| P4 | Profile-proven kernels and MTP batching | Exact output identity, explicit route contract, repeated hardware A/B, and a positive user-visible result |

## P0 — model-scoped lifecycle and admission

Each published `LiveState` now owns an admission controller in addition to the
process-wide controller. Admission acquires the model lease first and the
global lease second, then validates the model generation. A draining or stale
model therefore cannot consume process capacity.

Lifecycle behavior is target-scoped:

- `load_mode=add` builds and warms a detached generation, then publishes it;
- availability-first replacement keeps the outgoing generation available
  during the build and drains only that model immediately before publication;
- unload stops new admission only for the selected model and waits only for
  that model's work;
- failure before publication leaves every resident model untouched;
- the registry publication/swap remains a short write-locked operation;
- the process still serializes control-plane load/unload mutations, but this
  mutex no longer closes data-plane admission.

The process-wide execution arbiter now records turns, wait time, and hold time
by model and by `engine_step`/`bulk_command` work class.

M5 Max lifecycle replay:

- Qwen streamed all 192 tokens with HTTP 200;
- sibling Gemma unload completed in 1.70 ms;
- sibling Gemma add/load completed in 447.45 ms;
- both lifecycle operations returned HTTP 200;
- the Qwen output identity matched the non-lifecycle workload.

Evidence:
[lifecycle isolation](../../benchmarks/results/profiling/multimodel/2026-07-23-lifecycle-isolation-m5max.json).

## P1 — long-prefill sibling decode isolation

Multi-model publication automatically enables fair prefill for every resident
session unless `AX_SERVER_MULTI_MODEL_PREFILL_ISOLATION=0`.

The execution service changes the scheduler quantum online:

- sibling active, waiting, or active within the 250 ms grace window:
  one prefill token per model turn;
- no recent sibling activity: 256 prefill tokens per model turn.

This preserves the per-model scheduler and cache ownership while bounding the
time for which one model can hold the shared device. Prefill stays ordered;
portable prefix serialization is deferred until the final scheduler-split
chunk, avoiding quadratic snapshot work.

M5 Max replay (`Qwen3.5-9B` streaming 192 tokens while Gemma 4 performs an 8K
prefill):

| Metric | Isolation disabled | Adaptive isolation | Result |
|---|---:|---:|---:|
| Qwen stream-gap max | 1,570.11 ms | 30.03 ms | -98.1% |
| Qwen stream-gap p99 | 1,104.49 ms | 28.81 ms | -97.4% |
| Qwen end-to-end | 6,854.32 ms | 5,429.20 ms | -20.8% |
| Gemma prefill end-to-end | 4,899.97 ms | 12,405.95 ms | +153.2% |
| Aggregate output throughput | 28.15 tok/s | 15.24 tok/s | -45.9% |

This is an explicit tail-latency policy, not a throughput win. Quantum sweeps
of 4, 8, and 16 increased sibling latency; the 4-token run also selected a
different Gemma greedy token at a near tie. They are retained only as negative
profiling evidence. The production value stays fixed at one token.

The disabled and adaptive runs produced identical request output hashes.
Evidence:
[disabled](../../benchmarks/results/profiling/multimodel/2026-07-23-prefill-isolation-disabled-m5max.json),
[adaptive](../../benchmarks/results/profiling/multimodel/2026-07-23-prefill-isolation-adaptive-m5max.json).

## P2 — production Qwen/Gemma decode coalescing

The runtime has two distinct routes:

1. certified tensor batching, admitted only by artifact/device/runtime evidence;
2. production row-exact coalescing for otherwise eligible direct Qwen/Gemma
   cohorts.

Row-exact coalescing retains one graph, reduction, sampler, and KV cache per
request. It groups the independent MLX arrays into one evaluation barrier.
Sampled requests, MTP, incompatible cache layouts, and unsupported structural
capabilities fail closed to the established per-item route.

The certification matrix covers:

- batch 2 and 4;
- prompt lengths 32, 128, 512, and 992;
- two prompt seeds;
- ragged lengths;
- Gemma's sliding-window compaction boundary;
- an independent `model::forward` oracle and a second per-item runner.

The `model::forward` oracle now mirrors the production physical sliding-ring
topology after prefill. It remains an independent forward path, but no longer
compares ordered and slot-ordered K/V at a numerically near-tied argmax.

M5 Max results:

| Family | Verdict | Row-exact over sequential |
|---|---|---:|
| Qwen 3.5 9B 4-bit | pass, 7/7 scenarios | 1.12–1.33x |
| Gemma 4 12B 4-bit | pass, 7/7 scenarios | 1.06–1.19x |

Evidence:
[Qwen](../../benchmarks/results/profiling/batched-decode/2026-07-23-qwen35-9b-row-exact-m5max.json),
[Gemma](../../benchmarks/results/profiling/batched-decode/2026-07-23-gemma4-12b-row-exact-m5max.json).

## P3 — memory truth and hybrid KV

`/metrics` now separates measurements from attribution:

- process-wide MLX active, cache, and peak bytes;
- Metal recommended working-set bytes;
- current host RSS;
- exact on-disk weight-artifact bytes per model;
- logical KV bytes;
- contiguous KV capacity;
- recurrent/linear state bytes;
- physical paged-pool slab bytes;
- prefix-cache payload bytes;
- a non-double-counted physical-KV attribution;
- attributed lower bound, unattributed active bytes, excess, and coverage.

Per-model topology is emitted as typed Prometheus labels:

- attention: `contiguous` or `paged_pool`;
- sliding: `none`, `ordered`, or `rotating_ring`;
- recurrent state: `none` or `present`;
- rollback: `o1_trim`, `bounded_cursor_restore`, or `restore_replay`.

Correctness hardening also prevents a rotated Gemma ring from becoming a
portable prefix snapshot, requires an exact physical prefix length when
seeding attention KV, and requires exact state for Qwen recurrent layers.

The physical full-attention block pool remains default-off. Both M5 matrices
were exact, but normal-vs-pool decode results do not justify promotion:

- Qwen: the pool was slower in every shape by 1.6–3.0%;
- Gemma: -0.6% to +3.2%, mixed and too small for a stable claim.

Evidence:
[Qwen pool](../../benchmarks/results/profiling/batched-decode/2026-07-23-qwen35-9b-row-exact-hybrid-kv-m5max.json),
[Gemma pool](../../benchmarks/results/profiling/batched-decode/2026-07-23-gemma4-12b-row-exact-hybrid-kv-m5max.json).

## P4 — profile-proven MTP batching

Disabling n-gram acceleration previously selected the direct bootstrap/session
pipeline even when MTP was requested. The routing predicates now distinguish
“disable n-gram” from “disable MTP”; explicit unsupported-sampling fallback
still behaves as before.

Qwen MTP batching is not promoted in this phase because it has no repeated,
route-proven M5 result that meets the same output-identity and user-visible
performance contract.

Gemma assistant-MTP has a narrow production coalescing route:

- assistant attached and enabled;
- no target Qwen/GLM MTP head;
- MTP explicitly requested;
- assistant-only pending draft;
- deterministic greedy sampling;
- no logits processors, n-gram stacking, skip-state, or adaptive gate;
- at least two aligned rows.

Every request retains its own target graph, cache transaction, exact greedy
acceptance, rollback, and assistant draft. Only the target verification
completion barrier is shared.

Four cold-start, prefix-cache-disabled M5 runs were interleaved OFF/ON:

| Mode | Aggregate throughput | Mean client TPOT |
|---|---:|---:|
| OFF, mean of two | 78.03 tok/s | 23.41 ms |
| ON, mean of two | 81.36 tok/s | 22.37 ms |

Result: +4.3% aggregate throughput and -4.4% mean TPOT. Route contracts proved
both MTP and coalesced verification were used. Both 256-token request outputs
were hash-identical across all four runs. The feature defaults on and retains
`AX_MLX_GEMMA4_ASSISTANT_MTP_COALESCED_VERIFY=0` as a kill switch.

Evidence:
[OFF r1](../../benchmarks/results/profiling/mtp-batching/2026-07-23-gemma4-12b-assistant-mtp-b2-controlled-off-r1-m5max.json),
[OFF r2](../../benchmarks/results/profiling/mtp-batching/2026-07-23-gemma4-12b-assistant-mtp-b2-controlled-off-r2-m5max.json),
[ON r1](../../benchmarks/results/profiling/mtp-batching/2026-07-23-gemma4-12b-assistant-mtp-b2-controlled-on-r1-m5max.json),
[ON r2](../../benchmarks/results/profiling/mtp-batching/2026-07-23-gemma4-12b-assistant-mtp-b2-controlled-on-r2-m5max.json).

## Benchmark contract

`scripts/bench_ax_multimodel_serving.py` provides timed JSONL replay for
request/load/unload events and records:

- aggregate and per-model throughput;
- TTFT, client TPOT, stream-step intervals, and end-to-end latency;
- lifecycle latency and status;
- final route counters;
- SHA-256 identities of streamed token IDs;
- required route-counter contracts that fail the command after preserving the
  artifact.

The manifests under `benchmarks/manifests/replay/` are the fixed regression
workloads. mlxcel multi-process comparison remains a separate competitive
benchmark: these artifacts establish AX internal correctness and performance,
not a cross-project leaderboard claim.

## Next phases

The next work should remain evidence-gated:

1. Run longer concurrency/soak matrices and set product SLOs for p95/p99 stream
   gaps, load/unload, and memory-pressure behavior.
2. Convert model-attributed memory metrics into a global residency governor
   only after graph/temp/cache headroom is characterized on supported Macs.
3. Improve same-model batch formation so Gemma assistant-MTP rows remain
   aligned beyond the first few verify barriers.
4. Profile MoE routing/grouped GEMM and sampling/output-head costs; change
   kernels only where a captured profile identifies a dominant bottleneck.
5. Keep the physical KV pool opt-in until it demonstrates either a capacity
   win at equal latency or a repeatable speed win.

The intended product claim is narrow:

> AX Engine keeps Qwen and Gemma resident in one Mac process, isolates
> interactive decode from sibling lifecycle and long-prefill work, and only
> promotes batching paths that are exact and route-proven.
