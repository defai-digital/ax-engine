# ADR 0018: KV Cache and Scheduler Revamp Strategy

Status: Accepted
Date: 2026-05-09
Deciders: AX Engine

## Context

AX Engine v4 now routes user-facing inference to the repo-owned MLX runtime or
delegated compatibility backends. The next practical performance and reliability
question is whether the KV cache and scheduler should be rewritten, deeply
revamped, or improved through bounded slices.

The current system has three separate but related layers:

- `KvManager` in `ax-engine-core` owns logical block tables, refcounts, retained
  prompt-prefix metadata, and memory-pressure reporting.
- `Scheduler` remains a pure planning function over request snapshots,
  execution-plan hints, token budget, and current memory pressure.
- `MlxRunner` owns physical MLX KV state, including an MLX prefix snapshot cache
  that can restore `MlxKVCache` snapshots by reference-counted `MlxArray` clones.

The earlier concern that `SchedulerInput.memory_pressure` was ignored is now
stale. `Scheduler::plan()` already reads memory pressure: low pressure and
reclaimable-cache pressure cap prefill, while true `kv_exhausted` defers
prefill. Final allocation failure is still resolved by the engine's
post-schedule KV fallback.

The larger unresolved gap is the split between logical prefix reuse and physical
MLX prefix restore. `KvManager` can identify a prefix match and the engine can
advance request state before runner work. The MLX runner then independently
decides whether a physical prefix snapshot is available. On physical cache miss
or unsupported layer layout, the runner warms the reused prefix by replaying
prefill. This preserves correctness, but it can make performance and memory
behavior hard to predict.

Paged-attention-style rewrites are not currently practical. AX does not own a
scattered SDPA Metal kernel for MLX mode, and the active optimization policy
requires checked-in evidence before kernel, eval-boundary, or memory-layout
changes. The existing chunked KV buffers, O(1) rollback for supported paths,
route metadata, and prefix snapshot cache are useful foundations that should be
preserved.

Recent project changes strengthen the same conclusion:

- `KvManager` retained-prefix entries now keep block-token payloads and verify
  token equality on cache lookup/promotion, so hash collisions cannot silently
  produce a false prefix hit.
- Scheduler/request-manager invariants now reject schedule plans that omit a
  runnable request, and runner-output validation uses request-indexed lookups for
  stricter contract checks.
- The 2026-05-09 thermal-fix Gemma benchmark artifacts are useful MLX runtime
  performance evidence, but they are single-request direct/ngram rows with zero
  prefix reuse evidence. They do not satisfy the long-context/prefix evidence
  gate in this ADR.
- The recent sampling fast path is a decode CPU optimization outside the
  KV/scheduler ownership boundary. It does not change this ADR's decision.

## Decision

Do not rewrite the KV cache or scheduler.

AX Engine will pursue a bounded, evidence-gated revamp:

1. Keep `Scheduler` pure and small.
2. Improve scheduler memory-pressure policy only where the engine and KV manager
   can provide explicit pressure states.
3. Treat physical MLX prefix snapshot availability as runner/runtime state, not
   as a hidden scheduler responsibility.
4. Add long-context and prefix-reuse artifacts before changing cache ownership or
   eviction complexity.
5. Preserve current chunked KV growth and rollback semantics unless benchmark
   artifacts prove a replacement is faster and equally correct.
6. Keep TurboQuant and other compression work outside default runtime behavior
   until their separate promotion gates pass.

The immediate product direction is "KV revamp before scheduler revamp":

- Scheduler changes should be policy hardening and validation.
- KV changes should focus on runtime evidence, metadata, prefix coordination,
  and model-specific long-context behavior.
- Major memory-layout changes require explicit proof that the current
  MLX/kernel boundary can consume the new layout without hidden recomputation or
  broad synchronization cost.

## Architecture Rules

- The scheduler must not inspect or mutate MLX tensors.
- The scheduler must not depend on runner-local mutable caches.
- The engine may combine logical KV state, scheduler plans, and runner telemetry
  into route metadata and benchmark artifacts.
- Prefix reuse must report the path taken: logical hit, physical snapshot
  restored, physical snapshot missed and warmed, unsupported, or blocked.
- Prefix-cache keys may use hashes for lookup, but retained-prefix hits must
  remain content-validated before a block is treated as reusable.
- Memory-pressure policy must distinguish low capacity from exhausted capacity.
- User-visible preemption or early termination requires an explicit stop reason,
  route metadata, tests, and product review.
- Any O(log N) eviction index must preserve the current descendant-safe
  retained-prefix semantics.
- Experimental TurboQuant readbacks, profiling diagnostics, and production decode
  paths must remain separately labeled.

## Implementation Direction

### Phase 0: Safety Baseline

Status: Partially complete.

Completed:

- retained-prefix cache collision checks;
- schedule-plan completeness validation for runnable requests;
- runner-output validation hardening;
- low-memory and reclaimable-cache prefill throttling while preserving decode
  progress.
- exhausted-memory prefill deferral when no retained cache can be reclaimed.

Still open:

- physical MLX prefix restore/warmup must be measured separately from logical
  prefix reuse.

### Phase 1: Evidence and Metadata

Add or refresh long-context artifacts that report:

- TTFT and decode throughput;
- logical KV tokens and capacity;
- retained-prefix hits;
- MLX prefix snapshot hits, misses, stores, evictions, and warmup tokens;
- unsupported prefix-cache layouts;
- correctness shape and deterministic replay status where applicable.

### Phase 2: Scheduler Policy Hardening

Refine `memory_pressure` handling:

- `kv_low_free_blocks:*` throttles prefill while preserving decode progress.
- `kv_exhausted_reclaimable_cache` throttles prefill so allocation can trigger
  retained-cache eviction.
- `kv_exhausted` defers new prefill when no retained cache can be reclaimed.
- The engine's post-schedule KV fallback should remain as a safety net, not the
  primary policy.

### Phase 3: Prefix Coordination Contract

Make the split between logical and physical prefix reuse explicit in runtime
metadata and tests. The runner may continue warming reused prefixes on physical
cache miss, but artifacts must show the cost and frequency.

The first metadata slice is implemented: MLX route metadata now separates
physical snapshot hits, eligible misses, warmup tokens, policy-disabled blocks,
unsupported-layout blocks, and trim-failure blocks while preserving the existing
aggregate blocked counter. The MLX inference-stack benchmark summary also
propagates those reason-specific counters into `prefix_reuse_evidence`, so
artifact readers can distinguish eligible misses from unsupported or disabled
physical snapshot paths. That summary now carries an explicit
`physical_snapshot_coverage` label to prevent hit-only or zero-prefix artifacts
from being read as complete physical prefix-cache evidence. Public performance
claim validation now requires physical snapshot hit evidence before an artifact
may claim `prefix_reuse`; miss-warmup-only and blocked-only artifacts remain
diagnostic evidence.

Only after evidence shows material warmup cost should AX consider an engine-runner
coordination API for preflight physical-prefix availability.

### Phase 4: Targeted KV Data-Structure Improvements

Optimize retained-prefix eviction only when artifacts show meaningful CPU cost or
capacity churn. Candidate changes include maintaining an eviction index or cached
descendant counts, but the first implementation must be smaller than a new cache
subsystem.

### Phase 5: Long-Context Capacity Work

Evaluate physical trimming for sliding-window attention layers and conservative
compression only under the existing MLX KV cache improvement and TurboQuant
promotion gates.

## Consequences

Positive:

- Avoids a risky rewrite across scheduler, KV ownership, and MLX tensor state.
- Keeps scheduler behavior testable and deterministic.
- Turns prefix reuse from a hidden performance assumption into an observable
  runtime contract.
- Aligns KV work with the evidence-first optimization policy.

Negative / risks:

- Physical prefix cache misses may still pay warmup cost until Phase 3 evidence
  justifies deeper coordination.
- The existing retained-prefix eviction path remains O(N) until there is evidence
  that it matters.
- Full paged-attention benefits remain out of scope without MLX/Metal kernel
  support.

## Alternatives Considered

### Rewrite the scheduler

Rejected. The scheduler's pure-function shape is already the right boundary.
The current gaps are policy precision and metadata, not scheduler architecture.

### Rewrite KV cache as vLLM-style paged attention

Rejected for current MLX mode. Logical paged metadata is useful, but physical
scattered attention would require kernel support that AX does not currently own.

### Merge `KvManager` and MLX prefix snapshot cache

Rejected as a first step. Logical ownership and physical tensor lifetime have
different responsibilities. The safer contract is explicit telemetry and
coordination, not shared mutable ownership.

### Add full preemption and swap

Deferred. It is more relevant to multi-user server workloads than the current
single-device MLX runtime target. Any user-visible preemption requires a product
contract first.

## Related Documents

- ADR 0003: Paged KV and Prefix Caching
- ADR 0013: MLX KV Cache Improvement Strategy
- ADR 0016: Experimental TurboQuant KV Compression for MLX Runtime
- ADR 0017: MLX Runtime Optimization Governance
- `../planning/KV-SCHEDULER-REVAMP-PRD.md`
- `../planning/MLX-RUNTIME-PERFORMANCE-PRD.md`
