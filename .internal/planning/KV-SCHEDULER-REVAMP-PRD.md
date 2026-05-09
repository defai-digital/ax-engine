# KV Cache and Scheduler Revamp PRD

Status: Active
Date: 2026-05-09
Owner: AX Engine
ADR: ../adr/0018-kv-cache-and-scheduler-revamp-strategy.md

## 1. Summary

This PRD defines the execution plan for improving AX Engine's KV cache and
scheduler behavior without a rewrite.

The near-term goal is to make long-context and prefix-reuse behavior measurable,
predictable, and safe. The work should improve memory-pressure behavior and
prefix-cache observability while preserving the current scheduler boundary,
chunked MLX KV cache, and correctness-first fallback paths.

## 2. Current State

AX Engine currently has:

- logical KV ownership in `ax-engine-core::KvManager`;
- retained-prefix cache collision checks that verify block-token content before
  a cached hash match is treated as reusable;
- a pure `Scheduler::plan()` function that already reads memory pressure, caps
  prefill work under low-KV or reclaimable-cache pressure, and defers prefill
  under true KV exhaustion;
- request-manager validation that rejects schedule plans which omit runnable
  requests;
- post-schedule KV allocation fallback in `EngineCore`;
- an MLX runner prefix snapshot cache that restores physical KV snapshots when
  possible and warms reused prefixes when physical snapshots are missing;
- KV, prefix-cache, TurboQuant, and decode telemetry emitted through route
  metadata.

The main product gap is not that the cache or scheduler is missing. The gap is
that runtime artifacts do not yet prove where prefix reuse, memory pressure, and
long-context capacity help or fail.

Recent thermal-fix benchmark artifacts under
`benchmarks/results/mlx-inference/2026-05-09-thermal-fix/` are useful MLX runtime
performance evidence, but they are single-request direct/ngram rows. Their
`prefix_reuse_evidence` is zero, so they do not close this PRD's
long-context/prefix-reuse evidence requirement.

## 3. Goals

G1. Produce checked-in long-context and prefix-reuse artifacts before structural
KV changes.

G2. Make logical prefix reuse and physical MLX prefix snapshot behavior separately
observable.

G3. Refine memory-pressure scheduling so low-KV and exhausted-KV states have
different policy outcomes.

G4. Preserve decode progress and deterministic request-state transitions under
memory pressure.

G5. Keep scheduler changes small, pure, and unit-testable.

G6. Keep KV data-structure changes evidence-gated and behavior-preserving.

## 4. Non-Goals

- Do not rewrite scheduler architecture.
- Do not rewrite MLX KV cache around vLLM-style paged attention.
- Do not add a full swap/preemption system in this PRD.
- Do not make TurboQuant a default runtime feature.
- Do not introduce custom Metal kernels as part of this workstream.
- Do not merge logical `KvManager` ownership with runner-local MLX tensor
  lifetime.

## 5. User and Product Impact

This work supports:

- longer agent and coding sessions with clearer memory behavior;
- repeated-prompt workloads where prefix reuse should improve TTFT;
- predictable behavior when KV capacity is low;
- benchmark-defensible MLX runtime claims.

End users should not see new modes or confusing switches. Improvements should
surface through better defaults, route metadata, and benchmark artifacts.

## 6. Workstreams

### W0. Safety Baseline

**Status**: Complete for scheduler policy.

#### Done

- Retained-prefix cache entries validate block-token content to guard against
  hash-collision false hits.
- Cache promotion stops after a parent hash collision instead of refreshing
  descendant entries.
- Schedule-plan validation rejects omitted runnable requests.
- Runner-output validation uses request-indexed lookups and rejects duplicate or
  missing decode result sources.
- Low-KV and reclaimable-cache memory pressure cap prefill without starving
  decode.
- Exhausted-KV memory pressure defers prefill when no retained cache can be
  reclaimed.

#### Open

- Physical MLX prefix restore/warmup still needs artifact coverage separate from
  logical prefix reuse.

#### Exit Criteria

- The completed safety behavior remains covered by targeted unit tests.
- Open policy/evidence gaps are tracked by W1-W3 instead of being mixed into a
  rewrite-sized design.

### W1. Long-Context and Prefix-Reuse Evidence

**Status**: Not started.

#### Requirements

- Add or refresh replay/scenario coverage for:
  - retained prompt-prefix cache reuse;
  - MLX physical prefix snapshot hit;
  - MLX physical prefix snapshot miss with warmup;
  - prefix-cache unsupported layout;
  - KV-low and KV-exhausted scheduling paths.
- Produce artifacts that include:
  - TTFT;
  - decode tok/s;
  - prefill tok/s where applicable;
  - logical KV tokens and capacity;
  - MLX prefix hits, misses, warmup tokens, stores, evictions, entries, and bytes;
  - correctness shape and deterministic replay status.

#### Exit Criteria

- A checked-in artifact identifies the most important KV/prefix bottleneck.
- Unsupported or fallback paths are clearly labeled.
- No structural cache change is started before this evidence exists.
- Single-request throughput artifacts with zero prefix reuse do not count as W1
  completion.

### W2. Scheduler Memory-Pressure Policy

**Status**: Complete for scheduler policy.

#### Requirements

- Preserve the existing `kv_low_free_blocks:*` behavior: throttled prefill while
  decode remains schedulable.
- Treat `kv_exhausted_reclaimable_cache` as throttled prefill so allocation can
  trigger retained-cache eviction.
- Treat `kv_exhausted` as a stronger state that avoids scheduling new prefill
  when no retained cache can be reclaimed.
- Preserve decode progress when decode can reuse existing partial capacity.
- Keep post-schedule KV allocation fallback as a safety net.
- Add unit tests covering:
  - low pressure schedules decode plus capped prefill;
  - exhausted pressure schedules decode but defers prefill;
  - no runnable request is lost from selected/deferred/blocked outputs.

#### Done

- Low-pressure prefill capping is implemented and tested.
- Reclaimable-cache pressure prefill capping is implemented and tested.
- Exhausted-pressure prefill deferral is implemented and tested.
- Request-manager validation rejects missing runnable requests.

#### Exit Criteria

- `cargo test -p ax-engine-core scheduler --quiet` passes.
- `cargo test -p ax-engine-core request_manager --quiet` passes when schedule
  validation changes.
- Route metadata still records scheduled and skipped prefill/decode tokens.
- Existing request-state validation remains unchanged.

### W3. Prefix Coordination Metadata

**Status**: Partially complete.

#### Requirements

- Ensure route metadata distinguishes:
  - logical live share;
  - retained logical prefix hit;
  - MLX physical prefix snapshot hit;
  - MLX physical prefix snapshot miss;
  - MLX warmup tokens;
  - unsupported prefix snapshot cache layout;
  - policy-disabled prefix snapshot cache;
  - trim failures while storing a prefix snapshot.
- Add a replay or targeted test proving that a physical snapshot miss remains
  correct and observable.
- Do not make scheduler depend on runner-local mutable cache state.

#### Done

- MLX runner route metadata reports physical prefix snapshot hits, misses,
  warmup tokens, stores, evictions, entries, and bytes.
- Blocked physical snapshot paths now include reason-specific counters for
  disabled policy, unsupported layout, and trim failure while preserving the
  aggregate blocked counter.
- MLX inference-stack `prefix_reuse_evidence` now summarizes physical prefix
  hits, misses, warmup tokens, cache footprint, and blocked reason breakdowns
  for AX rows.
- MLX inference-stack `prefix_reuse_evidence` now includes explicit
  `physical_snapshot_coverage` classification so zero-prefix, hit-only,
  miss-warmup, blocked-only, and hit-plus-miss-warmup artifacts are not
  conflated.
- README/public performance claim validation now rejects `prefix_reuse` claims
  unless the artifact has physical snapshot hit evidence; miss-warmup-only and
  blocked-only artifacts remain diagnostics, not proof of physical reuse.
- README/public performance claim validation now recalculates prefix snapshot
  coverage from raw counters and rejects inconsistent flags, coverage labels,
  blocked-reason totals, and negative counters.
- README/public performance claim validation now rejects unknown
  `public_claims`; new claim names must add an explicit evidence mapping and
  checker coverage first.
- README/public performance claim validation now rejects `continuous_batching`
  claims unless overlap classification evidence is positive and internally
  consistent.
- `ax.mlx_prefix_warmup.v1` artifact validation now defines the contract for
  physical prefix snapshot miss/warmup correctness evidence.
- `build_mlx_prefix_warmup_artifact.py` now converts `ax-engine-bench` result
  directories into checked `ax.mlx_prefix_warmup.v1` artifacts.
- Prefix warmup artifact building now rejects non-replay manifests,
  non-MLX manifests, prefix-cache-disabled manifests, nondeterministic manifests,
  and manifests that do not require prefix reuse.
- Prefix warmup artifact building now also requires passing replay status,
  churn status, correctness, and determinism gates from `metrics.json`.

#### Open

- Produce a real checked-in replay/artifact that exercises a physical snapshot
  miss with warmup and passes `build_mlx_prefix_warmup_artifact.py` plus
  `check_mlx_prefix_warmup_artifact.py`.

#### Exit Criteria

- A benchmark or replay artifact can explain whether prefix reuse saved GPU work
  or only moved work into runner warmup.
- Prefix fallback correctness is covered by tests.

### W4. Retained Prefix Eviction Efficiency

**Status**: Evidence-gated.

#### Requirements

- Profile or instrument retained-prefix eviction under long-context churn before
  changing data structures.
- If the O(N) eviction candidate scan is material, implement the smallest safe
  index that preserves descendant-safe eviction.
- Candidate implementation:
  - maintain an LRU index keyed by touch tick;
  - maintain descendant counts or parent-child bookkeeping;
  - keep rollback and retained-cache tests passing.

#### Exit Criteria

- Before/after artifact shows reduced CPU overhead or fewer fallback retries.
- Existing prefix-cache semantics remain unchanged.

### W5. User-Visible Memory Exhaustion Behavior

**Status**: Deferred.

#### Requirements

- Define product semantics before implementing early termination or preemption.
- Any early finish must include:
  - explicit stop reason;
  - route metadata;
  - tests;
  - documentation of user-visible behavior.

#### Exit Criteria

- Product decision exists before code changes.
- No silent request termination is introduced.

### W6. Sliding-Window and Compression Follow-Up

**Status**: Owned by adjacent plans.

#### Requirements

- Sliding-window physical trimming remains under ADR 0013 and
  `MLX-RUNTIME-PERFORMANCE-PRD.md`.
- TurboQuant and lower-bit compression remain under ADR 0016 and
  `TURBOQUANT-PROMOTION-PRD.md`.
- This PRD may consume their telemetry, but should not promote their runtime
  behavior.

## 7. Milestones

### Milestone 1: Evidence Baseline

- Add or refresh targeted replay/scenario artifacts.
- Confirm route metadata includes enough KV and prefix counters.
- Identify the first measured bottleneck.
- Explicitly exclude the current 2026-05-09 thermal-fix single-request artifacts
  from prefix-reuse completion criteria.

### Milestone 2: Scheduler Hardening

- Preserve completed low-pressure prefill throttling and schedule completeness
  validation.
- Preserve implemented `kv_exhausted`-specific scheduler policy and targeted
  tests.
- Verify engine request-state invariants remain intact.

### Milestone 3: Prefix Runtime Contract

- Add metadata and tests that separate logical prefix reuse from MLX physical
  restore/warmup.
- Decide whether a runner preflight API is justified by evidence.

### Milestone 4: Data-Structure Optimization, If Proven

- Implement retained-prefix eviction index only if W1/W4 evidence shows it
  matters.

## 8. Success Metrics

- Long-context artifacts include memory, TTFT, decode, prefix, and correctness
  fields.
- KV-low and KV-exhausted scheduling behavior differ in tests and metadata.
- Physical prefix miss warmup is visible and quantifiable.
- Retained-prefix hash matches remain content-validated.
- No throughput claim is made without artifact provenance.
- No rewrite-sized diff is required to complete the first two milestones.

## 9. Verification Plan

Minimum verification by change type:

- Scheduler policy: `cargo test -p ax-engine-core scheduler --quiet`
- Schedule-plan validation: `cargo test -p ax-engine-core request_manager --quiet`
- KV manager changes: `cargo test -p ax-engine-core kv --quiet`
- Engine prefix-reuse changes: targeted `ax-engine-core` engine tests plus
  relevant replay artifact
- MLX runner metadata changes: targeted `ax-engine-mlx` tests plus route metadata
  artifact
- Cross-cutting changes: `cargo test --quiet`

## 10. Active Reading Path

1. `../adr/0018-kv-cache-and-scheduler-revamp-strategy.md`
2. this PRD
3. `../adr/0013-mlx-kv-cache-improvement-strategy.md`
4. `MLX-RUNTIME-PERFORMANCE-PRD.md`
5. `TURBOQUANT-PROMOTION-PRD.md` only for compression-specific work
