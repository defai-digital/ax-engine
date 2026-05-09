# Scheduler

The scheduler decides which requests to run each step and how many tokens to
process. It is a pure function: `Scheduler.plan()` takes `&self` and returns
a new `SchedulePlan` without mutating any state. This makes scheduling
decisions deterministic and easy to test.

## Data Types

```
SchedulerInput
├── step_id          StepId
├── request_snapshots  Vec<RequestSnapshot>  (Runnable requests only)
├── memory_pressure    Option<String>        (low/reclaimable → throttle, exhausted → defer prefill)
└── global_token_budget  u32

SchedulePlan
├── step_id
├── selected_requests    Vec<RequestId>   (will run this step)
├── deferred_requests    Vec<RequestId>   (skipped this step)
├── memory_blocked_requests  Vec<RequestId>   (set by engine after KV failure)
└── execution_batch      Option<ExecutionBatch>
```

The scheduler only sees `RequestState::Runnable` snapshots. Requests in
`Waiting`, `Running`, `BlockedOnMemory`, `Finished`, `Cancelled`, and `Failed`
states are filtered out before `plan()` is called.

## Selection Algorithm

```
1. Filter: keep only Runnable snapshots
2. Filter: keep only snapshots whose model_id matches the first request
           (multi-model batches are not supported; extras are deferred)
3. Classify each request as Prefill or Decode
4. Sort by (execution_mode_priority, arrival_sequence, request_id)
           Decode = priority 0, Prefill = priority 1
           → decode-first ordering within the sorted list
5. Walk sorted list, for each candidate:
   a. Check token budget (remaining_budget > 0)
   b. If memory_pressure is low/reclaimable and mode is Prefill,
      limit to MEMORY_PRESSURE_MAX_PREFILL_TOKENS_PER_STEP (= 1)
      If memory_pressure is kv_exhausted and mode is Prefill,
      defer the prefill request
   c. Check batch route compatibility (route_seed_can_join_batch)
   d. If all checks pass: add to selected, deduct tokens from budget
      Otherwise: add to deferred
```

### Decode-first priority

All active decode requests are sorted ahead of all prefill requests. This
minimises time-to-completion for requests already generating tokens. A
prefill request only gets budget after every decode request is scheduled.

### Chunked prefill

When a prefill request's remaining prompt tokens exceed `remaining_budget`,
the scheduler schedules only `remaining_budget` tokens and marks the rest as
the next step's prefill work. The request stays `Runnable` and continues
prefill in subsequent steps.

### Memory pressure throttle

When `memory_pressure` is `Some("kv_low_free_blocks:<free>/<total>")`, prefill
slots are limited to one token per step
(`MEMORY_PRESSURE_MAX_PREFILL_TOKENS_PER_STEP = 1`). This slows prompt ingestion
to reduce new KV block demand while existing decode requests drain.

When `memory_pressure` is `Some("kv_exhausted_reclaimable_cache")`, prefill is
also limited to one token per step. Although no block is currently free, the
engine may be able to satisfy the allocation by evicting retained prefix cache,
so the scheduler gives the KV manager a bounded allocation opportunity.

When `memory_pressure` is `Some("kv_exhausted")`, prefill requests are deferred
for the step. Decode requests are still eligible, because they may be able to
reuse existing partial block capacity or finish and free memory. Concrete KV
allocation remains the final authority; if decode cannot allocate, the engine
marks that request `BlockedOnMemory` and retries without it.

---

## Batch Routing and Route Compatibility

Each request carries a `RouteMetadata` that describes how the model runner
should execute it (execution plan name, KV mode, attention route, prefix cache
path, barrier mode). The scheduler enforces that all requests in one
`ExecutionBatch` are compatible.

### Same-mode batches

All-prefill or all-decode batches require full `RouteMetadata` equality plus
matching `execution_plan_ref`. Any mismatch defers the candidate.

### Mixed prefill+decode batches

When a batch already contains prefill requests and a decode candidate (or vice
versa), only a subset of fields must match:

```rust
fn mixed_route_metadata_compatible(anchor: &RouteMetadata, candidate: &RouteMetadata) -> bool {
    anchor.kv_mode == candidate.kv_mode
        && anchor.barrier_mode == candidate.barrier_mode
        && anchor.prefix_cache_path == candidate.prefix_cache_path
}
```

The execution plan is overridden to `"phase2.token_budget"` and the attention
route to `"mixed_prefill_decode"` for the combined batch.

### RouteMetadata source

`route_seed()` builds a request's `RouteMetadata` from its
`route_metadata_hint` (if non-empty) or falls back to just the
`execution_plan_ref`. The engine populates `route_metadata_hint` via
`DeterministicExecutionPlanResolver`, which maps model slug + phase to a
canonical execution plan name (e.g. `phase1.qwen3_5_9b.paged_decode`).

---

## TokenBudgetTelemetry

The scheduler accumulates per-step token counters and appends them to the
batch's `RouteMetadata.crossover_decisions`:

| Key | Meaning |
|---|---|
| `ax_scheduler_scheduled_prefill_tokens` | Prefill tokens dispatched this step |
| `ax_scheduler_scheduled_decode_tokens` | Decode tokens dispatched this step |
| `ax_scheduler_skipped_prefill_tokens` | Prefill tokens deferred due to budget |
| `ax_scheduler_skipped_decode_tokens` | Decode tokens deferred due to budget |
| `ax_scheduler_mixed_prefill_decode_batches` | 1 if this step mixed modes, else 0 |

These values appear in runner output and benchmark replay records.

---

## Three-Phase Scheduling Per Step

`engine.rs` may call `scheduler.plan()` up to three times per step:

```
Phase 1 — Initial plan
  scheduler.plan(all Runnable requests)
  → SchedulePlan

Phase 2 — Prefix reuse re-plan (if any prefix hits found)
  apply_prefix_reuse() discovers shared prefixes from KvManager
  → triggers full re-plan with updated request snapshots
  scheduler.plan(all Runnable requests, updated snapshots)
  → SchedulePlan

Phase 3 — KV fallback re-plan (if KV allocation fails)
  resolve_kv_schedule_plan() fails to allocate blocks for some requests
  → marks them BlockedOnMemory
  → if all candidates blocked: scheduler.plan(excluding blocked requests)
  → SchedulePlan without the memory-blocked set
```

Phase 3 only fires when `InsufficientCapacity` is returned during KV block
allocation and the engine cannot proceed with any of the initially selected
requests. In practice, Phase 3 is rare under normal working-set sizes.

---

## Request State Machine

Requests move through states managed by `RequestManager` in
`crates/ax-engine-core/src/request_manager.rs`.

```
            admit()
Waiting ──────────────→ Runnable
                            │
              scheduler.plan() selects
                            │
                            ↓
                         Running
                         /     \
               step OK  /       \  KV alloc fails
                        ↓       ↓
                     Runnable  BlockedOnMemory
                        │             │
              max_tokens │    memory   │ freed
              reached    ↓    freed    ↓
                      Finished     Runnable
                        │
              cancel_requested
                        ↓
                     Cancelled
                        │
              runner error
                        ↓
                      Failed
```

Key rules enforced by `transition_to()`:
- Only `Runnable → Running` is valid for the scheduler to select a request.
- `BlockedOnMemory` can only transition back to `Runnable`.
- Terminal states (`Finished`, `Cancelled`, `Failed`) have no outgoing
  transitions.
- `cancel_requested` is a flag, not an immediate transition — it is applied
  when the request exits `Running` at the end of the step.

### Prefill completion and sampling

A prefill request that completes its prompt in a given step becomes eligible
for token sampling only if it has **no generated tokens yet** and the step
**fully completes the prompt**. `prefill_completion_request_ids()` in
`engine.rs` enforces this. Partial-prefill steps (chunked prefill) do not
sample.

---

## Key Invariants

- `Scheduler.plan()` is `&self` — calling it never changes scheduler state.
  The engine can safely call it multiple times per step.
- All Runnable requests for the same model are candidates each step; none are
  silently skipped unless budget, routing, or memory pressure excludes them.
- The `execution_batch` in a `SchedulePlan` is `None` when no requests were
  selected, not an empty-item batch. Callers check `Option<ExecutionBatch>`.
- `deferred_requests` in the plan are still `Runnable`; the engine does not
  change their state. They will be candidates again next step.
- `memory_blocked_requests` in the plan are populated by `engine.rs` after
  KV allocation fails, not by the scheduler itself.
