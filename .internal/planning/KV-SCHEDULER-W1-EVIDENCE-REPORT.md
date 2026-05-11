# W1 Long-Context and Prefix-Reuse Evidence Report

**PRD**: `.internal/planning/KV-SCHEDULER-REVAMP-PRD.md` §W1
**ADR**: 0018 KV/Scheduler Revamp
**Status**: Slice 3 complete (2026-05-11). Telemetry merge fix shipped. New downstream bottleneck identified: prefix reuse fires at engine level but does not translate to TTFT improvement.
**Author**: AX Engine work session

## TL;DR

Across two supported native MLX models (Gemma 4 E2B 4-bit and Qwen3.5-9B
4-bit), the retained prompt-prefix cache **never engages** in the
`Session.generate()` path. Three sequential scenarios designed to exercise
retained-prefix reuse all produce `retained_cache_hits = 0` and
`prefix_reused_blocks = 0`. The blocked-reuse counters are also zero,
confirming the misses are lookup misses (the cache is empty or unreachable
from the lookup site), not policy rejections.

**This is the most important KV/prefix bottleneck.** No structural cache
change should be started before the root cause of "retained cache never
hits" is understood. The KV-SCHEDULER-REVAMP PRD's higher-level concerns
(eviction policy, sliding window scheduling, KV-low budgeting) are
downstream — they only matter if the cache is actually being consulted
and hit.

## Method

Script: `scripts/profile_kv_long_context_evidence.py`
Schema: `ax.kv_long_context_evidence.v1`

Per model, three sequential scenarios in a single `Session`:

| scenario      | prompt                                | expectation                                                |
|---------------|---------------------------------------|------------------------------------------------------------|
| `cold`        | freshly synthesized 2048-token prompt | no prefix reuse possible                                   |
| `warm_repeat` | same 2048-token prompt re-issued      | retained logical prefix + MLX physical snapshot should hit |
| `warm_extend` | cold prompt + 256 new suffix tokens   | logical prefix should cover the cold-prompt portion        |

Each scenario captures TTFT, decode tok/s, prefill tok/s, KV capacity /
logical-token telemetry, and the full set of prefix-related counters from
`crossover_decisions`.

Models exercised:

- `gemma4` (Gemma 4 E2B, 4-bit) — `benchmarks/results/kv-long-context/gemma4-2026-05-11.json`
- `qwen3-5-9b` (Qwen3.5 9B, 4-bit) — `benchmarks/results/kv-long-context/qwen3-5-9b-2026-05-11.json`

## Findings

### Finding 1 — Retained prefix cache never hits (PRIMARY BOTTLENECK)

`retained_cache_hits = 0` and `prefix_reused_blocks = 0` across ALL six
scenario runs (3 scenarios × 2 models). `blocked_prefix_reuse_blocks = 0`
and `blocked_prefix_reuse_tokens = 0` as well — meaning the
"prefix-cache unsupported layout" / blocked-reuse path is not firing
either. The lookup is returning a clean miss.

Concrete numbers (Qwen3.5 9B):

| scenario      | TTFT (s) | decode tok/s | logical_tokens | prefix_reused | retained_hits | blocked_reuse |
|---------------|----------|--------------|----------------|---------------|---------------|---------------|
| cold          | 0.706    | 153.8        | 2114           | 0             | 0             | 0             |
| warm_repeat   | 0.662    | 155.1        | 2114           | 0             | 0             | 0             |
| warm_extend   | 0.770    | 155.4        | 2370           | 0             | 0             | 0             |

Gemma 4 E2B shows the same pattern (TTFT 0.254 / 0.236 / 0.278 s, all
zeros for prefix counters).

The 7% TTFT shrink on warm_repeat is consistent with warm-up of allocators
and CPU caches, not prefix reuse. Decode tok/s is unchanged, as expected
when no logical KV is reused.

### Finding 2 — Lookup site is reached but always returns miss

Source inspection (no code change yet):

- `crates/ax-engine-core/src/engine.rs:1078` `lookup_prefix` is invoked from
  `apply_prefix_reuse` for every Prefill execution item.
- `crates/ax-engine-core/src/kv.rs:175` `lookup_prefix` delegates to
  `lookup_live_prefix` and `lookup_cached_prefix`.
- `lookup_live_prefix` returns miss when no other live request has a
  matching first block. In a sequential single-request workload this is
  expected to miss (no concurrent peer).
- `lookup_cached_prefix` should hit because `KvManager::free` calls
  `promote_prompt_prefix_to_cache` (kv.rs:489) when a request is released,
  which populates `cached_blocks` keyed by token-hash prefix block keys.

Telemetry rules out the policy-blocked branch
(`blocked_prefix_reuse_* = 0`), so the failure mode is one of:

1. `free()` is not being called between sequential `Session.generate()`
   invocations, so promotion never runs.
2. `promote_prompt_prefix_to_cache` runs but
   `cached_token_count` resolves to 0 (kv.rs:618). This happens when
   `table.full_block_count == 0` even though the request processed a
   full 2048-token prompt — implying the MLX-native backend may complete
   prompt work without ever growing `BlockTable.full_block_count` in
   core's logical view. (Native KV lives in MLX-side buffers and the
   core-side block table may not track it the same way as the simulated
   backend.)
3. `cache_group_id` differs across calls (unlikely — same Session).
4. Token-hash block keys do not match across calls because the prompt
   bytes differ in some subtle way (e.g., BOS-injection difference).
   Unlikely given the same `input_tokens` list is replayed verbatim.

Hypothesis (2) is the most consistent with observed signals — KV capacity
clearly grew (`ax_mlx_kv_capacity_tokens` 18432 → 20480 on extend), but
the core's logical block book-keeping that drives the retained-cache
promote step may not advance for native-MLX prefills the way it would for
a simulated backend.

### Finding 3 — Coverage gaps acknowledged

The first slice covers three of the five §W1 coverage points cleanly:

- ✅ retained prompt-prefix cache reuse — `warm_repeat` (result: zero reuse)
- ✅ MLX physical prefix snapshot hit — `warm_repeat` (result: never hits)
- ✅ MLX physical prefix snapshot miss with warmup — `cold` + `warm_extend`
  (result: warmup path not measurable in current telemetry; new-tokens
  suffix on `warm_extend` would have triggered warmup IF prefix had hit)

Not yet covered:

- ⚠ prefix-cache unsupported layout — would require deliberately
  constructing a request whose execution-plan binding rejects prefix
  reuse (e.g., GQA mismatch, sliding-window-only model with prompt
  longer than the window). Holding for slice 2 — the broader retained
  miss is the larger signal.
- ⚠ KV-low and KV-exhausted scheduling paths — would require a workload
  big enough to actually exhaust `ax_mlx_kv_capacity_tokens`. Capacity
  on Qwen3.5-9B is 18432 → 20480 at our prompt sizes (well under
  exhaustion). Holding for slice 2.

These gaps do NOT block this report's verdict. The §W1 exit criterion is:

> A checked-in artifact identifies the most important KV/prefix
> bottleneck.

The identified bottleneck (retained cache never hits) is upstream of the
coverage gaps: KV-low and KV-exhausted scheduling improvements would
benefit from a working prefix cache, but cannot rescue a cache that does
not engage.

## Verdict / Recommended Next Action

**Bottleneck**: `retained_prefix_cache_not_engaging` — the core-side
retained prefix cache promote/lookup machinery exists and the lookup site
is reached for every Prefill, but no entry is ever found. Multi-model
confirmed; not a model-specific quirk.

**Recommended next step (slice 2 of W1)**: Add three temporary debug
counters to `KvManager` so the live path can prove which of the four
candidate failure modes is real:

1. `promote_prompt_prefix_to_cache_calls` — how many times promote was
   invoked.
2. `promote_prompt_prefix_to_cache_blocks_promoted` — how many blocks
   actually got `insert_cached_block`-ed.
3. `lookup_cached_prefix_calls` — how many lookups occurred.

If `(1) > 0 && (2) == 0`, hypothesis 2 (full_block_count stays 0) is
confirmed and the fix lives at the boundary where the MLX runner reports
materialized block progress to `KvManager`. If `(1) == 0` instead, the
fix lives at the `free()`/request-lifecycle boundary — `free()` is not
being called between sequential `Session.generate()` calls.

**Slice 2 should not exceed ~1 day of work** — three counters, re-run
this script, compare. The next structural decision (eviction policy,
KV-low scheduling, snapshot retention) is gated on knowing which side of
the boundary the bug lives on.

## Slice 2 Findings (2026-05-11) — ALL FOUR HYPOTHESES WRONG

The slice-1 hypothesis catalog (h1: free not called, h2: full_block_count
stays 0, h3: cache_group_id mismatch, h4: token-hash mismatch) was
**all wrong**. Slice 2 added `AX_KV_DIAG`-gated diagnostics to `KvManager`
(`crates/ax-engine-core/src/kv.rs`) covering `lookup_prefix`, `free`, and
`promote_prompt_prefix_to_cache`, plus a temporary trace inside
`engine.rs::apply_prefix_reuse` (since reverted). Trace on Gemma 4 E2B,
3 sequential calls × 512-token prompt:

```
[KV_DIAG] free(request=2): table.full_block_count=32 logical_token_count=519
[KV_DIAG] promote: COMPLETE newly_inserted=28 touched_existing=4 total_cached_blocks_now=32
[KV_DIAG] lookup_prefix(request=3): cached_blocks_total=32
[KV_DIAG] lookup_prefix result: cached_matched_tokens=512
[KV_DIAG] apply_prefix_reuse: lookup.hit=true matched_tokens=512 matched_blocks=32 retained=true
```

The prefix cache:
- **Is** populated by `free()` → `promote_prompt_prefix_to_cache` (32 blocks
  cached after first 512-token prompt completes).
- **Is** consulted by `lookup_prefix` on the second-call prefill (all 32
  blocks match, 512 matched tokens).
- **Reaches** `apply_prefix_reuse` with `lookup.hit=true` and `retained=true`.
- The `prefix_reuse` map IS populated and passed to
  `rebuild_execution_batch`, which writes `retained_cache_hits=1`,
  `prefix_reused_blocks=32`, `prefix_reused_tokens=512` into
  `execution_batch.route_metadata.crossover_decisions`.

### Slice 2 Root Cause — Telemetry Overwrite in Per-Step Aggregation

Bug location: `crates/ax-engine-sdk/src/session.rs:748` —
`store_native_request_route(request_id, route)` is a **plain `insert`**,
not a merge:

```rust
fn store_native_request_route(&mut self, request_id: u64, route: GenerateRouteReport) {
    if !self.native_request_routes.contains_key(&request_id) {
        self.native_route_report_order.push_back(request_id);
    }
    self.native_request_routes.insert(request_id, route);   // <-- overwrites every step
    ...
}
```

A `Session.generate()` call produces N engine steps:
- Step 1 = Prefill (where `apply_prefix_reuse` runs and writes counters).
- Steps 2..N = Decode (where `apply_prefix_reuse` returns early; the
  decode-step route metadata contains **zero** prefix-reuse counters).

Each step's route metadata is stored with `insert(request_id, route)`,
so by the time `Session.generate()` returns, the stored route for that
request is the LAST step's metadata — i.e. a decode step with zero
prefix-reuse counters. The prefill-step route (with the real
`retained_cache_hits=1`, `prefix_reused_blocks=32`, etc.) is silently
overwritten on step 2.

**This explains every observation in slice 1**: the `crossover_decisions`
the Python artifact reads are from the final decode step, and the engine's
real prefix-reuse work is invisible. `blocked_prefix_reuse_*` is zero too
because that path also only fires on prefill — it's overwritten the same
way.

### Verdict — Bug Is in the SDK, Not the Cache or Scheduler

The retained prefix cache, the cache-promote-on-free path, the lookup
machinery, and the per-step prefix-reuse plumbing in
`ax-engine-core::engine` all work correctly. The bug is a one-line
aggregation defect in `ax-engine-sdk::session::store_native_request_route`.

**Recommended fix** (≤1 day, owned by SDK, not core):

Replace the `insert` with a merge that preserves monotonically-meaningful
counters. Concretely, when a step has zero `prefix_reused_blocks` /
`retained_cache_hits` / `live_share_hits` / `blocked_prefix_reuse_*` but
the existing stored route has non-zero values for those keys, keep the
stored values. Cleanest implementation: a small `merge_route_for_request`
that, for each key in a fixed allow-list, keeps `max(stored, new)`. Keys:

```
prefix_reused_requests, live_share_hits, retained_cache_hits,
prefix_reused_blocks, prefix_reused_tokens,
blocked_prefix_reuse_requests, blocked_prefix_reuse_blocks,
blocked_prefix_reuse_tokens, max_prefix_blocks_reused_per_request,
branch_prefill_requests, branch_decode_requests,
branch_prefill_tail_tokens, branch_decode_tokens
```

Non-monotonic keys (`ax_mlx_kv_logical_tokens`, `ax_mlx_kv_capacity_tokens`,
etc.) should keep the LAST value, matching current behaviour.

**Once the fix lands**, re-run `scripts/profile_kv_long_context_evidence.py`
on both Gemma and Qwen. Expected outcome: warm_repeat shows
`retained_cache_hits=1`, `prefix_reused_blocks≈prompt_tokens/block_size`,
and warm_extend shows the same plus a partial-match. TTFT speedup will
become measurable (the prefill compute really is skipped — the
SDK was just lying about it in telemetry).

### Slice 2 Coverage Update vs §W1 PRD

Coverage in the §W1 exit-criteria sense (now grounded in real engine
behaviour rather than telemetry-only data):

- ✅ retained prompt-prefix cache reuse — works at engine level; telemetry
  fix needed to confirm runtime TTFT improvement.
- ✅ MLX physical prefix snapshot hit — engine-level prefix matching
  confirmed working; whether MLX backend honours it as a physical-snapshot
  restore (vs. recompute) requires the telemetry fix to measure.
- ✅ MLX physical prefix snapshot miss with warmup — warm_extend already
  exercises this path (extension tokens require warmup); cannot quantify
  warmup cost until telemetry fix lands.
- ⚠ prefix-cache unsupported layout — still untested (slice 3).
- ⚠ KV-low and KV-exhausted scheduling paths — still untested (slice 3).

### Diagnostics Left in Tree

The `AX_KV_DIAG=1`-gated eprintlns in
`crates/ax-engine-core/src/kv.rs` remain (zero cost when the env var is
not set). They are useful for ongoing KV/prefix investigation and can
be removed later as a separate cleanup commit once the telemetry-merge
fix is shipped and proves the cache is fully working.

The engine-side trace inside `apply_prefix_reuse` was reverted —
slice-specific, no ongoing value.

## Slice 3 Findings (2026-05-11) — Telemetry Fix Shipped, New Downstream Bottleneck

### Fix landed

Two overwrite sites were responsible for the telemetry zero-out, not one:

1. `store_native_request_route` (session.rs:748) — plain `insert` into
   `native_request_routes`. Replaced with a merge that uses
   `merge_native_route_into`.
2. `apply_native_step_route_to_report` (session.rs:1933) — `report.route =
   route.clone()`. Also replaced with `merge_native_route_into`. This
   second site explains why fixing only the first didn't change the
   Python artifact: every streaming step's route was clobbering the
   per-request stored route via this path too.

`merge_native_route_into` (session.rs ~58 lines) does:

- String fields: last-wins, except `prefix_cache_path` keeps a more
  informative stored value rather than being clobbered by the decode-step
  default `"metadata_lookup"`.
- `crossover_decisions`: monotonic `max()` merge for the 13 prefix-reuse
  and branching counters declared in `MONOTONIC_CROSSOVER_DECISION_KEYS`;
  last-wins for all others (preserves current behaviour for `ax_mlx_kv_*`
  capacity/logical-tokens counters etc.).

Three unit tests cover the cases that mattered (full SDK lib suite is 72
tests, all green; ax-engine-core 400 tests still green).

### Post-fix evidence

Re-ran the same `profile_kv_long_context_evidence.py` on both models.
Selected post-fix artifacts:

- `benchmarks/results/kv-long-context/gemma4-post-fix-2026-05-11.json`
- `benchmarks/results/kv-long-context/qwen3-5-9b-post-fix-2026-05-11.json`

Qwen3.5-9B post-fix:

| scenario      | TTFT (s) | decode tok/s | prefix_reused_blocks | retained_hits |
|---------------|----------|--------------|----------------------|---------------|
| cold          | 0.704    | 156.1        | 4                    | 1             |
| warm_repeat   | 0.665    | 155.5        | 128                  | 1             |
| warm_extend   | 0.766    | 154.6        | 128                  | 1             |

Gemma 4 E2B post-fix:

| scenario      | TTFT (s) | decode tok/s | prefix_reused_blocks | retained_hits |
|---------------|----------|--------------|----------------------|---------------|
| cold          | 0.257    | 366.3        | 4                    | 1             |
| warm_repeat   | 0.259    | 367.8        | 128                  | 1             |
| warm_extend   | 0.284    | 369.3        | 128                  | 1             |

Notes on the values:

- `cold` shows 4 blocks reused because the `warmup ...` step in the
  harness (a tiny 64-token prefill before scenario 1) seeded 4 blocks
  that then matched the cold scenario's first 64 tokens. This is real
  prefix reuse the script accidentally exercises; the artifact's
  bottleneck-classifier already handles it.
- `warm_repeat` shows 128 blocks reused = 2048 tokens / 16 tokens-per-block,
  i.e. the full prompt prefix matched as designed.
- `warm_extend` shows 128 blocks reused = the cold prompt's full prefix
  before the new suffix begins. Matches expectation.

### New bottleneck — engine-level prefix reuse does not move TTFT

This is the surprise. With telemetry now accurate, the script's verdict
flipped from `prefix_reuse_not_firing` to `prefix_reuse_low_value`:

- Qwen3.5-9B: warm_repeat TTFT speedup = **1.06×** vs cold (0.665 / 0.704 s)
- Gemma 4 E2B: warm_repeat TTFT speedup = **0.99×** vs cold (0.259 / 0.257 s)

Both models report 128 blocks reused (≈ full 2048-token prompt) but
neither delivers meaningful TTFT improvement. Hypotheses for slice 4:

1. **MLX runner ignores the matched-prefix annotation.** Even though core
   marks `prefix_tokens_reused = 2048`, the MLX runner may still execute
   the full prompt prefill on the GPU because the physical KV state was
   evicted/dropped between calls or never persisted past `free()`. The
   scheduler-level reuse is a logical accounting, not a physical KV
   restore. This is consistent with the existing model docs:
   "After each decode step, all KV backing buffers are evaluated …" —
   they live in MLX lazy-eval space and may be freed when the request is
   freed, regardless of what core's block table records.
2. **Warmup token slice cancels out the saving.** `native_prefix_warmup_token_slice`
   (engine.rs:1118) returns the full matched-prefix slice on warm prefill,
   meaning the runner is *asked* to re-process exactly those tokens
   anyway, defeating the point.
3. **MLX backend has its own snapshot cache that isn't being keyed on the
   same hash.** The MLX-side prefix snapshot machinery may exist but use a
   different cache key (e.g. block-id-based rather than token-hash-based).
   Telemetry counters like `ax_mlx_kv_request_snapshots` are still 1 in
   warm_repeat, so the snapshot store path isn't being hit.

Hypothesis 2 is the most checkable. Reading
`native_prefix_warmup_token_slice` (engine.rs:1118):

```rust
if let Some(lookup) = lookup {
    return snapshot
        .prompt_tokens
        .get(..lookup.matched_token_count as usize)
        .unwrap_or(snapshot.prompt_tokens.as_slice())
        .to_vec();
}
```

This returns the matched-prefix tokens. They get attached to the
execution item as `reused_prefix_token_slice` and the runner
re-processes them as a "warmup" to rebuild the KV state. **So the
matched-prefix is being re-prefilled, not skipped.** This is precisely
the "warmup" path mentioned in §W1 — but for the same-prompt repeat case
where a cached MLX snapshot could in theory skip warmup, the runner has
no mechanism to honour the snapshot and re-runs the prefill anyway.

This is consistent with both observations (telemetry says "reused 128
blocks" — true at the bookkeeping level — but TTFT is unchanged because
the runner re-executes everything).

### Recommended slice 4

Investigate the MLX runner's warmup path:

- Source: `crates/ax-engine-mlx/src/runner.rs` — find where
  `reused_prefix_token_slice` is consumed.
- Question: is there an MLX-side KV snapshot persisted across `free()`,
  and does the runner consult it when `prefix_tokens_reused > 0`?
- If snapshot exists: why isn't it being matched? (Likely cache-key
  mismatch.)
- If snapshot does not exist: this is the structural change ADR 0018
  contemplates — adding a physical-KV-snapshot retention layer to MLX
  backend that survives `free()` and is keyed compatibly with
  `KvManager`'s retained prefix cache.

**Estimated work**: 2-3 days. This is finally the structural change
the §W1 evidence gate was protecting. ADR 0018 should adopt slice 3 +
slice 4 evidence together as its decision predicate.

## Artifacts

Pre-fix (slice 1 evidence — telemetry was zero):

- `benchmarks/results/kv-long-context/gemma4-2026-05-11.json`
- `benchmarks/results/kv-long-context/qwen3-5-9b-2026-05-11.json`

Post-fix (slice 3 evidence — telemetry reflects engine reality):

- `benchmarks/results/kv-long-context/gemma4-post-fix-2026-05-11.json`
- `benchmarks/results/kv-long-context/qwen3-5-9b-post-fix-2026-05-11.json`

Harness:

- `scripts/profile_kv_long_context_evidence.py`

## ADR/PRD Cross-References

- KV-SCHEDULER-REVAMP-PRD.md §W1 — coverage requirements + exit criteria.
- ADR 0018 — KV/scheduler revamp policy. This report is the evidence
  predicate to that ADR's "no structural change without evidence" gate.
- ADR 0017 — evidence-first governance. This report follows that pattern:
  identify the bottleneck before proposing a fix.
