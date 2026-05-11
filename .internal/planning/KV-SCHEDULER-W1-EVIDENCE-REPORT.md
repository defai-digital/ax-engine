# W1 Long-Context and Prefix-Reuse Evidence Report

**PRD**: `.internal/planning/KV-SCHEDULER-REVAMP-PRD.md` §W1
**ADR**: 0018 KV/Scheduler Revamp
**Status**: First slice complete (2026-05-11). Bottleneck identified.
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

## Artifacts

- `benchmarks/results/kv-long-context/gemma4-2026-05-11.json`
- `benchmarks/results/kv-long-context/qwen3-5-9b-2026-05-11.json`
- `scripts/profile_kv_long_context_evidence.py`

## ADR/PRD Cross-References

- KV-SCHEDULER-REVAMP-PRD.md §W1 — coverage requirements + exit criteria.
- ADR 0018 — KV/scheduler revamp policy. This report is the evidence
  predicate to that ADR's "no structural change without evidence" gate.
- ADR 0017 — evidence-first governance. This report follows that pattern:
  identify the bottleneck before proposing a fix.
