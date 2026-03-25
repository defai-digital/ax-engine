# AX Engine Performance PRDs

`PRDS/` is the active product-planning set.
`ADRS/` captures architecture decisions.
`TODOS/` is historical analysis (superseded).

Use [REFERENCE-FIRST-SUPPORT-WORKFLOW.md](/Users/akiralam/code/ax-engine-v2/PRDS/REFERENCE-FIRST-SUPPORT-WORKFLOW.md) for new quant/architecture support planning.

## Current Status

See **`STATUS-FINAL-2026-03-23.md`** for the complete picture: benchmark
results, competitive position, what worked, what didn't, and what's next.

**Position**: current Phase 3 cross-validation says AX is at or above parity
with local `llama.cpp` at `P=64` on Qwen3 8B and LLaMA 3 8B, and near parity on
Gemma3 4B. Longer-context decode remains below local `llama.cpp`, and the
sweeps did not justify any new default/profile changes. Against external
engines on the same short shape, AX is ahead of `mistralrs` where usable, but
still trails MLX decode by roughly `13-19%`. Prefill remains about half of
local `llama.cpp` on these model families.

**Next move**: move the remaining decode follow-up into PRD-0003 and stop
reopening PRD-0004. The full sweep and Phase 3 cross-validation are now done,
and they did not justify any default/profile promotion. PRD-0002 is
benchmarked and remains experimental because current speculative throughput is
far below the non-spec baseline.

## Priority Order

1. **PRD-0003** — Decode matvec tuning: **Mostly done.**
   NR2 multi-row kernel shipped. The remaining work is the LLaMA 3
   `P=256+` decode follow-up; PRD-0004 already showed no broader default/profile
   change is justified.

2. **PRD-0002** — Speculative decode: **Implemented, benchmarked, not shippable yet.**
   Real small same-vocab draft pairs now run, but current throughput is far
   below non-spec decode. The next work is draft-path optimization, not just
   more pair hunting.

3. **PRD-0004** — Parameter sweep: **Done.**
   Decode sweep, prefill sweep, and Phase 3 cross-validation all completed.
   Result: no stable default/profile promotion.

4. **PRD-0001** — Split-K attention: **Done.**
   Gemma3 hd=256 shipped (−2.4% degradation, better than llama.cpp).
   Qwen3 hd=128 tested and disproved.

5. **PRD-0005** — Selective decode fusion and sync cleanup: **Proposed.**
   Supersedes the broad TODO memo as the concrete next-step document for
   benchmark-backed decode-path cleanup.

6. **PRD-0006** — Prefill profile surface: **Implemented.**
   Adds a narrow, schema-backed prefill profile layer for routing and
   thresholds, without turning profiles into raw llama.cpp parameter dumps.

7. **PRD-0007** — Prefill gap structural close: **Proposed.**
   Adds a benchmarked prefill gap-workstream that learns from llama.cpp structure
   while staying within schema-valid AX controls.

8. **PRD-0009** — Architecture hardening roadmap: **Completed.**
   Keeps AX's design advantage while tuning continues, by centralizing policy,
   KV planning, execution planning, and experiment selection.

9. **PRD-0010** — Hardware study and microbench workbench: **Proposed.**
   Turns Apple Silicon hardware literacy into a concrete workflow: targeted
   reading, dedicated microbenches, and hardware-backed perf hypotheses.

10. **PRD-0011** — Q5_K support: **In progress.**
   Baseline decode-only Metal support is landed from the imported reference
   shape. Remaining work is a possible experimental AX-native batch/prefill
   track plus any justified second candidate.

11. **PRD-0012** — Qwen3.5 support: **Proposed.**
   Reference-first plan: external engines define the initial hybrid/recurrent
   basis, then AX microbench and end-to-end runs validate AX-specific choices
   for Qwen3.5.

## Quick experiments (not PRDs, just tests)

These can be run immediately without code changes:

| Experiment | Command | Expected |
|---|---|---|
| Barrier A/B | `bash scripts/bench_decode_barriers_ab.sh` | Measured neutral to slightly negative |
| Spec pair check | `bash scripts/bench_speculative_pairs.sh` | Older large-draft pairs are blocked/bad |
| Q4_K_V2 prefill A/B | `PROMPT_TOKENS=256 WARMUP_ITERS=1 MEASURE_ITERS=3 bash scripts/bench_prefill_v2_ab.sh` | `Q4K_V2=1` regressed in tested short prefill prompts |

## PRD Files

- `PRD-0001-split-k-decode-attention.md` — **Done.** Retained as implementation history.
- `PRD-0002-batched-speculative-verification.md` — **Implemented and benchmarked.** Still experimental.
- `PRD-0003-decode-matvec-tuning.md` — **NR2 shipped.** LLaMA 3 P=256+ open.
- `PRD-0004-kernel-parameter-sweep.md` — **Done.** No stable default/profile promotion.
- `PRD-0005-selective-decode-fusion-and-sync-cleanup.md` — **Proposed.**
  Focuses the remaining decode work on selective fusion and scoped synchronization.
- `PRD-0006-prefill-profile-surface.md` — **Implemented.**
  Defines the narrow prefill profile surface and rollout rules.
- `PRD-0007-prefill-gap-structural-close.md` — **Proposed.**
  Focuses on prefill parity through supported routing controls and schema-backed
  profile extraction.
- `PRD-0008-prefill-performance-decision.md` — **Done.**
  Documents the v2 opt-in regression finding and the profiling-first next step.
- `PRD-0009-architecture-hardening-roadmap.md` — **Completed.**
  Defines the design track that keeps tuning as typed policy and planning
  instead of ad hoc runtime branching.
- `PRD-0010-hardware-study-and-microbench-workbench.md` — **Proposed.**
  Defines what hardware topics to study, which microbenches to add, and how to
  use them before promoting tuning changes.
- `PRD-0011-q5k-support.md` — **In progress.**
  Defines the staged plan to move `Q5_K` from live-validated decode-only
  baseline support to full first-class optimized support.
- `PRD-0011-q5k-reference-matrix.md` — **Active artifact.**
  Capture sheet for imported `Q5_K` layout, kernel, fallback, and validation
  choices, now including the landed baseline decode import.
- `PRD-0011-q5k-metal-import-note.md` — **Baseline slice landed.**
  Concrete AX mapping and current-state note for the imported `llama.cpp`
  `Q5_K` Metal decode path.
- `PRD-0011-q5k-prefill-experimental-track.md` — **Proposed companion.**
  Defines the best-practice rules for any future AX-native `Q5_K` GPU
  prefill/batch work: opt-in, experimental, benchmark-gated, and non-default.
- `PRD-0012-qwen35-support.md` — **Proposed.**
  Defines a reference-first staged plan to make Qwen3.5 a first-class
  supported hybrid architecture rather than a special-case implementation.
- `PRD-0012-qwen35-reference-matrix.md` — **Planned artifact.**
  Capture sheet for imported Qwen3.5 hybrid/recurrent design and validation choices.

## Status Files

- `STATUS-FINAL-2026-03-23.md` — **Current.** Complete analysis with journey, disproved hypotheses, architecture review, and plan.
- `STATUS-2026-03-23.md` — Day's benchmark results (superseded by FINAL).
- `STATUS-2026-03-22.md` — Historical (superseded).

## External Comparison

- `../automatosx/tmp/engine-comparison-2026-03-23.md` — AX vs MLX Engine vs mistral.rs on the same local model families.
- `../automatosx/tmp/speculative-benchmark-2026-03-23.md` — Real small-draft speculative results vs non-spec baselines.
- `../automatosx/tmp/prd-0004-prefill-sweep-summary-2026-03-23.md` — Phase 2 prefill sweep summary; no stable default/profile winner.
- `../automatosx/tmp/prd-0004-phase3-summary-2026-03-23.md` — Final Phase 3 cross-validation summary; PRD-0004 closed with no stable promotion.

## Disproved Approaches (Do Not Re-Test)

| Approach | Result | Documented in |
|---|---|---|
| TG=256 for Q4_K matvec | −16% to −35% | PRD-0003 |
| BLK2 multi-block loop | −6% to −40% | PRD-0003 |
| x2 naive multi-row | −20% to −27% | PRD-0003 |
| Precomputed f16 decode | −63% | STATUS-2026-03-23 §5.1 |
| Split-K for hd=128 | −36% | PRD-0001 |
| f16 KV force-on (already auto) | Neutral | STATUS-2026-03-22 |
| Pair FFN kernel | Neutral | STATUS-2026-03-23 §5.3 |
| Decode barriers off | Neutral to slightly negative | STATUS-FINAL §1 |
| Metal 3.1 language version | Neutral | STATUS-FINAL §1 |
| target-cpu=native for GPU | Neutral | STATUS-FINAL §1 |
