# PRD: MTP + N-gram Gating From Lightning-MLX 0.7.0 Learnings

**Status**: Implemented
**Date**: 2026-06-02
**Scope**: `crates/ax-engine-mlx/src/runner.rs`, `crates/ax-engine-mlx/src/ngram_accel.rs`
**Related ADR**: `.internal/adr/ADR-018-mtp-ngram-gating-lightning-learnings.md`
**Related Tech Spec**: `.internal/planning/TECH-SPEC-2026-06-02-mtp-ngram-gating-lightning-learnings.md`

## Problem

AX Engine has an increasingly capable MTP + n-gram hybrid decode path, but the
hybrid path needs stronger production gating before new defaults or behavior
changes are considered complete.

The local `lightning-mlx` reference was reviewed at:

- `v0.7.0` tag: `63ef0e5 docs: revert raw decode section to original numbers`
- current `main`: `ec19b3d fix: streaming responses drop content for tool-less requests (#3)`
- comparison base: `v0.6.32`

Lightning 0.7.0 is useful less as a code copy target and more as a process
reference: it keeps changes only when experiment artifacts justify them, skips
low-leverage ideas, and gates n-gram when MTP is already strong or n-gram is
weak.

Current AX review found that the active `runner.rs` path implements several
Lightning-inspired ideas:

- cascade-corrected `mtp_only_accept_rate_ewma`;
- n-gram saturation gating;
- n-gram hurt gating using combined EWMA versus MTP-only EWMA;
- dual-condition auto-disable based on MTP acceptance and n-gram acceptance;
- per-request n-gram self-tune counters;
- n-gram pseudo log-probability changed toward a delta-distribution model.

The implementation review closed the high-risk gaps found in the earlier WIP:

- new n-gram gate env knobs are cached and clamped outside the decode hot loop;
- focused tests cover hurt gating, auto-disable, self-tune, acceptance modes,
  and route metadata;
- per-request self-tune increments drafted count only when an n-gram draft is
  actually submitted after cycle-guard suppression;
- n-gram-source acceptance mode is explicit: `confidence` remains the default,
  while `delta` and `greedy` require `AX_MLX_MTP_NGRAM_ACCEPTANCE_MODE`.

Benchmark `opt2` rows sampled before this implementation did not show the hurt
gate firing (`ax_mtp_ngram_hurt_gated_steps = 0` in sampled rows), so default
promotion still requires fresh fair benchmark evidence.

## Goals

- **G1**: Prevent n-gram stacking from regressing MTP-only decode throughput on
  agentic workloads.
- **G2**: Keep n-gram enabled when it is measurably beneficial, especially on
  long repetitive code and artifact-generation prompts.
- **G3**: Preserve MTP rejection-sampling correctness for MTP-sourced draft
  tokens.
- **G4**: Make n-gram-source acceptance semantics explicit and test-covered.
- **G5**: Expose enough telemetry to prove which gate fired and why.
- **G6**: Match repository hot-path best practice: env knobs are cached,
  clamped, documented, and covered by focused tests.
- **G7**: Require benchmark evidence before promoting any Lightning-inspired
  default.

## Non-goals

- Copying Lightning-MLX implementation internals wholesale.
- Enabling optimistic acceptance by default.
- Enabling n-gram by default for every model family.
- Promoting MTP draft depth changes without AX-specific benchmark artifacts.
- Implementing EAGLE-3/4, Metal4/NAX, dFlash, or `mx.compile` equivalents in
  this work item.
- Reworking the public OpenAI server streaming stack, except for opening a
  separate follow-up if AX has the same tool-mode bug fixed in Lightning `main`.

## Lightning-MLX 0.7.0 Evidence Reviewed

### Keep

- **E2 MTP depth sweep**: MTP depth 5 was kept only after measured gains:
  +10.3% short and +21.8% long in Lightning's M5 Max agentic rows.
- **E8 n-gram profile plumbing**: kept as infrastructure, with no behavioral
  default change.
- **E9 hardware/Metal4 detection**: kept as observability only.

### Skip or Revert

- **E12 sampler GPU path**: skipped because batch=1 agentic CPU syncs were
  small tensors and negligible.
- **E13 `mx.compile` coverage**: skipped because safe targets were absent.
- **E4 prefix cache tune**: reverted because short-prompt overhead outweighed
  cache benefit.

### Post-tag Bug Fix Worth Auditing Separately

Lightning `main` after `v0.7.0` fixes streaming content suppression by deriving
tool mode from `request.tools`, not from global parser configuration. AX should
audit the same bug class separately: content-suppression heuristics must be
request-scoped, not globally enabled by parser defaults.

## User Impact

| User / workload | Expected impact |
|---|---|
| MTP-only decode | No regression; n-gram gates must not affect pure MTP mode. |
| MTP + n-gram on strong MTP rows | Lower wasted verify work by suppressing weak n-gram. |
| Repetitive long-code rows | Keep n-gram available when acceptance evidence supports it. |
| Tool-call / structured output | Avoid stale n-gram guesses in structurally repetitive but content-sensitive regions. |
| Benchmark publication | More auditable route metadata explaining gate behavior. |

## Product Requirements

### R1: Gate n-gram when MTP is saturated

When cascade-corrected MTP-only EWMA exceeds a depth-aware threshold after a
minimum sample count, suppress n-gram drafting for that step.

The existing saturation gate is the right direction and should remain.

### R2: Gate n-gram when it is actively hurting

When combined accept-rate EWMA is lower than MTP-only EWMA by a configured
margin after warmup, suppress n-gram drafting and increment
`ax_mtp_ngram_hurt_gated_steps`.

Implemented with `mtp_ngram_hurt_margin()` and pure helper
`mtp_ngram_hurt_gate(...)`.

### R3: Add dual-condition auto-disable

When MTP global acceptance is strong and n-gram global acceptance is weak after
their respective warmups, suppress n-gram drafting. Defaults should mirror the
reviewed Lightning behavior unless AX benchmarks prove different thresholds:

- MTP warmup: 64 draft tokens
- n-gram warmup: 32 draft tokens
- MTP threshold: 0.85
- n-gram floor: 0.50

### R4: Add per-request self-tune

After 32 actually submitted n-gram draft tokens in a request, disable n-gram for
the rest of that request if accepted / drafted falls below 0.30.

Drafts suppressed by cycle guard or other pre-submit gates must not be counted
as drafted.

Implemented with request-local `NgramSelfTuneState`.

### R5: Make n-gram-source acceptance semantics explicit

N-gram-source draft tokens may use one of two explicit modes:

- `confidence`: previous AX pseudo distribution, `log_prob = ln(confidence)`;
- `delta`: Lightning-like point mass, `log_prob = 0.0`;
- optional future `greedy`: accept n-gram source only on target argmax match.

The default must remain whatever benchmarks prove safest. A silent semantic
change from confidence to delta is not complete without tests and benchmark
evidence.

Implemented default: `confidence`. `delta` and `greedy` are opt-in.

### R6: Telemetry and auditability

Route metadata must include:

- saturation-gated steps;
- hurt-gated steps;
- auto-disabled steps;
- self-tune-disabled steps;
- MTP-only EWMA and sample count;
- combined EWMA and sample count;
- n-gram submitted/accepted counters used by self-tune.

### R7: Kill switches and rollback

Every active gate must be individually disableable via cached env-var helpers.
Disabling all new gates should restore the prior n-gram stacking behavior.

## Acceptance Criteria

- `cargo fmt --check`
- `git diff --check`
- `cargo test -p ax-engine-mlx --quiet`
- `cargo clippy -p ax-engine-mlx --all-targets -- -D warnings`
- focused tests for hurt, auto-disable, self-tune, route telemetry, and n-gram
  pseudo-logprob/greedy modes;
- benchmark artifact where at least one row exercises each new gate or a
  documented explanation that the gate did not trigger;
- no n-gram+MTP row regresses against MTP-only by more than 2% unless the row is
  explicitly classified as opt-in experimental;
- route metadata contains non-ambiguous gate counters.

As of implementation, code-level acceptance is complete. Benchmark acceptance
remains a separate artifact requirement before default promotion.

## Current Evidence Snapshot

The local `opt2` benchmark directory contains useful evidence but is not enough
to close this PRD by itself:

- sampled 27B rows show n-gram variants improving mean decode throughput over
  MTP-only in flappy, long_code, and python_modules_long;
- sampled 35B flappy row shows n-gram variant above MTP-only;
- sampled `ax_mtp_ngram_hurt_gated_steps` is `0`, so the new hurt gate has not
  been proven by these rows;
- not all 35B rows were present in the sampled summary at review time.

## Open Questions

- Should AX keep confidence-based n-gram rejection sampling as the default, or
  move to delta/greedy mode only for specific model presets?
- Should auto-disable thresholds be model-family specific?
- Do server streaming heuristics have the same request-tools versus global-parser
  bug fixed in Lightning `ec19b3d`?
- Should n-gram self-tune feed counters from `NgramAccelerationTelemetry` or a
  dedicated request-local state object?
