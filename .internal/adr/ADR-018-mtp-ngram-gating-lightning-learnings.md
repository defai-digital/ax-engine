# ADR-018: MTP + N-gram Gating From Lightning-MLX 0.7.0 Learnings

**Status**: Accepted
**Date**: 2026-06-02

## Context

AX Engine already implements MTP speculative decode and n-gram stacking. Recent
work has made the hybrid path stronger, but it also increased the risk of
over-stacking: n-gram can help repetitive workloads, yet it can reduce
throughput when MTP is already strong or when n-gram guesses cascade-reject MTP
tail tokens.

The local `lightning-mlx` reference was reviewed at `v0.7.0` and current
`main`. The important lesson is not a single implementation detail. The lesson
is the release process:

- keep changes with benchmark evidence;
- skip low-leverage optimizations;
- revert defaults that regress target workloads;
- expose route metrics so performance claims are auditable.

Lightning 0.7.0 kept MTP depth 5 only after measured wins, skipped sampler GPU
path work for batch=1 agentic workloads, and reverted prefix-cache tuning when
short prompts regressed. Its n-gram path uses per-request self-tune and global
auto-disable when MTP is strong and n-gram is weak.

Current AX implementation in `runner.rs` implements similar ideas with the
cleanup required before treating the design as complete:

- hot-loop env parsing for the new gate knobs is replaced with cached helpers;
- self-tune counters count submitted drafts, not merely predicted drafts;
- n-gram acceptance mode is explicit and defaults to previous AX confidence
  semantics;
- focused tests cover gate predicates, self-tune, acceptance modes, and route
  metadata.

Benchmark evidence is still required before promoting a new default policy.

## Decision

Adopt Lightning-inspired n-gram gating in AX Engine, but only behind AX-style
cached configuration, explicit acceptance semantics, focused tests, and fair
benchmark gates.

### D1: Keep cascade-corrected MTP-only EWMA

MTP-only EWMA excludes cascade rejections caused by earlier n-gram failures.
This is the correct signal for deciding whether MTP itself is strong enough to
make n-gram stacking unnecessary.

### D2: Keep saturation gating

When MTP-only EWMA exceeds a depth-aware threshold after warmup, suppress n-gram
for that step. This preserves n-gram for depth=1 and weaker MTP rows while
avoiding wasted work when MTP is already saturated.

### D3: Add hurt gating

When combined EWMA trails MTP-only EWMA by a configured margin, suppress n-gram
and increment hurt-gate telemetry. This directly targets the hybrid cascade
failure mode.

This gate must be implemented with a cached/clamped helper and tests.

### D4: Add global auto-disable

When MTP global acceptance is strong and n-gram global acceptance is weak after
warmup, suppress n-gram. Use Lightning's reviewed defaults as initial values:

- MTP warmup: 64 tokens
- n-gram warmup: 32 tokens
- MTP threshold: 0.85
- n-gram floor: 0.50

Do not promote these thresholds as universal until AX-specific benchmarks cover
27B and 35B-A3B rows.

### D5: Add per-request self-tune

After a request submits enough n-gram draft tokens, disable n-gram for that
request if its acceptance rate is low. This mirrors Lightning's
`NgramRequestState` behavior and avoids punishing other requests.

Self-tune must count only submitted drafts. Cycle-guarded or otherwise
suppressed drafts must not affect the rate.

### D6: Make n-gram acceptance mode explicit

The n-gram-source draft distribution must not be a hidden semantic change.
Support explicit modes:

- `confidence`: previous AX behavior;
- `delta`: point-mass draft distribution;
- `greedy`: accept n-gram-sourced depths only on target argmax match.

The default remains an evidence decision. The ADR does not approve silently
switching all n-gram positions to delta mode.

Implementation default is `confidence`; `delta` and `greedy` are opt-in via
`AX_MLX_MTP_NGRAM_ACCEPTANCE_MODE`.

## Alternatives Considered

### A1: Always enable n-gram with MTP

Rejected. Lightning evidence and AX artifacts both show that n-gram can be
workload-dependent. Always-on stacking can add lookup overhead, verify width,
cache resets, and cascade rejections.

### A2: Disable n-gram globally for MTP models

Rejected. Some repetitive long-code rows benefit from n-gram. A global disable
throws away upside that can be preserved with gates.

### A3: Copy Lightning thresholds and defaults directly

Rejected as a final policy. Lightning's defaults are useful starting points,
but AX has different runtime, telemetry, MTP sidecar behavior, and benchmark
harnesses. AX thresholds must be proven on AX artifacts.

### A4: Use GPU sampler optimization as the main lever

Rejected for this ADR. Lightning E12 explicitly skipped sampler GPU path work
for batch=1 agentic workloads because the syncs were small and negligible. AX's
GPU top-k sampling work remains separately gated and benchmarked, but it should
not replace n-gram/MTP acceptance gating.

### A5: Use n-gram confidence as probability forever

Deferred. Confidence mode preserves prior AX behavior, but Lightning's delta
and greedy approaches may perform better for some low-temperature reasoning
workloads. The mode should be configurable and benchmarked.

## Consequences

### Positive

- Reduces n-gram regression risk when MTP is strong.
- Keeps n-gram available for workloads where it helps.
- Produces route metadata explaining gate behavior.
- Aligns AX with evidence-driven best practices observed in Lightning 0.7.0.

### Negative

- Adds more gate state and telemetry.
- Requires more focused tests.
- Thresholds are workload-sensitive and cannot be declared universal without
  benchmark evidence.
- Multiple acceptance modes increase audit surface.

### Risks

- False gating can suppress useful n-gram drafts early in a request.
- Counting predicted-but-suppressed drafts in self-tune can disable n-gram
  incorrectly.
- Delta n-gram probability can change rejection-sampling behavior relative to
  confidence mode.
- Benchmark rows where new gates never fire do not validate the gate logic.

## Validation Requirements

Before implementation is accepted:

- `cargo test -p ax-engine-mlx --quiet`
- `cargo clippy -p ax-engine-mlx --all-targets -- -D warnings`
- focused tests for each gate and acceptance mode;
- fair benchmark artifacts with MTP-only and MTP+ngram rows;
- route metadata showing whether each gate fired.

Code-level implementation acceptance is complete when the listed cargo gates
pass. Fair benchmark artifacts remain required before default promotion.

Before default promotion:

- root `cargo test --quiet --no-fail-fast`;
- root `cargo clippy --all-targets --all-features -- -D warnings`;
- benchmark report proving no material regression against MTP-only rows;
- explicit conclusion for each candidate default: keep, skip, or revert.

## Related

- `.internal/adr/ADR-008-mtp-ngram-stacking.md`
- `.internal/adr/ADR-013-mtp-optimization-phase2.md`
- `.internal/adr/ADR-015-mtp-decode-optimization.md`
- `.internal/adr/ADR-017-decode-speed-optimization.md`
- `.internal/prd/PRD-2026-06-02-mtp-ngram-lightning-learnings.md`
- `.internal/planning/TECH-SPEC-2026-06-02-mtp-ngram-gating-lightning-learnings.md`
