# Tech Spec: MTP + N-gram Gating From Lightning-MLX 0.7.0 Learnings

**Status**: Implemented
**Date**: 2026-06-02
**PRD**: `.internal/prd/PRD-2026-06-02-mtp-ngram-lightning-learnings.md`
**ADR**: `.internal/adr/ADR-018-mtp-ngram-gating-lightning-learnings.md`

## Reviewed Sources

AX Engine:

- `crates/ax-engine-mlx/src/runner.rs`
- `crates/ax-engine-mlx/src/ngram_accel.rs`
- `benchmarks/results/mtp-fair/2026-06-02-qwen36-fair-ax-opt2/`

Lightning-MLX local reference:

- `v0.6.32..v0.7.0`
- `reports/exp/E2/verdict.md`
- `reports/exp/E4/verdict.md`
- `reports/exp/E8/verdict.md`
- `reports/exp/E9/verdict.md`
- `reports/exp/E12/verdict.md`
- `reports/exp/E13/verdict.md`
- `vllm_mlx/scheduler.py`
- `vllm_mlx/speculative/ngram_drafter.py`
- `vllm_mlx/speculative/native_mtp/sampling.py`
- `ec19b3d` post-tag streaming fix in `vllm_mlx/routes/chat.py` and
  `vllm_mlx/service/helpers.py`

## Implemented Design

### Phase 0: Document and freeze current semantics

Tests describe existing and new behavior:

- cascade-corrected `mtp_only_accept_rate_ewma`;
- n-gram saturation gate;
- n-gram pseudo log-prob behavior;
- route metadata emitted by `MtpTelemetry::append_route_decisions`.

This prevents Lightning-inspired changes from silently shifting semantics.

### Phase 1: Extract cached env helpers

Added helpers near existing env helpers in `runner.rs`:

```rust
fn mtp_ngram_hurt_margin() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_NGRAM_HURT_MARGIN")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(0.02)
    })
}
```

Add equivalent helpers for:

- `AX_MLX_MTP_NGRAM_AUTO_DISABLE_MTP_THRESHOLD`, default `0.85`;
- `AX_MLX_MTP_NGRAM_AUTO_DISABLE_MIN_NGRAM`, default `0.50`;
- `AX_MLX_MTP_NGRAM_SELF_TUNE_THRESHOLD`, default `0.30`;
- `AX_MLX_MTP_NGRAM_SELF_TUNE_WARMUP`, default `32`;
- `AX_MLX_MTP_NGRAM_AUTO_DISABLE_MTP_WARMUP`, default `64`;
- `AX_MLX_MTP_NGRAM_AUTO_DISABLE_NGRAM_WARMUP`, default `32`;
- `AX_MLX_MTP_NGRAM_ACCEPTANCE_MODE`, default `confidence` until benchmark
  evidence supports another default.

Use existing repository convention: parse once with `OnceLock`, default safe,
and disable with explicit env values where useful.

### Phase 2: Factor gate predicates

Gate checks are factored out of the decode loop into small pure helpers so they can be
unit-tested without constructing a full runner:

```rust
fn mtp_ngram_saturated_gate(
    ngram_max: usize,
    mtp_depth: usize,
    mtp_only_ewma: f32,
    mtp_only_samples: u32,
) -> bool
```

```rust
fn mtp_ngram_hurt_gate(
    ngram_max: usize,
    combined_ewma: f32,
    combined_samples: u32,
    mtp_only_ewma: f32,
    mtp_only_samples: u32,
    min_samples: u32,
    margin: f32,
) -> bool
```

```rust
fn mtp_ngram_auto_disable_gate(
    ngram_max: usize,
    mtp_drafted: u32,
    mtp_accepted: u32,
    ngram_drafted: u32,
    ngram_accepted: u32,
    cfg: MtpNgramAutoDisableConfig,
) -> bool
```

Return a bitset/enum so telemetry can distinguish:

```rust
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MtpNgramGateDecision {
    gated: bool,
    saturated: bool,
    hurt: bool,
    auto_disabled: bool,
    self_tune_disabled: bool,
}
```

### Phase 3: Fix self-tune accounting

Current WIP increments `ngram_self_tune_drafted` immediately after
`predict_with_policy`, before cycle-guard suppression. Move the drafted counter
increment to the branch where the n-gram draft is actually submitted:

```rust
if !ngram_outcome.draft.is_empty() && !ngram_cycle_guarded {
    let ngram_len = ngram_outcome.draft.len();
    state.ngram_self_tune.record_drafted(ngram_len);
    ...
}
```

Accepted count should be updated only in the feedback path after verifier
acceptance is known.

Represent request-local self-tune as a small struct:

```rust
#[derive(Default)]
struct NgramSelfTuneState {
    drafted: u32,
    accepted: u32,
    disabled: bool,
}
```

Methods:

- `record_submitted(drafted: usize)`
- `record_verified(accepted: usize)`
- `should_disable(threshold: f32, warmup: u32) -> bool`

### Phase 4: Preserve or explicitly switch n-gram acceptance mode

The high-risk WIP changed `mtp_ngram_pseudo_log_prob(confidence)` to return
`0.0` for any finite confidence. That semantic change was removed as the
default. Implemented explicit mode selection:

```rust
enum MtpNgramAcceptanceMode {
    Confidence,
    Delta,
    Greedy,
}
```

Mode behavior:

- `Confidence`: `ln(confidence.clamp(...))`, previous AX behavior.
- `Delta`: `0.0`, point-mass draft distribution.
- `Greedy`: bypass probability-ratio acceptance for n-gram-sourced positions
  and accept only if target argmax equals draft token.

Default is `Confidence`. `Delta` and `Greedy` require
`AX_MLX_MTP_NGRAM_ACCEPTANCE_MODE`.

### Phase 5: Telemetry

Extended `MtpTelemetry` with:

- `ngram_hurt_gated_steps`;
- `ngram_auto_disabled_steps`;
- `ngram_self_tune_disabled_steps`;
- optionally `ngram_submitted_tokens` and `ngram_submitted_accepted_tokens`.

Added route decision keys:

- `ax_mtp_ngram_hurt_gated_steps`
- `ax_mtp_ngram_auto_disabled_steps`
- `ax_mtp_ngram_self_tune_disabled_steps`
- `ax_mtp_ngram_submitted_tokens`
- `ax_mtp_ngram_submitted_accepted_tokens`
- `ax_mtp_ngram_acceptance_mode`

Existing sampled artifacts did not contain hurt-gate firings, so explicit
counters are required to prove whether the gate is active or irrelevant.

### Phase 6: Tests

Added focused tests in `runner.rs`:

- `mtp_ngram_hurt_gate_fires_when_combined_trails_mtp_only_by_margin`
- `mtp_ngram_hurt_gate_does_not_fire_before_min_samples`
- `mtp_ngram_hurt_gate_does_not_fire_inside_margin`
- `mtp_ngram_auto_disable_requires_both_mtp_strong_and_ngram_weak`
- `ngram_self_tune_counts_only_submitted_drafts`
- `ngram_self_tune_disables_after_warmup_when_acceptance_low`
- `mtp_ngram_pseudo_log_probs_cover_ngram_only_draft_windows`
- `mtp_ngram_pseudo_logprob_delta_mode_returns_zero`
- `mtp_accept_count_ngram_greedy_mode_uses_argmax_match`
- existing telemetry tests now assert new gate counters and acceptance mode.

### Phase 7: Benchmark evidence

Use fair artifacts with MTP-only and MTP+ngram rows:

```text
python3 scripts/bench_qwen36_mtp_fair.py \
  --models 27b-4bit 35b-a3b-4bit \
  --engines ax_engine ax_engine_ngram \
  --suites flappy long_code python_modules_long \
  --depth-policy native \
  --generation-tokens 1000 \
  --repetitions 5
```

Required evidence:

- each completed row has route metadata containing gate counters;
- MTP+ngram must not regress MTP-only by more than 2%;
- at least one row should exercise the new gate, or the report must explicitly
  state that the gate did not trigger;
- if `Delta` or `Greedy` n-gram mode is proposed as default, compare against
  `Confidence` mode on the same prompts.

## Known Current-Code Issues To Resolve

1. Hot-loop env parsing:
   - `AX_MLX_MTP_NGRAM_HURT_MARGIN`
   - `AX_MLX_MTP_NGRAM_AUTO_DISABLE_MTP_THRESHOLD`
   - `AX_MLX_MTP_NGRAM_AUTO_DISABLE_MIN_NGRAM`

2. Self-tune drafted count is updated before cycle-guard suppression.

3. Auto-disable uses `state.mtp_telemetry.accepted_tokens / draft_tokens`,
   which includes n-gram-source effects in the numerator/denominator. Consider
   whether MTP-only counters should drive the MTP side of the gate.

4. N-gram pseudo log-prob mode is not configurable.

5. New gate counters are incomplete: auto-disable and self-tune do not yet have
   explicit route counters in the WIP diff.

## Validation Commands

```text
cargo fmt --check
git diff --check
cargo test -p ax-engine-mlx --quiet
cargo clippy -p ax-engine-mlx --all-targets -- -D warnings
```

Before default promotion:

```text
cargo test --quiet --no-fail-fast
cargo clippy --all-targets --all-features -- -D warnings
```

## Rollback

Rollback must be possible without reverting code:

- set hurt margin/gate disable env to disable hurt gating;
- set auto-disable threshold to `0` to disable auto-disable;
- set self-tune warmup to a high value or explicit disable env;
- set n-gram acceptance mode back to `confidence`.
