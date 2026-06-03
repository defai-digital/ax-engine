# Tech Spec: MTP Stochastic Draft Sampling and Source-Aware N-gram Hurt Gate

**Status**: Implemented (Phases 1–5; Phase 6 benchmark pending model hardware)
**Date**: 2026-06-03
**PRD**: `.internal/prd/PRD-2026-06-03-mtp-stochastic-draft-and-source-aware-gate.md`
**ADR**: `.internal/adr/ADR-019-mtp-stochastic-draft-and-source-aware-gate.md`

## Reviewed Sources

AX Engine:

- `crates/ax-engine-mlx/src/mtp.rs:1-596`
- `crates/ax-engine-mlx/src/runner.rs:1300-1394` (MtpTelemetry, EWMA update)
- `crates/ax-engine-mlx/src/runner.rs:6420-6515` (decode-loop n-gram gate site)
- `crates/ax-engine-mlx/src/runner.rs:7064-7152` (gate predicates)
- `crates/ax-engine-mlx/src/runner.rs:7244-7439` (target probs, accept count)

MTPLX:

- `.internal/reference/MTPLX/mtplx/sampling.py:15-148`
- `.internal/reference/MTPLX/mtplx/generation.py:1971-2017`

Lightning-MLX:

- `.internal/reference/lightning-mlx/vllm_mlx/speculative/native_mtp/sampling.py`
- `.internal/reference/lightning-mlx/vllm_mlx/scheduler.py:860-1010` (MTP draft
  + verify loop)
- `.internal/reference/lightning-mlx/vllm_mlx/cli.py:160-170` (default
  `mtp_draft_temperature = 0.5`)

Benchmark artifacts:

- `benchmarks/results/mtp-fair/2026-06-02-qwen36-fair-v0632/27b-4bit/python_modules_long/{ax_engine,ax_engine_ngram,mtplx}.json`

## Findings That Drive This Spec

### F1: `mtp_draft_tokens_sampled` is actually greedy

`crates/ax-engine-mlx/src/mtp.rs:572` calls `lazy_argmax_logits(&logits)`. The
`_rng` parameter at line 544 is prefixed `_` to suppress the unused-variable
warning. The function is dispatched at `mtp.rs:323-337` when
`head.draft_sampling.temperature > 0`, but produces argmax tokens.

Code comment at `mtp.rs:530-532`:

> The argmax-selected tokens are deterministic for a given hidden state,
> matching how MTPLX and Lightning-MLX produce MTP drafts.

This is incorrect for the MTPLX 0.3.7 and Lightning 0.7.0 versions tracked in
`.internal/reference`. Both use stochastic top-p/top-k sampling when
`temperature > 0`.

### F2: `mtp_only_accept_rate_ewma` has selection bias

`runner.rs:1373-1394`:

```rust
let mtp_only_accepted_count = sources
    .iter()
    .take(accepted)
    .filter(|s| matches!(s, MtpDraftSource::Mtp | MtpDraftSource::HybridMtp))
    .count();
let first_rejection_is_mtp = accepted < drafted
    && sources
        .get(accepted)
        .map(|s| matches!(s, MtpDraftSource::Mtp | MtpDraftSource::HybridMtp))
        .unwrap_or(true);
let mtp_only_drafted = mtp_only_accepted_count + usize::from(first_rejection_is_mtp);
if mtp_only_drafted > 0 {
    // update mtp_only_accept_rate_ewma
}
```

The intent was to exclude cascade rejections caused by n-gram failure (correct
per ADR-018 D1). The side effect is that steps where n-gram fails at position 0
never update `mtp_only_accept_rate_ewma`. Meanwhile `accept_rate_ewma`
(combined) updates with `step_rate = 0`.

Numerical example with n-gram per-token = 0.77, MTP per-token = 0.69, hybrid
step `[ngram, ngram, mtp, mtp, mtp]`:

| Outcome      | P     | combined contrib | mtp_only contrib | mtp_only updates? |
|:-------------|------:|-----------------:|-----------------:|:------------------|
| reject at 0  | 0.230 | 0.0              | —                | no                |
| reject at 1  | 0.177 | 0.2              | —                | no                |
| reject at 2  | 0.184 | 0.4              | 0.0              | yes (1/1)         |
| reject at 3  | 0.127 | 0.6              | 0.5              | yes (1/2)         |
| reject at 4  | 0.087 | 0.8              | 0.667            | yes (2/3)         |
| accept all   | 0.195 | 1.0              | 1.0              | yes (3/3)         |

E[combined_step_rate] = 0.449
E[mtp_only_step_rate \| updated] = 0.534

`combined < mtp_only - 0.02` → hurt gate fires.

### F3: `python_modules_long` v0632 telemetry

Run 2 of `ax_engine_ngram.json` (3 prompts aggregated):

- `decode_steps`: 1697
- `ngram_hurt_gated_steps`: 919 (54.2% of steps)
- `accepted_source_ngram_tokens / draft_source_ngram_tokens`: 2328/3367 (this
  appears as MTP-source — check counter ordering)
- `ngram_submitted_accepted_tokens / ngram_submitted_tokens`: 897/1167 (76.9%)
- `accepted_source_mtp_tokens / draft_source_mtp_tokens`: 2328/3367 (69.1%)
- `accept_rate_ewma_x1000`: 4047 (raw, scale TBD — see F4)
- `mtp_only_accept_rate_ewma_x1000`: 3837 (raw, scale TBD)

### F4: `ax_mtp_accept_rate_ewma_x1000` scale is suspicious

`runner.rs:1755-1758`:

```rust
decisions.upsert_route_decision(
    "ax_mtp_accept_rate_ewma_x1000",
    (self.accept_rate_ewma * 1000.0) as u32,
);
```

`accept_rate_ewma` is asserted in unit tests (`runner.rs:12858, 12877`) to be
in 0..=1. Multiplying by 1000 should yield 0..=1000. Observed JSON values are
4047, 4239, 4372, 4407 — all > 1000.

This is not load-bearing for the source-aware gate work (which uses raw
counters), but the metric name and emission should be re-audited as a separate
hygiene task. Suspect path: route_decisions across multiple prompts in the same
run are not deduplicated as expected, or the metric name has drifted from the
emission code.

## Implementation Plan

### Phase 0: Document current semantics with tests

Add tests asserting the existing greedy-with-temperature-rescale behavior and
the EWMA-based hurt gate semantics, so the new path can be A/B'd without
silent drift. Tests live in `crates/ax-engine-mlx/src/runner.rs` near the
existing `mtp_ngram_hurt_gate_*` tests.

New tests:

- `mtp_draft_tokens_sampled_is_currently_greedy` — asserts the documented
  intent matches reality (so future renames don't silently flip behavior).
- `mtp_ngram_hurt_gate_fires_under_selection_bias_scenario` — reproduces F2
  numerically.

### Phase 1: Add `mtp_draft_tokens_stochastic`

New function in `mtp.rs`, parallel to `mtp_draft_tokens_sampled`:

```rust
fn mtp_draft_tokens_stochastic(
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    vocab: i32,
    rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3])
```

Behavior per depth:

1. Forward MTP head, compute logits `[1, vocab]`.
2. Apply `top_p`/`top_k` from `head.draft_sampling`:
   - `softmax(logits / T_draft)` → `probs`.
   - Filter to top-k indices (when `top_k > 0`).
   - Within top-k, take cumulative sum sorted descending, keep tokens whose
     cumulative ≤ `top_p`, plus the first token that crosses.
   - Renormalize across the kept set → `filtered_probs`.
3. Sample one token from `filtered_probs` using the request RNG.
4. Compute `log_p_draft = log(filtered_probs[sampled_token])`.

Filter implementation lives in `crate::sampling` (existing CPU sampler
already supports top-p/top-k for the verify-pass tail token; reuse the same
helper if shapes allow).

Per-depth eval is unavoidable because sampling is CPU-side. ADR-012 documented
this as a known constraint of temperature-mode drafting; the lazy/fused path
remains available for greedy mode.

Dispatch in `mtp_draft_tokens` (`mtp.rs:306-349`) selects:

- `MtpDraftMode::Greedy` (current behavior, default).
- `MtpDraftMode::Stochastic` (new path, opt-in).

Mode source priority:

1. `AX_MLX_MTP_DRAFT_MODE` env (cached via `OnceLock`).
2. `head.draft_sampling.mode` field if the manifest carries it (future).
3. Fall back to greedy.

### Phase 2: Filter target distribution to match draft path

When draft mode is stochastic, `compute_mtp_target_probs`
(`runner.rs:7244-7320`) must apply the same top-p/top-k filter as the draft
distribution. Currently it supports a `topk` parameter from
`AX_MLX_MTP_TARGET_SOFTMAX_MODE` (full-vocab default), but no `top_p`.

Extend the function signature:

```rust
fn compute_mtp_target_probs(
    logits_all: &MlxArray,
    pending: &[u32],
    pending_log_probs: &[f32],
    vocab: i32,
    target_sampling: MlxSamplingParams,
    draft_filter: MtpDraftFilter,  // new
    workspace: &mut MtpTargetProbWorkspace,
) -> Option<LazyTargetProbs>
```

`MtpDraftFilter`:

```rust
struct MtpDraftFilter {
    top_p: f32,
    top_k: u32,
}
```

When `top_k > 0` or `top_p < 1.0`, take the existing top-k branch and add a
top-p cumulative-mask step. Reuse `argpartition_axis`/`take_along_axis` for the
top-k path already implemented at `runner.rs:7269-7297`.

When the draft path is greedy, pass `MtpDraftFilter::IDENTITY` (top_k = 0,
top_p = 1.0) to preserve existing behavior.

### Phase 3: Add source-aware hurt gate

New predicate in `runner.rs` next to `mtp_ngram_hurt_gate`:

```rust
fn mtp_ngram_source_hurt_gate(
    ngram_max: usize,
    mtp_drafted: u32,
    mtp_accepted: u32,
    ngram_drafted: u32,
    ngram_accepted: u32,
    min_samples: u32,
    margin: f32,
) -> bool {
    if ngram_max == 0
        || ngram_drafted < min_samples
        || mtp_drafted < min_samples
    {
        return false;
    }
    let ngram_rate = ngram_accepted as f32 / ngram_drafted.max(1) as f32;
    let mtp_rate = mtp_accepted as f32 / mtp_drafted.max(1) as f32;
    ngram_rate + margin < mtp_rate
}
```

Counters already exist:

- `state.mtp_telemetry.draft_source_ngram_tokens`
- `state.mtp_telemetry.accepted_source_ngram_tokens`
- `state.mtp_telemetry.draft_source_mtp_tokens` (plus hybrid_mtp)
- `state.mtp_telemetry.accepted_source_mtp_tokens` (plus hybrid_mtp)

Hook into `mtp_ngram_gate_decision` (`runner.rs:7064-7107`):

```rust
fn mtp_ngram_gate_decision(...) -> MtpNgramGateDecision {
    let saturated = mtp_ngram_saturated_gate(...);
    let hurt = match mtp_ngram_hurt_gate_mode() {
        HurtGateMode::SourceAware => mtp_ngram_source_hurt_gate(...),
        HurtGateMode::LegacyEwma  => mtp_ngram_hurt_gate(...),
    };
    let auto_disabled = mtp_ngram_auto_disable_gate(...);
    ...
}
```

Mode selection via `AX_MLX_MTP_NGRAM_HURT_GATE` env (`source` default after
benchmark evidence, `legacy` for rollback).

### Phase 4: Telemetry

Add to `MtpTelemetry`:

- `ngram_source_hurt_gated_steps: u32`
- `ngram_legacy_hurt_gated_steps: u32`
- (keep existing `ngram_hurt_gated_steps` as alias of whichever gate is
  active, to avoid breaking existing dashboards)

Add to route decisions:

- `ax_mtp_ngram_source_hurt_gated_steps`
- `ax_mtp_ngram_legacy_hurt_gated_steps`
- `ax_mtp_draft_mode` — string-encoded: 0=greedy, 1=stochastic
- `ax_mtp_hurt_gate_mode` — 0=source, 1=legacy

Re-audit `ax_mtp_accept_rate_ewma_x1000` (see F4) as a separate hygiene
follow-up. Out of scope for this spec, but a TODO comment is added at
`runner.rs:1757`:

```rust
// TODO(2026-06-03): observed values exceed 1000 in v0632 artifacts.
// See PRD-2026-06-03 F4. accept_rate_ewma is asserted 0..=1 in tests;
// emission scale or aggregation path has drifted.
```

### Phase 5: Tests

Add to `runner.rs::tests`:

- `mtp_ngram_source_hurt_gate_does_not_fire_when_ngram_better_than_mtp`
- `mtp_ngram_source_hurt_gate_fires_when_ngram_worse_than_mtp`
- `mtp_ngram_source_hurt_gate_respects_min_samples`
- `mtp_ngram_source_hurt_gate_respects_margin`
- `mtp_ngram_hurt_gate_mode_env_selects_predicate`

Add to `mtp.rs::tests` (or `runner.rs` if MtpWeights mocks are easier there):

- `mtp_draft_mode_env_selects_stochastic_path`
- `mtp_draft_stochastic_respects_top_k`
- `mtp_draft_stochastic_uses_request_rng`
- `mtp_draft_stochastic_log_prob_matches_filtered_distribution`

### Phase 6: Benchmark evidence

Re-run with:

```text
python3 scripts/bench_qwen36_mtp_fair.py \
  --models 27b-4bit \
  --engines ax_engine ax_engine_ngram \
  --suites flappy long_code python_modules_long \
  --depth-policy native \
  --generation-tokens 1000 \
  --repetitions 5
```

For each cell, vary:

- `AX_MLX_MTP_DRAFT_MODE` ∈ {`greedy`, `stochastic`}
- `AX_MLX_MTP_NGRAM_HURT_GATE` ∈ {`legacy`, `source`}

Acceptance criteria for default promotion:

| Default change                | Criterion                                                |
|:------------------------------|:---------------------------------------------------------|
| stochastic draft as default   | `python_modules_long` accept ≥ 85%, no row regresses > 2% on tok/s or accept |
| source-aware hurt as default  | MTP+ngram tok/s ≥ MTP-only tok/s on `python_modules_long`, no regression > 2% on `flappy`/`long_code` |

Document each cell with route metadata showing draft mode and gate mode.

## Known Risks

1. **Per-depth GPU sync cost**: stochastic mode adds 2-3 GPU syncs per draft
   step (one per depth). Benchmark must measure tok/s, not just accept rate.
2. **Source-aware gate margin tuning**: 0.02 margin inherited from legacy gate
   may be wrong for source-level rates (which are typically further apart than
   step-level rates). Likely needs benchmark-driven tuning.
3. **`top_p` semantics in target prob**: filtering the target distribution
   changes the rejection-sampling target. The Leviathan/Chen proof requires
   the same distribution on both sides — keep parity rigorous.
4. **MTPLX sidecar mismatch**: stochastic AX may diverge from MTPLX outputs on
   identical prompts even with same seed, because Xorshift64 ≠ NumPy RNG. AX
   artifacts must be labeled `seed=ax-rng` to avoid implying bit-identity.

## Validation Commands

```text
cargo fmt --check
cargo test -p ax-engine-mlx --quiet
cargo clippy -p ax-engine-mlx --all-targets -- -D warnings
```

Before default promotion:

```text
cargo test --quiet --no-fail-fast
cargo clippy --all-targets --all-features -- -D warnings
bash scripts/check-mlx-telemetry.sh
```

## Rollback

All new behavior is env-gated. To rollback in production without code revert:

- `AX_MLX_MTP_DRAFT_MODE=greedy`
- `AX_MLX_MTP_NGRAM_HURT_GATE=legacy`

Both restore pre-PRD-2026-06-03 behavior exactly.
