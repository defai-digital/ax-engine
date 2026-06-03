# ADR-019: MTP Stochastic Draft Sampling and Source-Aware N-gram Hurt Gate

**Status**: Accepted (runtime implemented, benchmark validation pending)
**Date**: 2026-06-03
**Amends**: `.internal/adr/ADR-018-mtp-ngram-gating-lightning-learnings.md` D3
**PRD**: `.internal/prd/PRD-2026-06-03-mtp-stochastic-draft-and-source-aware-gate.md`
**Tech Spec**: `.internal/planning/TECH-SPEC-2026-06-03-mtp-stochastic-draft-and-source-aware-gate.md`

## Context

Two diagnostic findings on the v0632 `python_modules_long` artifact require
amending the active MTP draft and n-gram gate policy:

1. **AX uses greedy MTP draft, MTPLX and Lightning use stochastic.** The
   function `mtp_draft_tokens_sampled` (`crates/ax-engine-mlx/src/mtp.rs:535-596`)
   uses `lazy_argmax_logits` regardless of `head.draft_sampling.temperature`.
   The function comment claims this matches MTPLX/Lightning. The
   `.internal/reference/MTPLX/mtplx/generation.py:1990-1991` and
   `.internal/reference/lightning-mlx/vllm_mlx/speculative/native_mtp/sampling.py`
   show both reference projects use `mx.random.categorical` /
   `sample_from_distribution` over a top-p/top-k filtered distribution when
   `temperature > 0`. The MTPLX artifact recommends
   `temperature=0.7, top_p=0.95, top_k=20`.

   On suites where the MTP head's argmax matches the main model's argmax
   (`flappy`, `long_code`), greedy and stochastic perform identically. On
   `python_modules_long`, where rank disagreement is more common, AX loses
   ~10pp acceptance because greedy cannot recover when the MTP head's argmax
   has low `p_target`.

2. **The hurt gate compares biased EWMAs, not source rates.** ADR-018 D3
   adopted `combined_ewma < mtp_only_ewma - margin` from Lightning. The
   `mtp_only_accept_rate_ewma` update path
   (`crates/ax-engine-mlx/src/runner.rs:1373-1394`) only fires when an MTP
   position was "meaningfully evaluated" — which excludes the cases where
   n-gram fails at position 0 (the exact failure mode the gate should detect
   *or* tolerate). The gate therefore fires structurally, not only in true
   regression scenarios. Run 2 of the v0632 artifact records
   `ngram_hurt_gated_steps = 919 / 1697 = 54%` while per-token n-gram
   acceptance (76.9%) exceeds per-token MTP acceptance (69.1%) — the gate
   suppressed a genuinely helpful n-gram path for more than half the run.

These findings do not invalidate ADR-018's overall direction (hurt-detect
n-gram, expose telemetry, keep evidence-driven defaults). They invalidate the
specific predicate adopted in D3 and add a previously missing MTP draft
behavior.

## Decision

### D1: Add a stochastic MTP draft path; keep greedy as default

Implement `mtp_draft_tokens_stochastic` in
`crates/ax-engine-mlx/src/mtp.rs`, dispatched from `mtp_draft_tokens` when
`MtpDraftMode::Stochastic` is selected. Mode source priority:

1. `AX_MLX_MTP_DRAFT_MODE` env, cached via `OnceLock`.
2. `head.draft_sampling.mode` field if a future manifest carries it.
3. Default: greedy.

Stochastic path applies the same `top_p`/`top_k`/`temperature` filter chain as
`.internal/reference/lightning-mlx/vllm_mlx/speculative/native_mtp/sampling.py`,
then samples with the per-request `Xorshift64` RNG. `log_p_draft` is computed
on the same filtered+renormalized distribution.

Greedy remains default until benchmark evidence on AX-specific artifacts
proves stochastic is a strict improvement.

### D2: Filter the target distribution when the draft is filtered

`compute_mtp_target_probs` (`runner.rs:7244-7320`) must apply the same
top-p/top-k filter as the draft path when `MtpDraftMode::Stochastic` is
active. This is required for rejection-sampling correctness — the
Leviathan/Chen accept rule `min(1, p_target / p_draft)` only preserves the
target distribution when both probabilities come from the same distribution
family.

When greedy is active, target prob stays full-vocab (current behavior, env
override via `AX_MLX_MTP_TARGET_SOFTMAX_MODE` preserved).

### D3: Replace the EWMA-based hurt gate with a source-aware predicate

Amends ADR-018 D3. New predicate:

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
    let mtp_rate   = mtp_accepted   as f32 / mtp_drafted.max(1)   as f32;
    ngram_rate + margin < mtp_rate
}
```

Counters are already tracked
(`draft_source_ngram_tokens`, `accepted_source_ngram_tokens`,
`draft_source_mtp_tokens` + hybrid variants). The predicate fires when n-gram
per-token acceptance is worse than MTP per-token acceptance by more than the
margin — the exact condition where n-gram is genuinely hurting.

The legacy EWMA-based gate remains available via
`AX_MLX_MTP_NGRAM_HURT_GATE=legacy` for rollback and A/B benchmarking. The
source-aware gate becomes default once a fair benchmark row proves no
regression on `flappy`, `long_code`, or `python_modules_long`.

### D4: Keep all other ADR-018 decisions

ADR-018 D1, D2, D4, D5, D6 are unchanged:

- D1 cascade-corrected `mtp_only_accept_rate_ewma` is still emitted as
  telemetry. Its role shifts from gate input to observability only.
- D2 saturation gate stays.
- D4 auto-disable stays.
- D5 per-request self-tune stays.
- D6 explicit n-gram acceptance mode stays.

### D5: Telemetry must distinguish gate modes and draft modes

New route decision keys:

- `ax_mtp_draft_mode` — 0 greedy, 1 stochastic.
- `ax_mtp_hurt_gate_mode` — 0 source, 1 legacy.
- `ax_mtp_ngram_source_hurt_gated_steps` — counter for the new gate.
- `ax_mtp_ngram_legacy_hurt_gated_steps` — counter for ADR-018 D3 gate.

The existing `ax_mtp_ngram_hurt_gated_steps` key continues to be emitted as
the union of whichever mode is active, to avoid breaking downstream readers.

## Design Rules

1. **Sampling parity for rejection sampling**: when draft uses
   `top_p/top_k/T_draft`, target probability must also use the same filter
   before the `min(1, p_target/p_draft)` test. Skipping this breaks the
   distributional guarantee of speculative decoding.
2. **Greedy is the safe fallback**: it must remain selectable via env
   regardless of manifest hints, because per-depth GPU sync cost can dominate
   acceptance gains on small batches.
3. **Source counters are the authoritative gate input**: EWMA values are
   useful for observability but not for binary decisions when their
   denominators carry selection bias.
4. **No silent default promotion**: stochastic draft and source-aware gate
   stay opt-in until at least one fair benchmark row per suite proves no
   regression and at least one row proves the intended gain.
5. **Telemetry must declare mode**: every artifact must record `draft_mode`
   and `hurt_gate_mode` so historical comparisons stay sound when defaults
   later change.

## Validation

Code-level acceptance (must pass before merge):

- `cargo fmt --check`
- `cargo test -p ax-engine-mlx --quiet`
- `cargo clippy -p ax-engine-mlx --all-targets -- -D warnings`
- Focused tests:
  - `mtp_draft_mode_env_selects_stochastic_path`
  - `mtp_draft_stochastic_respects_top_k`
  - `mtp_draft_stochastic_log_prob_matches_filtered_distribution`
  - `mtp_ngram_source_hurt_gate_does_not_fire_when_ngram_better_than_mtp`
  - `mtp_ngram_source_hurt_gate_fires_when_ngram_worse_than_mtp`
  - `mtp_ngram_hurt_gate_mode_env_selects_predicate`

Benchmark-level acceptance (required before default promotion):

- `bench_qwen36_mtp_fair.py` with `--repetitions 5` covering
  `{flappy, long_code, python_modules_long}` ×
  `{greedy, stochastic}` × `{legacy, source}` on 27B-4bit.
- Route metadata in each artifact records `ax_mtp_draft_mode` and
  `ax_mtp_hurt_gate_mode`.
- `python_modules_long` pure-MTP accept ≥ 85% under stochastic.
- `python_modules_long` MTP+ngram tok/s ≥ MTP-only tok/s under source-aware
  gate.
- No suite regresses tok/s or accept by more than 2% under either change.

## Consequences

### Positive

- Closes the ~10pp acceptance gap vs MTPLX on diverse-content suites without
  changing the verify path or sidecar weights.
- N-gram stacking can deliver its measured per-token acceptance gain instead
  of being structurally suppressed.
- Telemetry distinguishes draft mode and gate mode, enabling clean A/B
  comparisons across artifacts.
- Source-aware gate aligns the binary decision with the metric the original
  gate intent described ("n-gram is hurting").

### Negative

- Stochastic draft adds per-depth GPU sync (ADR-012 documented this as a
  3-4× per-depth path slowdown vs fused greedy). Net tok/s benefit depends on
  per-suite acceptance gain.
- Two gate predicates increase audit surface during the legacy-to-source
  transition window.
- Tightening rejection-sampling parity (D2) means the target softmax topk env
  knob is no longer free to set independently in stochastic mode.

### Risks

- The 0.02 margin inherited from the legacy gate may not be the right value
  for source-level rates. Benchmark-driven tuning is required.
- Xorshift64 RNG ≠ MTPLX NumPy RNG, so stochastic AX outputs will not match
  MTPLX bit-identically even on same seed. Artifacts must avoid implying
  bit-identity.
- Per-prompt EWMA scale audit (`ax_mtp_accept_rate_ewma_x1000` shows values
  > 1000 in v0632 artifacts despite tests asserting 0..=1) is a separate
  follow-up. The source-aware gate avoids this scale by using raw counters.

## Rejected Alternatives

### A1: Apply top-p/top-k filter to target prob only

Rejected. Filtering target without filtering draft breaks the
`p_target / p_draft` ratio for tokens at the boundary of the filter — those
where draft assigns positive probability but target filtered-distribution
assigns zero. The result is artificially low acceptance, not a fix.

### A2: Switch to stochastic and tune temperature to recover greedy behavior

Rejected. Temperature-zero stochastic is a known degenerate case (the
reference codes special-case it back to argmax). Keeping greedy as a
first-class mode is cleaner than emulating it.

### A3: Replace `mtp_only_accept_rate_ewma` instead of replacing the gate

Considered, rejected for this ADR. Repairing the EWMA would require counting
n-gram-prefix failures as MTP-evaluation events with `mtp_only_rate = 0`,
which conflates "MTP failed" with "n-gram failed before MTP ran". The signal
becomes ambiguous. Source-level rates are unambiguous and already available.

### A4: Remove the hurt gate entirely

Rejected. Workloads where n-gram is genuinely worse than MTP still exist
(short non-repetitive outputs). The gate must remain — only the predicate
changes.

### A5: Make stochastic the default before benchmark evidence

Rejected per ADR-018 D7. Lightning 0.7.0's release policy was the explicit
template: defaults change only with measured wins.

### A6: Fix `ax_mtp_accept_rate_ewma_x1000` scale in this ADR

Out of scope. The metric scale anomaly does not affect any decision in this
ADR (which uses raw counters). Tracked as a separate observability hygiene
follow-up.

## Related

- `.internal/adr/ADR-008-mtp-ngram-stacking.md`
- `.internal/adr/ADR-011-transactional-speculative-decode.md`
- `.internal/adr/ADR-012-mtp-fused-lazy-draft-skip-state.md`
- `.internal/adr/ADR-018-mtp-ngram-gating-lightning-learnings.md`
- `.internal/prd/PRD-2026-06-03-mtp-stochastic-draft-and-source-aware-gate.md`
- `.internal/planning/TECH-SPEC-2026-06-03-mtp-stochastic-draft-and-source-aware-gate.md`
- `.internal/planning/MTP-MTPLX-OUTPERFORMANCE-PLAN-2026-05-28.md`
