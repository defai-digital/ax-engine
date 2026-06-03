# PRD: MTP Stochastic Draft Sampling and Source-Aware N-gram Hurt Gate

**Status**: Implemented (Phases 1–4, tests; benchmark pending)
**Date**: 2026-06-03
**Scope**: `crates/ax-engine-mlx/src/mtp.rs`, `crates/ax-engine-mlx/src/runner.rs`
**Related ADR**: `.internal/adr/ADR-019-mtp-stochastic-draft-and-source-aware-gate.md`
**Related Tech Spec**: `.internal/planning/TECH-SPEC-2026-06-03-mtp-stochastic-draft-and-source-aware-gate.md`
**Supersedes-partial**: `.internal/adr/ADR-018-mtp-ngram-gating-lightning-learnings.md` D3 (hurt gate definition)

## Problem

Fresh `python_modules_long` artifacts in
`benchmarks/results/mtp-fair/2026-06-02-qwen36-fair-v0632/27b-4bit/` show two
unrelated regressions on AX Engine relative to MTPLX 0.3.7:

| Suite                | MTPLX tok/s / accept | AX (mtp) tok/s / accept | AX+ngram tok/s / accept |
|---------------------:|---------------------:|------------------------:|------------------------:|
| flappy               | 58.9 / 100.0%        | 65.2 / 98.2%            | 65.0 / 95.9%            |
| long_code            | 59.0 / 99.7%         | 63.4 / 95.3%            | 67.4 / 91.8%            |
| python_modules_long  | 54.4 / 87.6%         | 54.0 / 77.5%            | 52.1 / 73.5%            |

The `python_modules_long` row shows:

- **Pure MTP** loses ~10pp acceptance and ~0.4 tok/s on diverse Python module
  generation, but matches MTPLX on `flappy`/`long_code`.
- **MTP + n-gram** loses additional throughput vs pure MTP despite per-token
  n-gram acceptance (76.9-94.8% across runs) exceeding per-token MTP
  acceptance (65.7-72.4% across runs).

Source-code review against MTPLX (`.internal/reference/MTPLX/mtplx/sampling.py`,
`generation.py`) and Lightning-MLX
(`.internal/reference/lightning-mlx/vllm_mlx/speculative/native_mtp/sampling.py`)
identified two root causes:

### Root cause 1: AX uses greedy draft, MTPLX/Lightning use stochastic draft

`crates/ax-engine-mlx/src/mtp.rs:535-596` defines `mtp_draft_tokens_sampled`,
which despite the name uses `lazy_argmax_logits` for token selection regardless
of `head.draft_sampling.temperature`. The `_rng` parameter is prefixed `_` to
mark it intentionally unused. The function comment at `mtp.rs:530-532` asserts
this "matches how MTPLX and Lightning-MLX produce MTP drafts", but reference
code disagrees:

- MTPLX `generation.py:1990-1991`: `_sample_draft_from_logits` calls
  `_sample_from_logits` when `temperature > 0`, which does `mx.random.choice`
  over the top-p/top-k filtered distribution.
- Lightning `native_mtp/sampling.py:43-63`: `distribution_logprobs` applies
  `top_p`, `min_p`, `top_k` then temperature, then `sample_from_logprobs` does
  `mx.random.categorical`.
- MTPLX artifact (`v0632/27b-4bit/python_modules_long/mtplx.json`):
  `draft_sampling = {temperature: 0.7, top_p: 0.95, top_k: 20}` — stochastic.

When the MTP head's argmax disagrees with the main model's argmax, greedy AX is
forced to draft a low-`p_target` token and pays a guaranteed acceptance hit.
Stochastic sampling can land on a token the target prefers, recovering the
ratio. The gap is small on `flappy`/`long_code` because draft and target rank
agree most of the time, large on `python_modules_long` where Python boilerplate
introduces wider ranking disagreement.

### Root cause 2: Hurt gate compares biased EWMAs, not source-level rates

`runner.rs:7120-7133` defines `mtp_ngram_hurt_gate`:

```rust
combined_ewma < mtp_only_ewma - margin
```

Where `mtp_only_ewma` is updated by `record_step` at `runner.rs:1373-1394` only
when `mtp_only_drafted > 0`. The denominator excludes steps where:

- n-gram occupies a draft position before MTP, AND
- the first rejection happens at the n-gram position (so the MTP positions are
  never "meaningfully evaluated").

This creates a **selection bias**: `mtp_only_ewma` measures MTP acceptance
*conditional* on n-gram passing first. Meanwhile `combined_ewma` includes all
draft positions, including n-gram failures that contribute step_rate ≈ 0.

Symbolic example with per-token n-gram acceptance 0.77 and per-token MTP
acceptance 0.69 in a `[ngram, ngram, mtp, mtp, mtp]` hybrid step:

- `E[combined_step_rate] ≈ 0.449`
- `E[mtp_only_step_rate | updated] ≈ 0.534`
- Gap is 8.5pp > 2pp margin → hurt gate fires every step

Run 2 of the v0632 benchmark shows `ngram_hurt_gated_steps = 919` out of 1697
total steps (54%), with per-token n-gram acceptance 76.9% above per-token MTP
acceptance 69.1%. The gate is structurally biased toward firing even when
n-gram is genuinely helping.

ADR-018 D3 adopted this gate from Lightning-MLX's pattern, but the cascade
correction in `mtp_only_ewma` was designed to *exclude* unrelated rejections —
it inadvertently also excludes the exact cases where n-gram is most beneficial.

## Goals

- **G1**: Restore parity with MTPLX 0.3.7 on `python_modules_long` pure MTP
  acceptance, target ≥ 85%.
- **G2**: Prevent the hurt gate from suppressing n-gram when per-token n-gram
  acceptance exceeds per-token MTP acceptance.
- **G3**: Keep the gate effective at suppressing n-gram on workloads where
  n-gram is genuinely worse than MTP (no regression on `flappy`/`long_code`).
- **G4**: Preserve MTP rejection-sampling correctness for the target
  distribution.
- **G5**: Make draft sampling mode (greedy vs stochastic) explicit per request,
  not implicit.
- **G6**: Expose source-level acceptance counters in telemetry so future gate
  decisions can be made from raw counts.
- **G7**: Require benchmark evidence before default promotion of stochastic
  drafts or source-aware gating.

## Non-goals

- Changing the verify path or rejection-sampling acceptance criterion.
- Modifying MTP head weights or sidecar artifacts.
- Reimplementing MTPLX/Lightning sampling in mlx_lm/Python — AX stays native
  Rust.
- Removing the existing `mtp_ngram_hurt_gate` env knobs (kept as fallback).
- Reworking cycle guard, saturation gate, or auto-disable in this PRD
  (separate follow-up).
- Adding new MTP draft policies (delta/greedy n-gram acceptance modes are
  already covered by ADR-018 D6).

## Evidence Reviewed

### AX Engine code paths

- `crates/ax-engine-mlx/src/mtp.rs:33-50` — `gpu_draft_log_prob_lazy`
- `crates/ax-engine-mlx/src/mtp.rs:283-349` — `mtp_draft_tokens` dispatch
- `crates/ax-engine-mlx/src/mtp.rs:522-596` — `mtp_draft_tokens_sampled`
  (greedy despite the name)
- `crates/ax-engine-mlx/src/runner.rs:1333-1394` — `MtpTelemetry::record_step`
  with cascade correction
- `crates/ax-engine-mlx/src/runner.rs:6444-6465` — `mtp_ngram_gate_decision`
  call site
- `crates/ax-engine-mlx/src/runner.rs:7064-7152` — gate predicates
- `crates/ax-engine-mlx/src/runner.rs:7244-7320` — `compute_mtp_target_probs`
- `crates/ax-engine-mlx/src/runner.rs:7333-7439` — `mtp_accept_count` with
  temperature rescaling

### Reference projects

- `.internal/reference/MTPLX/mtplx/sampling.py:120-122` —
  `distribution_from_logits` applies filter then renormalizes
- `.internal/reference/MTPLX/mtplx/sampling.py:143-148` —
  `acceptance_probability` is the standard Leviathan/Chen ratio
- `.internal/reference/MTPLX/mtplx/generation.py:1971-1996` —
  `_sample_from_logits` and `_sample_draft_from_logits`
- `.internal/reference/lightning-mlx/vllm_mlx/speculative/native_mtp/sampling.py`
  — full mlx_lm filter chain + `mx.random.categorical`

### Benchmark artifacts

- `benchmarks/results/mtp-fair/2026-06-02-qwen36-fair-v0632/27b-4bit/python_modules_long/ax_engine.json`
- `benchmarks/results/mtp-fair/2026-06-02-qwen36-fair-v0632/27b-4bit/python_modules_long/ax_engine_ngram.json`
- `benchmarks/results/mtp-fair/2026-06-02-qwen36-fair-v0632/27b-4bit/python_modules_long/mtplx.json`

Per-run hurt-gate firings (3 runs aggregated across 3 `pymod_*` prompts):

| Run | decode_steps | hurt_gated | ngram accept | mtp accept |
|----:|-------------:|-----------:|-------------:|-----------:|
| 0   | 1382         | 444 (32%)  | 94.8%        | 72.4%      |
| 1   | 1545         | 496 (32%)  | 84.8%        | 65.7%      |
| 2   | 1697         | 919 (54%)  | 76.9%        | 69.1%      |

## Plan

### Phase 1: Add stochastic draft sampling path (greedy stays default)

Implement `mtp_draft_tokens_stochastic` alongside the existing greedy path. The
new path:

- Applies top-p/top-k filter to `softmax(logits / T_draft)` before sampling.
- Samples each depth's token with the per-request RNG.
- Computes `log_p_draft` on the same filtered+renormalized distribution.

Selection is per request, gated by env or sampler-config field. Greedy remains
default while benchmark evidence is collected.

### Phase 2: Match `compute_mtp_target_probs` to the draft distribution

When the draft path uses filter+temperature, the target path must also apply
the same filter (currently full-vocab only). This is required for correct
rejection sampling — `p_target / p_draft` must come from comparable
distributions.

Reuse the existing `topk` plumbing in `compute_mtp_target_probs`
(`runner.rs:7249-7320`) and extend it to support top-p when the draft path is
stochastic.

### Phase 3: Replace hurt gate with source-aware predicate

Add `mtp_ngram_source_hurt_gate` using the already-tracked source counters
(`accepted_source_ngram_tokens`, `draft_source_ngram_tokens`,
`accepted_source_mtp_tokens`, `draft_source_mtp_tokens`):

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
    if ngram_max == 0 || ngram_drafted < min_samples || mtp_drafted < min_samples {
        return false;
    }
    let ngram_rate = ngram_accepted as f32 / ngram_drafted.max(1) as f32;
    let mtp_rate = mtp_accepted as f32 / mtp_drafted.max(1) as f32;
    ngram_rate + margin < mtp_rate
}
```

Keep the existing EWMA-based hurt gate behind an env knob for rollback. Default
becomes the source-aware predicate.

### Phase 4: Telemetry

Extend telemetry with:

- `ax_mtp_ngram_source_hurt_gated_steps` — counter for the new gate.
- `ax_mtp_ngram_legacy_hurt_gated_steps` — rename of existing
  `ax_mtp_ngram_hurt_gated_steps` to make the comparison auditable.
- `ax_mtp_draft_mode` — `greedy` or `stochastic`, emitted per row.

### Phase 5: Benchmark evidence

Required rows before promoting any default:

- Pure MTP greedy vs stochastic on `flappy`, `long_code`,
  `python_modules_long`, on 27B-4bit and 35B-A3B-4bit.
- MTP + n-gram with legacy hurt gate vs source-aware hurt gate on the same
  suites.
- Acceptance rate must come from telemetry counters, not from summary numbers
  alone.

Promote stochastic draft as default only if:

- `python_modules_long` accept ≥ 85% on 27B-4bit (MTPLX baseline 87.6%);
- no regression > 2% on `flappy` or `long_code` acceptance or tok/s;
- per-depth GPU sync overhead from stochastic sampling does not drop tok/s
  below the greedy baseline.

Promote source-aware hurt gate as default only if:

- MTP+ngram tok/s ≥ MTP-only tok/s on `python_modules_long`;
- no regression > 2% on suites where the legacy gate did the right thing
  (`flappy`, `long_code`).

## Risks

- **Stochastic sampling adds per-depth GPU syncs**: temperature mode requires
  per-depth `eval` for CPU-side sampling. ADR-012 documented this as a 3-4×
  slowdown of the per-depth path. Mitigation: only enable stochastic when the
  acceptance gain outweighs latency.
- **Source-aware gate over-trusts early samples**: with `min_samples = 16` (the
  current EWMA gate's value), the new gate could fire too aggressively or too
  late. Mitigation: tune `min_samples` per artifact, fall back to the legacy
  gate via env.
- **Stochastic draft changes output distribution**: even though rejection
  sampling formally preserves the target distribution, in practice the path
  differs because greedy AX always produces argmax-equivalent text on accept.
  Mitigation: document in docs/PERFORMANCE.md and label artifacts explicitly.
- **MTP head argmax disagreement is artifact-specific**: the gap on
  `python_modules_long` may reflect this specific MTPLX sidecar's training,
  not a general property. Mitigation: also test on the `Quality` artifact.

## Rollback

All new behavior is env-gated:

- `AX_MLX_MTP_DRAFT_MODE=greedy` (default) restores the current draft path.
- `AX_MLX_MTP_NGRAM_HURT_GATE=legacy` (default while stabilizing) keeps the
  ADR-018 D3 gate.

No code revert is required to disable either change.

## References

- `.internal/adr/ADR-008-mtp-ngram-stacking.md`
- `.internal/adr/ADR-011-transactional-speculative-decode.md`
- `.internal/adr/ADR-012-mtp-fused-lazy-draft-skip-state.md`
- `.internal/adr/ADR-018-mtp-ngram-gating-lightning-learnings.md`
- `.internal/planning/MTP-MTPLX-OUTPERFORMANCE-PLAN-2026-05-28.md`
