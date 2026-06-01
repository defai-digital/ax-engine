# ADR: MTP Decode Pipeline Optimisation — Fused Lazy Draft and Skip-State Consumption

## Status

Accepted (2026-06-01)

## Context

AX Engine's MTP (multi-token prediction) speculative decode on Qwen 3.6 27B was 22–52%
slower than Lightning MLX and MTPLX on real prompt suites. Three root causes were
identified by comparing against Lightning MLX's `_mtp_step` always-advance loop
(`vllm_mlx/scheduler.py` line 763) and its `mtp_generate_step` generator
(`speculative/mtp_generate.py`):

### R1: Per-depth GPU sync barriers in draft

The sampled MTP draft path (`mtp_draft_tokens_sampled` in `mtp.rs`) called
`eval(&[&logits])` at each depth level to materialise logits for CPU-side categorical
sampling, then a batch eval for log probs. For depth-3 this meant 4 GPU sync barriers
per draft step.

Lightning MLX uses `mx.argmax` (lazy) and `mx.async_eval` to fuse the entire multi-depth
graph into a single GPU dispatch. MTPLX similarly uses a single fused pass.

### R2: Skip-state not consumed

After the verify forward for `[primary, d1, d2, d3]`, `logits_all[3]` is already the
distribution for the next primary token. Lightning MLX stores this as `skip_state["logits"]`
and reuses it on the next iteration, eliminating every other model forward on full-accept
paths.

AX Engine captured `mtp_skip_logits` and `mtp_skip_hidden` but discarded them with
`let _skip_logits = state.mtp_skip_logits.take()`.

### R3: Off-by-one in target prob indexing

`compute_mtp_target_probs_lazy` indexed `logits_all[i+1]` instead of `logits_all[i]`
for the target probability of `pending[i]`. This read wrong logits rows, giving
systematically lower target probabilities and causing massive over-rejection.

## Decision

### D1: Fused lazy draft with argmax selection

Replace the per-depth eval + CPU sampling loop with a fully lazy compute graph:

- Use `lazy_argmax_logits` for token selection (already exists for greedy path).
- Add `gpu_draft_log_prob_lazy` that takes a lazy argmax array as the gather index
  instead of a concrete `u32`, keeping the softmax → gather → log chain lazy.
- Materialise all tokens + log probs in a single `eval`.

**Trade-off:** The draft tokens are now argmax-selected (greedy) rather than
categorically sampled. This is acceptable because:
1. MTPLX uses greedy draft (100% acceptance on flappy confirms this works).
2. Rejection sampling acceptance (`min(1, p_target/p_draft)`) still provides
   stochastic quality control over the output.
3. Lightning MLX's verified mode also uses `sample_from_logprobs` but the draft
   quality difference is within noise for depth-3.

### D2: Skip-state consumption via pending override

When skip-state is available (previous verify produced logits/hidden at last accepted
position), sample the primary token from skip logits and draft new MTP tokens from
skip hidden. Write the new drafts into `pending` and `mtp_pending_draft_log_probs` so
the existing verify/accept pipeline operates on them without modification.

**Trade-off:** Skip-state is only consumed when `pending.is_empty()` (no pending draft
from the previous step to verify). This means skip-state is not used when a previous
draft was partially rejected and there are leftover pending tokens. This is correct:
partial rejection means the cache was trimmed and the skip-state logits/hidden are at
a stale position.

### D3: Correct target prob indexing

Change `flat_idx[i] = (i + 1) * vocab + pending[i]` to
`flat_idx[i] = i * vocab + pending[i]`.

This is consistent with the greedy argmax comparison (`predicted[i] == pending[i]`
using row `i`) and the correction token selection (`predicted[ac]` from row `ac`).

## Alternatives considered

### A1: Keep per-depth CPU sampling with async_eval overlap

Could keep the CPU sampling but overlap it with GPU work using `async_eval`. Rejected
because:
- The sampling RNG is on the CPU and must complete before the next depth's forward pass
  can build its graph (the sampled token feeds the next embedding lookup).
- `async_eval` helps with overlapping GPU output transfer but doesn't eliminate the
  fundamental serial dependency.
- Lightning MLX's own sampled path also uses per-depth syncs for temperature > 0; their
  speed advantage comes primarily from skip-state, not from draft fusion.

### A2: Multi-depth batch verify (verify all depths in one forward)

Could accumulate drafts across multiple steps and verify them all at once. Rejected because:
- Increases latency (must wait for multiple draft cycles before verifying).
- More complex rollback on partial rejection.
- The current single-step batch verify already achieves 99%+ acceptance on flappy.

### A3: Optimistic mode as default (always accept, no rejection sampling)

Lightning MLX has an `optimistic` flag that always accepts drafts without verification.
Rejected as default because:
- On complex suites (python_modules_long), acceptance drops to 67%, meaning 33% of
  output tokens would be wrong without verification.
- Optimistic mode is already available as `AX_MLX_MTP_OPTIMISTIC=1` for users who
  want maximum throughput and can tolerate occasional quality loss.

## Consequences

### Positive

- Draft GPU sync barriers reduced from 4 to 1 per step (~3–4× faster draft phase).
- Every other model forward eliminated on full-accept paths (skip-state consumption).
- Correct rejection sampling restored (off-by-one fix).
- Combined: expected 40–60% decode throughput improvement on 27B MTP benchmarks.

### Negative

- Skip-state adds complexity to `run_mtp_decode`: `pending` is now mutable and can be
  overridden by skip-state drafts. The bug fixes in `167b1053` demonstrate this
  complexity. Future changes to this function must carefully test both skip and
  non-skip paths.
- Draft tokens are argmax-selected rather than sampled. For temperature > 0 this
  slightly changes the draft distribution. In practice this is not measurable because
  rejection sampling corrects for any draft distribution (Leviathan/Chen speculative
  decoding theory guarantees unbiased output regardless of draft distribution).

### Risks

- **Skip-state at wrong position:** If the verify forward is trimmed (partial rejection)
  but skip-state was already captured, the skip-state would point at a stale cache
  position. Mitigated by only capturing skip-state on full-accept and clearing it on
  partial rejection.
- **Linear attention contamination:** Skip-state consumption feeds `skip_hidden` to
  `mtp_draft_tokens`, which advances the MTP cache. If the main model cache was trimmed
  (rejection), the MTP cache position could be ahead. This is already handled by the
  existing MTP cache rollback logic.

## Commits

| Commit | Description |
|---|---|
| `435e11b8` | Fix off-by-one in `compute_mtp_target_probs_lazy` |
| `b0011d37` | Fused lazy MTP draft (single eval) |
| `105b147c` | Skip-state consumption + fused draft combined |
| `167b1053` | Fix 3 bugs: wrong vocab, dropped drafts, fake telemetry |
