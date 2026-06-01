# ADR-012: MTP Fused Lazy Draft and Skip-State Consumption

**Status**: Active
**Date**: 2026-06-01

---

## Context

AX Engine's MTP speculative decode on Qwen 3.6 27B was 22–52% slower than Lightning MLX
and MTPLX on real prompt suites (May 2026 fair benchmark). The 35B-A3B model was already
leading. Three root causes were identified by comparing against Lightning MLX's
`_mtp_step` always-advance loop (`vllm_mlx/scheduler.py:763`):

1. **Per-depth GPU sync barriers in draft.** `mtp_draft_tokens_sampled` called
   `eval(&[&logits])` at each depth level for CPU categorical sampling, causing 4 GPU
   sync barriers for depth-3. Lightning uses a single fused lazy `argmax` + `async_eval`.

2. **Skip-state captured but never consumed.** After verify, logits at the last accepted
   position are the next primary token's distribution. Lightning stores these as
   `skip_state["logits"]` and reuses them on the next step. AX captured them in
   `mtp_skip_logits` but discarded them with `let _skip_logits = state.mtp_skip_logits.take()`.

3. **Off-by-one in rejection-sampling target probs.** `compute_mtp_target_probs_lazy`
   indexed `logits_all[i+1]` instead of `logits_all[i]`, causing systematic over-rejection
   of draft tokens and collapsing decode from ~40 tok/s to ~15 tok/s.

---

## Decision

Adopt two architectural changes to the MTP decode pipeline:

### D1: Fused lazy draft

Replace the per-depth eval + CPU sampling loop in `mtp_draft_tokens_sampled` with a
fully lazy compute graph:

- Use `lazy_argmax_logits` for token selection (same as greedy path).
- Add `gpu_draft_log_prob_lazy` that takes the lazy argmax array as the gather index,
  keeping the entire softmax → gather → log chain lazy.
- Materialise all depth tokens + log probs in a single `eval`.

**Design rules:**

- Draft tokens are argmax-selected regardless of temperature. Rejection sampling
  (`min(1, p_target/p_draft)`) provides stochastic quality control over output.
- `mtp_draft_tokens_sampled` must not contain any `eval` call inside the depth loop.
- The `gpu_draft_log_prob_lazy` function signature takes `&MlxArray` (lazy token) not
  `u32` (concrete token).

### D2: Skip-state consumption

When `mtp_skip_logits` and `mtp_skip_hidden` are available from a previous verify and
there is no pending draft to verify (`pending.is_empty()`):

1. Sample primary token from skip logits (no model forward).
2. Draft new MTP tokens from skip hidden.
3. Write new drafts into `pending`, `mtp_pending_draft_log_probs`, and
   `mtp_pending_draft_sources` so the existing verify/accept pipeline operates unchanged.

**Design rules:**

- Skip-state is only consumed when `pending.is_empty()`. Partial rejection produces stale
  skip-state positions.
- Skip-state must be cleared (`take()`) at the top of every `run_mtp_decode` call,
  regardless of whether it is consumed. Stale skip-state across iterations is a correctness
  hazard.
- The `pending` variable in `run_mtp_decode` is `let mut pending` to allow skip-state
  override.

### D3: Correct target prob indexing (already committed)

`flat_idx[i] = i * vocab + pending[i]`, not `(i+1) * vocab + pending[i]`.
Consistent with greedy argmax comparison and correction token selection.

---

## Validation

- `cargo test -p ax-engine-mlx` — 482 tests pass (unchanged).
- `cargo clippy -p ax-engine-mlx --all-targets -- -D warnings` — clean.
- `scripts/bench_qwen36_mtp_fair.py --models 27b-4bit --suites flappy long_code python_modules_long`
  — post-fix results should show ≥1.0× vs MTPLX on flappy/long_code.

---

## Consequences

- Draft GPU sync barriers: 4 → 1 per step (~3–4× faster draft phase).
- Model forwards: halved on full-accept paths (skip-state consumption).
- Correct rejection sampling restored (off-by-one fix).
- Combined: expected 40–60% decode throughput improvement on 27B benchmarks.

---

## Rejected Alternatives

- **Per-depth CPU sampling with async_eval overlap**: Sampling RNG is serial (each depth's
  token feeds the next embedding lookup). `async_eval` cannot break this dependency.
- **Multi-step batch verify**: Accumulating drafts across multiple steps increases latency
  and complicates rollback. Single-step batch verify already achieves 99%+ on flappy.
- **Optimistic mode as default**: Drops rejection sampling entirely; 33% wrong tokens on
  python_modules_long is unacceptable. Already available as `AX_MLX_MTP_OPTIMISTIC=1`.

---

## Commits

| Commit | Description |
|---|---|
| `435e11b8` | Fix off-by-one in `compute_mtp_target_probs_lazy` |
| `b0011d37` | Fused lazy MTP draft with `gpu_draft_log_prob_lazy` |
| `105b147c` | Skip-state consumption |
| `167b1053` | Fix 3 bugs: vocab=1, dropped drafts, fake telemetry |
| `7a8eff69` | Move PRD/ADR to `.internal/` |
