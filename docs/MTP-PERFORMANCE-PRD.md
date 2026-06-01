# MTP Decode Performance Improvement PRD

## Background

AX Engine MTP (multi-token prediction) speculative decode on the 27B Qwen 3.6 model
was 22–52% slower than Lightning MLX and MTPLX on the flappy/long_code/python_modules_long
prompt suites (May 2026 `bench_qwen36_mtp_fair.py` results). The 35B-A3B model was already
competitive.

Root-cause analysis identified three architectural gaps against Lightning MLX's
`_mtp_step` always-advance loop (`vllm_mlx/scheduler.py`):

1. **Per-depth GPU sync barriers** in the MTP draft path. Each depth level called
   `eval(&[&logits])` to materialise logits for CPU-side categorical sampling, causing
   3 GPU sync barriers for depth-3 plus a batch eval for log probs (4 total). Lightning
   uses a single fused `argmax` + `async_eval`.

2. **No skip-state consumption.** After verify, logits at the last accepted position
   are already the next primary token's distribution. Lightning reuses these on the next
   step (`skip_state["logits"]`), eliminating every other model forward. AX Engine captured
   skip-state but never consumed it.

3. **Off-by-one in rejection-sampling target probs.** `compute_mtp_target_probs_lazy`
   indexed `logits_all[i+1]` instead of `logits_all[i]`, reading wrong positions and
   causing systematic over-rejection of MTP draft tokens.

## Goals

- **G1** Match or exceed Lightning MLX MTP decode throughput on the 27B 4-bit fair
  benchmark suite (flappy, long_code, python_modules_long).
- **G2** Maintain or improve 35B-A3B decode throughput (already leading).
- **G3** Preserve acceptance rate parity (≥95% on flappy/long_code).
- **G4** No correctness regression: all existing unit tests pass, output tokens remain
  deterministic for greedy sampling.

## Non-goals

- Multi-depth MTP for Lightning MLX (Lightning is depth-1 only; AX already has depth-3).
- dFlash block-diffusion drafting (separate speculative decode architecture).
- N-gram + MTP stacking optimisation (already a separate path activated by `--enable-ngram`).
- Changing the MTPLX benchmark configuration or fairness rules.

## Benchmarks (pre-fix)

Source: `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/`

| Suite | MTPLX | Lightning MTP | AX MTP | AX/MTPLX | AX/Lightning |
|---|---:|---:|---:|---:|---:|
| flappy (27B, depth-3) | 55.5 | 48.8 | 37.6 | 0.68 | 0.77 |
| long_code (27B, depth-3) | 44.3 | 48.7 | 38.0 | 0.86 | 0.78 |
| python_modules_long (27B, depth-3) | 47.7 | 45.0 | 22.9 | 0.48 | 0.51 |
| flappy (35B, depth-1) | 107.3 | 147.9 | 182.5 | 1.70 | 1.23 |

## Changes

### C1: Off-by-one fix in `compute_mtp_target_probs_lazy`

**File:** `crates/ax-engine-mlx/src/runner.rs`
**Commit:** `435e11b8`

In a causal transformer, `logits_all[i]` is conditioned on tokens `0..=i` and predicts
`pending[i]`. The target probability for `pending[i]` must come from row `i`, not row
`i+1`. Changed `flat_idx[i] = (i+1) * vocab + pending[i]` back to
`flat_idx[i] = i * vocab + pending[i]`.

This fix alone restored decode from ~15 tok/s to ~41 tok/s on flappy_pipes.

### C2: Fused lazy MTP draft

**File:** `crates/ax-engine-mlx/src/mtp.rs`
**Commit:** `b0011d37`

Replaced the per-depth `eval(&[&logits])` + CPU sampling loop with a fully lazy graph:
`argmax` for token selection + `gpu_draft_log_prob_lazy` for log-prob computation,
all materialised in a single `eval`. Added `gpu_draft_log_prob_lazy` that takes a lazy
argmax array as the gather index instead of a concrete `u32`, keeping the entire chain
lazy.

Reduces GPU sync barriers from 4 (3 per-depth + 1 batch) to 1 per draft step.

### C3: Skip-state consumption (Lightning always-advance pattern)

**File:** `crates/ax-engine-mlx/src/runner.rs`
**Commits:** `105b147c`, `167b1053` (bug fixes)

When the previous verify forward captured logits at the last accepted position, reuse
them to sample the next primary token and draft new MTP tokens WITHOUT a fresh model
forward. The new drafts are written into `pending` so the existing verify/accept pipeline
operates on them unchanged.

On full-accept paths this eliminates every other model forward, matching Lightning MLX's
skip-state optimisation.

### C3 bug fixes (`167b1053`)

Three bugs were found and fixed in the initial skip-state implementation:

1. `sample_logit_row` called with `vocab=1` instead of actual vocab size, causing
   garbage primary token sampling from a 1-element slice.
2. Skip-state drafts generated into a separate vector invisible to verify/accept,
   causing all drafts to be silently dropped (accept_count=0 always).
3. Bogus `mtp_telemetry.record_step(0, 0, &[])` recording.

## Target benchmarks (post-fix, expected)

Based on the architectural analysis:

| Suite | MTPLX | Lightning MTP | AX MTP (expected) | Target AX/MTPLX |
|---|---:|---:|---:|---:|
| flappy (27B) | 55.5 | 48.8 | 55–65 | ≥ 1.0 |
| long_code (27B) | 44.3 | 48.7 | 50–60 | ≥ 1.1 |
| python_modules_long (27B) | 47.7 | 45.0 | 45–55 | ≥ 0.95 |

The 35B-A3B numbers should remain stable or improve slightly.

## Validation plan

1. `cargo test -p ax-engine-mlx` — all 482 tests pass.
2. `cargo clippy -p ax-engine-mlx --all-targets -- -D warnings` — clean.
3. Re-run `scripts/bench_qwen36_mtp_fair.py --models 27b-4bit --suites flappy long_code python_modules_long`
   with a release build.
4. Compare new results against `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/`.
5. Run the `lightning-ngram3` config to verify n-gram+MTP stacking still works.

## Out of scope for this PRD

- **Adaptive depth tuning** for `python_modules_long` (67% acceptance at depth-3 suggests
  the depth should drop to 2 faster on complex code). Separate optimisation.
- **Verify-eval wall time reduction** (AX verify-eval is ~3× MTPLX per step; may benefit
  from Metal kernel fusion for the softmax+gather path). Separate investigation.
- **Batched multi-request MTP** (current MTP is single-request; batching speculative decode
  across concurrent requests is a larger architectural change).

## Evidence

| Artifact | Path |
|---|---|
| Pre-fix full-rerun (3-engine) | `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/` |
| Pre-fix AX-only | `benchmarks/results/mtp-fair/2026-05-31-ax-engine-only/` |
| Lightning ngram+MTP | `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-lightning-ngram3/` |
| Lightning MLX source reference | `.internal/reference/lightning-mlx/vllm_mlx/scheduler.py` (`_mtp_step`) |
| Lightning MLX MTP sampling | `.internal/reference/lightning-mlx/vllm_mlx/speculative/native_mtp/sampling.py` |
| AX MTP implementation | `crates/ax-engine-mlx/src/mtp.rs` |
| AX MTP decode loop | `crates/ax-engine-mlx/src/runner.rs` (`run_mtp_decode`) |
