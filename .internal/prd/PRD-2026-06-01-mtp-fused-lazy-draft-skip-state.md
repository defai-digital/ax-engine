# PRD: MTP Decode Performance Improvement — Fused Lazy Draft and Skip-State

**Status**: Active
**Date**: 2026-06-01
**Current ADR**: `.internal/adr/ADR-012-mtp-fused-lazy-draft-skip-state.md`
**Scope**: `crates/ax-engine-mlx/src/mtp.rs`, `crates/ax-engine-mlx/src/runner.rs`

---

## Problem

AX Engine MTP decode on Qwen 3.6 27B was 22–52% slower than Lightning MLX and MTPLX
on the fair benchmark prompt suites. The gap was caused by three issues identified by
studying Lightning MLX's `_mtp_step` source:

1. Per-depth GPU sync barriers in the MTP draft path (4 barriers for depth-3).
2. Skip-state logits captured but never consumed (wasted model forward every other step).
3. Off-by-one in rejection-sampling target probs (systematic over-rejection).

## Goals

- **G1** Match or exceed Lightning MLX MTP decode throughput on 27B 4-bit.
- **G2** Maintain or improve 35B-A3B throughput (already leading).
- **G3** Preserve acceptance rate parity (≥95% on flappy/long_code).
- **G4** No correctness regression: all unit tests pass, output deterministic for greedy.

## Non-goals

- Multi-depth MTP for Lightning MLX (Lightning is depth-1 only).
- dFlash block-diffusion drafting (separate architecture).
- N-gram + MTP stacking optimisation (separate path, already active with `--enable-ngram`).
- Adaptive depth tuning for python_modules_long (separate optimisation).
- Changing MTPLX or Lightning MLX source code (can only tune our benchmark scripts).

## Current Evidence

Pre-fix benchmarks (`benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/`):

| Suite | MTPLX | Lightning | AX | AX/MTPLX | AX/Lightning |
|---|---:|---:|---:|---:|---:|
| flappy (27B d3) | 55.5 | 48.8 | 37.6 | 0.68 | 0.77 |
| long_code (27B d3) | 44.3 | 48.7 | 38.0 | 0.86 | 0.78 |
| python_modules_long (27B d3) | 47.7 | 45.0 | 22.9 | 0.48 | 0.51 |
| flappy (35B d1) | 107.3 | 147.9 | 182.5 | 1.70 | 1.23 |

Post-fix with n-gram+MTP stacking (`2026-05-31-qwen36-fair-lightning-ngram3/`):

| Suite | AX MTP | AX/MTPLX | AX/Lightning |
|---|---:|---:|---:|
| flappy (27B) | 65.9 | 1.28 | 1.33 |
| long_code (27B) | 65.6 | 1.23 | 1.27 |
| python_modules_long (27B) | 53.8 | 1.04 | 1.15 |

## Plan

### Phase 1: Off-by-one fix ✅

Commit `435e11b8`. Changed target prob indexing from `(i+1)*vocab` to `i*vocab`.
Restored decode from ~15 to ~41 tok/s.

### Phase 2: Fused lazy draft ✅

Commit `b0011d37`. Replaced per-depth eval loop with lazy argmax + lazy log-prob graph,
single eval. Added `gpu_draft_log_prob_lazy`.

### Phase 3: Skip-state consumption ✅

Commits `105b147c`, `167b1053`. Sample primary + draft from skip-state, write into
`pending`, run existing verify/accept. Bug fixes for wrong vocab, dropped drafts,
fake telemetry.

### Phase 4: Script-level competitor optimisation

The fair benchmark script was not configuring MTPLX and Lightning MLX at their best:

**MTPLX:** Script used `--profile stable` (conservative, long_response_exact_staged).
The `sustained` profile enables `MTPLX_SKIP_VERIFY_SNAPSHOT=1`,
`MTPLX_LAZY_VERIFY_LOGITS=1`, `MTPLX_DEFER_VERIFY_HIDDEN_EVAL=1`,
`MTPLX_BATCH_TARGET_ARRAYS=1`, and paged attention — the same flags MTPLX
uses for its own published benchmarks.

**Lightning MLX:** Script did not pass `--mtp-optimistic` or
`--mtp-draft-temperature`. Lightning's own Qwen3.6 serve preset defaults
`mtp_optimistic=True` and `mtp_draft_temperature=0.5` (line 128, 167 in
`cli.py`). Without optimistic, Lightning runs verified rejection sampling
which is ~10-15% slower. These flags are CLI arguments, not source changes.

### Phase 5: Re-benchmark

Run fair benchmark with release build to confirm targets:
```
scripts/bench_qwen36_mtp_fair.py --models 27b-4bit --suites flappy long_code python_modules_long
```

Expected: AX ≥ MTPLX on all 27B suites (≥1.0× ratio).

## Acceptance Criteria

- [ ] All 482 `ax-engine-mlx` tests pass.
- [ ] Clippy clean.
- [ ] Fair benchmark shows AX ≥1.0× vs MTPLX on flappy and long_code (27B).
- [ ] 35B-A3B throughput does not regress below previous run.
- [ ] Acceptance rate ≥95% on flappy and long_code (27B).

## Validation

```text
cargo test -p ax-engine-mlx
cargo clippy -p ax-engine-mlx --all-targets -- -D warnings
scripts/bench_qwen36_mtp_fair.py --models 27b-4bit 35b-a3b-4bit --suites flappy long_code python_modules_long
```

## Evidence Artifacts

| Artifact | Path |
|---|---|
| Pre-fix full-rerun | `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/` |
| Pre-fix AX-only | `benchmarks/results/mtp-fair/2026-05-31-ax-engine-only/` |
| Lightning ngram+MTP | `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-lightning-ngram3/` |
| Lightning source reference | `.internal/reference/lightning-mlx/vllm_mlx/scheduler.py` |
| AX MTP implementation | `crates/ax-engine-mlx/src/mtp.rs` |
| AX MTP decode loop | `crates/ax-engine-mlx/src/runner.rs` |
