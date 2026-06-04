# MTP/N-gram Acceptance Rate Optimization PRD

## Status

**Implemented.** All four phases shipped. Auto-optimistic hysteresis tightened from 0.99/0.95 to 0.98/0.96 to increase activation rate on harder prompts. Per-depth accept rate telemetry added (`ax_mtp_accept_rate_depth{0,1,2}_x1000`).

## Background

AX Engine's MTP (Multi-Token Prediction) speculative decode path already achieves high acceptance rates on Qwen3.6-27B-MTP: depth-0 ≈ 99.9%, depth-1 ≈ 99.8%, depth-2 ≈ 98.9%, with auto-optimistic mode activating on ~80% of decode steps. The hybrid n-gram+MTP path stacks n-gram prefix drafts with MTP tail drafts, and the skip-state optimization halves model forwards.

However, several design decisions limit acceptance rate:

1. **Greedy mode lacks draft log-probs**: `mtp_draft_tokens_greedy` returns empty `draft_log_probs`, falling back to argmax-only comparison. This caps acceptance at argmax match rate even when the draft token has high target probability.
2. **Draft-target temperature mismatch**: MTP draft uses `head.draft_sampling.temperature` (default 0.7) while target uses request temperature (e.g., 0.6). The rejection sampling formula assumes matched temperatures.
3. **N-gram tokens bypass rejection sampling**: In hybrid mode, n-gram-sourced tokens use greedy argmax comparison only, missing acceptance opportunities when target is uncertain.
4. **N-gram saturation gate threshold too high**: At 0.99, the gate disables n-gram stacking when MTP acceptance is already high, but n-gram still provides value at 97-98%.
5. **Adaptive depth floor too aggressive on complete miss**: Floor of 2 after `accept_count=0` wastes forward passes on low-acceptance prompts.

The latest benchmark (`2026-06-02-qwen36-fair-ax-opt1`) on Qwen3.6-27B-MTP 4-bit shows 3.7x tokens per forward (3723 bonus tokens / 1000 output). These optimizations target pushing toward 4.0x+.

## Goals

- **G1**: Enable rejection sampling in greedy mode by computing GPU draft log-probs in `mtp_draft_tokens_greedy`, increasing acceptance when argmax disagrees but draft token has high target probability.
- **G2**: Align draft-target temperature by rescaling draft log-probs when `head.draft_sampling.temperature` differs from request temperature, improving acceptance probability calibration.
- **G3**: Enable pseudo-rejection sampling for n-gram-sourced tokens in hybrid mode by deriving draft log-probs from n-gram confidence scores.
- **G4**: Lower n-gram saturation gate threshold from 0.99 to 0.97 for depth≥3, keeping n-gram active longer when MTP acceptance is high but not saturated.
- **G5**: Make adaptive depth floor responsive to complete misses — floor remains at 2 on the first miss (status quo), drops to 1 after the second consecutive miss, and drops to 0 after the third+ consecutive miss — reducing wasted forward passes on persistently low-acceptance prompts.

## Non-goals

- Changes to MTP head architecture (single recurrent layer, shared lm_head).
- Training-time MTP improvements (this is a runtime-only optimization).
- Whole-layer Metal kernel fusion (low ceiling per `PERFORMANCE-DECODE-GAP.md`).
- Changes to the double-buffer direct pipeline contract.
- KV cache memory layout changes (separate roadmap track).

## User Impact

| User | Impact |
|---|---|
| MTP speculative decode users (default on MTP models) | Higher tokens-per-forward, lower latency |
| N-gram+MTP hybrid users | More hybrid tail steps, higher acceptance on n-gram prefix |
| Greedy mode users with MTP | Rejection sampling now enabled, higher acceptance |
| Reasoning model users (think-block gating) | Faster recovery from low-acceptance regions |

## Metrics

| Metric | Baseline (Qwen3.6-27B-MTP 4-bit, flappy) | Target | Measurement |
|---|---|---|---|
| depth-0 accept rate | 99.9% | ≥ 99.9% (maintain) | `ax_mtp_accepted_depth0 / ax_mtp_drafted_depth0` |
| depth-1 accept rate | 99.8% | ≥ 99.8% (maintain) | `ax_mtp_accepted_depth1 / ax_mtp_drafted_depth1` |
| depth-2 accept rate | 98.9% | ≥ 99.5% | `ax_mtp_accepted_depth2 / ax_mtp_drafted_depth2` |
| Cycle acceptance | 99.0% | ≥ 99.5% | `(decode_steps - rejected_cycles) / decode_steps` |
| Bonus tokens per output | 3.7x | ≥ 4.0x | `ax_mlx_bonus_tokens / generation_tokens` |
| Auto-optimistic activation | 80.7% | ≥ 90% | `ax_mtp_auto_optimistic_steps / decode_steps` |
| N-gram hybrid tail steps | 0 (gated) | > 0 on repeating prompts | `ax_mtp_ngram_hybrid_tail_steps` |

## Phases

### Phase 1 — Greedy Mode Draft Log-Probs (P0)

Compute GPU-side draft log-probs in `mtp_draft_tokens_greedy` using the same `gpu_draft_log_prob_lazy` path as sampled mode, but with temperature=1.0 for the draft distribution. This enables rejection sampling in greedy mode without changing token selection (still argmax).

**Files**: `crates/ax-engine-mlx/src/mtp.rs`, `crates/ax-engine-mlx/src/runner.rs`

### Phase 2 — Temperature Alignment (P0)

When `head.draft_sampling.temperature` differs from request `sampling.temperature`, rescale draft log-probs: `log_prob_scaled = log_prob * (draft_T / target_T)`. This ensures `accept_prob = min(1, p_target/p_draft)` compares distributions at the same temperature.

**Files**: `crates/ax-engine-mlx/src/runner.rs` (in `mtp_accept_count`)

### Phase 3 — N-gram Pseudo Log-Probs (P1)

For n-gram-sourced tokens in hybrid mode, compute pseudo draft log-probs from n-gram confidence: `pseudo_log_prob = ln(confidence)`. This gives rejection sampling a signal for n-gram positions.

**Files**: `crates/ax-engine-mlx/src/ngram_accel.rs`, `crates/ax-engine-mlx/src/runner.rs`

### Phase 4 — Adaptive Gate and Depth (P1)

Lower n-gram saturation gate threshold to 0.97 for depth≥3. Make adaptive depth floor responsive to complete misses.

**Files**: `crates/ax-engine-mlx/src/runner.rs`

## Evidence Gates

All phases require:
- `cargo test --quiet --no-fail-fast` passes.
- `cargo clippy --all-targets --all-features -- -D warnings` passes.
- `cargo fmt --check` passes.
- A/B benchmark artifact on Qwen3.6-27B-MTP 4-bit showing improved acceptance rate or bonus tokens.
- No regression on greedy decode path (the common case).
- No regression on n-gram-only path (when MTP is disabled).
