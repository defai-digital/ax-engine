# Decode Speed Optimization PRD

## Background

AX Engine's decode path already implements several optimizations: double-buffer direct pipeline (`async_eval`/`eval` overlap), n-gram speculative decoding, MTP speculative decoding, TurboQuant KV compression, and 50+ Metal/C++ fast-path kernels. However, the non-greedy sampling path (temperature > 0 with top-k/top-p) carries per-step CPU overhead that scales with vocabulary size, and speculative decode parameters are static rather than adaptive.

The `PERFORMANCE-DECODE-GAP.md` investigation established that the residual 1-6% gap to `mlx_lm.benchmark` on dense models is dominated by per-MLX-op FFI overhead (~800-1300 op dispatches per step). Single-op fusion has a low ceiling. This PRD targets the **CPU-side decode hot path** — the work that happens between GPU forward completion and the next step's graph build — where we have direct control and proven leverage.

## Goals

- **G1**: Reduce per-step CPU overhead in the non-greedy sampling path (temperature + top-k/top-p) by replacing O(V log V) full-vocab sort with GPU-side `argpartition_axis` + CPU-side small-k sort.
- **G2**: Eliminate per-step heap allocations in the sampling path (probs `Vec<f32>`, repetition penalty logits copy) by pre-allocating reusable buffers in `RequestState`.
- **G3**: Improve n-gram speculative decode throughput by making draft length adaptive based on recent acceptance rate, rather than static `MAX_DRAFT_LEN = 6`.
- **G4**: Cache `full_vocab_token_logprob` normalization constants (max-logit, exp-sum) to avoid recomputing O(V) scans during MTP rejection sampling.

## Non-goals

- Whole-layer Metal kernel fusion (already investigated in `PERFORMANCE-DECODE-GAP.md`; low ceiling per fusion site).
- `mx.compile`-analog for per-layer forward (listed as out-of-scope future work in `PERFORMANCE-DECODE-GAP.md`).
- Changes to the double-buffer pipeline contract (already optimal for greedy decode).
- KV cache memory layout changes (separate roadmap track).

## User Impact

| User | Impact |
|---|---|
| Greedy decode users (default) | No change — already uses optimal double-buffer pipeline |
| Temperature + top-k/top-p users | Lower per-step latency, especially for large-vocab models (Qwen 150K+) |
| Repetition penalty users | Lower per-step allocation pressure |
| N-gram speculative decode users | Higher tokens-per-forward when patterns are stable |
| MTP speculative decode users | Lower rejection-sampling overhead |

## Metrics

| Metric | Baseline | Target | Measurement |
|---|---|---|---|
| Non-greedy decode step latency (Qwen 3.6 27B, top-k=50, top-p=0.9) | TBD | -10% CPU time | `decode-trace` with `AX_MLX_DIRECT_PIPELINE_STAGE_PROFILE=1` |
| Per-step heap allocations (non-greedy, repetition penalty) | 2 allocs/step | 0 allocs/step | `cargo bench` with allocation tracking |
| N-gram acceptance rate (repeating prompts) | TBD | +15% tokens/forward | `ax_ngram_accept_at_depth_*` route decisions |
| MTP rejection sampling overhead | TBD | -20% | MTP telemetry counters |

## Phases

### Phase 1 — Sampling Path Optimization (P0)

Replace the CPU-side `sort_by` in `apply_top_k_top_p` with GPU-side `argpartition_axis` to get top-k candidates, then transfer only k logits to CPU for top-p filtering. Pre-allocate sampling buffers in `RequestState` to eliminate per-step heap allocations.

**Files**: `crates/ax-engine-mlx/src/sampling.rs`, `crates/ax-engine-mlx/src/runner.rs`

### Phase 2 — Adaptive N-gram Draft Length (P1)

Make `MAX_DRAFT_LEN` dynamic based on a rolling window of acceptance rates. When acceptance rate > 80% over the last 32 steps, increase draft ceiling to 8. When < 30%, decrease to 3.

**Files**: `crates/ax-engine-mlx/src/ngram_accel.rs`, `crates/ax-engine-mlx/src/runner.rs`

### Phase 3 — MTP Logprob Caching (P1)

Cache the max-logit and exp-sum from the forward pass so `full_vocab_token_logprob` can compute log-prob in O(1) instead of O(V).

**Files**: `crates/ax-engine-mlx/src/sampling.rs`, `crates/ax-engine-mlx/src/mtp.rs`

## Evidence Gates

All phases require:
- `cargo test --quiet --no-fail-fast` passes (existing test suite).
- `cargo clippy --all-targets --all-features -- -D warnings` passes.
- `cargo fmt --check` passes.
- A/B benchmark artifact showing decode rate improvement on at least one model family.
- No regression on greedy decode path (the common case).
