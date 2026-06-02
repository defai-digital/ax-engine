# Decode Speed Optimization PRD

## Background

AX Engine's decode path already implements several optimizations: double-buffer direct pipeline (`async_eval`/`eval` overlap), n-gram speculative decoding, MTP speculative decoding, TurboQuant KV compression, and 50+ Metal/C++ fast-path kernels. However, the non-greedy sampling path (temperature > 0 with top-k/top-p) carries per-step CPU overhead that scales with vocabulary size, and the MTP target-probability path still has avoidable small allocation/extraction overhead.

The `PERFORMANCE-DECODE-GAP.md` investigation established that the residual 1-6% gap to `mlx_lm.benchmark` on dense models is dominated by per-MLX-op FFI overhead (~800-1300 op dispatches per step). Single-op fusion has a low ceiling. This PRD targets the **CPU-side decode hot path** — the work that happens between GPU forward completion and the next step's graph build — where we have direct control and proven leverage.

## Goals

- **G1**: Reduce per-step CPU overhead in exact non-greedy `top_k` sampling by replacing CPU full-vocab `sort_by` with GPU-side `argpartition_axis` + CPU-side small-k sampling while preserving the current probability semantics.
- **G2**: Eliminate per-step heap allocations in the CPU sampling path (probabilities, candidate tuples, repetition-penalty logits copy) by reusing request-local buffers or an equivalent decode-local workspace.
- **G3**: Reduce MTP rejection-sampling overhead by optimizing the existing lazy GPU target-probability path in `runner.rs` without materializing full-vocabulary logits on CPU.

## Non-goals

- Whole-layer Metal kernel fusion (already investigated in `PERFORMANCE-DECODE-GAP.md`; low ceiling per fusion site).
- `mx.compile`-analog for per-layer forward (listed as out-of-scope future work in `PERFORMANCE-DECODE-GAP.md`).
- Changes to the double-buffer pipeline contract (already optimal for greedy decode).
- KV cache memory layout changes (separate roadmap track).
- Adaptive n-gram draft length. It is deferred because it can regress linear-attention branch/recompute cost, MTP+n-gram hybrid source accounting, and workload-dependent acceptance behavior.

## User Impact

| User | Impact |
|---|---|
| Greedy decode users (default) | No change — already uses optimal double-buffer pipeline |
| Temperature + top-k/top-p users | Lower per-step latency, especially for large-vocab models (Qwen 150K+) |
| Repetition penalty users | Lower per-step allocation pressure |
| MTP speculative decode users | Lower rejection-sampling overhead |

## Metrics

| Metric | Baseline | Target | Measurement |
|---|---|---|---|
| Non-greedy decode step latency (Qwen 3.6 27B, top-k=50, top-p=0.9) | TBD | -10% CPU time | `decode-trace` with `AX_MLX_DIRECT_PIPELINE_STAGE_PROFILE=1` |
| Per-step heap allocations (non-greedy, repetition penalty) | 2 allocs/step | 0 allocs/step | `cargo bench` with allocation tracking |
| MTP rejection sampling overhead | TBD | -20% target-prob extraction/acceptance CPU time | MTP telemetry counters and decode-stage profile |

## Phases

### Phase 1 — Sampling Path Optimization (P0)

Replace the CPU-side `sort_by` in exact `top_k` sampling with GPU-side `argpartition_axis` to get top-k candidates, then transfer only k logits/indices to CPU for sampling. Pre-allocate or reuse sampling buffers in the decode workspace to eliminate avoidable per-step heap allocations.

`top_p` has a stricter correctness constraint: the current implementation computes nucleus filtering against full-vocabulary probability mass. Phase 1 must either preserve that full-domain cutoff exactly or fall back to the existing CPU path. In particular, `top_p < 1.0` with `top_k == 0` must not be approximated with a fixed candidate count.

**Files**: `crates/ax-engine-mlx/src/sampling.rs`, `crates/ax-engine-mlx/src/runner.rs`

### Phase 2 — MTP Target-Probability Workspace (P1)

Optimize the current `compute_mtp_target_probs` path in `runner.rs`. The current implementation already computes target probabilities lazily on GPU and gathers only the pending draft-token probabilities before `mtp_accept_count`. Phase 2 should reduce allocation, indexing, and extraction overhead in that path without introducing CPU full-logits readback.

**Files**: `crates/ax-engine-mlx/src/runner.rs`, `crates/ax-engine-mlx/src/mtp.rs`, `crates/ax-engine-mlx/src/sampling.rs`

## Deferred High-Risk Items

### Adaptive N-gram Draft Length

The adaptive n-gram draft-length proposal is intentionally disabled for this PRD. It may be revisited only after a separate benchmark plan proves:
- dense-model verifier batches win when the draft ceiling grows beyond the current `MAX_DRAFT_LEN = 6`;
- linear-attention requests do not regress from partial-reject branch/recompute cost;
- MTP+n-gram hybrid source-aware acceptance and telemetry remain unchanged;
- real workload suites improve, not only synthetic repeating prompts.

## Evidence Gates

All phases require:
- `cargo test --quiet --no-fail-fast` passes (existing test suite).
- `cargo clippy --all-targets --all-features -- -D warnings` passes.
- `cargo fmt --check` passes.
- A/B benchmark artifact showing decode rate improvement on at least one model family.
- No regression on greedy decode path (the common case).
