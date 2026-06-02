# Decode Speed Optimization — Technical Specification

## Overview

This spec details the implementation plan for two active decode path optimizations targeting CPU-side overhead in the sampling path and MTP rejection-sampling efficiency. Adaptive n-gram draft length is explicitly deferred as a high-risk research item.

## Phase 1 — Sampling Path Optimization

### 1.1 GPU-side argpartition for exact top-k sampling

**Problem**: `apply_top_k_top_p` in `sampling.rs:638-669` calls `candidates.sort_by()` on the full vocabulary (O(V log V)) when top-k or top-p filtering is active. For 150K vocabularies, this dominates CPU time in the non-greedy path.

**Solution**: Use MLX's `argpartition_axis` (already available in `mlx-sys/src/ops.rs`) to get the top-k indices on GPU, then transfer only k logits/indices to CPU for exact top-k sampling.

`top_p` must preserve the current full-vocabulary nucleus semantics. The existing CPU implementation computes the cumulative cutoff against full-vocabulary probability mass before truncating by `top_k`. Therefore:
- `top_k > 0 && top_p >= 1.0`: use the GPU top-k path.
- `top_k > 0 && top_p < 1.0`: use the GPU top-k path only if the implementation also computes the full-domain top-p cutoff exactly, for example by gathering full-domain probabilities for the top-k candidates or by using a GPU-computed full softmax denominator.
- `top_k == 0 && top_p < 1.0`: fall back to the existing CPU full-vocabulary path in Phase 1.

Do not approximate top-p-only sampling with a fixed candidate count such as 256. That changes the output distribution whenever the nucleus contains more candidates than the fixed cutoff.

**Implementation**:

```rust
// New function in sampling.rs
pub fn sample_categorical_with_topk_gpu(
    logits: &MlxArray,
    sampling: MlxSamplingParams,
    repetition_tokens: &[u32],
    rng: &mut Xorshift64,
) -> Option<u32> {
    // 1. Reject unsupported cases and fall back to the existing CPU path.
    // 2. Scale by 1/temperature
    // 3. argpartition_axis to get top-k indices
    // 4. take_along_axis to get top-k logits
    // 5. eval both indices and logits
    // 6. Apply exact top-k sampling, or exact top-k + top-p if full-domain
    //    top-p mass is available.
    // 7. Sample from filtered candidates
}
```

**Call site changes** in `decode_step_with_turboquant_context` (`generate.rs:651-706`):
- When `sampling.top_k > 0` AND `temperature > 0` AND no repetition penalty: attempt `sample_categorical_with_topk_gpu`.
- If it returns `None`, use the existing CPU `sample_categorical` path.
- When repetition penalty is active: fall back to existing CPU path (penalty requires random-access logits modification).

**Key constraints**:
- `argpartition_axis` is already used in `sample_logit_row_topk_gpu` (`ngram_accel.rs:1311-1337`) — reuse the same pattern.
- The `top_k` value from `MlxSamplingParams` is the k for argpartition.
- `top_p` is exact only if the cutoff is computed against full-vocabulary mass. Otherwise, keep the CPU fallback.
- GPU-side repetition penalty is NOT implemented in this phase — the CPU fallback path remains for that case.

### 1.2 Pre-allocated sampling buffers

**Problem**: Three per-step allocations in the sampling path:
1. `let probs: Vec<f32> = logits.iter().map(...).collect()` at `sampling.rs:178-184`
2. `adjusted_logits_buf = logits.to_vec()` at `sampling.rs:157` when repetition penalty is active
3. `candidates: Vec<(usize, f32)> = ...collect()` at `sampling.rs:206-213` when top-k/top-p filtering is active

**Solution**: Add reusable buffers to the request/decode workspace. `RequestState::new()` currently receives only `num_layers` and `request_id`, so either pass `cfg.vocab_size` into construction or lazily reserve the buffers on first sampling use.

```rust
// In RequestState (runner.rs)
struct RequestState {
    // ... existing fields ...
    sampling_probs_buf: Vec<f32>,       // Pre-allocated for vocab size
    sampling_logits_buf: Vec<f32>,       // Pre-allocated for vocab size (repetition penalty)
    sampling_candidates_buf: Vec<(usize, f32)>,
}
```

**Implementation**:
- Initialize buffers with `Vec::with_capacity(cfg.vocab_size)` once the vocab size is available, or lazily call `reserve_exact`/`reserve` before first use.
- In `sample_categorical`, accept `&mut Vec<f32>` buffers as parameters instead of allocating.
- Use `buf.clear(); buf.extend(...)` pattern to reuse capacity.
- Add an allocating wrapper for existing tests and non-runner callers, and a workspace-aware implementation for hot decode paths:

```rust
pub fn sample_categorical_into(
    logits: &[f32],
    sampling: MlxSamplingParams,
    repetition_tokens: &[u32],
    rng: &mut Xorshift64,
    probs_buf: &mut Vec<f32>,      // NEW: reusable buffer
    logits_buf: &mut Vec<f32>,     // NEW: reusable buffer for repetition penalty
    candidates_buf: &mut Vec<(usize, f32)>,
) -> u32
```

**Backward compatibility**: The existing `sample_categorical` signature is preserved as a wrapper that allocates locally, so tests and non-runner callers don't break.

**Call graph coverage**: To achieve the allocation target, update all hot call sites, not only `decode_step_with_turboquant_context`:
- `generate.rs` prefill final-token sampling paths;
- `generate.rs::decode_step_with_turboquant_context`;
- `ngram_accel.rs::single_decode_with_turboquant_context`;
- `ngram_accel.rs::sample_logit_row` fallback path.

### 1.3 Affected files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/sampling.rs` | Add `sample_categorical_with_topk_gpu`, add workspace-aware `sample_categorical_into` |
| `crates/ax-engine-mlx/src/generate.rs` | Update decode and prefill final-token sampling paths |
| `crates/ax-engine-mlx/src/runner.rs` | Add buffers to `RequestState`, pass them through decode call chain |
| `crates/ax-engine-mlx/src/ngram_accel.rs` | Update `single_decode_with_turboquant_context` to pass buffers |

## Phase 2 — MTP Target-Probability Workspace

### 2.1 Optimize the existing lazy target-probability path

**Problem**: MTP rejection sampling needs `p_target(draft_token_i)` for each pending draft position. The current runtime path already uses `compute_mtp_target_probs` in `runner.rs`, which builds a lazy GPU softmax/gather graph and extracts only the pending probabilities before `mtp_accept_count`. It does not call `full_vocab_token_logprob` in the hot MTP path.

**Solution**: Keep target-probability computation lazy and GPU-side. Optimize the small CPU-side workspace around the current path:
- reuse the flat index buffer used to gather pending draft probabilities;
- reuse the extracted `target_probs_cpu` buffer instead of allocating a new `Vec<f32>` per verify;
- keep `LazyTargetProbs` in the same eval batch as argmax, post-norm hidden, and KV refs to avoid adding a GPU sync point;
- do not materialize `[verify_len, vocab]` logits on CPU for a max-logit/exp-sum cache.

**Implementation**:

```rust
// In runner.rs, stored on RequestState or an MTP-local workspace
struct MtpTargetProbWorkspace {
    flat_indices: Vec<i32>,
    target_probs: Vec<f32>,
}

// compute_mtp_target_probs keeps returning a lazy GPU object. The workspace
// only owns small CPU buffers needed to build gather indices and extract the
// final pending-token probabilities.
```

**Call site changes**:
- In both MTP verify branches in `runner.rs`, pass the request-local workspace into `compute_mtp_target_probs`.
- Replace `LazyTargetProbs::extract_cpu(&pending) -> Option<Vec<f32>>` with a method that fills `workspace.target_probs` and returns `Option<&[f32]>`.
- Keep `full_vocab_token_logprob` as a cold CPU helper for tests or non-MTP callers; do not make it the MTP hot path.

**Memory tradeoff**: The workspace is O(draft_len), not O(vocab). For MTP depth <= 3, this is negligible and avoids CPU full-logits readback.

### 2.2 Affected files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/runner.rs` | Add MTP target-prob workspace, remove per-step small allocations in `compute_mtp_target_probs` / extraction |
| `crates/ax-engine-mlx/src/mtp.rs` | Preserve draft log-probability contract for sampled MTP tokens |
| `crates/ax-engine-mlx/src/sampling.rs` | Keep `full_vocab_token_logprob` as a cold helper/fallback; add tests if behavior changes |

## Testing Strategy

### Unit tests

- `sampling.rs`: Test `sample_categorical_with_topk_gpu` matches `sample_categorical` for exact `top_k` cases after candidate ordering/tie-breaking is normalized.
- `sampling.rs`: Test `top_p < 1.0 && top_k == 0` falls back to the CPU path unless full-domain top-p semantics are implemented.
- `sampling.rs`: Test pre-allocated buffer version produces identical results to allocating version.
- `runner.rs`: Test MTP target-prob workspace extraction matches the existing allocating path.

### Integration tests

- `cargo run -p ax-engine-mlx --release --bin decode-trace -- <model_dir> <steps>` on Qwen 3.6 27B 4-bit with top-k=50, top-p=0.9, temperature=0.7: verify no regression in output quality.
- `cargo run -p ax-engine-mlx --release --bin decode-trace -- <model_dir> <steps>` on Gemma 4 E2B 4-bit with greedy: verify no regression in decode rate.

### Benchmark gates

- `cargo bench` with existing inference harness: decode rate must not regress on any model.
- New benchmark row: non-greedy decode (temperature=0.7, top-k=50, top-p=0.9) on Qwen 3.6 27B 4-bit, comparing before/after CPU time per step.

## Rollback Plan

All changes are gated behind existing fast-path patterns:
- Phase 1: New GPU path only engages when `top_k > 0` AND `temperature > 0` AND no repetition penalty AND the `top_p` semantics are exact. Existing path is unchanged for unsupported cases.
- Phase 2: MTP target-prob workspace is an additive allocation optimization around the existing lazy GPU target-probability path.

Kill switches via env var (following `fastpath.rs` convention):
- `AX_MLX_DECODE_SAMPLING_GPU_TOPK=0` — disable GPU-side argpartition, fall back to CPU sort.
- `AX_MLX_DECODE_MTP_TARGET_PROB_WORKSPACE=0` — disable the MTP target-probability workspace, use the existing allocating extraction path.

## Deferred High-Risk Item — Adaptive N-gram Draft Length

Adaptive n-gram draft length is not active in this spec. Do not add `AdaptiveDraftPolicy`, do not change `DEFAULT_DRAFT_LEN`/`MAX_DRAFT_LEN` selection, and do not add `AX_MLX_DECODE_ADAPTIVE_DRAFT` as part of this PRD.

It can be reconsidered only as a separate work item with dedicated evidence that:
- dense-model draft ceilings above `MAX_DRAFT_LEN = 6` improve end-to-end decode throughput;
- linear-attention partial-reject branch/recompute cost does not regress;
- MTP+n-gram hybrid source-aware acceptance and telemetry remain unchanged;
- gains hold on real workload suites, not only synthetic repeating prompts.
