# Decode Speed Optimization — Technical Specification

## Overview

This spec details the implementation plan for three decode path optimizations targeting CPU-side overhead in the sampling path, speculative decode adaptivity, and MTP rejection sampling efficiency.

## Phase 1 — Sampling Path Optimization

### 1.1 GPU-side argpartition for top-k/top-p

**Problem**: `apply_top_k_top_p` in `sampling.rs:638-669` calls `candidates.sort_by()` on the full vocabulary (O(V log V)) when top-k or top-p filtering is active. For 150K vocabularies, this dominates CPU time in the non-greedy path.

**Solution**: Use MLX's `argpartition_axis` (already available in `mlx-sys/src/ops.rs`) to get the top-k indices on GPU, then transfer only k logits to CPU for top-p filtering.

**Implementation**:

```rust
// New function in sampling.rs
pub fn sample_categorical_with_topk_gpu(
    logits: &MlxArray,
    sampling: MlxSamplingParams,
    repetition_tokens: &[u32],
    rng: &mut Xorshift64,
) -> u32 {
    // 1. Apply repetition penalty on GPU if needed (or fall back to CPU path)
    // 2. Scale by 1/temperature
    // 3. argpartition_axis to get top-k indices
    // 4. take_along_axis to get top-k logits
    // 5. eval both indices and logits
    // 6. Apply top-p on CPU over k candidates (O(k log k), k << V)
    // 7. Sample from filtered candidates
}
```

**Call site changes** in `decode_step_with_turboquant_context` (`generate.rs:651-706`):
- When `sampling.top_k > 0 || sampling.top_p < 1.0` AND `temperature > 0` AND no repetition penalty: use `sample_categorical_with_topk_gpu` instead of `sample_categorical`.
- When repetition penalty is active: fall back to existing CPU path (penalty requires random-access logits modification).

**Key constraints**:
- `argpartition_axis` is already used in `sample_logit_row_topk_gpu` (`ngram_accel.rs:1311-1337`) — reuse the same pattern.
- The `top_k` value from `MlxSamplingParams` is the k for argpartition; if `top_k == 0` but `top_p < 1.0`, use a reasonable default (e.g., 256) for the GPU partition, then apply top-p on CPU.
- GPU-side repetition penalty is NOT implemented in this phase — the CPU fallback path remains for that case.

### 1.2 Pre-allocated sampling buffers

**Problem**: Two per-step allocations in the sampling path:
1. `let probs: Vec<f32> = logits.iter().map(...).collect()` at `sampling.rs:178-184`
2. `adjusted_logits_buf = logits.to_vec()` at `sampling.rs:157` when repetition penalty is active

**Solution**: Add reusable buffers to `RequestState` in `runner.rs`:

```rust
// In RequestState (runner.rs)
struct RequestState {
    // ... existing fields ...
    sampling_probs_buf: Vec<f32>,       // Pre-allocated for vocab size
    sampling_logits_buf: Vec<f32>,       // Pre-allocated for vocab size (repetition penalty)
}
```

**Implementation**:
- Initialize buffers in `RequestState::new()` with `Vec::with_capacity(cfg.vocab_size)`.
- In `sample_categorical`, accept `&mut Vec<f32>` buffers as parameters instead of allocating.
- Use `buf.clear(); buf.extend(...)` pattern to reuse capacity.
- The `sample_categorical` function signature changes to accept optional mutable buffer references:

```rust
pub fn sample_categorical(
    logits: &[f32],
    sampling: MlxSamplingParams,
    repetition_tokens: &[u32],
    rng: &mut Xorshift64,
    probs_buf: &mut Vec<f32>,      // NEW: reusable buffer
    logits_buf: &mut Vec<f32>,     // NEW: reusable buffer for repetition penalty
) -> u32
```

**Backward compatibility**: The existing `sample_categorical` signature is preserved as a wrapper that allocates locally, so tests and non-runner callers don't break.

### 1.3 Affected files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/sampling.rs` | Add `sample_categorical_with_topk_gpu`, modify `sample_categorical` to accept reusable buffers |
| `crates/ax-engine-mlx/src/generate.rs` | Update `decode_step_with_turboquant_context` to call new GPU path |
| `crates/ax-engine-mlx/src/runner.rs` | Add buffers to `RequestState`, pass them through decode call chain |
| `crates/ax-engine-mlx/src/ngram_accel.rs` | Update `single_decode_with_turboquant_context` to pass buffers |

## Phase 2 — Adaptive N-gram Draft Length

### 2.1 Rolling acceptance rate window

**Problem**: `MAX_DRAFT_LEN = 6` is static. When n-gram patterns are highly predictive (acceptance rate > 80%), longer drafts would yield more tokens per forward pass. When patterns are weak (< 30%), shorter drafts reduce wasted verification work.

**Solution**: Track a rolling window of acceptance rates and adjust draft length dynamically.

**Implementation**:

```rust
// In ngram_accel.rs or runner.rs RequestState
struct AdaptiveDraftPolicy {
    window: VecDeque<u32>,    // Recent accepted counts per step
    window_size: usize,        // Default 32
    base_draft_len: usize,     // Default DEFAULT_DRAFT_LEN (4)
    min_draft_len: usize,      // Default 2
    max_draft_len: usize,      // Default 8
}

impl AdaptiveDraftPolicy {
    fn resolve_draft_len(&self) -> usize {
        if self.window.len() < 8 {
            return self.base_draft_len;  // Not enough data
        }
        let avg: f32 = self.window.iter().map(|&c| c as f32).sum::<f32>()
            / self.window.len() as f32;
        if avg >= 6.0 {  // > 75% of max_draft_len(8)
            self.max_draft_len
        } else if avg >= 4.0 {  // > 50%
            self.base_draft_len
        } else if avg >= 2.0 {  // > 25%
            self.min_draft_len + 1
        } else {
            self.min_draft_len
        }
    }

    fn record_accept_count(&mut self, count: u32) {
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(count);
    }
}
```

**Call site changes**:
- In `run_model_decode` (`runner.rs`), resolve draft length via `AdaptiveDraftPolicy::resolve_draft_len()` instead of using `DEFAULT_DRAFT_LEN` or `MAX_DRAFT_LEN`.
- After each n-gram decode step, call `record_accept_count(accept_count)`.

**Fast-path interaction**: The adaptive policy only affects n-gram draft length. The direct pipeline and MTP paths are unchanged.

### 2.2 Affected files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/ngram_accel.rs` | Add `AdaptiveDraftPolicy` struct and methods |
| `crates/ax-engine-mlx/src/runner.rs` | Add policy to `RequestState`, integrate into `run_model_decode` |

## Phase 3 — MTP Logprob Caching

### 3.1 Cache normalization constants

**Problem**: `full_vocab_token_logprob` (`sampling.rs:561-587`) iterates the entire vocabulary to compute `max_l` and `sum` of exp-scaled logits. Called once per MTP rejection-sampling position, this is O(V) per draft token.

**Solution**: The forward pass already computes logits for all positions. Cache the max-logit and exp-sum per position so `full_vocab_token_logprob` can compute log-prob in O(1).

**Implementation**:

```rust
// In sampling.rs or a new module
pub struct LogprobCache {
    max_logits: Vec<f32>,     // max per position
    exp_sums: Vec<f32>,       // sum of exp((logit - max) / temp) per position
    temperature: f32,         // temperature used for normalization
}

impl LogprobCache {
    pub fn from_logits(logits_all: &[f32], seq_len: usize, vocab: usize, temperature: f32) -> Self {
        // Compute max and sum per position in one pass
        // seq_len × vocab flattened buffer
    }

    pub fn token_logprob(&self, token: u32, position: usize, logits_all: &[f32], vocab: usize) -> f32 {
        // O(1): use cached max and sum
        let idx = position * vocab + token as usize;
        let unnorm = ((logits_all[idx] - self.max_logits[position]) / self.temperature).exp();
        (unnorm / self.exp_sums[position]).max(1e-37_f32).ln().max(-30.0)
    }
}
```

**Call site changes**:
- In `verify_draft` (`ngram_accel.rs:1167-1264`), after `eval(&targets)` materializes `logits_all`, build a `LogprobCache` before the accept/reject loop.
- Replace `full_vocab_token_logprob` calls with `cache.token_logprob(...)`.

**Memory tradeoff**: The cache adds 2 × seq_len × 4 bytes (max + sum per position). For seq_len=8, vocab=150K, this is 64 bytes — negligible compared to the logits buffer (8 × 150K × 4 = 4.8MB).

### 3.2 Affected files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/sampling.rs` | Add `LogprobCache` struct, modify `full_vocab_token_logprob` to accept cache |
| `crates/ax-engine-mlx/src/ngram_accel.rs` | Build cache in `verify_draft`, use in accept/reject loop |
| `crates/ax-engine-mlx/src/mtp.rs` | Build cache in MTP verify path, use in rejection sampling |

## Testing Strategy

### Unit tests

- `sampling.rs`: Test `sample_categorical_with_topk_gpu` produces identical results to `sample_categorical` for the same seed, logits, and sampling params.
- `sampling.rs`: Test pre-allocated buffer version produces identical results to allocating version.
- `ngram_accel.rs`: Test `AdaptiveDraftPolicy` converges to correct draft length for known acceptance patterns.
- `sampling.rs`: Test `LogprobCache::token_logprob` matches `full_vocab_token_logprob` for same inputs.

### Integration tests

- `cargo run --release --bin decode-trace` on Qwen 3.6 27B 4-bit with top-k=50, top-p=0.9, temperature=0.7: verify no regression in output quality.
- `cargo run --release --bin decode-trace` on Gemma 4 E2B 4-bit with greedy: verify no regression in decode rate.
- N-gram acceleration test: run with repeating prompt, verify adaptive draft length increases over first 32 steps.

### Benchmark gates

- `cargo bench` with existing inference harness: decode rate must not regress on any model.
- New benchmark row: non-greedy decode (temperature=0.7, top-k=50, top-p=0.9) on Qwen 3.6 27B 4-bit, comparing before/after CPU time per step.

## Rollback Plan

All changes are gated behind existing fast-path patterns:
- Phase 1: New GPU path only engages when `top_k > 0 || top_p < 1.0` AND `temperature > 0` AND no repetition penalty. Existing path is unchanged.
- Phase 2: Adaptive policy defaults to `base_draft_len = DEFAULT_DRAFT_LEN` until window has enough data. Behavior is identical for first 8 steps.
- Phase 3: `LogprobCache` is an additive optimization; `full_vocab_token_logprob` remains as fallback.

Kill switches via env var (following `fastpath.rs` convention):
- `AX_MLX_DECODE_SAMPLING_GPU_TOPK=0` — disable GPU-side argpartition, fall back to CPU sort.
- `AX_MLX_DECODE_ADAPTIVE_DRAFT=0` — disable adaptive draft length, use static `DEFAULT_DRAFT_LEN`.
- `AX_MLX_DECODE_LOGPROB_CACHE=0` — disable logprob caching, use `full_vocab_token_logprob`.
