# ADR: Decode Speed Optimization Design Decisions

## Status

Proposed

## Context

AX Engine's decode path has three tiers: double-buffer direct pipeline (greedy), n-gram speculative decode, and MTP speculative decode. The `PERFORMANCE-DECODE-GAP.md` investigation established that the 1-6% gap to `mlx_lm.benchmark` on dense models is dominated by per-MLX-op FFI overhead (~800-1300 op dispatches per step). Single-op fusion has a low ceiling (~0.2% per fusion site).

This ADR records the design decisions for optimizing the CPU-side decode hot path — the work between GPU forward completion and the next step's graph build — where we have direct control and proven leverage.

## Decision 1: GPU-side argpartition for top-k/top-p sampling

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **A1: GPU `argpartition_axis` + CPU top-p** (chosen) | Uses existing MLX op; O(V) GPU partition + O(k log k) CPU sort; proven pattern in `sample_logit_row_topk_gpu` | Requires GPU round-trip for indices + logits |
| A2: Keep CPU `sort_by` | No code change; simple | O(V log V) CPU sort dominates for 150K vocab |
| A3: GPU `random_categorical` with full logits | Already implemented in `sample_categorical_gpu` | Cannot apply top-k/top-p filters; only works for plain temperature sampling |
| A4: Custom Metal top-k kernel | Maximum performance | High implementation cost; maintenance burden; MLX already provides `argpartition_axis` |

### Decision

**A1 is chosen.** MLX's `argpartition_axis` is already available in `mlx-sys` and used in `sample_logit_row_topk_gpu` (`ngram_accel.rs:1311-1337`). The pattern is:
1. `argpartition_axis(logits, -k, -1)` → indices of top-k elements
2. `slice(indices, [0, vocab-k], [1, vocab])` → top-k indices
3. `take_along_axis(logits, top_indices, -1)` → top-k logits
4. `eval` both → transfer to CPU
5. Apply top-p on CPU over k candidates

This reduces CPU work from O(V log V) to O(k log k) where k << V (typically k=50).

### Tradeoffs

- GPU round-trip cost for indices + logits: ~2 small array transfers. This is amortized by the elimination of the O(V log V) sort.
- Does NOT apply when repetition penalty is active (requires random-access logits modification on CPU). Falls back to existing CPU path.

## Decision 2: Pre-allocated sampling buffers in RequestState

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **B1: Per-request reusable buffers** (chosen) | Eliminates per-step allocation; bounded memory (2 × vocab_size × 4 bytes) | Requires `RequestState` changes; slightly larger per-request memory footprint |
| B2: Thread-local arena allocator | Shared across requests; no per-request overhead | Complex lifetime management; thread-local storage breaks if decode is ever parallelized |
| B3: Stack-allocated small buffer + heap fallback | Fast for small vocabs; no allocation for common case | Vocab size is not known at compile time; fallback still allocates |
| B4: Keep per-step allocation | Simple; no code change | ~2 allocs/step × 1000 steps = 2000 allocs per request; measurable GC pressure |

### Decision

**B1 is chosen.** Add two `Vec<f32>` buffers to `RequestState`:
- `sampling_probs_buf`: reused for temperature-scaled probability computation
- `sampling_logits_buf`: reused for repetition penalty logits copy

Each buffer is `Vec::with_capacity(cfg.vocab_size)` at construction. The `sample_categorical` function accepts `&mut Vec<f32>` parameters and uses `buf.clear(); buf.extend(...)` to reuse capacity.

### Tradeoffs

- Per-request memory overhead: 2 × 150K × 4 bytes = 1.2MB for large-vocab models. This is negligible compared to KV cache (hundreds of MB).
- Backward compatibility: The existing `sample_categorical` signature is preserved as a wrapper that allocates locally.

## Decision 3: Adaptive n-gram draft length

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **C1: Rolling acceptance rate window** (chosen) | Simple; self-tuning; no manual configuration | Requires 8+ steps of warmup before adapting |
| C2: Beta-Bernoulli posterior (already used for confidence gate) | Statistically principled; already implemented | Used for draft confidence, not draft length; conflates two different decisions |
| C3: Prompt-class-based static lengths | Zero warmup; uses existing `classify_prompt_class` | Coarse-grained; doesn't adapt to generation-time pattern changes |
| C4: Keep static `MAX_DRAFT_LEN = 6` | Simple; no code change | Leaves tokens on the table for highly predictable patterns |

### Decision

**C1 is chosen.** Track a rolling window of 32 recent acceptance counts. Resolve draft length as:
- avg >= 6.0: `max_draft_len = 8`
- avg >= 4.0: `base_draft_len = 4` (DEFAULT_DRAFT_LEN)
- avg >= 2.0: `min_draft_len + 1 = 3`
- avg < 2.0: `min_draft_len = 2`

The window starts empty; for the first 8 steps, use `base_draft_len` (no adaptation).

### Tradeoffs

- Warmup period: First 8 steps use static draft length. This is acceptable because n-gram tables also need warmup to build context.
- Memory: `VecDeque<u32>` of size 32 = 128 bytes per request. Negligible.
- Does NOT affect direct pipeline or MTP paths — only n-gram speculative decode.

## Decision 4: MTP logprob caching

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **D1: Cache max-logit and exp-sum per position** (chosen) | O(1) logprob lookup; minimal memory (2 × seq_len × 4 bytes) | Requires one O(V × seq_len) pass after forward |
| D2: Keep `full_vocab_token_logprob` O(V) scan | No code change; simple | O(V) per rejection position; called 1-3 times per MTP step |
| D3: GPU-side logprob computation | Parallel; no CPU work | Requires new MLX ops; complex graph construction; may not be faster for small seq_len |
| D4: Approximate logprob from top-k candidates only | Fast; no full-vocab scan | Incorrect for rejection sampling (needs full-vocab normalization domain) |

### Decision

**D1 is chosen.** After `verify_draft` materializes `logits_all`, compute max-logit and exp-sum per position in one pass. Store in `LogprobCache`. Replace `full_vocab_token_logprob` calls with `cache.token_logprob(...)` which is O(1).

### Tradeoffs

- Memory: 2 × seq_len × 4 bytes. For seq_len=8, this is 64 bytes — negligible.
- Compute: One O(V × seq_len) pass after forward. This replaces 1-3 O(V) scans, so it's a net win when seq_len >= 2.
- Correctness: `full_vocab_token_logprob` remains as fallback for callers that don't have a cache.

## Decision 5: Env-var kill switches for all phases

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **E1: Per-phase env-var kill switches** (chosen) | Follows existing `fastpath.rs` convention; observable; safe rollback | Adds 3 env vars to audit surface |
| E2: No kill switches | Simpler code | Harder to rollback if regression is found |
| E3: Compile-time feature flags | Zero runtime overhead | Requires rebuild to toggle; not suitable for production rollback |

### Decision

**E1 is chosen.** Following the `fastpath.rs` convention:
- `AX_MLX_DECODE_SAMPLING_GPU_TOPK` (default ON) — GPU-side argpartition for top-k/top-p
- `AX_MLX_DECODE_ADAPTIVE_DRAFT` (default ON) — adaptive n-gram draft length
- `AX_MLX_DECODE_LOGPROB_CACHE` (default ON) — MTP logprob caching

All use `env_flag_default_on!` macro with kill-switch via `=0`.

## Consequences

### Positive
- Reduced CPU overhead in non-greedy sampling path (Decision 1, 2)
- Higher tokens-per-forward for n-gram speculative decode (Decision 3)
- Lower MTP rejection sampling overhead (Decision 4)
- Safe rollback path via env-var kill switches (Decision 5)

### Negative
- Increased code complexity in sampling path (3 new functions, 2 new structs)
- Per-request memory overhead: ~1.2MB for large-vocab models (Decision 2) + 128 bytes for adaptive draft window (Decision 3) + 64 bytes for logprob cache (Decision 4)
- 3 new env vars to document and audit

### Risks
- GPU-side argpartition may have different numerical behavior than CPU sort for edge cases (NaN, -inf). Mitigated by existing NaN/inf handling in `sample_categorical`.
- Adaptive draft length may over-adapt to transient patterns. Mitigated by 32-step window smoothing.
- Logprob cache may become stale if temperature changes mid-generation. Mitigated by storing temperature in cache and checking on each call.
