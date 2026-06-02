# ADR: Decode Speed Optimization Design Decisions

## Status

Proposed

## Context

AX Engine's decode path has three tiers: double-buffer direct pipeline (greedy), n-gram speculative decode, and MTP speculative decode. The `PERFORMANCE-DECODE-GAP.md` investigation established that the 1-6% gap to `mlx_lm.benchmark` on dense models is dominated by per-MLX-op FFI overhead (~800-1300 op dispatches per step). Single-op fusion has a low ceiling (~0.2% per fusion site).

This ADR records the design decisions for optimizing the CPU-side decode hot path — the work between GPU forward completion and the next step's graph build — where we have direct control and proven leverage.

## Decision 1: GPU-side argpartition for exact top-k sampling

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **A1: GPU `argpartition_axis` + CPU small-k sampling** (chosen for exact top-k) | Uses existing MLX op; O(V) GPU partition + O(k log k) CPU sort; proven pattern in `sample_logit_row_topk_gpu` | Requires GPU round-trip for indices + logits; top-p needs extra correctness handling |
| A2: Keep CPU `sort_by` | No code change; simple | O(V log V) CPU sort dominates for 150K vocab |
| A3: GPU `random_categorical` with full logits | Already implemented in `sample_categorical_gpu` | Cannot apply top-k/top-p filters; only works for plain temperature sampling |
| A4: Custom Metal top-k kernel | Maximum performance | High implementation cost; maintenance burden; MLX already provides `argpartition_axis` |
| A5: Fixed-size top-p candidate approximation | Simple extension for top-p-only | Changes sampling semantics when the nucleus exceeds the fixed candidate count |

### Decision

**A1 is chosen for exact `top_k` sampling.** MLX's `argpartition_axis` is already available in `mlx-sys` and used in `sample_logit_row_topk_gpu` (`ngram_accel.rs:1311-1337`). The pattern is:
1. `argpartition_axis(logits, -k, -1)` → indices of top-k elements
2. `slice(indices, [0, vocab-k], [1, vocab])` → top-k indices
3. `take_along_axis(logits, top_indices, -1)` → top-k logits
4. `eval` both → transfer to CPU
5. Sample over the exact top-k candidate set

This reduces CPU work from O(V log V) to O(k log k) where k << V (typically k=50).

`top_p` is not automatically covered by this decision. The current implementation computes the nucleus cutoff against full-vocabulary probability mass. A GPU path for `top_k + top_p` is allowed only when it preserves that full-domain cutoff exactly. `top_p` without `top_k` remains on the existing CPU path in Phase 1.

### Tradeoffs

- GPU round-trip cost for indices + logits: ~2 small array transfers. This is amortized by the elimination of the O(V log V) sort.
- Exact `top_p` support may require a full-domain softmax denominator or gathered full-domain probabilities. Without that proof, fallback is required.
- Does NOT apply when repetition penalty is active (requires random-access logits modification on CPU). Falls back to existing CPU path.

## Decision 2: Pre-allocated sampling buffers in RequestState

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **B1: Per-request reusable buffers** (chosen) | Eliminates per-step allocation; bounded memory; follows request-local decode state ownership | Requires `RequestState` changes; slightly larger per-request memory footprint |
| B2: Thread-local arena allocator | Shared across requests; no per-request overhead | Complex lifetime management; thread-local storage breaks if decode is ever parallelized |
| B3: Stack-allocated small buffer + heap fallback | Fast for small vocabs; no allocation for common case | Vocab size is not known at compile time; fallback still allocates |
| B4: Keep per-step allocation | Simple; no code change | Repeated allocator pressure in the decode hot path |

### Decision

**B1 is chosen.** Add reusable sampling buffers to `RequestState` or an equivalent request-local decode workspace:
- `sampling_probs_buf`: reused for temperature-scaled probability computation
- `sampling_logits_buf`: reused for repetition penalty logits copy
- `sampling_candidates_buf`: reused for `(token, probability)` tuples in filtered CPU sampling

Because `RequestState::new()` currently receives only `num_layers` and `request_id`, the implementation must either pass `cfg.vocab_size` into construction or lazily reserve these buffers on first sampling use. The hot implementation accepts mutable workspace buffers and uses `buf.clear(); buf.extend(...)` to reuse capacity. The existing `sample_categorical` signature remains as an allocating wrapper for tests and cold callers.

### Tradeoffs

- Per-request memory overhead depends on which buffers are retained. The two `Vec<f32>` buffers are 2 × 150K × 4 bytes = 1.2MB for large-vocab models; retaining a full-vocabulary candidates buffer adds more CPU memory. This is still small compared to KV cache, but should be measured under multi-request concurrency.
- Backward compatibility: The existing `sample_categorical` signature is preserved as a wrapper that allocates locally.

## Decision 3: Keep n-gram draft length static for this PRD

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| C1: Rolling acceptance rate window | Simple; self-tuning; no manual configuration | Requires 8+ steps of warmup before adapting; can overfit transient patterns |
| C2: Beta-Bernoulli posterior (already used for confidence gate) | Statistically principled; already implemented | Used for draft confidence, not draft length; conflates two different decisions |
| C3: Prompt-class-based static lengths | Zero warmup; uses existing `classify_prompt_class` | Coarse-grained; doesn't adapt to generation-time pattern changes |
| **C4: Keep existing static draft-length policy** (chosen) | No behavior change; preserves existing linear-attention and MTP hybrid boundaries | May leave tokens on the table for highly predictable patterns |

### Decision

**C4 is chosen for this PRD.** Do not add adaptive n-gram draft length in the active decode-speed work. Keep the existing static policy and existing caps:
- dense models continue using the current `MAX_DRAFT_LEN = 6` path;
- linear-attention models continue using `DEFAULT_DRAFT_LEN` or lower where the current code caps branch/recompute cost;
- MTP+n-gram hybrid source-aware acceptance and telemetry are unchanged.

The rolling-window adaptive policy is deferred to a separate research item. It may be reconsidered only after a benchmark plan proves that longer dense-model verifier batches improve real workloads and that linear-attention partial-reject branch/recompute cost does not regress.

### Tradeoffs

- Lower implementation risk: no new request-local adaptive state, no new source-accounting interaction, and no new production kill switch for n-gram draft length.
- Potential opportunity cost: highly repetitive prompts might benefit from longer dense-model drafts, but that is not evidence-backed enough for this PRD.

## Decision 4: MTP target-probability workspace

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **D1: Reuse the existing lazy GPU target-probability path and add a small workspace** (chosen) | Preserves current GPU softmax/gather behavior; avoids CPU full-logits readback; removes small per-step allocations | Requires `runner.rs` workspace plumbing |
| D2: CPU max-logit + exp-sum cache over materialized logits | O(1) CPU lookup after cache build | Requires CPU materialization or scan of `[verify_len, vocab]`, which can regress the current lazy GPU path |
| D3: Keep current allocating `compute_mtp_target_probs` path | No behavior change | Leaves avoidable per-step small allocations and extraction overhead |
| D4: Approximate target probability from top-k candidates only | Fast; no full-vocab softmax | Incorrect unless explicitly configured as an approximation; rejection sampling needs full-vocab normalization |

### Decision

**D1 is chosen.** The hot MTP path already uses `compute_mtp_target_probs` in `runner.rs`, which builds target probabilities lazily on GPU and extracts only the pending draft-token probabilities before `mtp_accept_count`. Phase 3 keeps that contract and adds reusable request-local workspace for gather indices and extracted target probabilities.

`full_vocab_token_logprob` remains a cold CPU helper/fallback. It is not the current MTP rejection-sampling hot path and should not become one unless benchmarks prove CPU materialization is faster for a specific configuration.

### Tradeoffs

- Memory: O(draft_len) workspace for flat indices and gathered probabilities; negligible for current MTP depth.
- Compute: Keeps the existing GPU softmax/gather path and avoids a new CPU O(V × seq_len) scan.
- Correctness: Full-vocabulary normalization remains the default. Top-k target softmax approximation must stay explicit and auditable.

## Decision 5: Env-var kill switches for active phases

### Alternatives considered

| Alternative | Pros | Cons |
|---|---|---|
| **E1: Per-phase env-var kill switches** (chosen) | Follows existing `fastpath.rs` convention; observable; safe rollback | Adds 2 env vars to audit surface |
| E2: No kill switches | Simpler code | Harder to rollback if regression is found |
| E3: Compile-time feature flags | Zero runtime overhead | Requires rebuild to toggle; not suitable for production rollback |

### Decision

**E1 is chosen.** Following the `fastpath.rs` convention:
- `AX_MLX_DECODE_SAMPLING_GPU_TOPK` (default ON) — GPU-side argpartition for exact top-k sampling
- `AX_MLX_DECODE_MTP_TARGET_PROB_WORKSPACE` (default ON) — MTP target-probability workspace

All use `env_flag_default_on!` macro with kill-switch via `=0`.

## Consequences

### Positive
- Reduced CPU overhead in exact top-k non-greedy sampling path (Decision 1, 2)
- Lower MTP rejection-sampling extraction/allocation overhead without regressing the current lazy GPU path (Decision 4)
- Safe rollback path via env-var kill switches (Decision 5)

### Negative
- Increased code complexity in sampling and MTP target-probability paths
- Per-request memory overhead: ~1.2MB+ for large-vocab sampling buffers (Decision 2) + O(draft_len) MTP target-prob workspace (Decision 4)
- 2 new env vars to document and audit

### Risks
- GPU-side argpartition may have different tie/edge ordering than CPU sort for edge cases (NaN, -inf, equal logits). Mitigated by normalizing candidate ordering before CPU sampling and testing tie cases.
- Top-p semantics can be accidentally changed if the implementation applies nucleus filtering over only the top-k candidate mass. Mitigated by requiring full-domain top-p cutoff or fallback.
- MTP target-prob workspace may accidentally add a second GPU sync if it is not evaluated with argmax/KV refs. Mitigated by tests and stage-profile telemetry.
