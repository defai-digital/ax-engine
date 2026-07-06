# Implementation Plan: Gemma Model Family Improvements

**Date:** 2026-07-06
**Goal:** Improve safety documentation, observability, error handling, performance, and code hygiene across Gemma, DiffusionGemma, and EmbeddingGemma implementations
**PRD:** `ax-internal/prd/GEMMA-MODEL-FAMILY-IMPROVEMENT-PRD.md`
**Tech Spec:** `ax-internal/tech-spec/GEMMA-MODEL-FAMILY-IMPROVEMENT-TECH-SPEC.md`

---

## Phase Ordering and Dependencies

```text
Phase 1 (Safety & Observability)
  │
  ├──→ Phase 2 (Telemetry & Diagnostics) ──→ Phase 3 (Error Handling)
  │                                                │
  │                                                ▼
  │                                    Phase 4 (Assistant MTP Compile)
  │                                                │
  │                                                ▼
  │                                    Phase 5 (Layer Shell Extraction)
  │
  └──→ Evaluation Checkpoint
```

- Phase 1 has no dependencies (can start immediately)
- Phase 2 can run in parallel with Phase 1
- Phase 3 can run in parallel with Phase 2
- Phase 4 depends on Phase 3 (error types need unification first)
- Phase 5 depends on Phase 4 (layer shell extraction must not break compiled
  closures)
- Evaluation Checkpoint after all phases complete

---

## Phase 1 — Safety & Observability

- **Tech Spec:** Phase 1 in `ax-internal/tech-spec/GEMMA-MODEL-FAMILY-IMPROVEMENT-TECH-SPEC.md`
- **Files:** `crates/ax-engine-mlx/src/diffusion.rs`
- **Estimated effort:** 3h
- **Risk:** None

### Changes

**1-A** Add `// SAFETY:` comments to raw pointer captures in `diffusion.rs`:

- Lines 786–803: full-pipeline closure pointer capture
- Lines 893–903: denoiser closure pointer capture
- Document the lifetime invariant for each: closure consumed within function
  scope, never leaks beyond stack frame

**1-B** Add per-layer profiling to `forward_bidirectional` in `diffusion.rs`:

- Define `BidirectionalProfileStage` enum: `{ Embed, Layer { index }, LmHead }`
- Instrument each layer call with `profile_enter`/`profile_exit` gated behind
  `AX_MLX_DECODE_PROFILE`
- Mirror the existing `DecodeProfileStage` pattern from causal forward

**1-C** Add embedding cache hit/miss telemetry:

- Add `AtomicU64` fields `hits` and `misses` to `EmbeddingCache`
- Increment in `get_or_compute` on cache hit/miss
- Add accessor: `embedding_cache_stats() -> (u64, u64)`

### Validation

```bash
cargo test -p ax-engine-mlx diffusion
cargo test -p ax-engine-mlx embedding_cache
AX_MLX_DECODE_PROFILE=1 cargo test -p ax-engine-mlx bidirectional_profile
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

### Go/No-Go

- ✓ Go: All SAFETY comments present; profiling emits timing when enabled;
  cache counters increment correctly
- ✗ Stop: Any test regression in DiffusionGemma output quality

---

## Phase 2 — Telemetry & Diagnostics

- **Tech Spec:** Phase 2 in `ax-internal/tech-spec/GEMMA-MODEL-FAMILY-IMPROVEMENT-TECH-SPEC.md`
- **Files:** `crates/ax-engine-mlx/src/model/mod.rs`
- **Estimated effort:** 2h
- **Risk:** Low

### Changes

**2-A** Cache `build_bidirectional_padding_mask` in `model/mod.rs`:

- Thread-local `HashMap` keyed by `(batch_size, max_seq_len, actual_lens_hash)`
- 64-entry LRU eviction to bound memory
- Extends the existing all-zeros (no-padding) cache to the padded case

```rust
thread_local! {
    static PADDING_MASK_CACHE: RefCell<LruCache<(usize, usize, u64), MlxArray>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(64).unwrap()));
}
```

**2-B** Add diagnostic-only doc-comment to `normed_mean_pool_probe`:

- Clarify that per-row allocations are intentional for diagnostic use
- Add `#[cfg(feature = "diagnostic-probes")]` gate if not already present
- No code change to the probe body

### Validation

```bash
cargo test -p ax-engine-mlx embed_batch
cargo test -p ax-engine-mlx padding_mask_cache
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

### Go/No-Go

- ✓ Go: Padding mask cache hit rate >90% for typical batch sizes; no
  regression in embedding output
- ✗ Stop: Cache memory growth exceeds 64-entry bound

---

## Phase 3 — Error Handling Unification

- **Tech Spec:** Phase 3 in `ax-internal/tech-spec/GEMMA-MODEL-FAMILY-IMPROVEMENT-TECH-SPEC.md`
- **Files:** `crates/ax-engine-mlx/src/gemma4_unified.rs`, `crates/ax-engine-mlx/src/model/families/mod.rs`, `crates/ax-engine-mlx/src/gemma4_assistant_mtp.rs`, `crates/ax-engine-mlx/src/model/mod.rs`, `crates/ax-engine-mlx/src/model/families/standard.rs`
- **Estimated effort:** 4h
- **Risk:** Medium

### Changes

**3-A** Define `Gemma4UnifiedError` thiserror enum in `gemma4_unified.rs`:

```rust
#[derive(Debug, thiserror::Error)]
pub(crate) enum Gemma4UnifiedError {
    #[error("model configuration error: {0}")]
    ConfigError(String),
    #[error("weight loading failed: {0}")]
    WeightLoadError(String),
    #[error("forward pass failed: {0}")]
    ForwardError(String),
    #[error("kv_heads mismatch: expected {expected}, got {actual}")]
    KvHeadsMismatch { expected: usize, actual: usize },
}
```

- Migrate all `Result<_, String>` returns in `gemma4_unified.rs` to use
  `Gemma4UnifiedError`
- Audit callers that pattern-match on the old String error type

**3-B** Define shared `ModelError` enum in `model/families/mod.rs`:

```rust
#[derive(Debug, thiserror::Error)]
pub(crate) enum ModelError {
    #[error("kv_heads mismatch: expected {expected}, got {actual}")]
    KvHeadsMismatch { expected: usize, actual: usize },
    #[error("unsupported model configuration: {0}")]
    UnsupportedConfig(String),
    #[error("weight tensor missing: {0}")]
    MissingWeight(String),
}
```

- Migrate `&str` errors in `gemma4_assistant_mtp.rs` forward path to
  `ModelError` variants

**3-C** Standardize `kv_heads` error handling:

- `mod.rs:1232`: change to `cfg.num_kv_heads.expect("...descriptive message...")`
- `standard.rs:186`: change to `cfg.num_kv_heads.expect("...descriptive message...")`
- Use consistent descriptive messages across both sites

### Validation

```bash
cargo build --workspace
cargo test -p ax-engine-mlx gemma4_unified
cargo test -p ax-engine-mlx gemma4_assistant
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

### Go/No-Go

- ✓ Go: All Gemma modules compile clean; no `Result<_, String>` remains in
  Gemma model paths; clippy clean
- ✗ Stop: Any caller match-arm breakage without an updated pattern

---

## Phase 4 — Performance: Assistant MTP Compilation

- **Tech Spec:** Phase 4 in `ax-internal/tech-spec/GEMMA-MODEL-FAMILY-IMPROVEMENT-TECH-SPEC.md`
- **Files:** `crates/ax-engine-mlx/src/model/mod.rs`, `crates/ax-engine-mlx/src/fastpath.rs`
- **Estimated effort:** 6h
- **Risk:** Medium

### Changes

**4-A** Wrap `gemma4_assistant_forward_one` layer loop in `MlxClosure`:

- Create `build_assistant_forward_closure(cfg, weights)` that compiles the
  layer loop for a specific `(hidden_size, layer_count)` pair
- Cache in `ASSISTANT_COMPILE_CACHE: DashMap<(usize, usize), MlxClosure>`
- The assistant is stateless per step (reads target KV, no cache writes)

**4-B** Add `AX_MLX_GEMMA4_ASSISTANT_COMPILE` env flag in `fastpath.rs`:

- Default: off for initial rollout
- Opt-in: `AX_MLX_GEMMA4_ASSISTANT_COMPILE=1`
- Document the flag in a code comment

**4-C** Size-gate compilation:

- Only compile for known `(hidden_size, layer_count)` pairs
- Fall back to imperative path for unknown configurations

**4-D** Verify acceptance rate:

- Run assistant MTP depth-2 acceptance test with compile enabled
- Acceptance rate must be bit-exact with imperative baseline

### Validation

```bash
cargo test -p ax-engine-mlx gemma4_assistant
AX_MLX_GEMMA4_ASSISTANT_COMPILE=1 cargo test -p ax-engine-mlx mtp_acceptance
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

### Go/No-Go

- ✓ Go: Acceptance rate bit-exact with imperative baseline; compiled path
  shows measurable throughput improvement
- ✗ Stop: Acceptance rate diverges or compilation fails for any Gemma4
  assistant variant

---

## Phase 5 — Architecture: Layer Shell Extraction

- **ADR:** `ax-internal/adr/ADR-040-gemma-layer-shell-extraction.md`
- **Tech Spec:** Phase 5 in `ax-internal/tech-spec/GEMMA-MODEL-FAMILY-IMPROVEMENT-TECH-SPEC.md`
- **Files:** `crates/ax-engine-mlx/src/model/families/standard.rs`
- **Estimated effort:** 8h
- **Risk:** Medium

### Changes

**5-A** Extract `layer_shell` function in `standard.rs`:

- Signature: `fn layer_shell(cfg, weights, hidden, position, mask_opt, kv_cache, attn_fn) -> MlxArray`
- The `attn_fn` closure takes `(q, k, v, mask, kv_cache)` and returns
  `(attn_output, new_kv)`
- Add `#[inline]` to ensure the compiler can optimize through the closure

**5-B** Update `layer_forward` to delegate to `layer_shell`:

```rust
pub(crate) fn layer_forward(/* ... */) -> MlxArray {
    layer_shell(cfg, weights, hidden, position, mask_opt, kv_cache,
        |q, k, v, mask, cache| causal_attention(q, k, v, mask, cache))
}
```

**5-C** Update `layer_forward_bidirectional` to delegate to `layer_shell`:

```rust
pub(crate) fn layer_forward_bidirectional(/* ... */) -> MlxArray {
    layer_shell(cfg, weights, hidden, position, mask_opt, kv_cache,
        |q, k, v, mask, cache| bidirectional_attention(q, k, v, mask, cache))
}
```

**5-D** Verify bit-exact output:

- Compare causal forward output before/after extraction
- Compare bidirectional forward output before/after extraction
- Compare DiffusionGemma convergence before/after extraction

### Validation

```bash
cargo test -p ax-engine-mlx gemma_causal_layer_forward
cargo test -p ax-engine-mlx gemma_bidirectional_layer_forward
cargo test -p ax-engine-mlx diffusion_gemma
cargo build --workspace
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

### Go/No-Go

- ✓ Go: Both forward paths produce bit-exact output vs pre-extraction
  baseline; DiffusionGemma convergence unchanged
- ✗ Stop: Any numerical divergence in forward output or convergence
  regression

---

## Evaluation Checkpoint

After all phases complete, assess:

1. All High and Medium findings from PRD resolved (7 of 14 findings
   addressed; 7 deferred with documented rationale)
2. Zero regressions in DiffusionGemma convergence tests
3. Zero regressions in Gemma4 assistant MTP acceptance rate
4. Bit-exact output for both causal and bidirectional layer forwards
5. Observability: profiling stages emit timing, cache counters increment,
   error types are typed
6. CI gate clean: `cargo fmt`, `cargo clippy`, `cargo test`

---

## Risk Register

| Risk | Probability | Mitigation |
| --- | --- | --- |
| Layer shell extraction closure overhead in hot path | Low | `#[inline]` annotation; benchmark before/after |
| Assistant MTP compile cache invalidation under model reload | Low | Strict (hidden_size, layer_count) cache key; imperative fallback |
| Error type migration breaks caller match arms | Medium | Exhaustive match audit per module; incremental migration |
| Per-layer profiling overhead when env flag is unset | Very Low | Zero-cost when `AX_MLX_DECODE_PROFILE` is not set |
| Padding mask cache LRU eviction under unusual batch patterns | Very Low | 64-entry bound prevents unbounded growth |

---

## Quick Reference

| Item | File | Approx Line |
| --- | --- | --- |
| diffusion.rs full-pipeline closure | `crates/ax-engine-mlx/src/diffusion.rs` | ~786–803 |
| diffusion.rs denoiser closure | `crates/ax-engine-mlx/src/diffusion.rs` | ~893–903 |
| diffusion.rs EmbeddingCache | `crates/ax-engine-mlx/src/diffusion.rs` | ~1012–1047 |
| diffusion.rs forward_bidirectional | `crates/ax-engine-mlx/src/diffusion.rs` | ~1073–1141 |
| standard.rs layer_forward (causal) | `crates/ax-engine-mlx/src/model/families/standard.rs` | ~56–786 |
| standard.rs layer_forward_bidirectional | `crates/ax-engine-mlx/src/model/families/standard.rs` | ~962–1188 |
| standard.rs batched decode RoPE | `crates/ax-engine-mlx/src/model/families/standard.rs` | ~866–920 |
| gemma4_unified.rs errors | `crates/ax-engine-mlx/src/gemma4_unified.rs` | ~34 |
| model/mod.rs assistant MTP forward | `crates/ax-engine-mlx/src/model/mod.rs` | ~1010–1069 |
| model/mod.rs kv_heads handling | `crates/ax-engine-mlx/src/model/mod.rs` | ~1232 |
| model/mod.rs padding mask builder | `crates/ax-engine-mlx/src/model/mod.rs` | ~1531–1583 |
| model/mod.rs normed_mean_pool_probe | `crates/ax-engine-mlx/src/model/mod.rs` | ~1899–1928 |
| fastpath env flags | `crates/ax-engine-mlx/src/fastpath.rs` | ~204–688 |
| families/mod.rs kv_heads handling | `crates/ax-engine-mlx/src/model/families/standard.rs` | ~186 |
