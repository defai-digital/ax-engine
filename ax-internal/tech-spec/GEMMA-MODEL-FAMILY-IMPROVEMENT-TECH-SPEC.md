# Tech Spec: Gemma Model Family Improvements

**Status:** In Progress
**Date:** 2026-07-06
**Author:** Engineering
**PRD:** `ax-internal/prd/GEMMA-MODEL-FAMILY-IMPROVEMENT-PRD.md`
**ADR:** `ax-internal/adr/ADR-040-gemma-layer-shell-extraction.md`

## Summary

This tech spec details the implementation of five improvement phases identified
during the Gemma, DiffusionGemma, and EmbeddingGemma code review:

1. Safety & Observability (SAFETY comments, per-layer profiling, cache
   telemetry).
2. Telemetry & Diagnostics (padding mask caching, probe documentation).
3. Error Handling Unification (thiserror enums, kv_heads standardization).
4. Performance — Assistant MTP Compilation (compiled closure for Gemma4
   assistant draft forward).
5. Architecture — Layer Shell Extraction (shared causal/bidirectional shell).

Each phase is independently shippable and independently reversible. Phases are
ordered so that safety/documentation work (Phase 1) lands before diagnostics
(Phase 2), error handling unification (Phase 3) enables clean compilation
work (Phase 4), and architectural refactoring (Phase 5) lands last to avoid
disrupting the compiled closures.

---

## Phase 1 — Safety & Observability

### Objective

Document raw pointer safety invariants, add per-layer profiling to the
bidirectional denoiser forward, and expose embedding cache telemetry.

### Changes

**1.1 SAFETY comments for raw pointer captures (Finding 1)**

In `crates/ax-engine-mlx/src/diffusion.rs` (lines 786–803, 893–903), add
`// SAFETY:` comments to every closure that captures raw pointers. Document
the lifetime invariant: the closure is consumed within the same function scope
and never leaks beyond the stack frame.

```rust
// diffusion.rs — full-pipeline closure with raw pointer capture.
// SAFETY: `weights_ptr` is a shared reference to model weights loaded at
// session start and never freed during the closure's lifetime. The closure
// is consumed synchronously within this function scope via `mlx_sys::eval`
// and is not stored, cloned, or returned. The pointer remains valid for
// the duration of the eval call.
let closure = mlx_sys::compile(|inputs| {
    let w = unsafe { &*weights_ptr };
    // ... forward computation using w ...
});
```

Apply the same pattern to the denoiser closure at lines 893–903, documenting
the specific lifetime of each captured pointer.

**1.2 Per-layer profiling for bidirectional denoiser forward (Finding 6)**

Add `BidirectionalProfileStage` enum mirroring the existing
`DecodeProfileStage` pattern from causal `layer_forward`. Instrument
`forward_bidirectional` in `diffusion.rs` (lines 1073–1141) with per-stage
timing gated behind `AX_MLX_DECODE_PROFILE`.

```rust
// diffusion.rs — bidirectional profiling stages.
#[derive(Clone, Debug)]
pub(crate) enum BidirectionalProfileStage {
    Embed,
    Layer { index: usize },
    LmHead,
}

// In forward_bidirectional, gated behind AX_MLX_DECODE_PROFILE:
if profile_enabled() {
    profile_enter(BidirectionalProfileStage::Layer { index: layer_idx });
}
let hidden = layer_forward_bidirectional(/* ... */);
if profile_enabled() {
    profile_exit(BidirectionalProfileStage::Layer { index: layer_idx });
}
```

**1.3 Embedding cache hit/miss telemetry (Finding 13)**

Add `AtomicU64` counters to `EmbeddingCache` in `diffusion.rs`
(lines 1012–1047). Expose an accessor function.

```rust
// diffusion.rs — embedding cache telemetry.
pub(crate) struct EmbeddingCache {
    // ... existing fields ...
    hits: AtomicU64,
    misses: AtomicU64,
}

impl EmbeddingCache {
    pub(crate) fn get_or_compute(&self, key: &CacheKey) -> MlxArray {
        if let Some(cached) = self.lookup(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            cached
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            self.compute_and_store(key)
        }
    }

    /// Returns (hits, misses) for observability dashboards.
    pub(crate) fn embedding_cache_stats(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }
}
```

### Affected Files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/diffusion.rs` | Add `// SAFETY:` comments to pointer captures (lines 786–803, 893–903) |
| `crates/ax-engine-mlx/src/diffusion.rs` | Add `BidirectionalProfileStage` enum and profiling instrumentation (lines 1073–1141) |
| `crates/ax-engine-mlx/src/diffusion.rs` | Add `AtomicU64` hit/miss counters to `EmbeddingCache` (lines 1012–1047) |

### Validation

```bash
cargo test -p ax-engine-mlx diffusion
cargo test -p ax-engine-mlx embedding_cache
AX_MLX_DECODE_PROFILE=1 cargo test -p ax-engine-mlx bidirectional_profile
cargo clippy --all-targets --all-features -- -D warnings
```

### Go/No-Go

- ✓ **Go:** All SAFETY comments present; profiling stages emit timing when
  enabled; cache counters increment correctly.
- ✗ **Stop:** Any test regression in DiffusionGemma output quality.

---

## Phase 2 — Telemetry & Diagnostics

### Objective

Cache the bidirectional padding mask and document diagnostic-only probe
behavior.

### Changes

**2.1 Cache build_bidirectional_padding_mask (Finding 10)**

In `crates/ax-engine-mlx/src/model/mod.rs` (lines 1531–1583), cache the
padding mask in a thread-local `HashMap` keyed by
`(batch_size, max_seq_len, actual_lens_hash)`. Add 64-entry LRU eviction to
bound memory usage. The all-zeros (no-padding) cache already exists — this
extends caching to the padded case.

```rust
// model/mod.rs — cached bidirectional padding mask.
thread_local! {
    static PADDING_MASK_CACHE: RefCell<LruCache<(usize, usize, u64), MlxArray>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(64).unwrap()));
}

fn build_bidirectional_padding_mask_cached(
    batch_size: usize,
    max_seq_len: usize,
    actual_lens: &[usize],
) -> MlxArray {
    let key_hash = fxhash::hash64(actual_lens);
    let key = (batch_size, max_seq_len, key_hash);
    PADDING_MASK_CACHE.with(|c| {
        if let Some(cached) = c.borrow().get(&key) {
            return cached.clone();
        }
        let mask = build_bidirectional_padding_mask(batch_size, max_seq_len, actual_lens);
        c.borrow_mut().put(key, mask.clone());
        mask
    })
}
```

**2.2 normed_mean_pool_probe documentation (Finding 14)**

Add a doc-comment to `normed_mean_pool_probe` in `model/mod.rs`
(lines 1899–1928) clarifying that it is a diagnostic-only probe, not a
production path. Per-row allocations are acceptable for diagnostic use. No
code change unless the probe is promoted to production.

```rust
/// Diagnostic probe: compute mean-pooled embeddings with per-row norm.
///
/// **Not a production path.** Per-row allocations are intentional for
/// diagnostic clarity. If this probe is promoted to production, replace
/// per-row allocation with a single batched operation.
#[cfg(feature = "diagnostic-probes")]
pub(crate) fn normed_mean_pool_probe(/* ... */) -> MlxArray {
    // ...
}
```

### Affected Files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/model/mod.rs` | Add `build_bidirectional_padding_mask_cached` with LRU (lines 1531–1583) |
| `crates/ax-engine-mlx/src/model/mod.rs` | Add diagnostic-only doc-comment to `normed_mean_pool_probe` (lines 1899–1928) |

### Validation

```bash
cargo test -p ax-engine-mlx embed_batch
cargo test -p ax-engine-mlx padding_mask_cache
```

### Go/No-Go

- ✓ **Go:** Padding mask cache hit rate >90% for typical batch sizes; no
  regression in embedding output.
- ✗ **Stop:** Cache memory growth exceeds 64-entry bound.

---

## Phase 3 — Error Handling Unification

### Objective

Replace String/&str error types with thiserror enums across the Gemma model
family, and standardize `kv_heads` error handling.

### Changes

**3.1 Gemma4UnifiedError thiserror enum (Finding 8)**

In `crates/ax-engine-mlx/src/gemma4_unified.rs` (line 34), define a typed
error enum replacing `Result<_, String>`.

```rust
// gemma4_unified.rs — typed error enum.
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

// Migrate all Result<_, String> returns to Result<_, Gemma4UnifiedError>.
```

**3.2 Shared ModelError for cross-cutting errors (Finding 12)**

Define a shared error type in `model/families/mod.rs` covering errors common
across model families (kv_heads mismatch, unsupported configuration, etc.).

```rust
// model/families/mod.rs — shared model family errors.
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

Migrate `&str` errors in `gemma4_assistant_mtp.rs` forward path to use
`ModelError` variants.

**3.3 Standardize kv_heads error handling (Finding 11)**

Audit all `kv_heads` handling sites (`mod.rs:1232`, `standard.rs:186`) and
standardize on `expect` with a descriptive message. Configuration errors of
this nature should crash early rather than propagate silently.

```rust
// Before (inconsistent):
let kv_heads = cfg.num_kv_heads.unwrap_or(cfg.num_attention_heads);

// After (standardized):
let kv_heads = cfg.num_kv_heads
    .expect("model config must specify num_kv_heads; None is not valid for Gemma models");
```

### Affected Files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/gemma4_unified.rs` | Define `Gemma4UnifiedError`, migrate from `Result<_, String>` |
| `crates/ax-engine-mlx/src/model/families/mod.rs` | Define shared `ModelError` enum |
| `crates/ax-engine-mlx/src/gemma4_assistant_mtp.rs` | Migrate `&str` errors to `ModelError` |
| `crates/ax-engine-mlx/src/model/mod.rs` | Standardize `kv_heads` to `expect` (line ~1232) |
| `crates/ax-engine-mlx/src/model/families/standard.rs` | Standardize `kv_heads` to `expect` (line ~186) |

### Validation

```bash
cargo build --workspace
cargo test -p ax-engine-mlx gemma4_unified
cargo clippy --all-targets --all-features -- -D warnings
```

### Go/No-Go

- ✓ **Go:** All Gemma modules compile clean; no `Result<_, String>` remains
  in Gemma model paths; clippy clean.
- ✗ **Stop:** Any caller that pattern-matches on the old String error type
  breaks without an updated match arm.

---

## Phase 4 — Performance: Assistant MTP Compilation

### Objective

Wrap the Gemma4 assistant MTP forward path in an `MlxClosure` for improved
draft-throughput during speculative decoding.

### Changes

**4.1 Wrap assistant forward in MlxClosure (Finding 2)**

In `crates/ax-engine-mlx/src/model/mod.rs` (lines 1010–1069), wrap
`gemma4_assistant_forward_one`'s layer loop in an `MlxClosure`. The
assistant is stateless per step (reads target KV, writes no cache), making
it an ideal compilation target.

```rust
// model/mod.rs — compiled assistant MTP forward.
static ASSISTANT_COMPILE_CACHE: Lazy<DashMap<(usize, usize), MlxClosure>> =
    Lazy::new(DashMap::new);

fn gemma4_assistant_forward_compiled(
    cfg: &ModelConfig,
    weights: &Weights,
    hidden: &MlxArray,
    target_kv: &[KvCache],
    rope_offset: &MlxArray,
) -> MlxArray {
    if !assistant_compile_enabled() {
        return gemma4_assistant_forward_imperative(cfg, weights, hidden, target_kv, rope_offset);
    }

    let key = (cfg.hidden_size, cfg.num_layers);
    let closure = ASSISTANT_COMPILE_CACHE
        .entry(key)
        .or_insert_with(|| {
            build_assistant_forward_closure(cfg, weights)
        })
        .clone();

    closure.call(&[hidden.clone(), rope_offset.clone()])
}
```

**4.2 Environment flag gate**

Add `AX_MLX_GEMMA4_ASSISTANT_COMPILE` env flag in `fastpath.rs`, defaulting
to off for initial rollout.

```rust
// fastpath.rs — assistant MTP compile gate.
pub(crate) fn assistant_compile_enabled() -> bool {
    std::env::var("AX_MLX_GEMMA4_ASSISTANT_COMPILE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(false)
}
```

**4.3 Size-gated compilation**

Only compile for specific `(hidden_size, layer_count)` at first call. If the
model dimensions don't match a known assistant configuration, fall back to
the imperative path.

### Affected Files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/model/mod.rs` | Add compiled assistant forward + cache (lines 1010–1069) |
| `crates/ax-engine-mlx/src/fastpath.rs` | Add `assistant_compile_enabled()` env flag |

### Validation

```bash
cargo test -p ax-engine-mlx gemma4_assistant
# Verify acceptance rate unchanged:
AX_MLX_GEMMA4_ASSISTANT_COMPILE=1 cargo test -p ax-engine-mlx mtp_acceptance
```

### Go/No-Go

- ✓ **Go:** Assistant MTP depth-2 acceptance rate is bit-exact with
  imperative baseline; compiled path shows measurable throughput improvement.
- ✗ **Stop:** Acceptance rate diverges or compilation fails for any Gemma4
  assistant variant.

**Rollback:** Set `AX_MLX_GEMMA4_ASSISTANT_COMPILE=0` to revert to imperative
path.

---

## Phase 5 — Architecture: Layer Shell Extraction

### Objective

Extract a shared `layer_shell` function from `layer_forward` and
`layer_forward_bidirectional` in `standard.rs`, eliminating ~200 lines of
code duplication.

See **ADR-040** (`ax-internal/adr/ADR-040-gemma-layer-shell-extraction.md`)
for the full architectural decision.

### Changes

**5.1 Extract layer_shell (Finding 5)**

In `crates/ax-engine-mlx/src/model/families/standard.rs`, extract the shared
computation into `layer_shell(cfg, weights, hidden, position, mask_opt,
kv_cache, attn_fn)`. The attention closure takes `(q, k, v, mask, kv_cache)`
and returns `(attn_output, new_kv)`.

```rust
// standard.rs — shared layer shell.
#[inline]
fn layer_shell(
    cfg: &StandardConfig,
    weights: &StandardLayerWeights,
    hidden: &MlxArray,
    position: &MlxArray,
    mask_opt: Option<&MlxArray>,
    kv_cache: Option<&mut KvCache>,
    attn_fn: impl FnOnce(&MlxArray, &MlxArray, &MlxArray, Option<&MlxArray>, Option<&mut KvCache>)
        -> (MlxArray, Option<MlxArray>),
) -> MlxArray {
    // 1. QKV projection
    let q = weights.wq.matmul(hidden);
    let k = weights.wk.matmul(hidden);
    let v = weights.wv.matmul(hidden);

    // 2. Reshape + QK norm
    let (q, k) = reshape_and_qk_norm(cfg, weights, q, k);

    // 3. RoPE
    let (q, k) = apply_rope(cfg, q, k, position);

    // 4. Attention (injected via closure)
    let (attn_out, _new_kv) = attn_fn(&q, &k, &v, mask_opt, kv_cache);

    // 5. Output projection
    let proj = weights.wo.matmul(&attn_out);

    // 6. Residual + FFN (dense or MoE)
    let residual = hidden + &proj;
    let ffn_out = apply_ffn(cfg, weights, &residual);
    residual + &ffn_out
}
```

**5.2 Update layer_forward to call layer_shell**

```rust
// standard.rs — causal forward delegates to shell.
pub(crate) fn layer_forward(/* ... */) -> MlxArray {
    layer_shell(cfg, weights, hidden, position, mask_opt, kv_cache,
        |q, k, v, mask, cache| causal_attention(q, k, v, mask, cache))
}
```

**5.3 Update layer_forward_bidirectional to call layer_shell**

```rust
// standard.rs — bidirectional forward delegates to shell.
pub(crate) fn layer_forward_bidirectional(/* ... */) -> MlxArray {
    layer_shell(cfg, weights, hidden, position, mask_opt, kv_cache,
        |q, k, v, mask, cache| bidirectional_attention(q, k, v, mask, cache))
}
```

### Affected Files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/model/families/standard.rs` | Extract `layer_shell`, update both forward paths (lines 56–786, 962–1188) |

### Validation

```bash
# Bit-exact causal forward output
cargo test -p ax-engine-mlx gemma_causal_layer_forward

# Bit-exact bidirectional forward output
cargo test -p ax-engine-mlx gemma_bidirectional_layer_forward

# DiffusionGemma convergence unchanged
cargo test -p ax-engine-mlx diffusion_gemma

# Full workspace build + clippy
cargo build --workspace
cargo clippy --all-targets --all-features -- -D warnings
```

### Go/No-Go

- ✓ **Go:** Both forward paths produce bit-exact output vs pre-extraction
  baseline; DiffusionGemma convergence unchanged.
- ✗ **Stop:** Any numerical divergence in forward output or convergence
  regression.

---

## Deferred Findings

The following findings are intentionally deferred and require no code change
in this initiative:

| # | Finding | Reason for Deferral |
|---|---------|---------------------|
| 3 | KV concat buffer divergence reachable via opt-in | Already gated off by default. Retirement path is documented. No active test coverage; adding tests for dead code is not warranted. |
| 4 | No compiled closure for causal commit pass | Causal commit pass has side-effecting KV writes that prevent full compilation. Compilation would require restructuring the KV write pattern, which is out of scope. |
| 7 | Batched decode per-row RoPE is O(B) dispatches | Requires a batched RoPE Metal kernel. Out of scope for this Rust-level improvement initiative. |
| 9 | EmbeddingCache::needs_refresh forces GPU sync | Only affects the imperative fallback path, which is not the default. The compiled path bypasses this check entirely. |

---

## Rollback Plan (All Phases)

Each phase is independently reversible:

- **Phase 1:** Remove SAFETY comments (documentation only, zero functional
  impact). Remove profiling instrumentation (zero functional impact when
  env flag is unset). Remove cache counters (zero functional impact).
- **Phase 2:** Revert padding mask cache to per-call construction. Remove
  doc-comment (documentation only).
- **Phase 3:** Revert `Gemma4UnifiedError` to `Result<_, String>`. Revert
  `ModelError` to `&str` errors. Revert `expect` to `unwrap_or`.
- **Phase 4:** Set `AX_MLX_GEMMA4_ASSISTANT_COMPILE=0` to revert to
  imperative path.
- **Phase 5:** Revert `layer_shell` extraction, restore separate
  `layer_forward` and `layer_forward_bidirectional` implementations.

## Implementation Order

```text
Phase 1 (safety & observability)          ← lands first, zero-risk
  │
  ├── Phase 2 (telemetry & diagnostics)   ← can run parallel with Phase 3
  │
  └── Phase 3 (error handling)            ← enables clean Phase 4
        │
        └── Phase 4 (assistant MTP compile) ← depends on unified errors
              │
              └── Phase 5 (layer shell extraction)  ← lands last, highest risk
```
