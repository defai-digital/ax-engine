# Tech Spec: Qwen 3.6 / Qwen 3 Embedding Improvements

**Status:** In Progress
**Date:** 2026-07-06
**Author:** Engineering
**PRD:** `ax-internal/prd/QWEN-36-EMBEDDING-IMPROVEMENT-PRD.md`
**ADR:** `ax-internal/adr/ADR-039-embedding-bidirectional-attention-mask.md`

## Summary

This tech spec details the implementation of five improvement areas identified
during the Qwen 3.6 / Qwen 3 embedding code review:

1. Bidirectional attention mask for embeddings.
2. Dense FFN decode compile stabilization.
3. Compiled mean-pool closure.
4. Fused embedding dense head.
5. Compiled-closure fallback tracing.

Each phase is independently shippable and independently reversible. Phases are
ordered so that correctness work (Phase 1) lands before performance work
(Phases 2–4) and observability work (Phase 5) closes the loop.

---

## Phase 1 — Embedding Bidirectional Attention Mask

### Objective

Replace implicit causal masking with explicit bidirectional masking in
embedding forward paths.

### Changes

1.1 In `layer_forward_dense_embed` (`crates/ax-engine-mlx/src/model/mod.rs`,
    ~line 1333), replace `mask_opt = None` with an explicit all-zeros
    bidirectional mask tensor of shape `[1, 1, seq_len, seq_len]` in bfloat16.

1.2 Add a helper function `build_bidirectional_mask(seq_len: usize) -> MlxArray`
    that constructs the bf16 zero mask, cached per `seq_len` to avoid repeated
    allocation.

1.3 For EmbeddingGemma paths that already use bidirectional padding masks,
    verify unified behavior.

1.4 Add parametrized tests covering (Qwen3-last, Qwen3-mean, Qwen3-cls,
    EmbeddingGemma-mean) to validate correctness.

```rust
/// Build an all-zeros additive bidirectional attention mask of shape
/// `[1, 1, seq_len, seq_len]` in bf16. Zero additive bias means every
/// position may attend to every other position (no causal triangle).
///
/// Cached per `seq_len` to avoid re-allocating the mask on every forward.
fn build_bidirectional_mask(seq_len: usize) -> MlxArray {
    thread_local! {
        static CACHE: RefCell<HashMap<usize, MlxArray>> = RefCell::new(HashMap::new());
    }
    CACHE.with(|c| {
        c.borrow_mut()
            .entry(seq_len)
            .or_insert_with(|| {
                MlxArray::zeros(&[1, 1, seq_len as i32, seq_len as i32], Dtype::Bfloat16)
            })
            .clone()
    })
}
```

Call-site change in `layer_forward_dense_embed`:

```rust
// Before: implicit causal masking.
// let mask_opt = None;

// After: explicit bidirectional (all-zeros) mask for embedding encoders.
let mask_opt = Some(build_bidirectional_mask(seq_len));
```

### Affected Files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/model/mod.rs` | Add bidirectional mask to `layer_forward_dense_embed` |
| `crates/ax-engine-mlx/src/model/mod.rs` | Add `build_bidirectional_mask` helper |
| `crates/ax-engine-mlx/src/runner/mod.rs` | Update embedding tests |

### Validation

```bash
cargo test -p ax-engine-mlx embed
cargo clippy --all-targets --all-features -- -D warnings
```

### Go/No-Go

- ✓ **Go:** All embedding tests pass; last-token output is bit-exact with
  baseline.
- ✗ **Stop:** Any numerical regression in embedding outputs for last-token
  pooling.

---

## Phase 2 — Dense FFN Decode Compile Stabilization

### Objective

Re-enable `AX_MLX_DENSE_FFN_COMPILE` as default-on with proper stream-safety
guarantees.

### Changes

2.1 Audit MLX stream registry lifecycle in long-running server processes.
    Identify the specific failure mode that caused the revert (stream
    invalidation after N compilations).

2.2 Add a stream-scope guard that validates stream liveness before dispatch. If
    the stream is invalid, rebuild the compilation cache entry rather than
    panicking.

2.3 Add a compile-generation counter that triggers cache refresh after a
    configurable threshold (default: 10000 compilations) via
    `AX_MLX_COMPILE_CACHE_REFRESH_THRESHOLD`.

2.4 Promote `AX_MLX_DENSE_FFN_COMPILE` to default-on in `fastpath.rs` with a
    documented kill-switch.

2.5 Add a soak test that runs 50K decode steps and validates no stream-registry
    panic.

```rust
// per_layer_compile.rs — stream-scope guard + generation counter.
static COMPILE_GENERATION: AtomicU64 = AtomicU64::new(0);

fn compile_cache_refresh_threshold() -> u64 {
    std::env::var("AX_MLX_COMPILE_CACHE_REFRESH_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000)
}

fn dispatch_compiled(stream: &MlxStream, key: CompileKey, build: impl Fn() -> Closure) -> Closure {
    // Rebuild the entry if the stream is no longer valid, instead of panicking.
    if !stream.is_valid() {
        invalidate_cache_entry(&key);
    }
    let gen = COMPILE_GENERATION.fetch_add(1, Ordering::Relaxed);
    if gen % compile_cache_refresh_threshold() == 0 {
        refresh_cache();
    }
    get_or_build(key, build)
}
```

```rust
// fastpath.rs — promote to default-on with kill-switch.
env_flag_default_on!(
    /// `AX_MLX_DENSE_FFN_COMPILE` — enable compiled dense-FFN decode closure.
    ///
    /// **Default: ON** (opt-out). Set `AX_MLX_DENSE_FFN_COMPILE=0` to disable.
    dense_ffn_compile_enabled,
    "AX_MLX_DENSE_FFN_COMPILE"
);
```

### Affected Files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/fastpath.rs` | Change default for `AX_MLX_DENSE_FFN_COMPILE` to true |
| `crates/ax-engine-mlx/src/per_layer_compile.rs` | Add stream-scope guard and generation counter |
| `crates/ax-engine-mlx/src/model/shared/mlp.rs` | Update dense FFN compile closure with guard |

### Validation

```bash
cargo test -p ax-engine-mlx dense_ffn_compile
# Soak test (requires model artifacts):
AX_MLX_DENSE_FFN_COMPILE=1 cargo test -p ax-engine-mlx --test soak_decode_50k -- --ignored
```

### Go/No-Go

- ✓ **Go:** 50K decode soak passes without panic; throughput ≥5% improvement
  over split-only baseline.
- ✗ **Stop:** Any stream-registry crash or memory leak detected during soak.

**Rollback:** Set `AX_MLX_DENSE_FFN_COMPILE=0` (existing kill-switch).

---

## Phase 3 — Compiled Mean-Pool Closure

### Objective

Extend the compiled-closure path to mean-pooled embedding batches, reducing
dispatch overhead.

### Changes

3.1 Create `build_embedding_mean_pool_forward_closure` that compiles the
    layer-forward loop for a fixed `(batch_size, max_seq_len)` shape, returning
    full hidden states `[B, max_seq, H]`.

3.2 Cache key: `(thread_id, batch_size, max_seq_len)` in a new
    `embed_mean_pool_compile_cache: DashMap`.

3.3 After compiled forward, apply mean-pool masking imperatively (mask
    construction depends on `actual_lens` which varies per request).

3.4 Add a threshold: only compile when
    `batch_size * max_seq_len > AX_EMBED_MEAN_COMPILE_THRESHOLD`
    (default: 512 tokens total). Below the threshold, the imperative path is
    faster due to compilation overhead.

```rust
// runner/mod.rs — mean-pool compiled closure cache + threshold gate.
static EMBED_MEAN_POOL_COMPILE_CACHE: Lazy<DashMap<(u64, usize, usize), CompiledClosure>> =
    Lazy::new(DashMap::new);

fn embed_mean_compile_threshold() -> usize {
    std::env::var("AX_EMBED_MEAN_COMPILE_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512)
}

fn embed_mean_pool_forward(batch_size: usize, max_seq_len: usize, actual_lens: &[usize]) -> MlxArray {
    if batch_size * max_seq_len <= embed_mean_compile_threshold() {
        return embed_mean_pool_forward_imperative(batch_size, max_seq_len, actual_lens);
    }
    let key = (current_thread_id(), batch_size, max_seq_len);
    let closure = EMBED_MEAN_POOL_COMPILE_CACHE
        .entry(key)
        .or_insert_with(|| build_embedding_mean_pool_forward_closure(batch_size, max_seq_len))
        .clone();

    // Compiled forward returns [B, max_seq, H]; masking is applied imperatively
    // because actual_lens varies per request and cannot be baked into the graph.
    let hidden = closure.call(/* inputs */);
    apply_mean_pool_mask(&hidden, actual_lens)
}
```

### Affected Files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/runner/mod.rs` | Add mean-pool compiled closure builder and cache |
| `crates/ax-engine-mlx/src/model/mod.rs` | Extract mean-pool forward body for closure |

### Validation

```bash
cargo test -p ax-engine-mlx embed_batch_mean
python3 scripts/bench_embedding_fair.py --pooling mean --batch 8 --seq 256
```

### Go/No-Go

- ✓ **Go:** ≥10% throughput improvement for batch≥4, seq≥128 with mean pooling.
- ✗ **Stop:** Regression for batch=1 or short sequences (<64 tokens).

---

## Phase 4 — Fused Embedding Dense Head

### Objective

Integrate the post-norm dense head into the main compiled closure for Last/Cls
paths.

### Changes

4.1 Move `apply_embedding_dense_head` logic inside
    `build_embedding_forward_closure` output computation.

4.2 Update closure output shape from `[B, H]` (pre-head) to `[B, output_dim]`
    (post-head) for models with embedding heads.

4.3 For models without dense heads (majority), no change — the closure outputs
    the raw hidden state.

4.4 Update the compile cache key to include a `has_dense_head: bool`
    discriminant.

```rust
// runner/mod.rs — fold dense head into the compiled output.
struct EmbedCompileKey {
    thread_id: u64,
    batch_size: usize,
    max_seq_len: usize,
    has_dense_head: bool, // new discriminant (4.4)
}

fn build_embedding_forward_closure(cfg: &ModelConfig, w: &Weights) -> CompiledClosure {
    compile(move |inputs| {
        let hidden = run_embedding_layers(inputs);       // [B, H]
        if let Some(head) = w.embedding_dense_head.as_ref() {
            // 4.1 / 4.2: fused post-norm dense head → [B, output_dim].
            vec![apply_embedding_dense_head(head, &hidden)]
        } else {
            // 4.3: no dense head — emit raw hidden state.
            vec![hidden]
        }
    })
}
```

### Affected Files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/runner/mod.rs` | Integrate dense head into compiled closure |
| `crates/ax-engine-mlx/src/model/mod.rs` | Expose dense head weights to closure builder |

### Validation

```bash
cargo test -p ax-engine-mlx embed
```

### Go/No-Go

- ✓ **Go:** No regression in embedding output values; one fewer dispatch per
  inference call.
- ✗ **Stop:** Compile failure for models with dense head.

---

## Phase 5 — Compiled-Closure Fallback Tracing

### Objective

Add observability when compiled closures fail and fall back to imperative
execution.

### Changes

5.1 In all compiled-closure dispatch points (embedding, FFN, layer), add
    `tracing::warn!("compiled_closure_fallback", reason = %e, path = "embed|ffn|layer")`
    on fallback.

5.2 Add compile cache hit/miss counters: `AtomicU64` for `compile_cache_hits`
    and `compile_cache_misses`, exposed via runner telemetry.

5.3 Export counters in the server `/v1/runtime` endpoint response.

```rust
// runner/mod.rs — fallback tracing + counters.
static COMPILE_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static COMPILE_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

match try_compiled_forward(&inputs) {
    Ok(out) => {
        COMPILE_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        out
    }
    Err(e) => {
        COMPILE_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            target: "compiled_closure_fallback",
            reason = %e,
            path = "embed",
            "compiled closure failed; falling back to imperative execution"
        );
        imperative_forward(&inputs)
    }
}
```

```rust
// ax-engine-server/src/openai/runtime.rs — expose counters.
struct RuntimeResponse {
    // ... existing fields ...
    compile_cache_hits: u64,
    compile_cache_misses: u64,
}
```

### Affected Files

| File | Change |
|---|---|
| `crates/ax-engine-mlx/src/runner/mod.rs` | Add `tracing::warn` on fallback, add counters |
| `crates/ax-engine-mlx/src/per_layer_compile.rs` | Add `tracing::warn` on compile failure |
| `crates/ax-engine-server/src/openai/runtime.rs` | Expose compile counters in runtime response |

### Validation

```bash
AX_ENGINE_SERVER_LOG=warn cargo test -p ax-engine-server
cargo test -p ax-engine-mlx compile_fallback
```

### Go/No-Go

- ✓ **Go:** Fallback events visible in tracing output; counters increment
  correctly.
- ✗ **Stop:** Tracing overhead measurably impacts hot-path latency (>1%
  regression).

---

## Rollback Plan (All Phases)

Each phase is independently reversible with no cross-phase dependency on
rollback:

- **Phase 1:** Revert mask to `None` (restore causal behavior).
- **Phase 2:** `AX_MLX_DENSE_FFN_COMPILE=0` kill-switch.
- **Phase 3:** `AX_EMBED_NO_COMPILE=1` disables all embedding compilation.
- **Phase 4:** Remove dense head from closure, restore separate dispatch.
- **Phase 5:** Remove tracing instrumentation (zero functional impact).

## Implementation Order

```text
Phase 1 (bidirectional mask, correctness)   ← lands first
  │
  ├── Phase 2 (dense FFN compile stabilization)
  ├── Phase 3 (compiled mean-pool closure)
  └── Phase 4 (fused embedding dense head)
        │
        └── Phase 5 (fallback tracing, observability)  ← closes the loop
```
