# Implementation Plan: Qwen 3.6 & Qwen 3 Embedding Improvements

**Date:** 2026-07-06
**Goal:** Improve Qwen 3.6 decode performance and Qwen 3 embedding correctness, throughput, and observability
**PRD:** `ax-internal/prd/QWEN-36-EMBEDDING-IMPROVEMENT-PRD.md`
**Tech Spec:** `ax-internal/tech-spec/QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC.md`

---

## Phase Ordering and Dependencies

```text
Phase 1 (Correctness)          Phase 5 (Observability)
  │                                │
  ▼                                ▼
Phase 3 (Mean-Pool Compile) ──→ Phase 4 (Fused Dense Head)
                                   │
Phase 2 (FFN Compile) ────────────→│
                                   ▼
                              Evaluation Checkpoint
```

- Phase 1 and Phase 5 have no dependencies (can start immediately)
- Phase 2 is independent (can run in parallel with Phase 1)
- Phase 3 depends on Phase 1 (bidirectional mask affects compiled closure shape)
- Phase 4 depends on Phase 3 (builds on mean-pool closure infrastructure)
- Evaluation Checkpoint after all phases complete

---

## Phase 1 — Bidirectional Attention Mask for Embeddings

- **ADR:** `ax-internal/adr/ADR-039-embedding-bidirectional-attention-mask.md`
- **Tech Spec:** Phase 1 in `ax-internal/tech-spec/QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC.md`
- **Files:** `crates/ax-engine-mlx/src/model/mod.rs`, `crates/ax-engine-mlx/src/runner/mod.rs`
- **Estimated effort:** 4h
- **Risk:** Low

### Changes

**1-A** Add `build_bidirectional_mask(seq_len: usize) -> MlxArray` helper in `model/mod.rs`:

- Construct bf16 zeros tensor of shape `[1, 1, seq_len, seq_len]`
- Cache per seq_len using thread-local `HashMap<usize, MlxArray>`

**1-B** In `layer_forward_dense_embed`, replace `mask_opt = None` with the bidirectional mask:

```rust
let mask = build_bidirectional_mask(seq_len);
// Pass to attention as Some(mask) — zeros means "attend everywhere"
```

**1-C** Add parametrized tests:

- `test_embed_qwen3_last_token_determinism` — verify bit-exact output vs baseline
- `test_embed_qwen3_mean_pool_correctness` — verify all positions attend bidirectionally
- `test_embed_gemma_mean_pool_unchanged` — verify no regression for EmbeddingGemma

**1-D** Run full embedding benchmark suite to confirm no throughput regression.

### Validation

```bash
cargo test -p ax-engine-mlx embed
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

### Go/No-Go

- ✓ Go: All tests pass; last-token output bit-exact with pre-change baseline
- ✗ Stop: Any numerical difference in last-token output (indicates mask is applied incorrectly)

---

## Phase 2 — Dense FFN Decode Compile Stabilization

- **Tech Spec:** Phase 2 in `ax-internal/tech-spec/QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC.md`
- **Files:** `crates/ax-engine-mlx/src/fastpath.rs`, `crates/ax-engine-mlx/src/per_layer_compile.rs`, `crates/ax-engine-mlx/src/model/shared/mlp.rs`
- **Estimated effort:** 12h
- **Risk:** Medium

### Changes

**2-A** Audit stream-registry lifecycle:

- Instrument `MlxClosure::compile` with generation counter
- Log stream state before/after compilation
- Identify threshold at which stream invalidation occurs

**2-B** Implement stream-scope guard in `per_layer_compile.rs`:

```rust
fn validate_stream_liveness(closure: &CompiledClosure) -> bool {
    // Check if the stream backing this closure is still valid
    // If not, mark for rebuild
}
```

**2-C** Add compile-generation refresh mechanism:

- Counter per cache entry tracking compilation age
- Threshold `AX_MLX_COMPILE_CACHE_REFRESH_THRESHOLD` (default 10000)
- On threshold hit, rebuild closure with fresh stream

**2-D** Promote default in `fastpath.rs`:

- Change `AX_MLX_DENSE_FFN_COMPILE` default from `false` to `true`
- Document kill-switch behavior in code comment

**2-E** Write soak test:

- 50K decode steps with random prompts
- Assert no panic, no memory growth >10%, throughput stable ±2%

### Validation

```bash
cargo test -p ax-engine-mlx dense_ffn
cargo test -p ax-engine-mlx --test soak_decode_50k -- --ignored
```

### Go/No-Go

- ✓ Go: Soak passes; ≥5% decode throughput improvement
- ✗ Stop: Stream crash or memory leak in soak test

---

## Phase 3 — Compiled Mean-Pool Closure

- **Tech Spec:** Phase 3 in `ax-internal/tech-spec/QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC.md`
- **Files:** `crates/ax-engine-mlx/src/runner/mod.rs`, `crates/ax-engine-mlx/src/model/mod.rs`
- **Estimated effort:** 8h
- **Risk:** Low

### Changes

**3-A** Extract `forward_for_embedding_mean_pool_body` from existing mean-pool path:

- Takes `(token_ids, seq_len, batch_size)` as inputs
- Returns full hidden `[B, max_seq, H]` before masking

**3-B** Build `build_embedding_mean_pool_forward_closure`:

- Input shapes: `[(batch_size, max_seq_len)]` for token IDs
- Cache key: `(thread_id, batch_size, max_seq_len)`
- Store in `embed_mean_pool_compile_cache: DashMap`

**3-C** Add size-gating threshold:

- Only compile when `batch_size * max_seq_len > 512` (configurable via env)
- Below threshold, imperative path is faster

**3-D** After compiled forward, apply mean-pool mask imperatively:

- Mask application is O(B × max_seq) and doesn't benefit from compilation
- Keep existing bf16 mask construction and L2 normalization logic

### Validation

```bash
cargo test -p ax-engine-mlx embed_batch_mean
python3 scripts/bench_embedding_fair.py --pooling mean --batch 8 --seq 256
```

### Go/No-Go

- ✓ Go: ≥10% improvement at batch≥4, seq≥128; no regression at batch=1
- ✗ Stop: Compile failure or output divergence

---

## Phase 4 — Fused Embedding Dense Head

- **Tech Spec:** Phase 4 in `ax-internal/tech-spec/QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC.md`
- **Files:** `crates/ax-engine-mlx/src/runner/mod.rs`, `crates/ax-engine-mlx/src/model/mod.rs`
- **Estimated effort:** 4h
- **Risk:** Low

### Changes

**4-A** Move `apply_embedding_dense_head` into compiled closure:

- For Last/Cls paths: closure output becomes `[B, output_dim]` instead of `[B, H]`
- For models without dense head: no change (closure outputs raw hidden)

**4-B** Update compile cache key:

- Add `has_dense_head: bool` to key tuple
- Separate cache entries for models with/without head

**4-C** Verify numerics:

- Compare dense-head-in-closure output vs separate-dispatch output
- Must be bit-exact (same computation, just fused)

### Validation

```bash
cargo test -p ax-engine-mlx embed
```

### Go/No-Go

- ✓ Go: Bit-exact output; one fewer dispatch per call in profiling
- ✗ Stop: Compile failure for models with dense head

---

## Phase 5 — Compiled-Closure Fallback Tracing

- **Tech Spec:** Phase 5 in `ax-internal/tech-spec/QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC.md`
- **Files:** `crates/ax-engine-mlx/src/runner/mod.rs`, `crates/ax-engine-mlx/src/per_layer_compile.rs`, `crates/ax-engine-server/src/openai/runtime.rs`
- **Estimated effort:** 3h
- **Risk:** None

### Changes

**5-A** Add `tracing::warn!` at every compiled-closure fallback point:

```rust
tracing::warn!(
    path = "embed_batch",
    reason = %error,
    "compiled_closure_fallback"
);
```

**5-B** Add atomic counters:

```rust
static COMPILE_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static COMPILE_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
```

**5-C** Expose in `/v1/runtime` response:

```json
{
  "compile_cache_hits": 1234,
  "compile_cache_misses": 5,
  "compile_fallback_count": 2
}
```

### Validation

```bash
AX_ENGINE_SERVER_LOG=warn cargo test -p ax-engine-server
cargo test -p ax-engine-mlx compile_fallback
```

### Go/No-Go

- ✓ Go: Fallback visible in logs; counters work
- ✗ Stop: Measurable hot-path latency impact (>1%)

---

## Evaluation Checkpoint

After all phases complete, assess:

1. Embedding throughput vs baseline (target: ≥10% improvement at batch≥4)
2. Decode throughput with FFN compile (target: ≥5% improvement)
3. Prefill throughput gap (target: <15% vs mlx-lm)
4. Zero regressions in correctness tests
5. Observability: fallback events visible, counters accurate

---

## Risk Register

| Risk | Probability | Mitigation |
| --- | --- | --- |
| Stream-registry invalidation under load | Medium | Kill-switch + generation-counter refresh |
| Compiled closure exceeds MLX buffer for large batches | Low | Dynamic threshold gating |
| Bidirectional mask allocation overhead | Very Low | Per-seq_len caching |
| Tracing overhead on hot path | Very Low | Conditional compilation; warn-level only |
| Mean-pool closure shape mismatch on variable inputs | Low | Strict cache-key matching; imperative fallback |

---

## Quick Reference

| Item | File | Approx Line |
| --- | --- | --- |
| layer_forward_dense_embed | `crates/ax-engine-mlx/src/model/mod.rs` | ~1176 |
| fastpath flags | `crates/ax-engine-mlx/src/fastpath.rs` | ~204-688 |
| per_layer_compile | `crates/ax-engine-mlx/src/per_layer_compile.rs` | ~1-308 |
| dense FFN mlp | `crates/ax-engine-mlx/src/model/shared/mlp.rs` | ~1264-1550 |
| embedding runner | `crates/ax-engine-mlx/src/runner/mod.rs` | ~4523-4820 |
| server embeddings | `crates/ax-engine-server/src/openai/embeddings.rs` | ~1-150 |
| MTP weights | `crates/ax-engine-mlx/src/mtp.rs` | ~1-200 |
