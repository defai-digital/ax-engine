# Tech Spec: Decode & Prefill Speed Optimization — Phase 2

**Date:** 2026-06-16
**PRD:** [PRD-2026-06-16-decode-prefill-speed-phase2.md](../prd/PRD-2026-06-16-decode-prefill-speed-phase2.md)
**ADR:** [ADR-026-decode-prefill-speed-phase2.md](../adr/ADR-026-decode-prefill-speed-phase2.md)

## Change 1: Last-Position-Only for Linear-Attention Layers

### Current State

`layer_forward_with_turboquant_context_last_only` in `model/mod.rs` short-circuits
linear-attention layers:

```rust
if cfg.is_linear_attention_layer(layer_idx) {
    return families::qwen3_linear::layer_forward(cfg, w, hidden, cache, layer_idx);
}
```

`qwen3_linear::layer_forward` processes the full `[1, seq, hidden]` through FFN
unconditionally.

### Change

1. **`qwen3_linear::layer_forward`** — add `last_position_only: bool` parameter. When
   `last_position_only && seq > 1`:
   - After `hidden = add(hidden, &attn_proj)` (residual), slice `hidden` to last position:
     `slice(&hidden, &[0, last, 0], &[1, last+1, hs], &[1, 1, 1], None)`
   - The pre-FFN norm, FFN/MoE, post-FFN norm, and residual gate then operate on
     `[1, 1, hidden]`
   - **Correctness argument:** `linear_attention_forward` writes both conv1d state and
     recurrent state to `cache` before returning. The output `attn_proj` is position-wise
     after the gated-delta recurrence. The slice only affects the FFN path, which is
     position-wise (no cross-position interaction).

2. **`model/mod.rs::layer_forward_with_turboquant_context_last_only`** — instead of
   falling back, call `qwen3_linear::layer_forward` with `last_position_only=true`:

   ```rust
   if cfg.is_linear_attention_layer(layer_idx) {
       return families::qwen3_linear::layer_forward(cfg, w, hidden, cache, layer_idx, true);
   }
   ```

3. **`qwen3_linear::layer_forward` non-last-only callers** — update the existing call
   sites to pass `false`:
   - `layer_forward_with_turboquant_context` in `model/mod.rs` → calls through
     `layer_forward_with_turboquant_context` which doesn't use the last-only path

### Files Touched

- `crates/ax-engine-mlx/src/model/families/qwen3_linear.rs`
- `crates/ax-engine-mlx/src/model/mod.rs`

---

## Change 2: Qwen3 MoE Router Narrow Softmax (Opt-In)

### Current State

`moe_router_qwen3` in `shared/mlp.rs`:

```rust
let weights_all = softmax_precise(&logits, last_axis, None);  // softmax over ALL experts
let (top_k_indices, top_k_weights) = top_k_by_argpartition(&weights_all, ..., false);
// re-normalize if moe_norm_topk_prob
```

### Change

1. **`fastpath.rs`** — add env flag:

   ```rust
   env_flag!(
       qwen3_moe_narrow_softmax_enabled,
       "AX_MLX_QWEN3_MOE_NARROW_SOFTMAX"
   );
   ```

2. **`shared/mlp.rs::moe_router_qwen3`** — add narrow-softmax branch:

   ```rust
   if fastpath::qwen3_moe_narrow_softmax_enabled() {
       // Argpartition on raw logits (monotonic with softmax → same top-k set for
       // well-separated experts). Softmax only on the selected subset.
       let (top_k_indices, top_k_weights) = top_k_by_argpartition(
           &logits, cfg.moe_expert_count, cfg.moe_experts_per_token, true,
       );
       // ... re-normalize if moe_norm_topk_prob ...
       return (top_k_indices, top_k_weights);
   }
   // Existing full-width path preserved as default.
   ```

### Files Touched

- `crates/ax-engine-mlx/src/fastpath.rs`
- `crates/ax-engine-mlx/src/model/shared/mlp.rs`

---

## Change 3: Relax MoE Shared-Expert Fusion Gate

### Current State

`moe_experts_forward_with_shared` in `shared/mlp.rs`:

```rust
if seq == 1
    && let Some(shared) = shared_expert_out
    && fastpath::moe_fuse_shared_expert_add_enabled()
    && let Some(out) = qwen3_moe_weighted_sum_with_shared_metal(...)
```

### Change

Add a constant threshold and change the gate:

```rust
const MOE_SHARED_FUSION_SEQ_THRESHOLD: usize = 64;

if seq <= MOE_SHARED_FUSION_SEQ_THRESHOLD
    && let Some(shared) = shared_expert_out
    && fastpath::moe_fuse_shared_expert_add_enabled()
    && let Some(out) = qwen3_moe_weighted_sum_with_shared_metal(...)
```

### Files Touched

- `crates/ax-engine-mlx/src/model/shared/mlp.rs`

---

## Testing Strategy

1. **Compilation:** `cargo build --workspace` and `cargo clippy --all-targets --all-features -- -D warnings`
2. **Unit tests:** `cargo test --quiet --no-fail-fast`
3. **Correctness (Change 1):** Existing linear-attention tests validate that
   `layer_forward` produces the same output with and without `last_position_only` for
   `seq == 1` (decode). For `seq > 1`, the last-position output must match the
   last-position slice of the full-seq output.
4. **Correctness (Change 2):** Narrow softmax is default-OFF; no change to existing test
   behavior. When enabled, the top-k expert selection must be validated against the
   full-softmax reference for a representative set of router inputs.
5. **Correctness (Change 3):** The fused kernel produces the same output as the unfused
   path (weighted-sum + separate add) for all `seq <= 64`.
