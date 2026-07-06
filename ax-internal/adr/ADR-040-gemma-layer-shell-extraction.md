# ADR-040: Shared Layer Shell Extraction for Causal/Bidirectional Forwards

**Status:** Proposed
**Date:** 2026-07-06
**Deciders:** Engineering
**Superseded by:** —

## Context

The Gemma standard model implementation in ax-engine provides two layer forward
paths: `layer_forward` (causal attention for generative inference) and
`layer_forward_bidirectional` (bidirectional attention for diffusion denoising
and embedding). These two functions share approximately 200 lines of identical
code implementing the layer shell:

1. QKV projection (linear projections of hidden state into query, key, value).
2. QK normalization (RMSNorm applied to query and key heads).
3. RoPE application (rotary position embeddings applied to query and key).
4. Output projection (linear projection of attention output back to hidden dim).
5. FFN with MoE/dense routing (SwiGLU or MoE expert dispatch).
6. Residual connection with gating (layer-wise residual add + RMSNorm gating).

The **only** difference between the two paths is the attention mechanism:
causal attention uses a triangular mask for autoregressive generation, while
bidirectional attention allows all-to-all position interaction for denoising
and embedding tasks.

Relevant code locations:

- `crates/ax-engine-mlx/src/model/families/standard.rs`, lines 56–786
  (`layer_forward` — causal path).
- `crates/ax-engine-mlx/src/model/families/standard.rs`, lines 962–1188
  (`layer_forward_bidirectional` — bidirectional path).

This duplication was identified during a comprehensive code review of the
Gemma, DiffusionGemma, and EmbeddingGemma implementations. The duplication
increases maintenance burden: any change to the shared shell (e.g., a new norm
type, an FFN optimization, a residual gating variant) must be implemented in
two places and kept in sync manually.

## Decision

Extract a shared `layer_shell` function that encapsulates the common layer
computation and accepts an attention closure as a parameter. Both
`layer_forward` and `layer_forward_bidirectional` call `layer_shell` with
their specific attention implementation.

The attention closure takes `(q, k, v, mask, kv_cache)` as inputs and returns
`(attn_output, new_kv)`. The shell function handles everything else:
projection, normalization, RoPE, FFN, residual, and gating.

```rust
/// Attention function signature accepted by `layer_shell`.
/// Takes projected Q, K, V tensors plus optional mask and KV cache,
/// returns attention output and updated KV cache entry.
type AttentionFn = Box<
    dyn Fn(&MlxArray, &MlxArray, &MlxArray, Option<&MlxArray>, Option<&mut KvCache>)
        -> (MlxArray, Option<MlxArray>),
>;

/// Shared layer shell: QKV projection → QK-norm → RoPE → attention →
/// output projection → FFN → residual + gating. The attention mechanism
/// is injected via `attn_fn` so that causal and bidirectional paths share
/// a single implementation.
#[inline]
fn layer_shell(
    cfg: &ModelConfig,
    weights: &LayerWeights,
    hidden: &MlxArray,
    position: &MlxArray,
    mask_opt: Option<&MlxArray>,
    kv_cache: Option<&mut KvCache>,
    attn_fn: AttentionFn,
) -> MlxArray {
    // 1. QKV projection
    let (q, k, v) = project_qkv(weights, hidden);

    // 2. QK normalization
    let (q, k) = apply_qk_norm(cfg, weights, &q, &k);

    // 3. RoPE
    let (q, k) = apply_rope(cfg, &q, &k, position);

    // 4. Attention (injected)
    let (attn_out, new_kv) = attn_fn(&q, &k, &v, mask_opt, kv_cache);

    // 5. Output projection
    let proj = apply_output_projection(weights, &attn_out);

    // 6. Residual + FFN
    let residual = hidden + &proj;
    let ffn_out = apply_ffn(cfg, weights, &residual);
    residual + &ffn_out
}
```

Causal path calls the shell:

```rust
fn layer_forward(/* ... */) -> MlxArray {
    layer_shell(cfg, weights, hidden, position, mask_opt, kv_cache,
        Box::new(|q, k, v, mask, cache| {
            causal_attention(q, k, v, mask, cache)
        }))
}
```

Bidirectional path calls the shell:

```rust
fn layer_forward_bidirectional(/* ... */) -> MlxArray {
    layer_shell(cfg, weights, hidden, position, mask_opt, kv_cache,
        Box::new(|q, k, v, mask, cache| {
            bidirectional_attention(q, k, v, mask, cache)
        }))
}
```

## Consequences

### Positive

- Eliminates ~200 lines of code duplication between causal and bidirectional
  layer forwards.
- Future changes to the layer shell (new norm types, FFN optimizations,
  residual gating variants) require a single implementation.
- Self-documenting: the attention closure makes the causal vs bidirectional
  distinction explicit at the call site.
- Reduces the surface area for bugs — a fix to the shell automatically applies
  to both paths.

### Negative

- Introduces a closure-based abstraction in the hot path. Even with
  `#[inline]`, the closure indirection may inhibit some compiler
  optimizations (e.g., cross-function constant propagation).
- The `Box<dyn Fn(...)>` signature uses dynamic dispatch. For hot-path
  sensitivity, a generic parameter (`impl Fn(...)`) may be preferable, but
  this requires monomorphization at each call site.
- Increases the conceptual complexity of the layer forward — readers must
  understand the shell/closure split rather than reading a single linear
  function.

### Neutral

- No numerical change to model output (same computation, restructured).
- No API changes visible to users or external callers.
- Profiling instrumentation must be adapted to the shell structure but is
  functionally equivalent.

## Alternatives Considered

### Alternative 1: Keep duplication (status quo)

**Rejected.** The maintenance burden grows with each new feature. Recent
examples: QK-norm addition required edits in both paths; any future MoE
routing change will require the same. The risk of divergence between the two
implementations increases over time.

### Alternative 2: Trait-based polymorphism for attention strategy

**Rejected.** Defining an `AttentionStrategy` trait with `causal()` and
`bidirectional()` implementations would achieve the same deduplication, but
introduces dynamic dispatch via trait objects in the hot path. The closure
approach is lighter-weight and easier to inline.

### Alternative 3: Macro-based code generation

**Rejected.** A declarative macro that generates both forward variants from
a single template would eliminate duplication without runtime overhead.
However, macros obscure the control flow and make debugging significantly
harder. The closure approach preserves readability while achieving the
same deduplication goal.

### Alternative 4: Generic parameter instead of Box<dyn Fn>

**Considered for future.** Using `fn layer_shell<A: AttentionFn>(..., attn: A)`
with a generic parameter avoids dynamic dispatch entirely. This is the
preferred long-term direction if profiling reveals closure overhead, but
adds monomorphization cost at compile time. The `Box<dyn Fn>` approach is
sufficient for the initial extraction and can be upgraded later.

## Validation

```bash
# Unit tests: causal forward output bit-exact with pre-extraction baseline
cargo test -p ax-engine-mlx gemma_causal_layer_forward

# Unit tests: bidirectional forward output bit-exact with pre-extraction baseline
cargo test -p ax-engine-mlx gemma_bidirectional_layer_forward

# DiffusionGemma convergence: output quality unchanged
cargo test -p ax-engine-mlx diffusion_gemma

# Full workspace build + clippy
cargo build --workspace
cargo clippy --all-targets --all-features -- -D warnings
```

## References

- PRD: `ax-internal/prd/GEMMA-MODEL-FAMILY-IMPROVEMENT-PRD.md` (Finding 5)
- Tech Spec: `ax-internal/tech-spec/GEMMA-MODEL-FAMILY-IMPROVEMENT-TECH-SPEC.md` (Phase 5)
- Implementation Plan: `ax-internal/plan/IMPL-2026-07-06-gemma-model-improvement.md` (Phase 5)
- Related code: `crates/ax-engine-mlx/src/model/families/standard.rs` (layer_forward, layer_forward_bidirectional)
