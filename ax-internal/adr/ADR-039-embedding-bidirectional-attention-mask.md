# ADR-039: Explicit Bidirectional Attention Mask for Qwen 3 Embedding Models

**Status:** Accepted
**Date:** 2026-07-06
**Deciders:** Engineering
**Superseded by:** —

## Context

The Qwen 3 embedding implementation in ax-engine currently uses `mask_opt = None`
in `layer_forward_dense_embed`, which resolves to **causal** attention masking in
`full_precision_attention`. This is semantically incorrect for embedding models,
which should use bidirectional attention (all positions attend to all other
positions).

The bug is latent for **last-token pooling** (Qwen3's default) because the final
position correctly attends to all preceding positions under causal masking — the
output is numerically identical. However, for **mean pooling**, intermediate
positions only attend to their left context, which is semantically wrong and may
degrade embedding quality for non-final positions.

The issue was identified during a comprehensive code review of the Qwen 3.6 and
Qwen 3 embedding implementations.

Relevant code location: `crates/ax-engine-mlx/src/model/mod.rs`, line ~1333 in
`layer_forward_dense_embed`.

## Decision

Add explicit bidirectional attention mask (all-zeros or `None` handled as
bidirectional in embedding context) for all embedding forward paths.
Specifically:

1. Introduce a `mask_mode` parameter or conditional in `layer_forward_dense_embed`
   that constructs a bidirectional mask (allowing all-to-all attention) when the
   forward is for embedding purposes.
2. The mask should be a tensor of zeros (no masking) with shape
   `[1, 1, seq_len, seq_len]` in bfloat16 for consistency with existing bf16 mask
   optimizations.
3. For `EmbeddingGemma`, which already uses bidirectional padding masks, ensure
   the mask path is unified.

This preserves correctness for all pooling strategies (last, cls, mean) and makes
the code self-documenting.

## Consequences

### Positive

- Semantic correctness for all pooling strategies, including mean pooling.
- Self-documenting code (explicit bidirectional intent vs implicit causal
  default).
- Future-proofs against changes in attention defaults or new pooling strategies.
- Unifies mask handling between EmbeddingGemma and Qwen3 embedding paths.

### Negative

- Minor memory allocation for the all-zeros mask tensor (negligible for typical
  seq lengths).
- Compiled closure cache keys may need to account for mask presence (potential
  cache invalidation on first call after change).

### Neutral

- No numerical change for last-token pooling (existing default behavior
  preserved).
- No API changes visible to users.

## Alternatives Considered

### Alternative 1: Leave causal mask for Qwen3 embeddings (status quo)

**Rejected.** While last-token pooling is unaffected, mean pooling produces
semantically incorrect intermediate representations. As embedding models are
increasingly used with mean pooling (e.g., RAG applications), this technical debt
will compound.

### Alternative 2: Only add bidirectional mask when pooling=mean

**Rejected.** Creates conditional behavior that is harder to reason about and
test. The bidirectional mask has negligible cost and is always semantically
correct for embeddings.

### Alternative 3: Modify `full_precision_attention` to accept a mode enum (causal/bidirectional/custom)

**Rejected for now.** Larger refactor with wider blast radius. The targeted fix
(explicit mask in embedding path) is lower risk and achieves the same correctness
guarantee.

## Validation

```bash
# Unit tests for bidirectional attention correctness
cargo test -p ax-engine-mlx embedding_bidirectional

# Parametrized pooling tests (last, cls, mean)
cargo test -p ax-engine-mlx embed_batch_pooling

# Regression: ensure last-token output is bit-exact
cargo test -p ax-engine-mlx embed_last_token_determinism

# Full workspace build + clippy
cargo build --workspace
cargo clippy --all-targets --all-features -- -D warnings
```

## References

- Tech Spec: `ax-internal/tech-spec/QWEN-EMBEDDING-IMPROVEMENT-TECH-SPEC.md`
- Related code: `crates/ax-engine-mlx/src/model/mod.rs` (layer_forward_dense_embed)
- Related code: `crates/ax-engine-mlx/src/runner/mod.rs` (embedding batch dispatch)
