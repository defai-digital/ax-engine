# Findings — embeddings-primary

## Merge notes
- Codex: DI-W2-002 singleton embed path skipped EmbeddingGemma bidirectional route
- Fixed: forward_for_embedding routes embeddinggemma to forward_for_embedding_gemma3_batch

## Work queue
| ID | Class | Sev | Title | Status |
| --- | --- | --- | --- | --- |
| DI-W2-002 | IMPL | P1 | EmbeddingGemma single vs batch path drift | fixed |

EXIT closed-code-only


## Post-fix note
DI-W2-002 singleton EmbeddingGemma path fixed
Dual-agent batches: wave1-4-codex-batch.md, wave1-4-axcode-batch.md

## DI-W2-002 complete fix (post-skeptic)

Production single-item path `MlxRunner::embedding_forward` now early-returns for
`embeddinggemma` via `embedding_single_item_uses_gemma3_path`, using
`embedding_gemma_batch_compiled_forward` (or imperative batch-of-one).
`build_embedding_forward_closure` fail-closes via `dense_embed_closure_forbidden_reason`.

Regression tests:
- `embeddinggemma_single_item_dispatch_matches_batch_of_one`
- `dense_embed_closure_forbidden_for_embeddinggemma`
