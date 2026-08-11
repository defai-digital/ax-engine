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
