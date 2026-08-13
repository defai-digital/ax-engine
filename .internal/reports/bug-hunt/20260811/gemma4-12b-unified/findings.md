# Findings — gemma4-12b-unified

## Merge notes
- Codex batch: DI-W1-001 (P1) GeGLU/RoPE omit gemma4_unified after convert label fix
- Fixed in-branch: ModelConfig/is_gemma4, uses_geglu, build_layer_configs, architecture::uses_geglu
- Tests: gemma4_unified_uses_geglu_like_gemma4, converts_gemma4_unified_text

## Work queue
| ID | Class | Sev | Title | Status |
| --- | --- | --- | --- | --- |
| DI-W1-001 | IMPL | P1 | gemma4_unified missing from GeGLU/RoPE family gates | fixed |

EXIT closed-code-only; open P0/P1 = 0


## Post-fix note
DI-W1-001 GeGLU/RoPE gates fixed
Dual-agent batches: wave1-4-codex-batch.md, wave1-4-axcode-batch.md
