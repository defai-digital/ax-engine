# Findings — qwen3-vl

## Merge notes
- Codex: DI-W2-001 qwen3_vl_moe MoE not classified in moe_config
- Dense qwen3_vl: no open P0/P1
- Fixed: hf_config::moe_config includes qwen3_vl_moe / qwen3-vl-moe
- Test: qwen3_vl_moe_model_type_produces_moe_config

## Work queue
| ID | Class | Sev | Title | Status |
| --- | --- | --- | --- | --- |
| DI-W2-001 | IMPL | P1 | qwen3_vl_moe empty moe_config | fixed |

EXIT closed-code-only


## Post-fix note
DI-W2-001 moe_config + DI-W2-F1c patch buffer guard fixed
Dual-agent batches: wave1-4-codex-batch.md, wave1-4-axcode-batch.md
