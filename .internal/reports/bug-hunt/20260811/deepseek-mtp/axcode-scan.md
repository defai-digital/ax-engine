# ax-code scan — deepseek-mtp

**Model:** `zai-coding-plan/glm-5.2[1m]` (session present; findings merged with Codex)

## Coverage

- `mtp.rs` DeepSeek V4 nextn draft/verify/warmup
- `runner/mod.rs` draft/accept temperature plumbing, hybrid forced-prefix
- `mtp_model_policy.rs` fail-closed V4 certification candidate
- `chat.rs` DeepSeekChat fullwidth-bar template + thinking detection

## Findings

| ID | Sev | Note |
| --- | --- | --- |
| DI-DS-MTP-001 | P1 | Fixed — sample/log T lock |
| DI-DS-MTP-002 | P1 | Fixed — hybrid draft T parameter |
| DI-DS-MTP-003 | P1 | Fixed — pending draft T carry on state |
| DI-DS-TPL-001 | P2 | Fixed — V4 Flash template tests |

## Completeness

Static exactness path closed for temperature contract. Formal weight A/B still LIMIT.
