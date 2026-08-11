# ax-code scan — deepseek-mtp (post-fix)

**Agent:** ax-code CLI · model `zai-coding-plan/glm-5.2[1m]`  
**Mode:** read-only post-fix verification + follow-up min_p fix  
**Date:** 2026-08-11

## Coverage

| Surface | Status |
|---|---|
| `think_token_ids_from_manifest` deepseek defaults | ✅ |
| `parse_think_token_ids` content match | ✅ |
| Runner think-boundary draft T + pending lock | ✅ |
| MTP sample/log T lock | ✅ |
| DeepSeekChat template | ✅ |
| Fail-closed cert policy | ✅ IMPL |
| min_p target-prob / residual parity | ✅ fixed this pass |

## Open unparked P0/P1

**Zero.** All requested fix areas verified; min_p rejection-sampling gap closed after Codex triage.

## Parked

- **IMPL:** V4 nextn fail-closed without cert env.
- **LIMIT:** weight-backed Tier-2 A/B; sampled multi-token residual; HF jinja framing deltas vs native renderer (tests lock R1-style fullwidth bars).
