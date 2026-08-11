# Findings — deepseek-mtp (exactness + template)

**Agents:** Codex `gpt-5.6-sol` reasoning max + ax-code `zai-coding-plan/glm-5.2[1m]` (this pass)  
**Date:** 2026-08-11

## Work queue

| ID | Class | Sev | Title | Status |
| --- | --- | --- | --- | --- |
| DI-DS-MTP-001 | MTP | P1 | Stochastic think draft T vs accept log-prob T mismatch | fixed |
| DI-DS-MTP-002 | MTP | P1 | Hybrid `after_forced_prefix` hard-coded draft T=0.7 | fixed |
| DI-DS-MTP-003 | MTP | P1 | Pending-draft accept used recomputed T across steps | fixed |
| DI-DS-TPL-001 | DOC/TEST | P2 | V4 Flash package IDs must select DeepSeekChat template | fixed (test lock) |

## Details

### DI-DS-MTP-001 — fixed

**Problem:** Pure MTP draft used `deepseek_v4_mtp_effective_draft_temperature` (think → 1.0) while accept-path rescale used `deepseek_v4_mtp_draft_log_prob_temperature()` (stochastic mode → 0.7). Rejection sampling saw wrong `q(token)`.

**Fix:** `deepseek_v4_mtp_sample_and_log_temperature` locks sample + log T (greedy always 1.0; stochastic uses think-aware effective). Runner uses this for both draft and accept.

### DI-DS-MTP-002 — fixed

**Problem:** `deepseek_v4_mtp_draft_tokens_after_forced_prefix` always passed `DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE` (0.7), ignoring think-aware / mode-aware T.

**Fix:** Function takes `draft_temperature`; hybrid and pure paths pass the shared sample/log temperature.

### DI-DS-MTP-003 — fixed

**Problem (Codex residual):** Accept on step N+1 recomputed draft log-prob T from current think state, while pending log-probs were written at step N's draft T.

**Fix:** Carry `mtp_pending_draft_log_prob_temperature` on request state; accept uses the stored T.

### DI-DS-TPL-001 — fixed

**Problem:** Missing regression lock that V4 Flash / AutomatosX AX packages select `DeepSeekChat` and are **not** default-thinking (unlike R1).

**Fix:** Extended `deepseek_model_ids_select_chat_template` test.

## Residual LIMIT

- Product default for DeepSeek V4 nextn remains **fail-closed** until Tier 2 certification (`AX_MLX_DEEPSEEK_V4_MTP_CERTIFICATION_CANDIDATE`).
- Weight-backed formal A/B exactness on host still deferred without V4 weights in this environment.
- V3 has no nextn MTP in this codebase (dense MLA only).
