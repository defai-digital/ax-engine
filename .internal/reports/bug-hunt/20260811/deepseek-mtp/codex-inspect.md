# Codex inspect — deepseek-mtp (post-fix)

**Agent:** Codex CLI · model `gpt-5.6-sol` · `model_reasoning_effort=max`  
**Date:** 2026-08-11  
**Mode:** read-only audit (report finalized from agent session after sandbox blocked write)

## Named fixes verified

| Claim | Verdict | Anchors |
|---|---|---|
| DI-DS-MTP sample/log T lock | **fixed** | `mtp.rs` `deepseek_v4_mtp_sample_and_log_temperature` |
| Hybrid draft T parameter | **fixed** | `runner/mod.rs` hybrid uses `deepseek_next_draft_temperature` |
| Pending log-prob T carry | **fixed** | `mtp_pending_draft_log_prob_temperature` store/load |
| Think-token family defaults (A001) | **fixed** | `config.rs` deepseek_v3/v32/v4 arms (128798/799, 128821/822) |
| Think-boundary next-draft T | **fixed** | `think_state_after_result` → `deepseek_next_draft_temperature` |
| Greedy sequential verify | **fixed** (prior) | `sequential_greedy_deepseek_v4_mtp_verify` |

## Additional findings this pass

| ID | Sev | Title | Status |
|---|---|---|---|
| DI-DS-MTP-004 | P1 | `token_distribution` ignored `min_p` (residual correction law ≠ sampler) | **fixed** |
| DI-DS-MTP-005 | P1 | MTP target probs ignored `min_p` (DeepSeek thinking default 0.05) | **fixed** (`FullRows` + TopK min_p) |
| DI-DS-MTP-006 | MEDIUM | Sampled multi-token verify residual compressor drift (A004/A005) | parked — greedy sequential exact; product MTP fail-closed |
| DI-DS-TPL-002 | P2/LIMIT | Native renderer vs official V4 HF jinja framing deltas | parked — tests lock intentional R1-style fullwidth bars; not broken generation |
| DI-DS-CERT | IMPL | V4 nextn fail-closed without cert env | by design (ADR-020) |

## Open unparked P0/P1

**Zero open unparked P0/P1** on the DeepSeek MTP exactness + think-token + draft/accept temperature + min_p rejection-sampling surface after this pass.

## Residual LIMIT

- Product default: DeepSeek V4 nextn remains direct-decode until `AX_MLX_DEEPSEEK_V4_MTP_CERTIFICATION_CANDIDATE=1` + Tier-2 evidence.
- Weight-backed formal A/B on host deferred (no V4 weights in this environment).
- Sampled multi-token production-cache adopt under cert remains a known MEDIUM residual vs singleton greedy.
