# Findings — deepseek-mtp (exactness + template)

**Agents:** Codex `gpt-5.6-sol` reasoning max + ax-code `zai-coding-plan/glm-5.2[1m]`  
**Date:** 2026-08-11

## Work queue

| ID | Class | Sev | Title | Status |
| --- | --- | --- | --- | --- |
| DI-DS-MTP-001 | MTP | P1 | Stochastic think draft T vs accept log-prob T mismatch | fixed |
| DI-DS-MTP-002 | MTP | P1 | Hybrid `after_forced_prefix` hard-coded draft T=0.7 | fixed |
| DI-DS-MTP-003 | MTP | P1 | Pending-draft accept used recomputed T across steps | fixed |
| DI-DS-A001 | MTP | P1 | DeepSeek think-token family defaults missing | fixed |
| DI-DS-MTP-BND | MTP | P1 | Next draft used pre-result `ngram_in_think` | fixed |
| DI-DS-MTP-004 | MTP | P1 | `token_distribution` ignored min_p | fixed |
| DI-DS-MTP-005 | MTP | P1 | MTP target probs ignored min_p (thinking default 0.05) | fixed |
| DI-DS-TPL-001 | DOC/TEST | P2 | V4 Flash package IDs select DeepSeekChat / non-thinking | fixed (test lock) |
| DI-DS-A006 | DOC | LOW | Stochastic draft doc overstated top-p/k | fixed (doc) |

## Open unparked P0/P1

**None.**

## Details (this pass)

### DI-DS-A001 — fixed

**Problem:** `think_token_ids_from_manifest` had no deepseek arms → `think_start_token_id=None` → `ngram_in_think` never transitions → think-aware draft T inert.

**Fix:** Family defaults from official tokenizers:
- `deepseek_v4`: 128821 / 128822
- `deepseek_v3` \| `deepseek_v32`: 128798 / 128799  
Converter already parses content strings `<think>`/`</think>` when tokenizer.json is present.

### DI-DS-MTP-BND — fixed

**Problem:** Next draft used stale `state.ngram_in_think` while `think_state_after_result` was available for n-gram policy.

**Fix:** `deepseek_next_draft_temperature` / `next_draft_log_prob_temperature` from post-result think state; pending stores that T for the following accept.

### DI-DS-MTP-004 / 005 — fixed

**Problem:** DeepSeek thinking defaults `min_p=0.05`, but rejection-sampling target probs and `token_distribution` residual correction ignored min_p → wrong `p(token)` vs primary sampler.

**Fix:**
- `token_distribution` applies min_p before top-k/top-p (sampler parity).
- `compute_mtp_target_probs` uses `FullRows` when min_p set; TopK extract renorms under min_p.
- Qwen linear exact profile rejects min_p (`mtp_exact_sampling_supported`).

## Residual LIMIT

- Product default for DeepSeek V4 nextn remains **fail-closed** until Tier 2 certification (`AX_MLX_DEEPSEEK_V4_MTP_CERTIFICATION_CANDIDATE`).
- Weight-backed formal A/B exactness on host still deferred without V4 weights.
- Sampled multi-token verify residual (compressor/latent-K vs singleton) parked MEDIUM on opt-in cert path; greedy sequential path is exact.
- V3 has no nextn MTP in this codebase (dense MLA only).
