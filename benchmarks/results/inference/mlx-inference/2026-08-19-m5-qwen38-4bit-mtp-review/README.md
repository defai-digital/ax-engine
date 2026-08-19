# 2026-08-19 M5: Qwen3.8-27B 4-bit MTP review (V4-2bit discipline)

CLI `ax-engine-bench generate`, one natural 200-token greedy prompt,
binary at `0edc42b1` (v7.1.4 line). Route counters — not request flags —
verify which drafter ran (the lesson from the corrected DeepSeek V4
2-bit evidence): all MTP rows show `ax_mtp_decode_steps ~104`,
`accepted_source_mtp`, zero n-gram tokens.

| run | tok/s | vs direct | d0 accept |
| --- | --- | --- | --- |
| 4bit direct | 33.40 | — | — |
| **4bit MTP (default)** | **39.95** | **1.20×** | 76.9% |
| 4bit MTP + exact env | 39.72 | 1.19× | 75.9% |
| 4bit MTP + `AX_MLX_MTP_LINEAR_EXACT_REPLAY=1` | 18.75 | 0.56× | 78.6% |
| 6bit direct | 34.25 | — | — |
| 6bit MTP (default) | 28.10 | 0.82× | 72.3% |

## Verdict

1. **Qwen 4-bit MTP is healthy.** 1.20× net with 76.9% depth-0
   acceptance — the highest of the 3.8 family. The 2026-08-19 harness
   "acceptance collapse" (3/8 then bypass) was a random-prompt,
   8-step-sample artifact with the utility gate doing its job; it was
   not a defect, and this review retires that verdict.
2. **The 6-bit sibling loses under the same default protocol** (0.82×)
   while winning under yesterday's harness protocol (1.15× at
   p128/random with the exact env): heavier 6-bit forwards do not
   absorb the non-exact replay verify the way 4-bit does.
3. **Exact-profile telemetry correction**: the exact-env artifact records
   `eligible: 1` and `selection: 2`, proving that the explicit profile was
   resolved at model load. Its final `enabled: 0` was a terminal-step
   telemetry artifact: the last short-budget/direct-fallback step dropped the
   per-step arithmetic scope and overwrote the stable runner decision. It was
   not evidence that the 3.8 pack ignored the opt-in. v7.1.4 keeps `enabled`
   model-stable and reports per-step activity separately as `active`. Formal
   Tier-2 re-certification for the 3.8 family remains pending.

## Addendum (same day): joint review with ax-code / DeepSeek V4 Pro

Full second-opinion report in `axcode-deepseek-v4-pro-review.md`
(`ax-code run -m deepseek/deepseek-v4-pro --sandbox read-only`). Outcomes:

1. **Verdict #3 above is corrected**: the default protocol is
   **Auto-exact ON** (`resolve_qwen_linear_mtp_exact` returns Auto=true
   for eligible packs with the env unset), and `exact_enabled: 0` in
   these artifacts is a terminal-step telemetry overwrite —
   `selection: 2` on the exact-env run proves the profile engaged.
   `341aace5` has since split selection from per-step activity. The
   6-bit loss was therefore never replay economics.
2. **Real cause found and fixed**: the row-by-row verify projection
   re-read the 6-bit pack's dense lm_head once per row (2.54 GB × S) —
   verify_eval 25.3 ms/step vs the 4-bit sibling's 9.2. Fix `5910f0de`
   routes dense-weight_t heads through one batched wide GEMV
   (bit-exactness across Leading pinned by `82aa8789`). Re-measured
   (`q6-fixed.json`): 6-bit MTP **28.10 → 32.99 tok/s**, verify_eval →
   15.9 ms/step; now 0.96× vs direct 34.25.
3. **Remaining lever, deferred with reasons**: `verify_forward_wall`
   ~30-33 ms/step is host-side and dominates the step (GPU eval is
   ~9-16 ms) — the verify loop is host-bound. The per-layer compiled
   closures that would cut it further are bounded by recorded
   fail-hash evidence (`bbcc72ad` compiling out_proj into the closure
   reproduced `f4b5490d`), and the unhooked `fa_attn_norm_qkv` helper
   batches QKV inside `mx::compile` where the leading-invariant custom
   kernels cannot run — hooking it without the formal Tier-2 parity
   harness would be speculative. Next step when that harness runs:
   evaluate the FA QKV closure and a compiled verify trunk together.
4. Confirmed non-targets: cache clone (~2.2 µs/step) and the draft
   path (3-4 ms/step, already async-overlapped) — leave alone.
