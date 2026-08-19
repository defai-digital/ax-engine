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
