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
3. **Open engine question consolidated**: every Qwen3.8 pack reports
   `ax_mlx_qwen_linear_mtp_exact_enabled: 0` (with `eligible: 1`) even
   when the env opt-in is set, while the certified Qwen3.6 pack reports
   1 — and the 4-bit exact-env run's throughput matched default. Either
   the telemetry field is recorded before the effective resolution or
   the 3.8 packs never actually engage the checkpoint path. This gates
   the 6-bit default-serve MTP economics and belongs to the same
   investigation as the pending Tier-2 re-cert for the 3.8 family.
