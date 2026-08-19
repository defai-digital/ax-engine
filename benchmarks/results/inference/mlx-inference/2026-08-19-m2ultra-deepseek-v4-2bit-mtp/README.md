# 2026-08-19 M2 Ultra: DeepSeek V4 Flash 2-bit + MTP validation

Host `df-macstudio-m2` (Mac14,14 M2 Ultra, 192 GB), freshly
bootstrapped (rustup + MLX 0.32 wheel lib via `MLX_LIB_DIR`, repo at
`eec3478b`). Pack `AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP`
(122.3 GB, 37 files), loaded with `AX_ENGINE_2BIT_EXPERIMENTAL=1`.
CLI `ax-engine-bench generate`, pre-tokenized 200-token greedy runs.

| config | decode tok/s | steps/199 tok | output sha256[:16] |
| --- | --- | --- | --- |
| direct (AX_NO_SPEC) ×2 | 16.82 / 16.80 | 199 | `7442f1d94f9c5bb3` both |
| MTP (default) ×2 | **31.99 / 31.90** | **65** | `7442f1d94f9c5bb3` both |

- **MTP speedup 1.90×** — 199 tokens in 65 steps (~2.06 accepted
  drafts/step through the V4 nextn head).
- **All four streams token-identical**: MTP-on ≡ MTP-off and
  deterministic on the V4 nextn path, with no exact-profile
  complication (unlike the Qwen-linear packs).
- TTFT ~370-424 ms at p~30; 122 GB weights resident on 192 GB.
- Telemetry quirk to file: the `performance.mtp` block reports
  `available: false / draft_tokens: 0` while `step_count` (65) and the
  route's `ax_mtp_requested: 1` prove the nextn path ran — the CLI perf
  block maps the qwen-linear MTP counters, not the deepseek nextn ones.
- Probe harness note: wrapping the generate in `/usr/bin/time env ...`
  kills the run ("signal: Invalid argument"); invoke `env` directly.
