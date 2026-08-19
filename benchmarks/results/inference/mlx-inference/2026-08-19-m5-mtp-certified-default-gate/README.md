# 2026-08-19 M5: MTP certified-default gate validation

Host `df-macbookpro-m5`, binary built from `feat/mtp-certified-default-gate`
(`15401017`), CLI `ax-engine-bench generate --mlx --deterministic true
--ignore-eos`, 28-token prompt (same as the 08-19 dense-head-fix runs),
200 output tokens, one GPU-wake generate before each measured run.

The gate: default-on model MTP now requires the pack's
`axquant_runtime.json` `"mtp"` block to certify a win
(`enabled_by_default` && (`optimized` || `measured_speedup >= 1.0`)),
fail-closed. Today every published MTP pack carries
`optimized: false, measured_speedup: null`, so all default to MTP-off
until stamped.

## Plan correction (load-bearing)

The ax-code/DeepSeek-V4-Pro plan scoped the gate to `QwenCalibrated`
(route 1), asserting the Qwen3.8 AXQ family lives there. Production
telemetry disproved this: **all 3.8 packs are linear-attention and reach
route 2** (`QwenLinearCertificationCandidateDepthOne`) by default,
because the exact profile auto-enables on eligible packs and
`resolve_qwen_linear_certification_candidate` accepts the exact profile
alone. The gate therefore covers routes 1/2/3, with the explicit
`AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE` env opt-in keeping
default-on (formal-harness recipe unchanged). Confirmed by a second
DeepSeek-V4-Pro review, which also caught that **MXFP4-MTP is excluded
from route 2 upstream** (`!has_mxfp4_linears`) — it is already
default-off today, its 1.50× was candidate-env-forced, and stamping its
metadata would be inert; MXFP4 default-on needs a separate
auto-promotion decision.

## Validation matrix (6bit = `AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP`, 4bit = `…-4bit-MTP`)

| run | config | tok/s | steps | mtp_requested | mtp_decode_steps | certified_default_on |
| --- | --- | --- | --- | --- | --- | --- |
| a | 6bit, no envs (the 0.96× row) | 32.83 | 197 | **0** | 0 | 0 |
| b | 6bit + `AX_MLX_MTP_FORCE_REQUESTED=1` | 33.26 | 123 | 1 | 104 | 0 |
| c | 6bit + `…_CERTIFICATION_CANDIDATE=1` | 33.18 | 123 | 1 | 104 | **1** (env carve-out) |
| d | 4bit, no envs, unstamped metadata | 32.63 | 200 | 0 | 0 | 0 |
| e | 4bit, metadata stamped locally (`optimized:true`, `measured_speedup:1.20`) | **39.26** | 122 | 1 | 107 | 1 |
| f | 6bit + `AX_NO_SPEC=1` (pure direct) | 34.25 | 200 | 0 | 0 | 0 |

- **a**: the flagship no longer pays the 0.96× MTP tax by default; n-gram
  acceleration stays active (3 ngram steps — its stream drift vs pure
  direct is the pre-existing non-exact batched-verify behavior, not
  introduced here).
- **e**: metadata alone flips a winner back to default-on MTP (1.20×
  reproduced: 32.63 → 39.26), no envs. Run against a symlink-farm copy
  of the snapshot with only `axquant_runtime.json` edited.
- **f**: pure-direct stream is **bit-identical to the pre-gate binary**
  (`cli-fix38-a1`), 34.25 tok/s — the gate changes request policy only,
  no arithmetic.
- Telemetry keys `ax_mlx_mtp_certified_default_on` /
  `ax_mlx_mtp_runtime_{enabled_by_default,optimized,measured_speedup_x1000}`
  make the gating reason visible in every row above.

## Follow-ups

1. Stamp HF metadata for the two affine winners (4bit `1.20`, 8bit
   `1.55`) — outward-facing, needs owner confirmation.
2. MXFP4 auto-promotion (lift `!has_mxfp4_linears` after certification)
   — separate engine change, not a metadata stamp.
3. 6-bit (0.96×) and Qwen3.6-6bit (~1.0×) stay uncertified by design;
   revisit after the compiled-verify-trunk host-bound work.
