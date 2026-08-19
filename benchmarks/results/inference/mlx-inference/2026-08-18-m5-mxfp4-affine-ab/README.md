# 2026-08-18 M5 A/B: MXFP4 fused parity + affine-bits levers

Host `df-macbookpro-m5`, harness `scripts/bench_mlx_inference_stack.py`
(`--skip-mlx-lm --no-build-ax-engine`, 5 reps, 15 s cooldown, GPU wake ON).
Baseline binary built at `30caec32` (has the DVFS wake fix, predates the
MXFP4 fused-parity and affine-bits merges); "new" binary at `670adf2d`.
All runs `AutomatosX/AX-Qwen3.8-27B` AXQ packs from the local HF cache.

## R1 vs R2 — MXFP4-MTP pack, direct policy, baseline vs new binary

| lane | prefill tok/s | decode tok/s | verdict |
| --- | --- | --- | --- |
| p128 | 621.0 → 616.1 (0.992×) | 35.77 → 35.73 | flat |
| p512 | 904.1 → 902.3 (0.998×) | 35.62 → 35.60 | flat |
| p2048 | 960.7 → 960.4 (1.000×) | 35.14 → 35.17 | flat |

The scales-only MXFP4 fused-helper hosting (65c2b05e) is **perf-neutral on
the M5 server path** — its value is the correctness fixes (mislabeled-pack
panic gate) and capability parity. Context: this pack's prefill was already
ahead of the mixed-recipe "6bit" sibling (621/904/961 vs 369/694/882 in the
2026-08-18 coder-vl campaign), so the review's "MXFP4 lags its affine
sibling" premise does not hold on the M5 server path.

## R3 vs R4 — "6bit" pack, `AX_MLX_QWEN_DENSE_FFN_MATVEC_EXT_BITS` off/on

p128 decode 33.49 → 33.50 (1.000×) — **invalid as a 6-bit kernel A/B**: route
telemetry showed the 4-bit matvec kernel engaged 320/320 in *both* runs.
Per-tensor inspection of `AX-Qwen3.8-27B-MLX-AXQ-6bit` shows the AXQ "6bit"
recipe is mixed: **all 64 dense-FFN gate/up/down tensors are 4-bit gs32**;
the 6-bit budget sits in `linear_attn.in_proj_qkv` (48×, gs64) plus a
minority of `in_proj_a/b`, `k/v_proj`, and the 8-bit embed. The 6/8-bit
matvec kernels therefore have **no engagement surface on AXQ packs** and
stay default-OFF; they are relevant only to uniform community 6/8-bit
checkpoints.

## Parity probe + R8 vs R9 — `AX_MLX_EXACT_RMS_GATE_METAL`

4-way greedy probe (256 tokens, fixed prompt, server default MTP policy):
`a` MTP-off, `b` MTP-on, `c` MTP-on + gate-metal, `d` MTP-off + gate-metal
— **all four token-identical** (sha256 in `parity-hashes.txt`). The lever is
parity-clean on this probe, supporting the SDPA-drift root-cause hypothesis.
Perf (R8→R9, exact MTP p128): decode 53.75 → 53.63 (0.998×), prefill
693 → 697 — **neutral**, so the lever stays default OFF: the portable
RMS+SiLU chain is not a measurable bottleneck at this lane.

## R6 vs R7 — 6bit-MTP, `AX_MLX_MTP_MAX_DEPTH=4`

Decode 26.82 → 26.81 (1.000×) — raising the max-depth cap is a **no-op**:
the adaptive gate's acceptance-driven depth choice, not the cap, governs
actual draft depth. The qmv_wide S≤5 free-verify argument needs an adaptive
policy retune, not a cap change.

## Not exercisable this campaign

`AX_MLX_DENSE_WIDE_GEMV` needs a dense (unquantized) lm_head; every AXQ
Qwen3.8 pack quantizes it (MXFP4-MTP: affine 8-bit gs64), so the hook never
fires here. The M3 Max micro-bench evidence (1.615×/1.673× at S=2/4) stands;
the model-level A/B waits for a dense-lm_head target.
