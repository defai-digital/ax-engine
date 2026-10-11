# Qwen 3.8 27B MXFP4: host-adaptive activation dtype and dense-GEMM prefill — 2026-10-10

Raw `scripts/bench_mlx_inference_stack.py` artifacts behind
[`docs/performance/qwen38-27b-pre-m3-prefill-2026-10-10.md`](../../../../../docs/performance/qwen38-27b-pre-m3-prefill-2026-10-10.md).
Pack `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP` @
`b2c5354f779e430d0c1733143db848a72b71c16e`, MLX 0.32.3 pip wheel, greedy.
Hosts are identified by hardware only. Neither host is the 27B SKU (Mac mini
M4 Pro 64 GB); nothing here qualifies anything.

## `m2-ultra-192gb/` (Mac Studio M2 Ultra 192 GB, macOS 27, rustup 1.97.1)

- `baseline-direct-fe13a04b.json`, `baseline-mtp-flappy-fe13a04b.json`:
  the branch base (3 reps + 2 warmups).
- `reference-mlx-lm-0.31.3.json`: `mlx_lm.benchmark` on the same pack.
- `ab-direct-*.json`, `ab-mtp-*.json`: one binary (defaults = `auto`),
  every arm sets `AX_MLX_ACTIVATION_DTYPE`, `AX_MLX_QWEN_PREFILL_DEQUANT_DENSE`
  (and for MTP `AX_MLX_EXACT_LM_HEAD_WINDOW_INVARIANT`) explicitly;
  2 reps + 1 warmup; `output_token_ids` captured per trial for identity diffs.
- `decode-trace-baseline.log`: direct pipeline host/GPU split.
- `qmm-dtype-probe.log`, `prefill-floor-1024.log`, `prefill-floor-2048.log`,
  `bandwidth-probe-idle.log`: stock-MLX per-op probes (bf16 vs fp16
  `quantized_matmul`, per-op model floor, achievable read bandwidth).
- `mlxlm-dtype-bfloat16.log`, `mlxlm-dtype-float16.log`: stock `mlx_lm`
  with the model cast to each dtype, prompt tok/s and greedy token heads.

## `m5-max-128gb/` (MacBook Pro M5 Max 128 GB, macOS 27, Homebrew cargo 1.98.0, Python 3.13 venv)

- `baseline-direct-e8c60bb1.json`, `baseline-mtp-flappy-e8c60bb1.json`:
  defaults of the WIP commit (= the pre-change behaviour on this host:
  bf16, qmm, row-by-row lm_head window); 3 reps + 2 warmups.
- `ab-direct-fp16.json`, `ab-direct-dequant-resident.json`,
  `ab-direct-dequant-transient.json`, `ab-mtp-window.json`,
  `ab-mtp-fp16.json`: single-flag arms, 2 reps + 1 warmup, token ids captured.

## `qwen3_5-9b-m2-ultra-192gb/` (Mac Studio M2 Ultra 192 GB, final commit)

`mlx-community/Qwen3.5-9B-MLX-4bit` @ `938d8919` (affine 4-bit g64, no MTP
head), 3 reps + 2 warmups, token ids captured: `before-direct-with-mlx-lm.json`
(bf16 / qmm / row loop forced by env, plus the `mlx_lm.benchmark` rows),
`after-direct-defaults.json` (`auto` policies), `after-direct-fp16-only.json`.
