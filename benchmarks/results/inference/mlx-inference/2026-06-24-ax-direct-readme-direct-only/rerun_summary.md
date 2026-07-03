# AX direct README rerun summary

- completed rows: 7/8
- mode: `scripts/bench_mlx_inference_stack.py --ax-direct`
- reference rows: reused existing `mlx_lm` JSONs from `2026-05-26-direct-mode-clean-refresh`
- prompt tokens: 128, 512, 2048; generation tokens: 128; repetitions: 5; cooldown: 15s

| slug | status | decode tok/s medians 128 / 512 / 2048 | prefill tok/s medians 128 / 512 / 2048 | TTFT ms medians 128 / 512 / 2048 |
|---|---|---:|---:|---:|
| gemma-4-e2b-it-4bit | ok | 234.0 / 226.2 / 216.8 | 5230.3 / 14149.5 / 23988.3 | 24.5 / 36.2 / 85.4 |
| gemma-4-e2b-it-6bit | ok | 186.1 / 179.9 / 173.6 | 5156.7 / 15512.3 / 23063.3 | 24.8 / 33.0 / 88.8 |
| gemma-4-e4b-it-4bit | ok | 143.3 / 140.3 / 135.1 | 3516.1 / 7070.8 / 8789.2 | 36.4 / 72.4 / 233.0 |
| gemma-4-26b-a4b-it-4bit | ok | 129.1 / 131.0 / 125.3 | 1332.2 / 3021.7 / 3295.0 | 96.1 / 169.4 / 621.5 |
| gemma-4-31b-it-4bit | ok | 28.9 / 28.4 / 27.1 | 509.1 / 735.3 / 768.9 | 251.4 / 696.3 / 2663.6 |
| qwen3_6-27b-4bit | ok | 34.1 / 34.0 / 33.4 | 581.7 / 835.2 / 932.5 | 220.0 / 613.0 / 2196.3 |
| qwen3_6-27b-6bit | ok | 25.1 / 25.0 / 24.9 | 508.7 / 759.8 / 866.4 | 251.6 / 673.9 / 2363.9 |
| qwen3_6-35b-a3b-4bit | blocked_stalled_before_first_measured_rep | n/a | n/a | n/a |
