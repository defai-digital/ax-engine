# AX direct README current AX-only sweep

- started_at: 2026-07-07T01:20:12-0400
- finished_at: 2026-07-07T02:22:45-0400
- updated_at: 2026-07-07T02:32:03-0400
- method: run_ax_engine_direct_only_skip_mlx_lm; retained README reference sources are used only during README/table validation
- prompt_tokens: 128,512,2048
- generation_tokens: 128
- repetitions: 5
- warmup_repetitions: 1
- cooldown: 15.0
- elapsed_seconds: 3753.1
- publication_candidate: True

| slug | status | elapsed_s | reference_for_readme_delta | output | log | note |
|---|---:|---:|---|---|---|---|
| gemma-4-e2b-it-4bit | ok | 246.0 | benchmarks/results/inference/mlx-inference/2026-05-26-direct-mode-clean-refresh/gemma-4-e2b-it-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/gemma-4-e2b-it-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/gemma-4-e2b-it-4bit.log |  |
| gemma-4-e2b-it-6bit | ok | 250.3 | benchmarks/results/inference/mlx-inference/2026-05-26-direct-mode-clean-refresh/gemma-4-e2b-it-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/gemma-4-e2b-it-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/gemma-4-e2b-it-6bit.log |  |
| gemma-4-e4b-it-4bit | ok | 256.0 | benchmarks/results/inference/mlx-inference/2026-05-26-direct-mode-clean-refresh/gemma-4-e4b-it-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/gemma-4-e4b-it-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/gemma-4-e4b-it-4bit.log |  |
| gemma-4-e4b-it-6bit | ok | 262.7 | none | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/gemma-4-e4b-it-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/gemma-4-e4b-it-6bit.log |  |
| gemma-4-26b-a4b-it-4bit | ok | 265.8 | benchmarks/results/inference/mlx-inference/2026-05-26-direct-mode-clean-refresh/gemma-4-26b-a4b-it-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/gemma-4-26b-a4b-it-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/gemma-4-26b-a4b-it-4bit.log |  |
| gemma-4-26b-a4b-it-6bit | ok | 274.9 | benchmarks/results/inference/mlx-inference/2026-07-02-gemma4-6bit-direct-refresh/gemma-4-26b-a4b-it-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/gemma-4-26b-a4b-it-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/gemma-4-26b-a4b-it-6bit.log |  |
| gemma-4-31b-it-4bit | ok | 356.4 | benchmarks/results/inference/mlx-inference/2026-05-26-direct-mode-clean-refresh/gemma-4-31b-it-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/gemma-4-31b-it-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/gemma-4-31b-it-4bit.log |  |
| gemma-4-31b-it-6bit | ok | 570.8 | benchmarks/results/inference/mlx-inference/2026-07-02-gemma4-6bit-direct-refresh/gemma-4-31b-it-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/gemma-4-31b-it-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/gemma-4-31b-it-6bit.log |  |
| qwen3_6-27b-4bit | ok | 337.8 | benchmarks/results/inference/mlx-inference/2026-06-26-qwen36-direct-refresh/qwen3_6-27b-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/qwen3_6-27b-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/qwen3_6-27b-4bit.log |  |
| qwen3_6-27b-6bit | ok | 370.3 | benchmarks/results/inference/mlx-inference/2026-06-26-qwen36-direct-refresh/qwen3_6-27b-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/qwen3_6-27b-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/qwen3_6-27b-6bit.log |  |
| qwen3_6-35b-a3b-4bit | ok | 270.7 | benchmarks/results/inference/mlx-inference/2026-06-26-qwen36-direct-refresh/qwen3_6-35b-a3b-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/qwen3_6-35b-a3b-4bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/qwen3_6-35b-a3b-4bit-rerun.log | initial artifact failed run_stability_summary tail_regression on prompt=128 |
| qwen3_6-35b-a3b-6bit | ok | 291.3 | benchmarks/results/inference/mlx-inference/2026-06-26-qwen36-direct-refresh/qwen3_6-35b-a3b-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/qwen3_6-35b-a3b-6bit.json | benchmarks/results/inference/mlx-inference/2026-07-07-ax-direct-readme-current-ax-only/logs/qwen3_6-35b-a3b-6bit.log |  |
