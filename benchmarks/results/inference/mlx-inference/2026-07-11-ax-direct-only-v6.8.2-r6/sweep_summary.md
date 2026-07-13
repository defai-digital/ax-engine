# AX-only sweep summary

- publication_candidate: false
- failed_row_count: 14
- status_counts: bench_failed=12, model_dir_missing=2
- completed_row_count: 14/14
- elapsed: 2s

| slug | status | notes |
|---|---|---|
| gemma-4-e2b-it-4bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/gemma-4-e2b-it-4bit.log |
| gemma-4-e2b-it-6bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/gemma-4-e2b-it-6bit.log |
| gemma-4-e2b-it-8bit | model_dir_missing | No HF cache snapshot for mlx-community/gemma-4-e2b-it-8bit |
| gemma-4-e4b-it-4bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/gemma-4-e4b-it-4bit.log |
| gemma-4-e4b-it-6bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/gemma-4-e4b-it-6bit.log |
| gemma-4-26b-a4b-it-4bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/gemma-4-26b-a4b-it-4bit.log |
| gemma-4-26b-a4b-it-6bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/gemma-4-26b-a4b-it-6bit.log |
| gemma-4-31b-it-4bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/gemma-4-31b-it-4bit.log |
| gemma-4-31b-it-6bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/gemma-4-31b-it-6bit.log |
| qwen3_6-27b-4bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/qwen3_6-27b-4bit.log |
| qwen3_6-27b-6bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/qwen3_6-27b-6bit.log |
| qwen3_6-27b-8bit | model_dir_missing | MLX cache snapshot for mlx-community/Qwen3.6-27B-8bit missing model-manifest.json |
| qwen3_6-35b-a3b-4bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/qwen3_6-35b-a3b-4bit.log |
| qwen3_6-35b-a3b-6bit | bench_failed | exit_code=1; log=benchmarks/results/inference/mlx-inference/2026-07-11-ax-direct-only-v6.8.2-r6/logs/qwen3_6-35b-a3b-6bit.log |
