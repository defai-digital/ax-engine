# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-5bit/snapshots/46f9b268ffe53528a8ad4ce8f684a480b3ef0d18`  
Sampling: greedy (T=0), random-token prompts (mlx_lm.benchmark format)  
Gen tokens: 128, Reps: 2+1w

| Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx PLD |
|---:|---:|---:|---:|
| 128 | 26.1 | 26.4 | 26.3 |
| 512 | 26.2 | 26.2 | 26.2 |
| 2048 | 26.0 | 26.0 | 26.0 |
