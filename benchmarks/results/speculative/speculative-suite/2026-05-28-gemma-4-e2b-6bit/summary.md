# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-6bit/snapshots/6fe8c3cfab2910e5bc3439568f6f89413b4d1dca`  
Sampling: greedy (T=0), random-token prompts (mlx_lm.benchmark format)  
Gen tokens: 128, Reps: 3+1w

| Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx PLD |
|---:|---:|---:|---:|
| 128 | 143.5 | 143.6 | 143.7 |
| 512 | 141.1 | 141.2 | 141.1 |
| 2048 | 136.7 | 136.7 | 136.6 |
