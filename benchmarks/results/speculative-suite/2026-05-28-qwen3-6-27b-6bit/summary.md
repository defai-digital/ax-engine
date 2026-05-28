# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-6bit/snapshots/9bf976157e09080fbc11ccd971d4e9c57554889d`  
Sampling: greedy (T=0), random-token prompts (mlx_lm.benchmark format)  
Gen tokens: 128, Reps: 2+1w

| Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx PLD |
|---:|---:|---:|---:|
| 128 | 22.4 | 22.5 | 22.5 |
| 512 | 22.4 | 22.4 | 22.5 |
| 2048 | 22.6 | 22.6 | 22.6 |
