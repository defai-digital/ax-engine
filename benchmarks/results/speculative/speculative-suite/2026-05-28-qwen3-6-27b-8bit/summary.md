# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-8bit/snapshots/c5a593c1475a746e43a543b0a02bd2b357e5745f`  
Sampling: greedy (T=0), random-token prompts (mlx_lm.benchmark format)  
Gen tokens: 128, Reps: 2+1w

| Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx PLD |
|---:|---:|---:|---:|
| 128 | 17.7 | 17.8 | 17.8 |
| 512 | 17.7 | 17.6 | 17.6 |
| 2048 | 17.5 | 17.5 | 17.5 |
