# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-6bit/snapshots/938d4fb4ebff2df7f6c8200977cf82a06d20f5b9`  
Prompt mode: `random`  
Sampling: greedy (T=0)  
Gen tokens: 128, Reps: 2+1w

Measurement gates: decode_time >= 0.5s, tok/s <= 500.0.  

All four columns share the same Python mlx_lm decode loop with argmax-only acceptance. This is a fair algorithmic comparison of drafting strategies.

| Case | Category | Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx suffix | ax-ngram (same-loop) |
|---|---|---:|---:|---:|---:|---:|
| random_p128 | random | 128 | 19.0 | 41.4 (ar=100.0%) | 40.2 (ar=100.0%) | 42.1 (ar=100.0%) |
| random_p512 | random | 512 | 18.5 | 40.8 (ar=100.0%) | 40.0 (ar=100.0%) | 41.5 (ar=100.0%) |
| random_p2048 | random | 2048 | 18.2 | 38.1 (ar=100.0%) | 39.2 (ar=100.0%) | 40.0 (ar=100.0%) |
