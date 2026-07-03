# Speculative Suite Benchmark
Date: 2026-05-29  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05`  
Prompt mode: `random`  
Sampling: greedy (T=0)  
Gen tokens: 128, Reps: 5+2w

Measurement gates: decode_time >= 0.5s, tok/s <= 500.0.  

All four columns share the same Python mlx_lm decode loop with argmax-only acceptance. This is a fair algorithmic comparison of drafting strategies.

| Case | Category | Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx suffix | ax-ngram (same-loop) |
|---|---|---:|---:|---:|---:|---:|
| random_p128 | random | 128 | 23.4 | — | — | 57.7 (ar=100.0%) |
| random_p512 | random | 512 | 22.9 | — | — | 56.0 (ar=100.0%) |
| random_p2048 | random | 2048 | 22.0 | — | — | 53.0 (ar=99.0%) |
