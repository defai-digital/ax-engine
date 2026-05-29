# Speculative Suite Benchmark
Date: 2026-05-29  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-5bit/snapshots/7e2d6526209badeacaf09510e86528a107369316`  
Prompt mode: `random`  
Sampling: greedy (T=0)  
Gen tokens: 128, Reps: 5+2w

Measurement gates: decode_time >= 0.5s, tok/s <= 500.0.  

All four columns share the same Python mlx_lm decode loop with argmax-only acceptance. This is a fair algorithmic comparison of drafting strategies.

| Case | Category | Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx suffix | ax-ngram (same-loop) |
|---|---|---:|---:|---:|---:|---:|
| random_p128 | random | 128 | 125.0 | — | — | — (ar=100.0%) |
| random_p512 | random | 512 | 126.0 | — | — | — (ar=88.9%) |
| random_p2048 | random | 2048 | 122.4 | — | — | — (ar=100.0%) |
