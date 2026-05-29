# Speculative Suite Benchmark
Date: 2026-05-29  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd`  
Prompt mode: `random`  
Sampling: greedy (T=0)  
Gen tokens: 128, Reps: 5+2w

Measurement gates: decode_time >= 0.5s, tok/s <= 500.0.  

All four columns share the same Python mlx_lm decode loop with argmax-only acceptance. This is a fair algorithmic comparison of drafting strategies.

| Case | Category | Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx suffix | ax-ngram (same-loop) |
|---|---|---:|---:|---:|---:|---:|
| random_p128 | random | 128 | 140.3 | — | — | 87.8 (ar=0.0%) |
| random_p512 | random | 512 | 135.9 | — | — | 114.4 (ar=6.5%) |
| random_p2048 | random | 2048 | 137.7 | — | — | — (ar=100.0%) |
