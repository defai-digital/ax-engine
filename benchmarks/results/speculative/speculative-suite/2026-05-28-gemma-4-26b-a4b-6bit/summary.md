# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-6bit/snapshots/5f81a7a6f29e280f4bd5a4ce79d07d7a67fb867b`  
Prompt mode: `random`  
Sampling: greedy (T=0)  
Gen tokens: 128, Reps: 2+1w

Measurement gates: decode_time >= 0.5s, tok/s <= 500.0.  

All four columns share the same Python mlx_lm decode loop with argmax-only acceptance. This is a fair algorithmic comparison of drafting strategies.

| Case | Category | Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx suffix | ax-ngram (same-loop) |
|---|---|---:|---:|---:|---:|---:|
| random_p128 | random | 128 | 89.8 | 189.9 (ar=96.9%) | 230.7 (ar=100.0%) | 201.8 (ar=99.0%) |
| random_p512 | random | 512 | 89.2 | 181.0 (ar=97.9%) | 189.5 (ar=90.9%) | 167.7 (ar=82.1%) |
| random_p2048 | random | 2048 | 86.4 | 175.5 (ar=100.0%) | 85.6 (ar=27.9%) | 79.1 (ar=24.2%) |
