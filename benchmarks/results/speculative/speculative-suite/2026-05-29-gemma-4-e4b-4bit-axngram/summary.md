# Speculative Suite Benchmark
Date: 2026-05-29  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/deb1db712068b1c9f83fb1c97f08c1204b9459a1`  
Prompt mode: `random`  
Sampling: greedy (T=0)  
Gen tokens: 128, Reps: 5+2w

Measurement gates: decode_time >= 0.5s, tok/s <= 500.0.  

All four columns share the same Python mlx_lm decode loop with argmax-only acceptance. This is a fair algorithmic comparison of drafting strategies.

| Case | Category | Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx suffix | ax-ngram (same-loop) |
|---|---|---:|---:|---:|---:|---:|
| random_p128 | random | 128 | 92.5 | — | — | 61.2 (ar=0.0%) |
| random_p512 | random | 512 | 93.0 | — | — | 231.3 (ar=80.0%) |
| random_p2048 | random | 2048 | 91.0 | — | — | 253.2 (ar=100.0%) |
