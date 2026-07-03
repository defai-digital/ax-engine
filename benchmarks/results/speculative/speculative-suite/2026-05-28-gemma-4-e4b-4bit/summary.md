# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/cc3b666c01c20395e0dcebd53854504c7d9821f9`  
Sampling: greedy (T=0), random-token prompts (mlx_lm.benchmark format)  
Gen tokens: 128, Reps: 3+1w

| Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx PLD |
|---:|---:|---:|---:|
| 128 | 115.1 | 283.5 (ar=97%) | 289.9 (ar=96%) |
| 512 | 113.7 | 284.7 (ar=100%) | 282.2 (ar=100%) |
| 2048 | 111.7 | 303.9 (ar=100%) | 302.0 (ar=100%) |
