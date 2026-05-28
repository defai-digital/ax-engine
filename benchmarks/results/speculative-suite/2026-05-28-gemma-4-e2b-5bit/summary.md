# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-5bit/snapshots/7e2d6526209badeacaf09510e86528a107369316`  
Sampling: greedy (T=0), random-token prompts (mlx_lm.benchmark format)  
Gen tokens: 128, Reps: 3+1w

| Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx PLD |
|---:|---:|---:|---:|
| 128 | 158.9 | 158.8 | 158.8 |
| 512 | 220.8 | 216.3 | 216.7 |
| 2048 | 150.4 | 381.8 (ar=100%) | 378.4 (ar=100%) |
