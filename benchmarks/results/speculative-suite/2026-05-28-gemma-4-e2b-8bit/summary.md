# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-8bit/snapshots/0cc7ae1721072b5bcd2716d161e4a3b5e786a11e`  
Sampling: greedy (T=0), random-token prompts (mlx_lm.benchmark format)  
Gen tokens: 128, Reps: 3+1w

| Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx PLD |
|---:|---:|---:|---:|
| 128 | 129.7 | 129.6 | 129.6 |
| 512 | 127.2 | 127.3 | 127.3 |
| 2048 | 124.0 | 368.3 (ar=100%) | 364.8 (ar=100%) |
