# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/dcb78c3f5d6becacbfce71cd4851ad98c4f08a05`  
Sampling: greedy (T=0), random-token prompts (mlx_lm.benchmark format)  
Gen tokens: 128, Reps: 2+1w

| Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx PLD |
|---:|---:|---:|---:|
| 128 | 27.6 | 63.7 (ar=100%) | 63.3 (ar=100%) |
| 512 | 27.1 | 61.6 (ar=100%) | 61.3 (ar=100%) |
| 2048 | 26.0 | 41.0 (ar=99%) | 41.6 (ar=100%) |
