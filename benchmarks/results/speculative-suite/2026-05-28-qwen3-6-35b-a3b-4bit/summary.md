# Speculative Suite Benchmark
Date: 2026-05-28  
Model: `/Users/akiralam/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46`  
Prompt mode: `rapid`  
Sampling: greedy (T=0)  
Gen tokens: 128, Reps: 2+1w

Measurement gates: decode_time >= 0.5s, tok/s <= 500.0.  

| Case | Category | Prompt tok | baseline (tok/s) | lightning n-gram | rapid-mlx suffix |
|---|---|---:|---:|---:|---:|
| chat | low_repeat | 30 | 107.8 | — (ar=non_trimmable_cache) | — (ar=non_trimmable_cache) |
| json_array | structured | 62 | 108.0 | — (ar=non_trimmable_cache) | — (ar=non_trimmable_cache) |
| tool_loop | agentic | 569 | 108.9 | — (ar=non_trimmable_cache) | — (ar=non_trimmable_cache) |
| code_edit | structured | 85 | 109.6 | — (ar=non_trimmable_cache) | — (ar=non_trimmable_cache) |
