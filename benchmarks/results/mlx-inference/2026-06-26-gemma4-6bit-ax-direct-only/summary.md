# Gemma 4 6-bit AX direct-only refresh

- Date: 2026-06-26
- Harness: `scripts/bench_mlx_inference_stack.py`
- Mode: AX direct baseline only (`--ax-direct --skip-mlx-lm --no-build-ax-engine`)
- Shape: prompt tokens `128,512,2048`, generation tokens `128`, repetitions `5`, cooldown `15s`
- Build: release `ax-engine-server`, commit `5e48d22217d1df4473af872418ed30e318e0fd35`

| Model | Prompt tok | Prefill tok/s | Decode tok/s | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 E4B 6-bit | 128 | 3,010.1 | 108.9 | 42.5 |
|  | 512 | 6,388.3 | 107.8 | 80.1 |
|  | 2048 | 8,202.4 | 107.4 | 249.7 |
| Gemma 4 26B A4B 6-bit | 128 | 1,162.8 | 111.0 | 110.1 |
|  | 512 | 2,788.7 | 107.6 | 183.6 |
|  | 2048 | 4,357.3 | 104.0 | 470.0 |
| Gemma 4 31B 6-bit | 128 | 427.7 | 20.0 | 299.3 |
|  | 512 | 652.1 | 19.7 | 785.2 |
|  | 2048 | 706.5 | 18.8 | 2,898.7 |

Gemma 4 E4B 6-bit has no `mlx_lm` percentage deltas in the README because the
matching `mlx_lm.benchmark` run failed to load the upstream MLX checkpoint.
The AX row still uses the same random-token prompt contract and direct
same-policy decode baseline as the other README AX rows.
