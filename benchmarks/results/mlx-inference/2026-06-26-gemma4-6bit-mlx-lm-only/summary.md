# Gemma 4 6-bit mlx-lm-only refresh

- Date: 2026-06-26
- Harness: `scripts/bench_mlx_inference_stack.py`
- Mode: `mlx_lm.benchmark` only (`--skip-ax-engine --no-build-ax-engine`)
- Shape: prompt tokens `128,512,2048`, generation tokens `128`, repetitions `5`, cooldown `15s`

| Model | Prompt tok | Prefill tok/s | Decode tok/s | TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 26B A4B 6-bit | 128 | 414.0 | 103.7 | 309.2 |
|  | 512 | 1,285.0 | 101.1 | 398.4 |
|  | 2048 | 3,312.9 | 97.9 | 618.2 |
| Gemma 4 31B 6-bit | 128 | 259.6 | 19.6 | 493.1 |
|  | 512 | 548.6 | 19.3 | 933.3 |
|  | 2048 | 675.0 | 18.6 | 3,033.9 |

Gemma 4 E4B 6-bit was attempted after downloading
`mlx-community/gemma-4-e4b-it-6bit` and generating `model-manifest.json`.
`mlx_lm.benchmark` failed while loading the model because the installed
`mlx-lm` model instance rejected extra attention weights for language-model
layers 24 through 41. No numeric E4B 6-bit `mlx_lm` row is published from this
failed run.
