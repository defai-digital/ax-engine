# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['mtplx', 'lightning_mlx', 'ax_engine']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`
- ax_pure_mtp: `True`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | Lightning tok/s | Lightning accept | AX tok/s | AX accept | AX/MTPLX | AX/Lightning |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | flappy | 3 | 51.5 | 100.0% | 49.5 | 96.5% | 37.6 | 99.1% | 0.731 | 0.760 |
| Qwen3.6 27B 4-bit | long_code | 3 | 53.5 | 99.7% | 51.7 | 93.3% | 38.0 | 98.3% | 0.711 | 0.736 |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 51.6 | 87.6% | 46.9 | 76.5% | 22.9 | 67.0% | 0.443 | 0.488 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 107.3 | 50.8% | 147.9 | 99.0% | 84.2 | 99.9% | 0.785 | 0.569 |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 106.4 | 50.5% | 149.0 | 98.5% | 81.5 | 99.8% | 0.766 | 0.547 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 102.7 | 42.6% | 148.8 | 97.2% | 79.0 | 93.2% | 0.769 | 0.531 |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/27b-4bit/flappy/lightning_mlx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/27b-4bit/long_code/lightning_mlx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/27b-4bit/python_modules_long/lightning_mlx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/27b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/35b-a3b-4bit/flappy/lightning_mlx.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/35b-a3b-4bit/long_code/lightning_mlx.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/35b-a3b-4bit/python_modules_long/lightning_mlx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-final/35b-a3b-4bit/python_modules_long/ax_engine.json`
