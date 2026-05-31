# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['mtplx', 'ax_engine', 'lightning_mlx']`
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
| Qwen3.6 27B 4-bit | flappy | 3 | 39.2 | 100.0% | 48.8 | 95.9% | 37.2 | 99.1% | 0.949 | 0.762 |
| Qwen3.6 27B 4-bit | long_code | 3 | 44.3 | 99.7% | 48.7 | 92.8% | 27.6 | 98.3% | 0.625 | 0.568 |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 47.7 | 87.6% | 45.0 | 73.7% | 22.9 | 67.0% | 0.480 | 0.508 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 88.1 | 48.8% | 140.1 | 99.1% | 84.2 | 99.9% | 0.956 | 0.601 |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 105.2 | 52.3% | 140.4 | 98.6% | 81.5 | 99.8% | 0.775 | 0.580 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 95.2 | 42.3% | 137.8 | 96.8% | 79.0 | 93.2% | 0.829 | 0.573 |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/flappy/lightning_mlx.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / long_code / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/long_code/lightning_mlx.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/python_modules_long/ax_engine.json`
- 27b-4bit / python_modules_long / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/python_modules_long/lightning_mlx.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/35b-a3b-4bit/flappy/lightning_mlx.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / long_code / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/35b-a3b-4bit/long_code/lightning_mlx.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/35b-a3b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/35b-a3b-4bit/python_modules_long/lightning_mlx.json`
