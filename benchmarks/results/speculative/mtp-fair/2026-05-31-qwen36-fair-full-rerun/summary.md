# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit']`
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
| Qwen3.6 27B 4-bit | flappy | 3 | 55.5 | 100.0% | 48.8 | 96.8% | 37.6 | 99.1% | 0.679 | 0.771 |
| Qwen3.6 27B 4-bit | long_code | 3 | 44.3 | 99.7% | 48.7 | 93.3% | 38.0 | 98.3% | 0.859 | 0.782 |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 47.7 | 87.6% | 45.0 | 73.7% | 22.9 | 67.0% | 0.480 | 0.508 |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/flappy/lightning_mlx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/long_code/lightning_mlx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/python_modules_long/lightning_mlx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-qwen36-fair-full-rerun/27b-4bit/python_modules_long/ax_engine.json`
