# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['mtplx', 'ax_engine']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal/samuelfaj optimized bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown
- fixed-depth rows and tuned-best-of rows are separate benchmark contracts

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | AX MTP tok/s | AX MTP accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 3 | 57.8 | 100.0% | 61.9 | 99.7% |
| Qwen3.6 27B 4-bit | long_code | 3 | 55.7 | 99.7% | 57.1 | 99.6% |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 50.5 | 87.6% | 48.6 | 97.8% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 98.4 | 48.7% | 156.8 | 100.0% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 91.4 | 49.5% | 154.9 | 99.9% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 90.3 | 43.2% | 157.6 | 97.9% |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/27b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-23-qwen36-4bit-mtp-rerun/35b-a3b-4bit/python_modules_long/ax_engine.json`
