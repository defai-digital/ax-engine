# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['mtplx']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 3 | 58.8 | 100.0% |
| Qwen3.6 27B 4-bit | long_code | 3 | 58.4 | 99.7% |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 54.6 | 87.6% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 105.0 | 49.1% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 103.8 | 51.3% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 100.0 | 42.8% |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/flappy/mtplx.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/long_code/mtplx.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/python_modules_long/mtplx.json`
