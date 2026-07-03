# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['ax_engine']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `native`
- mode: `sampled`
- max_tokens: `128`
- repetitions: `3`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | AX tok/s | AX accept | AX/MTPLX |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | flappy | 3 | - | - | 36.5 | 61.7% | - |
| Qwen3.6 27B 4-bit | long_code | 3 | - | - | 54.3 | 78.1% | - |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | - | - | 46.2 | 53.8% | - |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | - | - | 139.0 | 59.4% | - |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | - | - | 173.8 | 90.7% | - |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | - | - | 159.3 | 77.1% | - |

Artifacts:

- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-ax-engine-only/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-ax-engine-only/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-ax-engine-only/27b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-ax-engine-only/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-ax-engine-only/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-ax-engine-only/35b-a3b-4bit/python_modules_long/ax_engine.json`
