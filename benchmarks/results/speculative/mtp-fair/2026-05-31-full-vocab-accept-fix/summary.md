# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['ax_engine']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `native`
- mode: `sampled`
- max_tokens: `128`
- repetitions: `3`
- ax_pure_mtp: `True`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | AX tok/s | AX accept | AX/MTPLX |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | flappy | 3 | - | - | 65.2 | 88.0% | - |
| Qwen3.6 27B 4-bit | long_code | 3 | - | - | 63.0 | 84.2% | - |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | - | - | 45.4 | 57.9% | - |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | - | - | 161.2 | 95.4% | - |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | - | - | 163.8 | 93.8% | - |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | - | - | 168.3 | 85.4% | - |

Artifacts:

- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-full-vocab-accept-fix/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-full-vocab-accept-fix/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-full-vocab-accept-fix/27b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-full-vocab-accept-fix/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-full-vocab-accept-fix/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-31-full-vocab-accept-fix/35b-a3b-4bit/python_modules_long/ax_engine.json`
