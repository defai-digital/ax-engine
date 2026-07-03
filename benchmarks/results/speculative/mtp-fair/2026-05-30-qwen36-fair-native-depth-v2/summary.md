# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['35b-a3b-4bit']`
- engines: `['mtplx', 'ax_engine']`
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
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 106.5 | 51.6% | 144.6 | 59.4% | 1.358 |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 94.8 | 48.0% | 164.6 | 90.7% | 1.737 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 100.0 | 48.6% | 151.3 | 77.1% | 1.513 |

Artifacts:

- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-native-depth-v2/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-native-depth-v2/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `7/8` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-native-depth-v2/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-native-depth-v2/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-native-depth-v2/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-native-depth-v2/35b-a3b-4bit/python_modules_long/ax_engine.json`
