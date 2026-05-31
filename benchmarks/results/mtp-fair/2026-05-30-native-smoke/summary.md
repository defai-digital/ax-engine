# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit']`
- engines: `['mtplx', 'ax_engine']`
- suites: `['long_code', 'python_modules_long']`
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
| Qwen3.6 27B 4-bit | long_code | 3 | 58.9 | 99.1% | 53.1 | 77.0% | 0.901 |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 50.4 | 84.6% | 45.0 | 51.6% | 0.894 |

Artifacts:

- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `7/8` `benchmarks/results/mtp-fair/2026-05-30-native-smoke/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-30-native-smoke/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-30-native-smoke/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-30-native-smoke/27b-4bit/python_modules_long/ax_engine.json`
