# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit']`
- engines: `['rapid_mlx']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown
- tri-engine Rapid comparison uses shared depth 1 unless --depth overrides it
- Rapid-MLX server path exposes throughput but not accepted/drafted token telemetry

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | Rapid tok/s | Rapid accept | AX tok/s | AX accept | AX/MTPLX | AX/Rapid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | flappy | 1 | - | - | - | - | - | - | - | - |
| Qwen3.6 27B 4-bit | long_code | 1 | - | - | - | - | - | - | - | - |
| Qwen3.6 27B 4-bit | python_modules_long | 1 | - | - | - | - | - | - | - | - |

Artifacts:

- 27b-4bit / flappy / rapid_mlx: `no_valid_runs` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-corrected/27b-4bit/flappy/rapid_mlx.json`
- 27b-4bit / long_code / rapid_mlx: `no_valid_runs` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-corrected/27b-4bit/long_code/rapid_mlx.json`
- 27b-4bit / python_modules_long / rapid_mlx: `no_valid_runs` `benchmarks/results/mtp-fair/2026-05-30-qwen36-fair-corrected/27b-4bit/python_modules_long/rapid_mlx.json`
