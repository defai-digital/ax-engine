# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['mtplx', 'rapid_mlx', 'ax_engine']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `128`
- repetitions: `1`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown
- tri-engine Rapid comparison uses shared depth 1 unless --depth overrides it
- Rapid-MLX server path exposes throughput but not accepted/drafted token telemetry

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | Rapid tok/s | Rapid accept | AX tok/s | AX accept | AX/MTPLX | AX/Rapid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | flappy | 1 | 23.5 | 1.4% | - | - | 25.6 | 12.4% | 1.087 | - |
| Qwen3.6 27B 4-bit | long_code | 1 | 21.7 | 1.0% | - | - | 43.2 | 68.9% | 1.990 | - |
| Qwen3.6 27B 4-bit | python_modules_long | 1 | 22.7 | 2.5% | - | - | 40.1 | 68.0% | 1.766 | - |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 88.7 | 7.0% | - | - | - | - | - | - |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 73.0 | 10.5% | - | - | - | - | - | - |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 77.0 | 15.8% | - | - | - | - | - | - |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / rapid_mlx: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/27b-4bit/flappy/rapid_mlx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `7/8` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / rapid_mlx: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/27b-4bit/long_code/rapid_mlx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / rapid_mlx: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/27b-4bit/python_modules_long/rapid_mlx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/27b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / rapid_mlx: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/35b-a3b-4bit/flappy/rapid_mlx.json`
- 35b-a3b-4bit / flappy / ax_engine: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `6/8` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / rapid_mlx: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/35b-a3b-4bit/long_code/rapid_mlx.json`
- 35b-a3b-4bit / long_code / ax_engine: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / rapid_mlx: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/35b-a3b-4bit/python_modules_long/rapid_mlx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-all-suites-final/35b-a3b-4bit/python_modules_long/ax_engine.json`
