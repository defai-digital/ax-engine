# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['mtplx', 'rapid_mlx', 'ax_engine']`
- suites: `['flappy']`
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
| Qwen3.6 27B 4-bit | flappy | 1 | 23.3 | 1.4% | - | - | 25.3 | 12.4% | 1.087 | - |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 87.0 | 7.0% | - | - | - | - | - | - |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-flappy-final/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / rapid_mlx: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-flappy-final/27b-4bit/flappy/rapid_mlx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-flappy-final/27b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-flappy-final/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / rapid_mlx: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-flappy-final/35b-a3b-4bit/flappy/rapid_mlx.json`
- 35b-a3b-4bit / flappy / ax_engine: `error` `benchmarks/results/mtp-fair/2026-05-28-qwen36-fair-flappy-final/35b-a3b-4bit/flappy/ax_engine.json`
