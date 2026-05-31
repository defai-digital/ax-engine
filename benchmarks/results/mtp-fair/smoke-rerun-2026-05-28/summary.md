# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['35b-a3b-4bit']`
- engines: `['rapid_mlx', 'ax_engine']`
- suites: `['flappy']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `16`
- repetitions: `1`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown
- tri-engine Rapid comparison uses shared depth 1 unless --depth overrides it
- Rapid-MLX server path exposes throughput but not accepted/drafted token telemetry

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | Rapid tok/s | Rapid accept | AX tok/s | AX accept | AX/MTPLX | AX/Rapid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | - | - | - | - | 99.5 | 17.5% | - | - |

Artifacts:

- 35b-a3b-4bit / flappy / rapid_mlx: `no_valid_runs` `benchmarks/results/mtp-fair/smoke-rerun-2026-05-28/35b-a3b-4bit/flappy/rapid_mlx.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/smoke-rerun-2026-05-28/35b-a3b-4bit/flappy/ax_engine.json`
