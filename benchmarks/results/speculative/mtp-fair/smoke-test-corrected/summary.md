# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit']`
- engines: `['mtplx', 'rapid_mlx', 'ax_engine']`
- suites: `['flappy']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `200`
- repetitions: `1`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown
- tri-engine Rapid comparison uses shared depth 1 unless --depth overrides it
- Rapid-MLX server path exposes throughput but not accepted/drafted token telemetry

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | Rapid tok/s | Rapid accept | AX tok/s | AX accept | AX/MTPLX | AX/Rapid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B 4-bit | flappy | 1 | 47.8 | 100.0% | 31.6 | - | 48.2 | 74.5% | 1.009 | 1.527 |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/smoke-test-corrected/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / rapid_mlx: `ok` `benchmarks/results/mtp-fair/smoke-test-corrected/27b-4bit/flappy/rapid_mlx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/smoke-test-corrected/27b-4bit/flappy/ax_engine.json`
