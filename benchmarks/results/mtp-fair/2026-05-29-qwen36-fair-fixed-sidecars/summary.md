# Qwen3.6 MTP Fair Benchmark

> **NOTE — MTPLX 27B data in this run is invalid and must not be used.**
> The benchmark ran MTPLX with the `sustained` profile, which applies
> `MTPLX_PREFILL_OMLX_EXTERNAL=1`. That flag activates a Youssofal-bundle-specific
> external prefill path; with a standard mlx-community 4-bit base it produces
> incoherent hidden states, garbage output tokens, and a spurious 1–3% accept rate.
> The harness has been fixed (`stable` profile, no omlx external prefill).
> Regenerate the sidecar and re-run to obtain valid MTPLX 27B numbers.

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['mtplx', 'rapid_mlx', 'ax_engine']`
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
| Qwen3.6 27B 4-bit | flappy | 1 | 21.0 | 1.8% | 25.3 | - | 39.4 | 87.5% | 1.875 | 1.561 |
| Qwen3.6 27B 4-bit | long_code | 1 | 20.7 | 1.0% | 25.2 | - | 42.0 | 91.9% | 2.025 | 1.663 |
| Qwen3.6 27B 4-bit | python_modules_long | 1 | 23.8 | 3.0% | 28.0 | - | 38.3 | 68.5% | 1.606 | 1.365 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | - | - | 66.5 | - | 144.7 | 85.0% | - | 2.175 |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | - | - | 70.2 | - | 145.2 | 89.6% | - | 2.070 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | - | - | 71.4 | - | 119.7 | 66.2% | - | 1.676 |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / rapid_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/27b-4bit/flappy/rapid_mlx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `6/8` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / rapid_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/27b-4bit/long_code/rapid_mlx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / rapid_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/27b-4bit/python_modules_long/rapid_mlx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/27b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / flappy / mtplx: `error` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / rapid_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/35b-a3b-4bit/flappy/rapid_mlx.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / long_code / mtplx: `error` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / rapid_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/35b-a3b-4bit/long_code/rapid_mlx.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `error` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / rapid_mlx: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/35b-a3b-4bit/python_modules_long/rapid_mlx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-05-29-qwen36-fair-fixed-sidecars/35b-a3b-4bit/python_modules_long/ax_engine.json`
