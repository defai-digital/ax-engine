# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['lightning_mlx', 'lightning_mtp_ngram']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `native`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | Light. MTP tok/s | Light. MTP accept | Light. ngram+MTP tok/s | Light. ngram+MTP accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 3 | - | - | - | - | - | - |
| Qwen3.6 27B 4-bit | long_code | 3 | - | - | - | - | - | - |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | - | - | - | - | - | - |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | - | - | - | - | - | - |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | - | - | - | - | - | - |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | - | - | - | - | - | - |

Artifacts:

- 27b-4bit / flappy / lightning_mlx: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/27b-4bit/flappy/lightning_mlx.json`
- 27b-4bit / flappy / lightning_mtp_ngram: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/27b-4bit/flappy/lightning_mtp_ngram.json`
- 27b-4bit / long_code / lightning_mlx: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/27b-4bit/long_code/lightning_mlx.json`
- 27b-4bit / long_code / lightning_mtp_ngram: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/27b-4bit/long_code/lightning_mtp_ngram.json`
- 27b-4bit / python_modules_long / lightning_mlx: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/27b-4bit/python_modules_long/lightning_mlx.json`
- 27b-4bit / python_modules_long / lightning_mtp_ngram: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/27b-4bit/python_modules_long/lightning_mtp_ngram.json`
- 35b-a3b-4bit / flappy / lightning_mlx: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/35b-a3b-4bit/flappy/lightning_mlx.json`
- 35b-a3b-4bit / flappy / lightning_mtp_ngram: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/35b-a3b-4bit/flappy/lightning_mtp_ngram.json`
- 35b-a3b-4bit / long_code / lightning_mlx: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/35b-a3b-4bit/long_code/lightning_mlx.json`
- 35b-a3b-4bit / long_code / lightning_mtp_ngram: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/35b-a3b-4bit/long_code/lightning_mtp_ngram.json`
- 35b-a3b-4bit / python_modules_long / lightning_mlx: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/35b-a3b-4bit/python_modules_long/lightning_mlx.json`
- 35b-a3b-4bit / python_modules_long / lightning_mtp_ngram: `error` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-only-fresh/35b-a3b-4bit/python_modules_long/lightning_mtp_ngram.json`
