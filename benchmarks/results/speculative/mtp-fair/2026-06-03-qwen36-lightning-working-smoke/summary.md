# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['lightning_mlx', 'lightning_mtp_ngram']`
- suites: `['flappy']`
- depth_policy: `native`
- mode: `sampled`
- max_tokens: `512`
- repetitions: `1`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | Light. MTP tok/s | Light. MTP accept | Light. ngram+MTP tok/s | Light. ngram+MTP accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 3 | - | - | 8.1 | 100.0% | 9.4 | 100.0% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | - | - | 19.0 | 100.0% | 33.2 | 100.0% |

Artifacts:

- 27b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-working-smoke/27b-4bit/flappy/lightning_mlx.json`
- 27b-4bit / flappy / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-working-smoke/27b-4bit/flappy/lightning_mtp_ngram.json`
- 35b-a3b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-working-smoke/35b-a3b-4bit/flappy/lightning_mlx.json`
- 35b-a3b-4bit / flappy / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-03-qwen36-lightning-working-smoke/35b-a3b-4bit/flappy/lightning_mtp_ngram.json`
