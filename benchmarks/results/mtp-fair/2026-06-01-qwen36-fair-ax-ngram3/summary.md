# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['35b-a3b-4bit']`
- engines: `['ax_engine_ngram']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | AX MTP+n-gram tok/s | AX MTP+n-gram accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | - | - | 259.4 | 88.8% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | - | - | 270.3 | 92.0% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | - | - | 195.0 | 83.5% |

Artifacts:

- 35b-a3b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-ngram3/35b-a3b-4bit/flappy/ax_engine_ngram.json`
- 35b-a3b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-ngram3/35b-a3b-4bit/long_code/ax_engine_ngram.json`
- 35b-a3b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-ngram3/35b-a3b-4bit/python_modules_long/ax_engine_ngram.json`
