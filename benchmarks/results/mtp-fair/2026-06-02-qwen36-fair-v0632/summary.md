# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['35b-a3b-4bit']`
- engines: `['mtplx', 'ax_engine', 'ax_engine_ngram']`
- suites: `['python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | AX MTP tok/s | AX MTP accept | AX MTP+n-gram tok/s | AX MTP+n-gram accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 96.3 | 41.5% | 170.9 | 93.6% | 167.5 | 90.7% |

Artifacts:

- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-02-qwen36-fair-v0632/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-02-qwen36-fair-v0632/35b-a3b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-02-qwen36-fair-v0632/35b-a3b-4bit/python_modules_long/ax_engine_ngram.json`
