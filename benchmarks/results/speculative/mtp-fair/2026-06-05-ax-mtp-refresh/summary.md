# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['mtplx', 'ax_engine', 'ax_engine_ngram']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal/samuelfaj optimized bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown
- fixed-depth rows and tuned-best-of rows are separate benchmark contracts

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | AX MTP tok/s | AX MTP accept | AX MTP+n-gram tok/s | AX MTP+n-gram accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 3 | 59.6 | 100.0% | 62.9 | 97.8% | 61.3 | 96.5% |
| Qwen3.6 27B 4-bit | long_code | 3 | 59.3 | 99.7% | 61.5 | 96.7% | 67.0 | 95.3% |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 54.4 | 87.6% | 47.4 | 82.2% | 47.0 | 81.0% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 105.3 | 48.5% | 180.4 | 99.9% | 181.8 | 99.1% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 106.0 | 51.7% | 177.9 | 99.4% | 183.8 | 98.1% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 102.1 | 43.4% | 170.3 | 94.2% | 170.7 | 94.2% |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/27b-4bit/flappy/ax_engine_ngram.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/27b-4bit/long_code/ax_engine_ngram.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/27b-4bit/python_modules_long/ax_engine.json`
- 27b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/27b-4bit/python_modules_long/ax_engine_ngram.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/35b-a3b-4bit/flappy/ax_engine_ngram.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/35b-a3b-4bit/long_code/ax_engine_ngram.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/35b-a3b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-refresh/35b-a3b-4bit/python_modules_long/ax_engine_ngram.json`
