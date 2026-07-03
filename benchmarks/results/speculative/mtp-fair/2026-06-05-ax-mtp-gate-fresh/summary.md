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
| Qwen3.6 27B 4-bit | flappy | 3 | 59.6 | 100.0% | 61.4 | 99.9% | 61.7 | 99.4% |
| Qwen3.6 27B 4-bit | long_code | 3 | 59.3 | 99.7% | 55.9 | 99.9% | 53.4 | 99.4% |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 54.4 | 87.6% | 47.5 | 99.3% | 47.1 | 98.9% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 105.3 | 48.5% | 181.0 | 100.0% | 182.1 | 99.2% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 106.0 | 51.7% | 178.9 | 100.0% | 185.3 | 98.3% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 102.1 | 43.4% | 183.2 | 99.6% | 183.2 | 98.7% |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/27b-4bit/flappy/ax_engine_ngram.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/27b-4bit/long_code/ax_engine_ngram.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/27b-4bit/python_modules_long/ax_engine.json`
- 27b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/27b-4bit/python_modules_long/ax_engine_ngram.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/35b-a3b-4bit/flappy/ax_engine_ngram.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/35b-a3b-4bit/long_code/ax_engine_ngram.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/35b-a3b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-05-ax-mtp-gate-fresh/35b-a3b-4bit/python_modules_long/ax_engine_ngram.json`
