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
| Qwen3.6 27B 4-bit | flappy | 3 | 56.1 | 100.0% | 60.6 | 99.9% | 57.4 | 99.1% |
| Qwen3.6 27B 4-bit | long_code | 3 | 57.9 | 99.7% | 54.9 | 99.9% | 59.3 | 99.1% |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 52.7 | 87.6% | 47.8 | 97.6% | 50.2 | 97.2% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 104.3 | 49.5% | 180.6 | 100.0% | 182.3 | 99.6% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 105.6 | 51.4% | 179.1 | 100.0% | 224.2 | 99.8% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 98.2 | 42.6% | 182.4 | 99.3% | 169.4 | 97.6% |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / ax_engine: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / flappy / ax_engine_ngram: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/flappy/ax_engine_ngram.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / ax_engine: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / long_code / ax_engine_ngram: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/long_code/ax_engine_ngram.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/python_modules_long/ax_engine.json`
- 27b-4bit / python_modules_long / ax_engine_ngram: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/python_modules_long/ax_engine_ngram.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / flappy / ax_engine_ngram: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/flappy/ax_engine_ngram.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / long_code / ax_engine_ngram: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/long_code/ax_engine_ngram.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / ax_engine_ngram: `ok` `/Users/akiralam/code/ax-engine_v5/benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/python_modules_long/ax_engine_ngram.json`
