# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['mtplx', 'lightning_mlx', 'lightning_mtp_ngram', 'ax_engine', 'ax_engine_ngram']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | Light. MTP tok/s | Light. MTP accept | Light. ngram+MTP tok/s | Light. ngram+MTP accept | AX MTP tok/s | AX MTP accept | AX MTP+n-gram tok/s | AX MTP+n-gram accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 3 | 58.8 | 100.0% | 70.2 | 100.0% | 67.2 | 100.0% | 65.0 | 98.2% | 65.0 | 86.5% |
| Qwen3.6 27B 4-bit | long_code | 3 | 58.4 | 99.7% | 70.4 | 100.0% | 72.7 | 100.0% | 63.1 | 94.7% | 76.5 | 96.8% |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 54.6 | 87.6% | 68.3 | 100.0% | 69.5 | 100.0% | 54.9 | 80.4% | 70.3 | 89.5% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 105.0 | 49.1% | 188.1 | 100.0% | 203.5 | 100.0% | 183.9 | 100.0% | 318.7 | 96.2% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 103.8 | 51.3% | 92.0 | 100.0% | 159.8 | 100.0% | 182.6 | 99.7% | 292.7 | 95.5% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 100.0 | 42.8% | 187.4 | 100.0% | 207.8 | 100.0% | 179.9 | 94.0% | 184.7 | 75.8% |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/flappy/lightning_mlx.json`
- 27b-4bit / flappy / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/flappy/lightning_mtp_ngram.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/flappy/ax_engine_ngram.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/long_code/lightning_mlx.json`
- 27b-4bit / long_code / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/long_code/lightning_mtp_ngram.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/long_code/ax_engine_ngram.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/python_modules_long/lightning_mlx.json`
- 27b-4bit / python_modules_long / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/python_modules_long/lightning_mtp_ngram.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/python_modules_long/ax_engine.json`
- 27b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/python_modules_long/ax_engine_ngram.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/flappy/lightning_mlx.json`
- 35b-a3b-4bit / flappy / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/flappy/lightning_mtp_ngram.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/flappy/ax_engine_ngram.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/long_code/lightning_mlx.json`
- 35b-a3b-4bit / long_code / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/long_code/lightning_mtp_ngram.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/long_code/ax_engine_ngram.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/python_modules_long/lightning_mlx.json`
- 35b-a3b-4bit / python_modules_long / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/python_modules_long/lightning_mtp_ngram.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/python_modules_long/ax_engine_ngram.json`
