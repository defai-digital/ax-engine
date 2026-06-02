# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['mtplx', 'lightning_mlx', 'lightning_mtp_ngram', 'ax_engine_ngram']`
- suites: `['flappy', 'long_code', 'python_modules_long']`
- depth_policy: `fair-shared`
- mode: `sampled`
- max_tokens: `1000`
- repetitions: `5`

Fairness rules:

- standard Qwen source MTP shards plus mlx-community 4-bit base only
- Youssofal MTPLX bundles are excluded
- same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown

| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | Light. MTP tok/s | Light. MTP accept | Light. ngram+MTP tok/s | Light. ngram+MTP accept | AX MTP+n-gram tok/s | AX MTP+n-gram accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 3 | 58.8 | 100.0% | 70.2 | 100.0% | 67.2 | 100.0% | 62.5 | 80.2% |
| Qwen3.6 27B 4-bit | long_code | 3 | 58.4 | 99.7% | 70.4 | 100.0% | 72.7 | 100.0% | 70.6 | 90.0% |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 54.6 | 87.6% | 68.3 | 100.0% | 69.5 | 100.0% | 60.5 | 77.7% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 105.0 | 49.1% | 188.1 | 100.0% | 203.5 | 100.0% | 257.9 | 88.8% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 103.8 | 51.3% | 92.0 | 100.0% | 159.8 | 100.0% | 272.0 | 92.0% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 100.0 | 42.8% | 187.4 | 100.0% | 207.8 | 100.0% | 191.6 | 83.5% |

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/flappy/lightning_mlx.json`
- 27b-4bit / flappy / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/flappy/lightning_mtp_ngram.json`
- 27b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/flappy/ax_engine_ngram.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/long_code/lightning_mlx.json`
- 27b-4bit / long_code / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/long_code/lightning_mtp_ngram.json`
- 27b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/long_code/ax_engine_ngram.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/python_modules_long/lightning_mlx.json`
- 27b-4bit / python_modules_long / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/python_modules_long/lightning_mtp_ngram.json`
- 27b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/27b-4bit/python_modules_long/ax_engine_ngram.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/flappy/lightning_mlx.json`
- 35b-a3b-4bit / flappy / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/flappy/lightning_mtp_ngram.json`
- 35b-a3b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/flappy/ax_engine_ngram.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/long_code/lightning_mlx.json`
- 35b-a3b-4bit / long_code / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/long_code/lightning_mtp_ngram.json`
- 35b-a3b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/long_code/ax_engine_ngram.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / lightning_mlx: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/python_modules_long/lightning_mlx.json`
- 35b-a3b-4bit / python_modules_long / lightning_mtp_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/python_modules_long/lightning_mtp_ngram.json`
- 35b-a3b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-01-qwen36-fair-ax-rerun4/35b-a3b-4bit/python_modules_long/ax_engine_ngram.json`
