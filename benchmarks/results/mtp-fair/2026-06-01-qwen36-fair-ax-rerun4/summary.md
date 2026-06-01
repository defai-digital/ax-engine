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
| Qwen3.6 27B 4-bit | flappy | 3 | 51.5 | 100.0% | 49.5 | 96.5% | 52.4 | 85.4% | 62.5 | 80.2% |
| Qwen3.6 27B 4-bit | long_code | 3 | 53.5 | 99.7% | 51.7 | 93.3% | 54.9 | 87.6% | 70.6 | 90.0% |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 51.6 | 87.6% | 46.9 | 76.5% | 45.0 | 72.2% | 60.5 | 77.7% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 107.3 | 50.8% | 147.9 | 99.0% | 173.9 | 91.0% | 257.9 | 88.8% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 106.4 | 50.5% | 149.0 | 98.5% | 194.9 | 92.1% | 272.0 | 92.0% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 102.7 | 42.6% | 148.8 | 97.2% | 136.1 | 91.8% | 191.6 | 83.5% |

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
