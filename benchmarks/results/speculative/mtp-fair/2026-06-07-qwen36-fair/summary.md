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
| Qwen3.6 27B 4-bit | flappy | 3 | 56.1 | 100.0% | 62.9 | 99.7% | 62.2 | 99.7% |
| Qwen3.6 27B 4-bit | long_code | 3 | 57.9 | 99.7% | 58.6 | 99.6% | 60.9 | 99.6% |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 52.7 | 87.6% | 51.0 | 97.5% | 50.8 | 97.5% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 104.3 | 49.5% | 176.0 | 100.0% | 174.2 | 100.0% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 105.6 | 51.4% | 174.7 | 100.0% | 174.2 | 100.0% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 98.2 | 42.6% | 177.5 | 98.6% | 177.3 | 98.6% |

## Same-artifact AX survival comparison

AX direct is the same sidecar package decoded without MTP or n-gram. Use this table to decide whether AX MTP should be a default and whether AX MTP+n-gram should remain opt-in.

| Model | Engine | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Drafted | Classification |
|---|---|---|---:|---:|---:|---:|:---:|---|
| 27b-4bit | AX MTP+n-gram | AX MTP | 60.9 | 58.6 | +3.9% | -1.1% | — | keep-default |
| 35b-a3b-4bit | AX MTP+n-gram | AX MTP | 174.2 | 176.0 | -1.0% | -1.0% | — | remove-claim |

### Warnings

- 27b-4bit: no AX direct row; AX MTP survival cannot be judged against same-artifact direct decode.
- 35b-a3b-4bit: no AX direct row; AX MTP survival cannot be judged against same-artifact direct decode.

Artifacts:

- 27b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/flappy/mtplx.json`
- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/flappy/ax_engine_ngram.json`
- 27b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/long_code/mtplx.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/long_code/ax_engine_ngram.json`
- 27b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/python_modules_long/mtplx.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/python_modules_long/ax_engine.json`
- 27b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/27b-4bit/python_modules_long/ax_engine_ngram.json`
- 35b-a3b-4bit / flappy / mtplx: `ok` validation `4/4` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/flappy/mtplx.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/flappy/ax_engine_ngram.json`
- 35b-a3b-4bit / long_code / mtplx: `ok_validation_warnings` validation `4/8` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/long_code/mtplx.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/long_code/ax_engine_ngram.json`
- 35b-a3b-4bit / python_modules_long / mtplx: `ok` validation `3/3` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/python_modules_long/mtplx.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-07-qwen36-fair/35b-a3b-4bit/python_modules_long/ax_engine_ngram.json`
