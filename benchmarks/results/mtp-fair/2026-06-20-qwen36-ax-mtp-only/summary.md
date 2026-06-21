# Qwen3.6 MTP Fair Benchmark

Contract:

- models: `['27b-4bit', '35b-a3b-4bit']`
- engines: `['ax_engine', 'ax_engine_ngram']`
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

| Model | Suite | Depth | AX MTP tok/s | AX MTP accept | AX MTP+n-gram tok/s | AX MTP+n-gram accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 3 | 61.4 | 99.7% | 61.6 | 99.7% |
| Qwen3.6 27B 4-bit | long_code | 3 | 60.5 | 99.6% | 61.0 | 99.6% |
| Qwen3.6 27B 4-bit | python_modules_long | 3 | 52.0 | 97.8% | 51.6 | 97.8% |
| Qwen3.6 35B-A3B 4-bit | flappy | 1 | 169.0 | 100.0% | 168.8 | 100.0% |
| Qwen3.6 35B-A3B 4-bit | long_code | 1 | 164.7 | 99.9% | 166.8 | 99.9% |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1 | 166.7 | 97.9% | 163.3 | 97.9% |

## Same-artifact AX survival comparison

AX direct is the same sidecar package decoded without MTP or n-gram. Use this table to decide whether AX MTP should be a default and whether AX MTP+n-gram should remain opt-in.

| Model | Engine | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Drafted | Classification |
|---|---|---|---:|---:|---:|---:|:---:|---|
| 27b-4bit | AX MTP+n-gram | AX MTP | 61.0 | 60.5 | +0.8% | -0.7% | — | keep-opt-in |
| 35b-a3b-4bit | AX MTP+n-gram | AX MTP | 166.8 | 166.7 | +0.1% | -2.0% | — | keep-opt-in |

### Warnings

- 27b-4bit: no AX direct row; AX MTP survival cannot be judged against same-artifact direct decode.
- 35b-a3b-4bit: no AX direct row; AX MTP survival cannot be judged against same-artifact direct decode.

Artifacts:

- 27b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/27b-4bit/flappy/ax_engine.json`
- 27b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/27b-4bit/flappy/ax_engine_ngram.json`
- 27b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/27b-4bit/long_code/ax_engine.json`
- 27b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/27b-4bit/long_code/ax_engine_ngram.json`
- 27b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/27b-4bit/python_modules_long/ax_engine.json`
- 27b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/27b-4bit/python_modules_long/ax_engine_ngram.json`
- 35b-a3b-4bit / flappy / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/35b-a3b-4bit/flappy/ax_engine.json`
- 35b-a3b-4bit / flappy / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/35b-a3b-4bit/flappy/ax_engine_ngram.json`
- 35b-a3b-4bit / long_code / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/35b-a3b-4bit/long_code/ax_engine.json`
- 35b-a3b-4bit / long_code / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/35b-a3b-4bit/long_code/ax_engine_ngram.json`
- 35b-a3b-4bit / python_modules_long / ax_engine: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/35b-a3b-4bit/python_modules_long/ax_engine.json`
- 35b-a3b-4bit / python_modules_long / ax_engine_ngram: `ok` `benchmarks/results/mtp-fair/2026-06-20-qwen36-ax-mtp-only/35b-a3b-4bit/python_modules_long/ax_engine_ngram.json`
