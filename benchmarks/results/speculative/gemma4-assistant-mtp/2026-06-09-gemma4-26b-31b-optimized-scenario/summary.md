# Gemma 4 Assistant MTP Benchmark

Output: `benchmarks/results/gemma4-assistant-mtp/2026-06-09-gemma4-26b-31b-optimized-scenario`

| Model | Suite | Profile | Mode | Depth | Decode tok/s | Affine max-bits | 8-bit tensors | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B A4B 4-bit | flappy | assistant_mtp_default | mtp | 1 | 126.1 | 8 | 120 | 99.3% | 99.5% | n/a | 0 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | flappy | assistant_mtp_ngram_default | mtp-ngram | 1 | 129.1 | 8 | 120 | 99.5% | 99.2% | 78.8% | 80 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | long_code | assistant_mtp_default | mtp | 1 | 125.5 | 8 | 120 | 99.1% | 99.3% | n/a | 0 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | long_code | assistant_mtp_ngram_default | mtp-ngram | 1 | 127.1 | 8 | 120 | 99.1% | 98.6% | 79.2% | 75 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | python_modules_long | assistant_mtp_default | mtp | 1 | 124.0 | 8 | 120 | 98.5% | 98.6% | n/a | 0 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | python_modules_long | assistant_mtp_ngram_default | mtp-ngram | 1 | 124.1 | 8 | 120 | 98.6% | 98.3% | 54.9% | 36 | 0 | 0 |
| Gemma 4 31B 4-bit | flappy | assistant_mtp_default | mtp | 1 | 37.9 | 4 | 0 | 99.3% | 99.5% | n/a | 0 | 0 | 0 |
| Gemma 4 31B 4-bit | flappy | assistant_mtp_ngram_default | mtp-ngram | 1 | 38.2 | 4 | 0 | 99.2% | 98.8% | 68.6% | 75 | 0 | 0 |
| Gemma 4 31B 4-bit | long_code | assistant_mtp_default | mtp | 1 | 37.8 | 4 | 0 | 99.2% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 31B 4-bit | long_code | assistant_mtp_ngram_default | mtp-ngram | 1 | 38.9 | 4 | 0 | 99.1% | 98.6% | 79.6% | 83 | 0 | 0 |
| Gemma 4 31B 4-bit | python_modules_long | assistant_mtp_default | mtp | 1 | 37.4 | 4 | 0 | 98.4% | 98.6% | n/a | 0 | 0 | 0 |
| Gemma 4 31B 4-bit | python_modules_long | assistant_mtp_ngram_default | mtp-ngram | 1 | 37.3 | 4 | 0 | 98.6% | 98.4% | 71.7% | 48 | 0 | 0 |

## Assistant-MTP relative comparison

No direct-decode rows are present in this artifact set, so this table is scoped to MTP+n-gram versus pure assistant-MTP.

| Model | Profile | Mode | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Parity | Drafted | Classification |
|---|---|---|---|---:|---:|---:|---:|:---:|:---:|---|
| 26b-a4b-4bit | assistant_mtp_ngram_default | mtp-ngram | assistant_mtp_default | 127.1 | 125.5 | +1.3% | +0.1% | — | — | keep-opt-in |
| 31b-4bit | assistant_mtp_ngram_default | mtp-ngram | assistant_mtp_default | 38.2 | 37.8 | +1.0% | -0.0% | — | — | keep-opt-in |

### Warnings

- 26b-a4b-4bit: no direct-decode row; assistant-MTP cannot be judged against same-artifact direct decode (PRD R1/R3).
- 31b-4bit: no direct-decode row; assistant-MTP cannot be judged against same-artifact direct decode (PRD R1/R3).
