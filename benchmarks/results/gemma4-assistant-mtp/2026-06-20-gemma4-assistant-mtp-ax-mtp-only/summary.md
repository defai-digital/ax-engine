# Gemma 4 Assistant MTP Benchmark

Output: `benchmarks/results/gemma4-assistant-mtp/2026-06-20-gemma4-assistant-mtp-ax-mtp-only`

| Model | Suite | Profile | Mode | Depth | Decode tok/s | Affine max-bits | 8-bit tensors | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B 4-bit-FFN | flappy | assistant_mtp_default | mtp | 2 | 96.8 | 4 | 0 | 98.7% | 98.8% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | flappy | assistant_mtp_ngram_default | mtp-ngram | 2 | 95.0 | 4 | 0 | 98.7% | 98.8% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | assistant_mtp_default | mtp | 2 | 92.3 | 4 | 0 | 99.1% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | assistant_mtp_ngram_default | mtp-ngram | 2 | 95.2 | 4 | 0 | 99.1% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | assistant_mtp_default | mtp | 2 | 82.9 | 4 | 0 | 97.5% | 97.9% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | assistant_mtp_ngram_default | mtp-ngram | 2 | 82.5 | 4 | 0 | 97.5% | 97.9% | n/a | 0 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | flappy | assistant_mtp_default | mtp | 1 | 128.8 | 8 | 120 | 99.2% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | flappy | assistant_mtp_ngram_default | mtp-ngram | 1 | 137.3 | 8 | 120 | 99.2% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | long_code | assistant_mtp_default | mtp | 1 | 136.7 | 8 | 120 | 99.0% | 99.2% | n/a | 0 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | long_code | assistant_mtp_ngram_default | mtp-ngram | 1 | 136.9 | 8 | 120 | 99.0% | 99.2% | n/a | 0 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | python_modules_long | assistant_mtp_default | mtp | 1 | 130.1 | 8 | 120 | 98.7% | 98.8% | n/a | 0 | 0 | 0 |
| Gemma 4 26B A4B 4-bit | python_modules_long | assistant_mtp_ngram_default | mtp-ngram | 1 | 125.3 | 8 | 120 | 98.7% | 98.8% | n/a | 0 | 0 | 0 |
| Gemma 4 31B 4-bit | flappy | assistant_mtp_default | mtp | 1 | 39.4 | 4 | 0 | 99.2% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 31B 4-bit | flappy | assistant_mtp_ngram_default | mtp-ngram | 1 | 39.1 | 4 | 0 | 99.2% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 31B 4-bit | long_code | assistant_mtp_default | mtp | 1 | 40.0 | 4 | 0 | 99.1% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 31B 4-bit | long_code | assistant_mtp_ngram_default | mtp-ngram | 1 | 40.4 | 4 | 0 | 99.1% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 31B 4-bit | python_modules_long | assistant_mtp_default | mtp | 1 | 37.4 | 4 | 0 | 97.3% | 97.5% | n/a | 0 | 0 | 0 |
| Gemma 4 31B 4-bit | python_modules_long | assistant_mtp_ngram_default | mtp-ngram | 1 | 37.1 | 4 | 0 | 97.3% | 97.5% | n/a | 0 | 0 | 0 |

## Assistant-MTP relative comparison

No direct-decode rows are present in this artifact set, so this table is scoped to MTP+n-gram versus pure assistant-MTP.

| Model | Profile | Mode | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Parity | Drafted | Classification |
|---|---|---|---|---:|---:|---:|---:|:---:|:---:|---|
| 12b-4bit-ffn4 | assistant_mtp_ngram_default | mtp-ngram | assistant_mtp_default | 95.0 | 92.3 | +3.0% | -1.8% | — | — | keep-opt-in |
| 26b-a4b-4bit | assistant_mtp_ngram_default | mtp-ngram | assistant_mtp_default | 136.9 | 130.1 | +5.2% | -3.7% | — | — | keep-opt-in |
| 31b-4bit | assistant_mtp_ngram_default | mtp-ngram | assistant_mtp_default | 39.1 | 39.4 | -0.7% | -0.9% | — | — | remove-claim |

### Warnings

- 12b-4bit-ffn4: no direct-decode row; assistant-MTP cannot be judged against same-artifact direct decode (PRD R1/R3).
- 26b-a4b-4bit: no direct-decode row; assistant-MTP cannot be judged against same-artifact direct decode (PRD R1/R3).
- 31b-4bit: no direct-decode row; assistant-MTP cannot be judged against same-artifact direct decode (PRD R1/R3).
