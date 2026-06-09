# Gemma 4 Assistant MTP Benchmark

Output: `benchmarks/results/gemma4-assistant-mtp/2026-06-09-gemma4-12b-ffn4-mtp-phase4-focused`

| Model | Suite | Profile | Mode | Depth | Decode tok/s | Affine max-bits | 8-bit tensors | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B 4-bit-FFN | flappy | direct | direct | 2 | 36.1 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | flappy | assistant_mtp_default | mtp | 2 | 102.3 | 4 | 0 | 95.5% | 96.0% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | flappy | assistant_mtp_ngram_default | mtp-ngram | 2 | 104.0 | 4 | 0 | 95.6% | 95.6% | 79.2% | 51 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | direct | direct | 2 | 35.0 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | assistant_mtp_default | mtp | 2 | 99.4 | 4 | 0 | 95.2% | 95.6% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | assistant_mtp_ngram_default | mtp-ngram | 2 | 101.0 | 4 | 0 | 94.5% | 93.9% | 71.7% | 39 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | direct | direct | 2 | 35.9 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | assistant_mtp_default | mtp | 2 | 105.0 | 4 | 0 | 93.5% | 93.8% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | assistant_mtp_ngram_default | mtp-ngram | 2 | 103.5 | 4 | 0 | 93.3% | 93.1% | 81.4% | 73 | 0 | 0 |

## Same-artifact survival comparison

All compared profiles share the direct-baseline target artifact.

| Model | Profile | Mode | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Parity | Drafted | Classification |
|---|---|---|---|---:|---:|---:|---:|:---:|:---:|---|
| 12b-4bit-ffn4 | assistant_mtp_default | mtp | direct | 102.3 | 35.9 | +184.6% | +183.4% | yes | yes | keep-default |
| 12b-4bit-ffn4 | assistant_mtp_ngram_default | mtp-ngram | direct | 103.5 | 35.9 | +187.9% | +187.9% | yes | yes | keep-default |
| 12b-4bit-ffn4 | assistant_mtp_ngram_default | mtp-ngram | assistant_mtp_default | 103.5 | 102.3 | +1.2% | -1.5% | — | — | keep-opt-in |
