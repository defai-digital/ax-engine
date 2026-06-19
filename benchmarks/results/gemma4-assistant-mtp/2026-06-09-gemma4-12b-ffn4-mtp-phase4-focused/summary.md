# Gemma 4 Assistant MTP Benchmark

Output: `benchmarks/results/gemma4-assistant-mtp/2026-06-09-gemma4-12b-ffn4-mtp-phase4-focused`

| Model | Suite | Profile | Mode | Depth | Decode tok/s | Affine max-bits | 8-bit tensors | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B 4-bit-FFN | flappy | direct | direct | 2 | 35.5 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | flappy | assistant_mtp_default | mtp | 2 | 100.7 | 4 | 0 | 98.7% | 98.9% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | flappy | assistant_mtp_ngram_default | mtp-ngram | 2 | 100.6 | 4 | 0 | 98.7% | 98.9% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | direct | direct | 2 | 35.8 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | assistant_mtp_default | mtp | 2 | 100.3 | 4 | 0 | 98.8% | 99.1% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | long_code | assistant_mtp_ngram_default | mtp-ngram | 2 | 100.3 | 4 | 0 | 98.8% | 99.1% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | direct | direct | 2 | 35.4 | 4 | 0 | n/a | n/a | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | assistant_mtp_default | mtp | 2 | 90.1 | 4 | 0 | 97.2% | 97.4% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit-FFN | python_modules_long | assistant_mtp_ngram_default | mtp-ngram | 2 | 90.2 | 4 | 0 | 97.2% | 97.4% | n/a | 0 | 0 | 0 |

## Optimized scenario

| Model | Profile | Mode | Decode tok/s | Direct tok/s | Δ vs direct | Worst suite Δ | Classification |
|---|---|---|---:|---:|---:|---:|---|
| 12b-4bit-ffn4 | assistant_mtp_default | mtp | 100.3 | 35.5 | +182.5% | +154.7% | keep-default |

## Same-artifact survival comparison

Direct-baseline rows share the same target artifact; assistant-MTP+n-gram rows use pure assistant-MTP as their baseline.

| Model | Profile | Mode | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Parity | Drafted | Classification |
|---|---|---|---|---:|---:|---:|---:|:---:|:---:|---|
| 12b-4bit-ffn4 | assistant_mtp_default | mtp | direct | 100.3 | 35.5 | +182.5% | +154.7% | yes | yes | keep-default |
| 12b-4bit-ffn4 | assistant_mtp_ngram_default | mtp-ngram | direct | 100.3 | 35.5 | +182.4% | +154.8% | yes | yes | keep-default |
| 12b-4bit-ffn4 | assistant_mtp_ngram_default | mtp-ngram | assistant_mtp_default | 100.3 | 100.3 | -0.0% | -0.1% | — | — | remove-claim |
