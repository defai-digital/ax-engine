# Gemma 4 Assistant MTP Benchmark

Output: `benchmarks/results/gemma4-assistant-mtp/2026-06-08-gemma4-12b-assistant-mtp`

| Model | Suite | Profile | Mode | Depth | Decode tok/s | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B 4-bit | flappy | assistant_mtp_default | mtp | 1 | 60.8 | 99.2% | 99.4% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | flappy | assistant_mtp_ngram_default | mtp-ngram | 1 | 61.9 | 99.1% | 98.8% | 72.6% | 68 | 0 | 0 |
| Gemma 4 12B 4-bit | long_code | assistant_mtp_default | mtp | 1 | 62.0 | 99.0% | 99.2% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | long_code | assistant_mtp_ngram_default | mtp-ngram | 1 | 62.1 | 99.0% | 98.6% | 72.3% | 53 | 0 | 0 |
| Gemma 4 12B 4-bit | python_modules_long | assistant_mtp_default | mtp | 1 | 56.9 | 98.4% | 98.5% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | python_modules_long | assistant_mtp_ngram_default | mtp-ngram | 1 | 57.2 | 98.6% | 98.3% | 56.9% | 35 | 0 | 0 |
