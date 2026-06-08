# Gemma 4 Assistant MTP Benchmark

Output: `benchmarks/results/gemma4-assistant-mtp/2026-06-08-gemma4-12b-assistant-mtp`

| Model | Suite | Profile | Mode | Depth | Decode tok/s | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B 4-bit | flappy | assistant_mtp_default | mtp | 2 | 64.3 | 98.7% | 98.9% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | flappy | assistant_mtp_ngram_default | mtp-ngram | 2 | 61.9 | 98.7% | 98.7% | 86.3% | 106 | 0 | 0 |
| Gemma 4 12B 4-bit | long_code | assistant_mtp_default | mtp | 2 | 67.1 | 98.7% | 99.0% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | long_code | assistant_mtp_ngram_default | mtp-ngram | 2 | 68.8 | 98.6% | 98.4% | 65.9% | 46 | 0 | 0 |
| Gemma 4 12B 4-bit | python_modules_long | assistant_mtp_default | mtp | 2 | 62.7 | 98.0% | 98.2% | n/a | 0 | 0 | 0 |
| Gemma 4 12B 4-bit | python_modules_long | assistant_mtp_ngram_default | mtp-ngram | 2 | 62.5 | 97.6% | 97.5% | 82.0% | 67 | 0 | 0 |
