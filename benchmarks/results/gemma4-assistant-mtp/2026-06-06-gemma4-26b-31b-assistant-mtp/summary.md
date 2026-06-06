# Gemma 4 Assistant MTP Benchmark

Output: `benchmarks/results/gemma4-assistant-mtp/2026-06-06-gemma4-26b-31b-assistant-mtp`

| Model | Suite | Mode | Depth | Decode tok/s | Assistant accept | MTP accept | n-gram accept | n-gram hits |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B A4B 4-bit | flappy | mtp | 1 | 123.3 | 99.3% | 99.5% | n/a | 0 |
| Gemma 4 26B A4B 4-bit | flappy | mtp-ngram | 1 | 122.9 | 99.4% | 98.8% | 33.3% | 40 |
| Gemma 4 26B A4B 4-bit | long_code | mtp | 1 | 119.3 | 99.1% | 99.3% | n/a | 0 |
| Gemma 4 26B A4B 4-bit | long_code | mtp-ngram | 1 | 117.6 | 99.1% | 98.3% | 16.0% | 44 |
| Gemma 4 26B A4B 4-bit | python_modules_long | mtp | 1 | 120.0 | 98.5% | 98.6% | n/a | 0 |
| Gemma 4 26B A4B 4-bit | python_modules_long | mtp-ngram | 1 | 120.5 | 98.7% | 97.8% | 13.9% | 26 |
| Gemma 4 31B 4-bit | flappy | mtp | 1 | 37.3 | 99.3% | 99.5% | n/a | 0 |
| Gemma 4 31B 4-bit | flappy | mtp-ngram | 1 | 37.0 | 99.3% | 98.8% | 38.3% | 40 |
| Gemma 4 31B 4-bit | long_code | mtp | 1 | 35.9 | 99.2% | 99.4% | n/a | 0 |
| Gemma 4 31B 4-bit | long_code | mtp-ngram | 1 | 35.2 | 99.4% | 98.6% | 14.4% | 43 |
| Gemma 4 31B 4-bit | python_modules_long | mtp | 1 | 35.9 | 98.4% | 98.6% | n/a | 0 |
| Gemma 4 31B 4-bit | python_modules_long | mtp-ngram | 1 | 35.8 | 98.4% | 97.5% | 0.0% | 25 |
