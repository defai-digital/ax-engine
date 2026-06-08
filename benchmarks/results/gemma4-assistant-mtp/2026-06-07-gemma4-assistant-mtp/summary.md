# Gemma 4 Assistant MTP Benchmark

Output: `/Users/akiralam/code/ax-engine_v5/benchmarks/results/gemma4-assistant-mtp/2026-06-07-gemma4-assistant-mtp`

| Model | Suite | Mode | Depth | Decode tok/s | Assistant accept | MTP accept | n-gram accept | n-gram hits |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B A4B 4-bit | flappy | mtp | 1 | 126.1 | 99.3% | 99.5% | n/a | 0 |
| Gemma 4 26B A4B 4-bit | flappy | mtp-ngram | 1 | 129.1 | 99.5% | 99.2% | 78.8% | 80 |
| Gemma 4 26B A4B 4-bit | long_code | mtp | 1 | 125.5 | 99.1% | 99.3% | n/a | 0 |
| Gemma 4 26B A4B 4-bit | long_code | mtp-ngram | 1 | 127.1 | 99.1% | 98.6% | 79.2% | 75 |
| Gemma 4 26B A4B 4-bit | python_modules_long | mtp | 1 | 124.0 | 98.5% | 98.6% | n/a | 0 |
| Gemma 4 26B A4B 4-bit | python_modules_long | mtp-ngram | 1 | 124.1 | 98.6% | 98.3% | 54.9% | 36 |
| Gemma 4 31B 4-bit | flappy | mtp | 1 | 37.9 | 99.3% | 99.5% | n/a | 0 |
| Gemma 4 31B 4-bit | flappy | mtp-ngram | 1 | 38.2 | 99.2% | 98.8% | 68.6% | 75 |
| Gemma 4 31B 4-bit | long_code | mtp | 1 | 37.8 | 99.2% | 99.4% | n/a | 0 |
| Gemma 4 31B 4-bit | long_code | mtp-ngram | 1 | 38.9 | 99.1% | 98.6% | 79.6% | 83 |
| Gemma 4 31B 4-bit | python_modules_long | mtp | 1 | 37.4 | 98.4% | 98.6% | n/a | 0 |
| Gemma 4 31B 4-bit | python_modules_long | mtp-ngram | 1 | 37.3 | 98.6% | 98.4% | 71.7% | 48 |
