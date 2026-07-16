# 6-bit MTP AX acceleration summary

This artifact summarizes exact AX MTP acceleration.

The acceleration ratio is `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 22.7 tok/s | 61.2 tok/s | 2.69x | 546.0 tok/s | 589 ms | 99.3% |
| `qwen3.6-27b-6bit` | `long_code` | 22.7 tok/s | 51.5 tok/s | 2.26x | 672.4 tok/s | 1067 ms | 98.6% |
| `qwen3.6-27b-6bit` | `python_modules_long` | 22.8 tok/s | 42.7 tok/s | 1.88x | 564.7 tok/s | 615 ms | 96.3% |
| `qwen3.6-35b-a3b` | `flappy` | 100.0 tok/s | 143.6 tok/s | 1.44x | 971.2 tok/s | 334 ms | 99.9% |
| `qwen3.6-35b-a3b` | `long_code` | 99.5 tok/s | 139.6 tok/s | 1.40x | 1759.1 tok/s | 408 ms | 98.5% |
| `qwen3.6-35b-a3b` | `python_modules_long` | 100.0 tok/s | 143.9 tok/s | 1.44x | 1054.9 tok/s | 327 ms | 98.4% |
| `gemma-4-12b` | `flappy` | 38.0 tok/s | 96.8 tok/s | 2.55x | 1113.0 tok/s | 313 ms | 99.9% |
| `gemma-4-12b` | `long_code` | 37.8 tok/s | 95.3 tok/s | 2.52x | 1465.8 tok/s | 558 ms | 100.0% |
| `gemma-4-12b` | `python_modules_long` | 38.1 tok/s | 76.2 tok/s | 2.00x | 1123.0 tok/s | 329 ms | 97.8% |
| `gemma-4-26b` | `flappy` | 88.5 tok/s | 146.9 tok/s | 1.66x | 1285.4 tok/s | 275 ms | 99.9% |
| `gemma-4-26b` | `long_code` | 87.6 tok/s | 144.4 tok/s | 1.65x | 2279.6 tok/s | 359 ms | 100.0% |
| `gemma-4-26b` | `python_modules_long` | 88.9 tok/s | 131.7 tok/s | 1.48x | 1358.5 tok/s | 273 ms | 98.4% |
| `gemma-4-31b` | `flappy` | 17.7 tok/s | 45.5 tok/s | 2.58x | 441.6 tok/s | 788 ms | 99.9% |
| `gemma-4-31b` | `long_code` | 18.0 tok/s | 45.4 tok/s | 2.53x | 572.0 tok/s | 1430 ms | 100.0% |
| `gemma-4-31b` | `python_modules_long` | 18.3 tok/s | 40.2 tok/s | 2.20x | 436.7 tok/s | 822 ms | 98.1% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
