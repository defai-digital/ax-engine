# 6-bit MTP AX acceleration summary

This artifact summarizes exact AX MTP acceleration.

The acceleration ratio is `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 23.2 tok/s | 62.5 tok/s | 2.69x | 525.6 tok/s | 612 ms | 99.3% |
| `qwen3.6-27b-6bit` | `long_code` | 23.3 tok/s | 52.6 tok/s | 2.26x | 672.5 tok/s | 1067 ms | 98.6% |
| `qwen3.6-27b-6bit` | `python_modules_long` | 22.8 tok/s | 42.7 tok/s | 1.87x | 562.6 tok/s | 619 ms | 96.3% |
| `qwen3.6-35b-a3b` | `flappy` | 98.8 tok/s | 141.3 tok/s | 1.43x | 611.4 tok/s | 529 ms | 99.9% |
| `qwen3.6-35b-a3b` | `long_code` | 99.8 tok/s | 143.3 tok/s | 1.44x | 1204.1 tok/s | 596 ms | 98.5% |
| `qwen3.6-35b-a3b` | `python_modules_long` | 101.9 tok/s | 147.9 tok/s | 1.45x | 656.5 tok/s | 525 ms | 98.4% |
| `gemma-4-12b` | `flappy` | 38.7 tok/s | 99.4 tok/s | 2.57x | 1004.4 tok/s | 357 ms | 99.9% |
| `gemma-4-12b` | `long_code` | 38.8 tok/s | 97.6 tok/s | 2.51x | 1364.2 tok/s | 600 ms | 100.0% |
| `gemma-4-12b` | `python_modules_long` | 38.8 tok/s | 75.9 tok/s | 1.96x | 876.1 tok/s | 431 ms | 97.8% |
| `gemma-4-26b` | `flappy` | 81.6 tok/s | 146.5 tok/s | 1.79x | 1318.5 tok/s | 264 ms | 99.9% |
| `gemma-4-26b` | `long_code` | 89.6 tok/s | 147.9 tok/s | 1.65x | 2258.3 tok/s | 362 ms | 100.0% |
| `gemma-4-26b` | `python_modules_long` | 88.6 tok/s | 131.7 tok/s | 1.49x | 1369.2 tok/s | 273 ms | 98.4% |
| `gemma-4-31b` | `flappy` | 17.8 tok/s | 46.4 tok/s | 2.61x | 452.4 tok/s | 770 ms | 99.9% |
| `gemma-4-31b` | `long_code` | 17.9 tok/s | 45.4 tok/s | 2.54x | 573.9 tok/s | 1426 ms | 100.0% |
| `gemma-4-31b` | `python_modules_long` | 18.1 tok/s | 40.0 tok/s | 2.21x | 443.7 tok/s | 822 ms | 98.1% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
