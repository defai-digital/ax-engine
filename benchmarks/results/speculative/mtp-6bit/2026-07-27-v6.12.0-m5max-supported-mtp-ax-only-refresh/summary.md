# 6-bit MTP AX acceleration summary

This artifact summarizes exact AX MTP acceleration.

The acceleration ratio is `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 23.5 tok/s | 57.0 tok/s | 2.43x | 395.9 tok/s | 812 ms | 99.8% |
| `qwen3.6-27b-6bit` | `long_code` | 23.5 tok/s | 41.9 tok/s | 1.78x | 520.0 tok/s | 1380 ms | 98.9% |
| `qwen3.6-27b-6bit` | `python_modules_long` | 23.5 tok/s | 31.6 tok/s | 1.34x | 405.9 tok/s | 856 ms | 98.4% |
| `qwen3.6-35b-a3b` | `flappy` | 102.4 tok/s | 142.9 tok/s | 1.40x | 774.9 tok/s | 416 ms | 100.0% |
| `qwen3.6-35b-a3b` | `long_code` | 100.9 tok/s | 133.6 tok/s | 1.32x | 1273.6 tok/s | 563 ms | 98.8% |
| `qwen3.6-35b-a3b` | `python_modules_long` | 101.9 tok/s | 126.3 tok/s | 1.24x | 804.7 tok/s | 434 ms | 99.7% |
| `gemma-4-12b` | `flappy` | 37.8 tok/s | 96.0 tok/s | 2.54x | 1031.0 tok/s | 338 ms | 99.9% |
| `gemma-4-12b` | `long_code` | 38.5 tok/s | 96.4 tok/s | 2.50x | 1240.1 tok/s | 660 ms | 100.0% |
| `gemma-4-12b` | `python_modules_long` | 38.8 tok/s | 77.5 tok/s | 2.00x | 1029.4 tok/s | 358 ms | 98.1% |
| `gemma-4-26b` | `flappy` | 92.5 tok/s | 150.5 tok/s | 1.63x | 1153.4 tok/s | 304 ms | 99.9% |
| `gemma-4-26b` | `long_code` | 91.2 tok/s | 147.8 tok/s | 1.62x | 1816.4 tok/s | 450 ms | 100.0% |
| `gemma-4-26b` | `python_modules_long` | 92.5 tok/s | 133.8 tok/s | 1.45x | 1073.1 tok/s | 339 ms | 98.0% |
| `gemma-4-31b` | `flappy` | 18.8 tok/s | 48.0 tok/s | 2.55x | 381.6 tok/s | 912 ms | 100.0% |
| `gemma-4-31b` | `long_code` | 18.7 tok/s | 47.1 tok/s | 2.51x | 499.0 tok/s | 1639 ms | 100.0% |
| `gemma-4-31b` | `python_modules_long` | 19.0 tok/s | 41.8 tok/s | 2.20x | 375.4 tok/s | 965 ms | 98.2% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
