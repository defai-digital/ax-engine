# 6-bit MTP AX acceleration summary

This artifact summarizes AX MTP acceleration as `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 24.3 tok/s | 45.8 tok/s | 1.89x | 219.8 tok/s | 1469 ms | 100.0% |
| `qwen3.6-27b-6bit` | `long_code` | 24.6 tok/s | 66.0 tok/s | 2.69x | 243.8 tok/s | 2943 ms | 100.0% |
| `qwen3.6-27b-6bit` | `python_modules_long` | 24.7 tok/s | 66.6 tok/s | 2.70x | 227.7 tok/s | 1534 ms | 100.0% |
| `qwen3.6-35b-a3b` | `flappy` | 121.4 tok/s | 151.7 tok/s | 1.25x | 714.8 tok/s | 452 ms | 100.0% |
| `qwen3.6-35b-a3b` | `long_code` | 121.0 tok/s | 121.6 tok/s | 1.00x | 1129.3 tok/s | 635 ms | 100.0% |
| `qwen3.6-35b-a3b` | `python_modules_long` | 121.5 tok/s | 152.5 tok/s | 1.26x | 774.2 tok/s | 444 ms | 100.0% |
| `gemma-4-12b` | `flappy` | 41.3 tok/s | 99.5 tok/s | 2.41x | 462.9 tok/s | 752 ms | 99.2% |
| `gemma-4-12b` | `long_code` | 41.1 tok/s | 98.3 tok/s | 2.39x | 529.6 tok/s | 1545 ms | 99.8% |
| `gemma-4-12b` | `python_modules_long` | 41.4 tok/s | 86.1 tok/s | 2.08x | 462.2 tok/s | 808 ms | 95.2% |
| `gemma-4-26b` | `flappy` | 105.0 tok/s | 151.2 tok/s | 1.44x | 892.6 tok/s | 390 ms | 99.7% |
| `gemma-4-26b` | `long_code` | 103.9 tok/s | 145.1 tok/s | 1.40x | 1245.6 tok/s | 647 ms | 100.0% |
| `gemma-4-26b` | `python_modules_long` | 105.2 tok/s | 141.5 tok/s | 1.35x | 940.6 tok/s | 402 ms | 96.9% |
| `gemma-4-31b` | `flappy` | 18.7 tok/s | 46.2 tok/s | 2.47x | 173.1 tok/s | 2013 ms | 99.8% |
| `gemma-4-31b` | `long_code` | 18.8 tok/s | 45.6 tok/s | 2.42x | 196.5 tok/s | 4165 ms | 99.9% |
| `gemma-4-31b` | `python_modules_long` | 19.4 tok/s | 43.1 tok/s | 2.22x | 173.1 tok/s | 2140 ms | 96.5% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
