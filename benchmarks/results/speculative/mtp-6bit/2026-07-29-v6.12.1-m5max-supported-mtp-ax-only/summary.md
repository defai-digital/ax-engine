# 6-bit MTP AX acceleration summary

This artifact summarizes exact AX MTP acceleration.

The acceleration ratio is `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 23.7 tok/s | 62.6 tok/s | 2.64x | 376.4 tok/s | 854 ms | 99.4% |
| `qwen3.6-27b-6bit` | `long_code` | 23.6 tok/s | 52.3 tok/s | 2.21x | 503.0 tok/s | 1427 ms | 98.6% |
| `qwen3.6-27b-6bit` | `python_modules_long` | 23.7 tok/s | 42.2 tok/s | 1.78x | 389.9 tok/s | 898 ms | 97.4% |
| `qwen3.6-35b-a3b` | `flappy` | 104.1 tok/s | 144.6 tok/s | 1.39x | 516.9 tok/s | 626 ms | 99.8% |
| `qwen3.6-35b-a3b` | `long_code` | 103.1 tok/s | 136.6 tok/s | 1.32x | 953.3 tok/s | 753 ms | 98.4% |
| `qwen3.6-35b-a3b` | `python_modules_long` | 103.7 tok/s | 133.2 tok/s | 1.28x | 559.5 tok/s | 616 ms | 99.0% |
| `gemma-4-12b` | `flappy` | 39.5 tok/s | 100.4 tok/s | 2.54x | 1032.0 tok/s | 337 ms | 99.9% |
| `gemma-4-12b` | `long_code` | 39.2 tok/s | 99.1 tok/s | 2.53x | 1407.5 tok/s | 581 ms | 100.0% |
| `gemma-4-12b` | `python_modules_long` | 39.6 tok/s | 79.8 tok/s | 2.01x | 1056.5 tok/s | 350 ms | 98.1% |
| `gemma-4-26b` | `flappy` | 92.9 tok/s | 152.7 tok/s | 1.64x | 1027.4 tok/s | 339 ms | 99.9% |
| `gemma-4-26b` | `long_code` | 91.9 tok/s | 150.1 tok/s | 1.63x | 1944.3 tok/s | 421 ms | 100.0% |
| `gemma-4-26b` | `python_modules_long` | 92.1 tok/s | 133.6 tok/s | 1.45x | 868.3 tok/s | 426 ms | 98.0% |
| `gemma-4-31b` | `flappy` | 18.7 tok/s | 47.8 tok/s | 2.55x | 379.5 tok/s | 918 ms | 100.0% |
| `gemma-4-31b` | `long_code` | 18.4 tok/s | 46.8 tok/s | 2.55x | 527.1 tok/s | 1552 ms | 100.0% |
| `gemma-4-31b` | `python_modules_long` | 18.9 tok/s | 42.0 tok/s | 2.22x | 388.5 tok/s | 942 ms | 98.2% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
