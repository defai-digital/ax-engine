# 6-bit MTP AX acceleration summary

This artifact summarizes exact AX MTP acceleration.

The acceleration ratio is `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 22.6 tok/s | 60.2 tok/s | 2.66x | 196.5 tok/s | 1638 ms | 99.4% |
| `qwen3.6-27b-6bit` | `long_code` | 22.9 tok/s | 51.4 tok/s | 2.24x | 231.2 tok/s | 3103 ms | 98.4% |
| `qwen3.6-27b-6bit` | `python_modules_long` | 23.0 tok/s | 42.6 tok/s | 1.85x | 210.3 tok/s | 1659 ms | 96.0% |
| `qwen3.6-35b-a3b` | `flappy` | 94.0 tok/s | 134.9 tok/s | 1.43x | 535.3 tok/s | 601 ms | 99.9% |
| `qwen3.6-35b-a3b` | `long_code` | 97.5 tok/s | 137.0 tok/s | 1.41x | 926.6 tok/s | 774 ms | 98.5% |
| `qwen3.6-35b-a3b` | `python_modules_long` | 96.8 tok/s | 139.6 tok/s | 1.44x | 576.8 tok/s | 590 ms | 98.2% |
| `gemma-4-12b` | `flappy` | 37.9 tok/s | 95.6 tok/s | 2.52x | 460.9 tok/s | 747 ms | 100.0% |
| `gemma-4-12b` | `long_code` | 37.5 tok/s | 93.7 tok/s | 2.50x | 522.5 tok/s | 1566 ms | 99.9% |
| `gemma-4-12b` | `python_modules_long` | 37.9 tok/s | 74.9 tok/s | 1.98x | 470.6 tok/s | 793 ms | 98.5% |
| `gemma-4-26b` | `flappy` | 88.1 tok/s | 144.3 tok/s | 1.64x | 872.8 tok/s | 418 ms | 99.9% |
| `gemma-4-26b` | `long_code` | 88.1 tok/s | 141.9 tok/s | 1.61x | 1269.1 tok/s | 645 ms | 100.0% |
| `gemma-4-26b` | `python_modules_long` | 89.3 tok/s | 130.4 tok/s | 1.46x | 967.3 tok/s | 389 ms | 98.6% |
| `gemma-4-31b` | `flappy` | 17.7 tok/s | 44.4 tok/s | 2.51x | 171.3 tok/s | 2039 ms | 99.9% |
| `gemma-4-31b` | `long_code` | 17.8 tok/s | 44.0 tok/s | 2.48x | 193.1 tok/s | 4238 ms | 99.8% |
| `gemma-4-31b` | `python_modules_long` | 17.9 tok/s | 38.5 tok/s | 2.15x | 171.2 tok/s | 2171 ms | 98.0% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
