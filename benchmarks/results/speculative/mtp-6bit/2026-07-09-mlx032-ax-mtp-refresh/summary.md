# 6-bit MTP AX acceleration summary

This artifact summarizes AX MTP acceleration as `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 22.8 tok/s | 65.7 tok/s | 2.89x | 242.2 tok/s | 1330 ms | 100.0% |
| `qwen3.6-27b-6bit` | `long_code` | 22.8 tok/s | 65.3 tok/s | 2.86x | 252.7 tok/s | 2839 ms | 100.0% |
| `qwen3.6-27b-6bit` | `python_modules_long` | 22.9 tok/s | 65.6 tok/s | 2.87x | 250.7 tok/s | 1396 ms | 100.0% |
| `qwen3.6-35b-a3b` | `flappy` | 96.6 tok/s | 148.0 tok/s | 1.53x | 1294.4 tok/s | 249 ms | 100.0% |
| `qwen3.6-35b-a3b` | `long_code` | 99.8 tok/s | 147.3 tok/s | 1.48x | 1648.8 tok/s | 435 ms | 100.0% |
| `qwen3.6-35b-a3b` | `python_modules_long` | 99.5 tok/s | 150.1 tok/s | 1.51x | 1399.5 tok/s | 248 ms | 100.0% |
| `gemma-4-12b` | `flappy` | 38.8 tok/s | 95.4 tok/s | 2.46x | 560.9 tok/s | 614 ms | 100.0% |
| `gemma-4-12b` | `long_code` | 38.1 tok/s | 94.6 tok/s | 2.48x | 571.5 tok/s | 1413 ms | 100.0% |
| `gemma-4-12b` | `python_modules_long` | 38.7 tok/s | 74.9 tok/s | 1.94x | 568.6 tok/s | 665 ms | 99.0% |
| `gemma-4-26b` | `flappy` | 89.3 tok/s | 148.2 tok/s | 1.66x | 1376.5 tok/s | 253 ms | 100.0% |
| `gemma-4-26b` | `long_code` | 89.2 tok/s | 144.5 tok/s | 1.62x | 1605.7 tok/s | 507 ms | 100.0% |
| `gemma-4-26b` | `python_modules_long` | 90.6 tok/s | 135.7 tok/s | 1.50x | 1430.3 tok/s | 264 ms | 99.0% |
| `gemma-4-31b` | `flappy` | 17.7 tok/s | 45.6 tok/s | 2.57x | 211.6 tok/s | 1625 ms | 99.9% |
| `gemma-4-31b` | `long_code` | 17.7 tok/s | 44.1 tok/s | 2.48x | 212.3 tok/s | 3802 ms | 100.0% |
| `gemma-4-31b` | `python_modules_long` | 18.1 tok/s | 39.6 tok/s | 2.18x | 214.1 tok/s | 1766 ms | 98.7% |
| `glm-4.7-flash` | `flappy` | 74.8 tok/s | 122.9 tok/s | 1.64x | 1063.1 tok/s | 260 ms | 98.1% |
| `glm-4.7-flash` | `long_code` | 74.5 tok/s | 100.9 tok/s | 1.35x | 1268.6 tok/s | 538 ms | 98.6% |
| `glm-4.7-flash` | `python_modules_long` | 76.4 tok/s | 91.2 tok/s | 1.19x | 1134.4 tok/s | 300 ms | 94.3% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
