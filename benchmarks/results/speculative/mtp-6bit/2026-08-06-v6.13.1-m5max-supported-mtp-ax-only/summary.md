# 6-bit MTP AX comparison summary

This artifact compares exact AX MTP decode with AX direct decode.

Measured binary: AX Engine v6.13.1 at `bff75300b9854cadc675e0ac22955e4314f93dd3`.

The comparison ratio is `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX MTP/direct | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 23.9 tok/s | 44.9 tok/s | 1.88x | 510.0 tok/s | 631 ms | 99.6% |
| `qwen3.6-27b-6bit` | `long_code` | 23.8 tok/s | 37.6 tok/s | 1.58x | 648.4 tok/s | 1107 ms | 98.3% |
| `qwen3.6-27b-6bit` | `python_modules_long` | 23.9 tok/s | 28.2 tok/s | 1.18x | 525.6 tok/s | 663 ms | 96.2% |
| `qwen3.6-35b-a3b` | `flappy` | 110.2 tok/s | 120.2 tok/s | 1.09x | 874.8 tok/s | 368 ms | 99.9% |
| `qwen3.6-35b-a3b` | `long_code` | 109.9 tok/s | 111.4 tok/s | 1.01x | 1627.3 tok/s | 441 ms | 98.6% |
| `qwen3.6-35b-a3b` | `python_modules_long` | 110.0 tok/s | 96.9 tok/s | 0.88x | 968.8 tok/s | 359 ms | 98.4% |
| `gemma-4-12b` | `flappy` | 39.5 tok/s | 100.4 tok/s | 2.54x | 1058.6 tok/s | 329 ms | 99.9% |
| `gemma-4-12b` | `long_code` | 39.2 tok/s | 99.1 tok/s | 2.53x | 1428.4 tok/s | 573 ms | 100.0% |
| `gemma-4-12b` | `python_modules_long` | 39.6 tok/s | 79.8 tok/s | 2.02x | 1069.8 tok/s | 345 ms | 98.1% |
| `gemma-4-26b` | `flappy` | 91.8 tok/s | 153.2 tok/s | 1.67x | 1180.4 tok/s | 295 ms | 99.9% |
| `gemma-4-26b` | `long_code` | 92.0 tok/s | 150.5 tok/s | 1.64x | 2173.1 tok/s | 376 ms | 100.0% |
| `gemma-4-26b` | `python_modules_long` | 93.3 tok/s | 136.1 tok/s | 1.46x | 1285.4 tok/s | 294 ms | 98.0% |
| `gemma-4-31b` | `flappy` | 18.7 tok/s | 47.8 tok/s | 2.56x | 436.0 tok/s | 829 ms | 100.0% |
| `gemma-4-31b` | `long_code` | 18.5 tok/s | 46.8 tok/s | 2.52x | 551.3 tok/s | 1484 ms | 100.0% |
| `gemma-4-31b` | `python_modules_long` | 18.8 tok/s | 41.4 tok/s | 2.20x | 413.7 tok/s | 879 ms | 98.2% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
