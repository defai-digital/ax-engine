# 6-bit MTP AX acceleration summary

This artifact summarizes exact AX MTP acceleration.

The diagnostic ratio is `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 23.2 tok/s | 55.4 tok/s | 2.39x | 234.6 tok/s | 1374 ms | 98.9% |
| `qwen3.6-35b-a3b` | `flappy` | 99.7 tok/s | 142.8 tok/s | 1.43x | 1282.0 tok/s | 252 ms | 100.0% |
| `gemma-4-12b` | `flappy` | 38.9 tok/s | 94.3 tok/s | 2.42x | 551.4 tok/s | 620 ms | 100.0% |
| `gemma-4-26b` | `flappy` | 91.1 tok/s | 144.7 tok/s | 1.59x | 1368.3 tok/s | 256 ms | 99.4% |
| `gemma-4-31b` | `flappy` | 17.9 tok/s | 45.6 tok/s | 2.55x | 187.3 tok/s | 1862 ms | 100.0% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
