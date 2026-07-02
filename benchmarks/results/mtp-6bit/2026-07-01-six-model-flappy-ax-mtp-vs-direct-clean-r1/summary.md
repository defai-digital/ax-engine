# 6-bit MTP AX acceleration summary

This artifact summarizes AX MTP acceleration as `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 18.5 tok/s | 43.3 tok/s | 2.33x | 764.2 tok/s | 423 ms | 100.0% |
| `qwen3.6-35b-a3b` | `flappy` | 41.0 tok/s | 140.9 tok/s | 3.44x | 1806.0 tok/s | 179 ms | 100.0% |
| `gemma-4-12b` | `flappy` | 27.7 tok/s | 77.5 tok/s | 2.80x | 1811.2 tok/s | 191 ms | 100.0% |
| `gemma-4-26b` | `flappy` | 41.7 tok/s | 114.2 tok/s | 2.74x | 2367.9 tok/s | 148 ms | 100.0% |
| `gemma-4-31b` | `flappy` | 15.2 tok/s | 28.5 tok/s | 1.87x | 701.5 tok/s | 479 ms | 100.0% |
| `glm-4.7-flash` | `flappy` | 51.2 tok/s | 97.3 tok/s | 1.90x | 1693.4 tok/s | 163 ms | 100.0% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
