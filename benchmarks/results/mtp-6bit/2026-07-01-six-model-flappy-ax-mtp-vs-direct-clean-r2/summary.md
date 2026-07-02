# 6-bit MTP AX acceleration summary

This artifact summarizes AX MTP acceleration as `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 18.6 tok/s | 43.6 tok/s | 2.34x | 766.6 tok/s | 420 ms | 100.0% |
| `qwen3.6-35b-a3b` | `flappy` | 41.3 tok/s | 142.7 tok/s | 3.45x | 1818.4 tok/s | 178 ms | 100.0% |
| `gemma-4-12b` | `flappy` | 27.9 tok/s | 79.3 tok/s | 2.84x | 1827.2 tok/s | 189 ms | 100.0% |
| `gemma-4-26b` | `flappy` | 46.2 tok/s | 120.0 tok/s | 2.60x | 2411.1 tok/s | 144 ms | 100.0% |
| `gemma-4-31b` | `flappy` | 15.2 tok/s | 28.3 tok/s | 1.87x | 699.8 tok/s | 478 ms | 100.0% |
| `glm-4.7-flash` | `flappy` | 51.2 tok/s | 97.5 tok/s | 1.90x | 1694.3 tok/s | 163 ms | 100.0% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
