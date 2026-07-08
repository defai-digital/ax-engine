# 6-bit MTP AX acceleration summary

This artifact summarizes AX MTP acceleration as `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 22.9 tok/s | 44.0 tok/s | 1.92x | 754.8 tok/s | 426 ms | 100.0% |
| `qwen3.6-35b-a3b` | `flappy` | 79.7 tok/s | 141.1 tok/s | 1.77x | 1805.9 tok/s | 179 ms | 100.0% |
| `gemma-4-12b` | `flappy` | 38.0 tok/s | 75.8 tok/s | 2.00x | 1792.2 tok/s | 194 ms | 100.0% |
| `gemma-4-26b` | `flappy` | 83.6 tok/s | 110.7 tok/s | 1.32x | 2286.9 tok/s | 152 ms | 100.0% |
| `gemma-4-31b` | `flappy` | 17.5 tok/s | 28.0 tok/s | 1.60x | 697.5 tok/s | 483 ms | 99.9% |
| `glm-4.7-flash` | `flappy` | 73.0 tok/s | 112.6 tok/s | 1.54x | 1635.5 tok/s | 171 ms | 98.0% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
