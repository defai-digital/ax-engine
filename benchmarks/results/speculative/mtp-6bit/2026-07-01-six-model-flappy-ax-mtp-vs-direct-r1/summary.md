# 6-bit MTP AX acceleration summary

This artifact summarizes AX MTP acceleration as `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 17.8 tok/s | 42.2 tok/s | 2.37x | 735.0 tok/s | 437 ms | 100.0% |
| `qwen3.6-35b-a3b` | `flappy` | 39.4 tok/s | 141.3 tok/s | 3.58x | 1813.0 tok/s | 179 ms | 100.0% |
| `gemma-4-12b` | `flappy` | 27.9 tok/s | 79.6 tok/s | 2.85x | 1825.4 tok/s | 190 ms | 100.0% |
| `gemma-4-26b` | `flappy` | 46.3 tok/s | 120.0 tok/s | 2.59x | 2413.5 tok/s | 144 ms | 100.0% |
| `gemma-4-31b` | `flappy` | 15.3 tok/s | 28.3 tok/s | 1.85x | 692.1 tok/s | 487 ms | 100.0% |
| `glm-4.7-flash` | `flappy` | 50.1 tok/s | 96.5 tok/s | 1.93x | 1663.9 tok/s | 167 ms | 100.0% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
