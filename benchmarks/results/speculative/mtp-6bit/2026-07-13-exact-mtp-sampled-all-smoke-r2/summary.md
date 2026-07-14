# 6-bit MTP AX acceleration summary

This artifact summarizes exact AX MTP acceleration.

The diagnostic ratio is `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 23.3 tok/s | 55.5 tok/s | 2.38x | 237.0 tok/s | 1362 ms | 98.9% |
| `qwen3.6-35b-a3b` | `flappy` | 100.5 tok/s | 142.8 tok/s | 1.42x | 1281.5 tok/s | 252 ms | 100.0% |
| `gemma-4-12b` | `flappy` | 38.8 tok/s | 94.1 tok/s | 2.42x | 552.2 tok/s | 621 ms | 100.0% |
| `gemma-4-26b` | `flappy` | 91.2 tok/s | 145.2 tok/s | 1.59x | 1370.1 tok/s | 258 ms | 99.4% |
| `gemma-4-31b` | `flappy` | 18.3 tok/s | 46.8 tok/s | 2.56x | 201.0 tok/s | 1716 ms | 100.0% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
