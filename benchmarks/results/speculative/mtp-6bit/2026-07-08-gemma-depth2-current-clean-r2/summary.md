# 6-bit MTP AX acceleration summary

This artifact summarizes AX MTP acceleration as `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `gemma-4-26b` | `flappy` | 89.6 tok/s | 136.2 tok/s | 1.52x | 2401.6 tok/s | 145 ms | 99.9% |
| `gemma-4-31b` | `flappy` | 17.4 tok/s | 33.0 tok/s | 1.89x | 695.6 tok/s | 483 ms | 99.8% |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
