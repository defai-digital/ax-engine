# 6-bit MTP AX acceleration summary

This publication artifact keeps the 2026-07-01 six-model flappy baseline and replaces only the Gemma 4 12B row with the 2026-07-05 rerun.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 18.6 tok/s | 43.6 tok/s | 2.34x | 766.6 tok/s | 420 ms | 100.0% |
| `qwen3.6-35b-a3b` | `flappy` | 41.3 tok/s | 142.7 tok/s | 3.45x | 1818.4 tok/s | 178 ms | 100.0% |
| `gemma-4-12b` | `flappy` | 37.8 tok/s | 75.1 tok/s | 1.99x | 1780.3 tok/s | 195 ms | 100.0% |
| `gemma-4-26b` | `flappy` | 46.2 tok/s | 120.0 tok/s | 2.60x | 2411.1 tok/s | 144 ms | 100.0% |
| `gemma-4-31b` | `flappy` | 15.2 tok/s | 28.3 tok/s | 1.87x | 699.8 tok/s | 478 ms | 100.0% |
| `glm-4.7-flash` | `flappy` | 51.2 tok/s | 97.5 tok/s | 1.90x | 1694.3 tok/s | 163 ms | 100.0% |

Gemma 4 12B refresh source: `benchmarks/results/speculative/mtp-6bit/2026-07-05-gemma4-12b-6bit-flappy-ax-mtp-vs-direct-refresh/summary.json`.
Other rows are retained from `benchmarks/results/speculative/mtp-6bit/2026-07-01-six-model-flappy-ax-mtp-vs-direct-clean-r2/summary.json`.

Pure-MTP verification: all listed AX MTP artifacts record zero n-gram accepted, proposed, submitted, and hit-step telemetry.
