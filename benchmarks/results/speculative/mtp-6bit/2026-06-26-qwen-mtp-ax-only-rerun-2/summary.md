# 6-bit MTP AX acceleration summary

This artifact summarizes AX MTP acceleration as `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept | MTPLX | lightning-mlx |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `qwen3.6-27b-6bit` | `flappy` | 17.6 tok/s | 39.6 tok/s | 2.25x | 628.2 tok/s | 514 ms | 99.4% | N/A | N/A |
| `qwen3.6-27b-6bit` | `long_code` | 17.5 tok/s | 38.2 tok/s | 2.18x | 726.0 tok/s | 988 ms | 99.5% | N/A | N/A |
| `qwen3.6-27b-6bit` | `python_modules_long` | 17.4 tok/s | 33.2 tok/s | 1.91x | 635.2 tok/s | 551 ms | 96.7% | N/A | N/A |
| `qwen3.6-35b-a3b` | `flappy` | 38.4 tok/s | 118.2 tok/s | 3.08x | 1477.9 tok/s | 218 ms | 99.8% | N/A | N/A |
| `qwen3.6-35b-a3b` | `long_code` | 38.1 tok/s | 118.1 tok/s | 3.10x | 2318.4 tok/s | 310 ms | 99.8% | N/A | N/A |
| `qwen3.6-35b-a3b` | `python_modules_long` | 38.0 tok/s | 118.9 tok/s | 3.13x | 1637.0 tok/s | 210 ms | 98.4% | N/A | N/A |
| `gemma-4-12b` | `flappy` | 26.7 tok/s | 62.2 tok/s | 2.33x | 1701.7 tok/s | 214 ms | 99.3% | N/A | N/A |
| `gemma-4-12b` | `long_code` | 26.6 tok/s | 70.5 tok/s | 2.65x | 1951.6 tok/s | 409 ms | 99.1% | N/A | N/A |
| `gemma-4-12b` | `python_modules_long` | 27.1 tok/s | 63.2 tok/s | 2.33x | 1753.3 tok/s | 205 ms | 98.0% | N/A | N/A |
| `gemma-4-26b` | `flappy` | 45.7 tok/s | 112.9 tok/s | 2.47x | 2395.0 tok/s | 148 ms | 99.8% | N/A | N/A |
| `gemma-4-26b` | `long_code` | 46.8 tok/s | 113.6 tok/s | 2.43x | 3754.7 tok/s | 219 ms | 99.3% | N/A | N/A |
| `gemma-4-26b` | `python_modules_long` | 45.9 tok/s | 107.2 tok/s | 2.34x | 2597.7 tok/s | 147 ms | 98.9% | N/A | N/A |
| `gemma-4-31b` | `flappy` | 15.4 tok/s | 28.1 tok/s | 1.82x | 701.9 tok/s | 516 ms | 99.6% | N/A | N/A |
| `gemma-4-31b` | `long_code` | 15.3 tok/s | 27.3 tok/s | 1.78x | 747.8 tok/s | 1067 ms | 99.5% | N/A | N/A |
| `gemma-4-31b` | `python_modules_long` | 15.0 tok/s | 25.5 tok/s | 1.70x | 678.5 tok/s | 512 ms | 98.9% | N/A | N/A |
| `glm-4.7-flash` | `flappy` | 52.6 tok/s | 91.5 tok/s | 1.74x | 1694.5 tok/s | 163 ms | 98.2% | N/A | N/A |
| `glm-4.7-flash` | `long_code` | 51.9 tok/s | 77.9 tok/s | 1.50x | 2727.6 tok/s | 250 ms | 98.2% | N/A | N/A |
| `glm-4.7-flash` | `python_modules_long` | 52.4 tok/s | 72.9 tok/s | 1.39x | 1948.2 tok/s | 174 ms | 97.7% | N/A | N/A |

The Qwen rows were refreshed AX-only in this run. Gemma and GLM rows are carried forward from `benchmarks/results/mtp-6bit/2026-06-26-six-model-mtp-ax-only-refresh/summary.json` and were not rerun here.

Peer rows are `N/A` when the peer cannot run the same prepared 6-bit `download-mtp` package under a comparable prompt-suite contract. MTPLX 0.3.7 rejects the Qwen dense runtime contract and has no Gemma assistant-MTP or GLM built-in sidecar runner. Lightning-MLX remains diagnostic-only under current policy after the silent-thinking pathology and does not provide a comparable promoted row for these prepared packages.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
