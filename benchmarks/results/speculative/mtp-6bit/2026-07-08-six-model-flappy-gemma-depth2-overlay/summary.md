# 6-bit MTP AX acceleration summary

This overlay keeps the full six-model flappy table while replacing the Gemma 4 26B and Gemma 4 31B rows with clean depth-2 assistant-MTP reruns from `2026-07-08-gemma-depth2-current-clean-r2`.

| Target | Suite | AX direct decode | AX MTP decode | AX speedup | AX MTP prefill | AX MTP TTFT | AX accept |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 22.9 tok/s | 44.0 tok/s | 1.92x | 754.8 tok/s | 426 ms | 100.0% |
| `qwen3.6-35b-a3b` | `flappy` | 79.7 tok/s | 141.1 tok/s | 1.77x | 1805.9 tok/s | 179 ms | 100.0% |
| `gemma-4-12b` | `flappy` | 38.0 tok/s | 75.8 tok/s | 2.00x | 1792.2 tok/s | 194 ms | 100.0% |
| `gemma-4-26b` | `flappy` | 89.6 tok/s | 136.2 tok/s | 1.52x | 2401.6 tok/s | 145 ms | 99.9% |
| `gemma-4-31b` | `flappy` | 17.4 tok/s | 33.0 tok/s | 1.89x | 695.6 tok/s | 483 ms | 99.8% |
| `glm-4.7-flash` | `flappy` | 73.0 tok/s | 112.6 tok/s | 1.54x | 1635.5 tok/s | 171 ms | 98.0% |

Source overlay:

- Gemma 4 26B and Gemma 4 31B: clean depth-2 rerun at commit `08df1a96e820`.
- Other rows: retained from `2026-07-07-six-model-flappy-current-code-refresh`.
- Qwen3.6 35B-A3B clean default probe: 142.3 tok/s, below the older 142.7 tok/s high-water record, so not promoted.

Pure-MTP verification: promoted AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
