# Long-prompt prefill chunk 512 vs 768 vs 1024

## Residual
Prior M5 sweep: 512 best thr/s; 1536/2048 slower. Intermediate sizes untested.
Exclusive S1 thr 1.036 needs ~11% pure wall cut for thr≥1.15.

## Result
| chunk | cold median | ratio vs 512 |
|------:|------------:|-------------:|
| 512 | 9118 | 1.000 |
| 768 | 9002 | 0.987 |
| 1024 | 8948 | 0.981 |

Decision: **keep_512** (need ≤0.925). Opt-in `AX_MLX_LONG_PROMPT_PREFILL_CHUNK`.
