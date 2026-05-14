# Gemma 4 E2B Decode Profile

This diagnostic run captures AX direct decode stage counters for
`.internal/models/gemma-4-e2b-it-4bit`.

- Artifact: `gemma-4-e2b-it-4bit-decode-profile.json`
- Prompt shape: 128 prompt tokens, 128 generation tokens
- AX policy: `direct_no_ngram_acceleration`
- Repetitions: 1 measured run after 1 AX warmup
- Build commit: `89a5cf24787970c749d2a95455b681bfaf512d3a`

`--ax-decode-profile` sets `AX_MLX_DECODE_PROFILE=1`. The profile materializes
lazy graphs between stages, so decode throughput is intentionally slower than
the production direct pipeline. Use the stage ratios for hotspot ranking, not as
headline throughput evidence.

| Stage | Wall us | Share of profiled stage time |
|---|---:|---:|
| Per-layer input | 43,957 | 0.7% |
| Pre-SDPA | 1,819,034 | 28.4% |
| SDPA | 770,235 | 12.0% |
| Post-attention | 3,687,381 | 57.6% |
| LM head | 81,065 | 1.3% |

Additional breakdown:

| Substage | Wall us | Parent share | Total profile share |
|---|---:|---:|---:|
| QKV projection | 336,442 | 18.5% of pre-SDPA | 5.3% |
| QK norm | 745,856 | 41.0% of pre-SDPA | 11.7% |
| RoPE + KV append | 726,504 | 39.9% of pre-SDPA | 11.3% |
| FFN | 1,239,538 | 33.6% of post-attention | 19.4% |
| Attention output projection | 769,014 | 20.9% of post-attention | 12.0% |
| Attention residual + pre-FFN norm | 715,806 | 19.4% of post-attention | 11.2% |
| FFN residual + per-layer gate | 951,826 | 25.8% of post-attention | 14.9% |

Initial reading:

- The per-layer-input path is measurable but not the dominant direct-decode
  bottleneck in this profiled 4-bit E2B run.
- The best next investigation target is post-attention, especially the FFN
  residual + per-layer gate path and FFN itself.
- The second target is the pre-SDPA tail after QKV projection: QK norm and
  RoPE/KV append each account for more profiled time than QKV projection.
