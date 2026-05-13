# Gemma 4 E2B Decode Profile

This diagnostic run captures AX direct decode stage counters for
`.internal/models/gemma-4-e2b-it-4bit`.

- Artifact: `gemma-4-e2b-it-4bit-decode-profile.json`
- Prompt shape: 128 prompt tokens, 128 generation tokens
- AX policy: `direct_no_ngram_acceleration`
- Repetitions: 1 measured run after 1 AX warmup
- Build commit: `9b14733778b2b28c279b1ae5e3aaf8d2763bf242`

`--ax-decode-profile` sets `AX_MLX_DECODE_PROFILE=1`. The profile materializes
lazy graphs between stages, so decode throughput is intentionally slower than
the production direct pipeline. Use the stage ratios for hotspot ranking, not as
headline throughput evidence.

| Stage | Wall us | Share of profiled stage time |
|---|---:|---:|
| Per-layer input | 42,180 | 1.0% |
| Pre-SDPA | 1,145,132 | 26.3% |
| SDPA | 790,848 | 18.2% |
| Post-attention | 2,302,256 | 52.8% |
| LM head | 75,988 | 1.7% |

Additional breakdown:

| Substage | Wall us | Parent share | Total profile share |
|---|---:|---:|---:|
| QKV projection | 345,650 | 30.2% of pre-SDPA | 7.9% |
| Pre-SDPA tail | 799,482 | 69.8% of pre-SDPA | 18.4% |
| FFN | 1,311,116 | 56.9% of post-attention | 30.1% |
| Post-attention non-FFN | 991,140 | 43.1% of post-attention | 22.8% |

Initial reading:

- The per-layer-input path is measurable but not the dominant direct-decode
  bottleneck in this profiled 4-bit E2B run.
- The best next investigation target is post-attention, especially FFN and the
  non-FFN tail around attention output, residual, gating, and layer scalar work.
- The second target is the pre-SDPA tail after QKV projection: QK/V norm,
  reshape/transpose, RoPE, and KV append account for more profiled time than
  QKV projection itself.
