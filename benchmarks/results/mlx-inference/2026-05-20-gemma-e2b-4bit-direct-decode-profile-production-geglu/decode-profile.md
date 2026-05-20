# Gemma 4 E2B 4-bit Production GeGLU Decode Profile

This report renders diagnostic `AX_MLX_DECODE_PROFILE=1` counters from inference-stack artifacts. Use it to choose direct-decode cache/fusion experiments, not as a production throughput claim.

## mlx-community/gemma-4-e2b-it-4bit p512 g128

- Artifact: `benchmarks/results/mlx-inference/2026-05-20-gemma-e2b-4bit-direct-decode-profile-production-geglu/gemma-4-e2b-it-4bit.json`
- Decode steps: 127
- Profiled layers: 4,445

| Stage | Wall us | Share of profiled stage time |
|---|---:|---:|
| per-layer input | 36,601 | 0.4% |
| pre-SDPA | 2,062,370 | 24.6% |
| SDPA | 797,299 | 9.5% |
| post-attention | 5,402,661 | 64.5% |
| LM head | 72,781 | 0.9% |

| Substage | Wall us | Parent share | Total profile share |
|---|---:|---:|---:|
| QKV projection | 315,530 | 15.3% | 3.8% |
| QK norm | 721,203 | 35.0% | 8.6% |
| RoPE + KV append | 1,013,744 | 49.2% | 12.1% |
| Attention output projection | 742,513 | 13.7% | 8.9% |
| Attention residual + FFN norm | 708,869 | 13.1% | 8.5% |
| FFN | 3,052,687 | 56.5% | 36.5% |
| FFN residual + layer gate | 888,047 | 16.4% | 10.6% |
| Unsplit pre-SDPA tail | 11,893 | n/a | 0.1% |
| Unsplit post-attention tail | 10,545 | n/a | 0.1% |
### Candidate Gate

- Dominant parent stage: post-attention, 64.5% of profiled stage time.
- Dominant leaf or unsplit region: FFN, 36.5% of profiled stage time.
- Recommended next probe: probe a post-attention/FFN block or one-layer closure; do not revive the old standalone FFN route.
- Promotion gate: require a real-artifact A/B with both decode throughput and ops/step evidence.

### Reading Notes

- `n/a` means the artifact predates that finer-grained counter.
- This profile uses timing barriers and disables production decode pipelining; do not use it as headline throughput evidence.
