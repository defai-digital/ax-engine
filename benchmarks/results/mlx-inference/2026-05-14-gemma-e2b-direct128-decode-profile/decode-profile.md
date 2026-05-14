# Gemma 4 E2B Direct 128 Decode Profile

This report renders diagnostic `AX_MLX_DECODE_PROFILE=1` counters from inference-stack artifacts. Use it to choose direct-decode cache/fusion experiments, not as a production throughput claim.

## .internal/models/gemma-4-e2b-it-4bit p128 g128

- Artifact: `benchmarks/results/mlx-inference/2026-05-14-gemma-e2b-direct128-decode-profile/gemma-4-e2b-it-4bit-decode-profile.json`
- Decode steps: 127
- Profiled layers: 4,445

| Stage | Wall us | Share of profiled stage time |
|---|---:|---:|
| per-layer input | 44,570 | 0.7% |
| pre-SDPA | 1,879,021 | 29.1% |
| SDPA | 798,415 | 12.3% |
| post-attention | 3,664,742 | 56.7% |
| LM head | 78,565 | 1.2% |

| Substage | Wall us | Parent share | Total profile share |
|---|---:|---:|---:|
| QKV projection | 346,574 | 18.4% | 5.4% |
| QK norm | 766,608 | 40.8% | 11.9% |
| RoPE + KV append | 756,847 | 40.3% | 11.7% |
| Attention output projection | 808,923 | 22.1% | 12.5% |
| Attention residual + FFN norm | 756,554 | 20.6% | 11.7% |
| FFN | 1,161,465 | 31.7% | 18.0% |
| FFN residual + layer gate | 927,463 | 25.3% | 14.3% |
| Unsplit pre-SDPA tail | 8,992 | n/a | 0.1% |
| Unsplit post-attention tail | 10,337 | n/a | 0.2% |

### Reading Notes

- Dominant parent stage: post-attention, 56.7% of profiled stage time.
- `n/a` means the artifact predates that finer-grained counter.
- This profile uses timing barriers and disables production decode pipelining; do not use it as headline throughput evidence.

## Comparison To 2026-05-13 Profile

Reference artifact:
`benchmarks/results/mlx-inference/2026-05-13-gemma-e2b-decode-profile/gemma-4-e2b-it-4bit-decode-profile.json`.

| Metric | 2026-05-13 | Current | Delta |
|---|---:|---:|---:|
| Build commit | `89a5cf24787970c749d2a95455b681bfaf512d3a` | `be0f5bd9f7d71fdf0cdb08cd9185c3223a521053` | n/a |
| Profile-mode decode tok/s | 19.8 | 19.6 | -1.1% |
| Total profiled stage wall us | 6,401,672 | 6,465,313 | +1.0% |
| Per-layer input wall us | 43,957 | 44,570 | +1.4% |
| Pre-SDPA wall us | 1,819,034 | 1,879,021 | +3.3% |
| SDPA wall us | 770,235 | 798,415 | +3.7% |
| Post-attention wall us | 3,687,381 | 3,664,742 | -0.6% |
| LM head wall us | 81,065 | 78,565 | -3.1% |

Substage deltas:

| Substage | 2026-05-13 wall us | Current wall us | Delta |
|---|---:|---:|---:|
| QKV projection | 336,442 | 346,574 | +3.0% |
| QK norm | 745,856 | 766,608 | +2.8% |
| RoPE + KV append | 726,504 | 756,847 | +4.2% |
| Attention output projection | 769,014 | 808,923 | +5.2% |
| Attention residual + FFN norm | 715,806 | 756,554 | +5.7% |
| FFN | 1,239,538 | 1,161,465 | -6.3% |
| FFN residual + layer gate | 951,826 | 927,463 | -2.6% |

## Follow-up Reading

The current profile shape is nearly unchanged from the 2026-05-13 profile:
total profiled stage time increased by only 1.0%, and no single stage accounts
for the production direct-decode slowdown seen in the v4.8.0 protocol rerun.

This keeps the hotspot ranking intact: post-attention remains the largest
profiled cost center, then the pre-SDPA QK norm and RoPE/KV-append tail. The
production slowdown should be investigated in the non-profiled direct pipeline
path or in run-condition variance before changing the n-gram path.
