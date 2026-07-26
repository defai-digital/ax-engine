# thr-b8 config residual: linear pack + Qwen background (mbp-m5, 2026-07-26)

**Decision: `reject` / not_yet.** Gates unchanged. No S0–S3.

## Motivation

thr-b8 + Qwen utility smoke thr **1.146** (ax 20.96 / mlxcel 18.28) is ~0.3%
from the thr gate. Common env forced `AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS=0`
(product default is ON). Hypotheses:

1. Restore pack linear → faster Qwen decode → thr and/or gap.
2. Qwen `process_qos_clamp=background` under thr-b8 → more Gemma GPU duty.

## 1-rep dual-target S1 smokes

Peer: `mlxcel-v0.4.2-qwen-gemma-m5max`. Same tip binary as tail ladder.

| config | thr | gap | ttft | ax thr | ax gap |
|--------|----:|----:|-----:|-------:|-------:|
| thr-b8-util baseline (pack=0, util) | **1.146** | 1.172 | 0.869 | **20.96** | 42.2 |
| thr-b8-util **linear-pack=1** | 1.142 | 1.265 | 0.871 | 20.93 | 42.3 |
| thr-b8 **qwen background** | 1.123 | 1.222 | 0.888 | 20.60 | 41.8 |
| thr-b8 **bg + linear-pack** | 1.111 | 1.239 | 0.897 | 20.38 | 43.0 |

Need thr ≥ **1.15**, gap ≤ **0.90**.

## Conclusion

- Keep **pack linear force-OFF** on thr-b8 stack (ON taxes gap, no thr win).
- Keep **Qwen utility** (background taxes thr ~2%, gap not improved).
- Combined is worse on both. Not product-on changes.
- Flip remains **not_yet**. Gates unchanged.
