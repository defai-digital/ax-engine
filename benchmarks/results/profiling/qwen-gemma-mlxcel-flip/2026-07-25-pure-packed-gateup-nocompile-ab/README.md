# Pure-wall A/B: packed gate/up without prefill compile (mbp-m5)

Profile residual: post_attn_ffn_gate_up dual 8-bit qmm (~3.35s).
Hypothesis: prior packed reject mixed in prefill compile; single packed qmm
could reuse X bandwidth vs two split qmm.

## 3-rep median cold wall

| variant | median cold ms |
|--|--:|
| split (default) | **8838** |
| packed + no compile | **9121** (1.03×) |
| packed + prefill compile | **9382** (1.06×) |

**Decision: keep split.** Packing does not beat dual MLX qmm for pure Gemma 13.8k
even without compile (mlxcel also uses two separate qmm for bits=8 multi-token).
