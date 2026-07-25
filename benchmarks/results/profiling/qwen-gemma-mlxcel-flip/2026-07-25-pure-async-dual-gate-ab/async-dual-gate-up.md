# Lever: async dual gate/up submit (gate_up residual)

## Residual

Profile pure Gemma: `post_attn_ffn_gate_up` ~**3.26s**. Dual Metal / dual_gate
compile / #705 under cache_eval max ~0.7% pure (need ≤0.96).

## mlxcel

`gemma4.rs` ~917–920 multi-token bits=8:
```
gate = gate_proj.forward(x);
up = up_proj.forward(x);
hidden = compiled_geglu_approx_activation(gate, up);
```
Both qmm stay lazy until activation/eval; MLX can co-schedule. AX Metal GEGLU
may serialize materialization.

## Change

`AX_MLX_ASYNC_DUAL_GATE_UP=1`: after dual gate/up tensors are built,
`async_eval([gate, up])` before GEGLU. Default OFF.

## Success

Pure under cache_eval ≤0.96 vs base → cool multi-process S1 / S0–S3.

## Result (mbp-m5 pure cache_eval, 2026-07-25, 3-rep)

| variant | median cold ms | ratio |
|---------|---------------:|------:|
| base | 8209 | 1.000 |
| async_du | 8264 | **1.007** |

Decision **keep_base** / reject. Default OFF.
