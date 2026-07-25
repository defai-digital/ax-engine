# Lever: dual affine qmm + GEGLU one C++ call (gate_up residual)

## Profile residual (pure-reprofile3)

| Stage | wall | thr≥21 headroom? |
|-------|------|------------------|
| **`post_attn_ffn_gate_up`** | **~3.26s** | **Yes** — 20% of stage ≈ 7.5% pure wall |
| post_attn_ffn_down | ~2.08s | Yes if ≥32% stage cut (prior fuses reject) |
| sdpa / qkv / o_proj / rope | ≤1.22s | No (measured rejects; physics short) |

Only **gate_up** still has clear thr≥21 headroom after measured rejects (dual Metal
v1/v2 8–25×, dual-qmm compile +2.1%, packed no-compile +3%, full MLP shaped +2%).

## mlxcel source

`gemma4.rs` dense MLP multi-token bits=8 (~917–920, post-#680):

```
gate = gate_proj.forward(x);   // quantized_matmul
up   = up_proj.forward(x);
hidden = compiled_geglu_approx_activation(&gate, &up);
down_proj.forward(&hidden);
```

Full-MLP compile disabled for multi-token 8-bit (#680). Activation is a separate
compiled subgraph; the two qmm stay op-at-a-time.

## AX residual

- Default pure: **split** gate/up (two `qw` / MLX qmm) + **Metal GEGLU** (default ON).
- Dual Metal custom kernel: rejected (correctness + 8–25× wall).
- `compiled_dual_gate_up_qmm` / full `compiled_gelu_approx_split_mlp` shaped: rejected.

**Untried:** collapse the mlxcel *sequence* (two qmm + gelu product) into **one
C++ FFI** without `mx::compile` — pure host-graph residual on the stage with
thr headroom.

## Change

- `ax_mlx_dual_qmm_geglu` / `dual_qmm_geglu`: two affine qmm + `gelu_approx_mul`.
- Opt-in `AX_MLX_DUAL_QMM_GEGLU=1` on multi-token split GEGLU path. Default OFF.

## Success metric

Pure 13.8k cold median ≤ 0.925× OFF → cool S1 thr≥21 → S0–S3. Else reject.


## A/B results (mbp-m5, 2026-07-25-pure-dual-qmm-geglu-ab)

3-rep alternating pure Gemma 13.8k (usage OK, text `" The"` both arms):

| arm | cold samples (ms) | median | mean |
|-----|-------------------|--------|------|
| OFF (portable dual qmm + Metal GEGLU) | 8577, 9287, 8949 | **8949** | 8938 |
| ON (dual_qmm_geglu C++) | 10185, 9763, 9633 | **9763** | 9860 |

- **ratio_median ≈ 1.091**
- **ratio_mean ≈ 1.103**
- **decision: `reject_keep_off`**

Collapsing FFI without Metal GEGLU regresses pure wall ~9–10%. Host-FFI collapse of gate_up is not a flip lever.

### Impassable note (gate_up residual for thr≥21)

Measured rejects on the only stage with thr≥21 pure-cut headroom (~3.3s gate_up):

| Lever | pure ratio | note |
|-------|------------|------|
| Dual Metal v1 | ~8.5× | empty usage |
| Dual Metal v2 | ~25× | empty text |
| dual-qmm compile | ~1.02 | |
| packed no-compile | ~1.03 | |
| full MLP shaped compile | ~1.02 | |
| dual_qmm_geglu FFI (this) | ~1.09 | loses Metal GEGLU |

GPU dual-qmm is already MLX-optimal; custom Metal loses badly; host collapses do not yield ≥7.5% pure cut. Exclusive thr≥21 under pure-sum bound remains blocked without a new GPU-level gate_up win or dual-hold gap fix (out of measured residual path). Skip cool S1 and S0–S3 flip; gates not relaxed.
