# MLX main quantized-matmul probe, M5 Max 128 GB — 2026-10-24

Same probe as `../mlx-main-qmv-probe-m2-20261024/`
(`crates/ax-engine-mlx/src/bin/mlx_main_qmv_probe.rs`, 2 warm-ups, 5 timed
iterations, upper median) on the NAX GPU (`applegpu_g17s`): the pinned MLX 0.32.4
wheel versus MLX main `56c2bf9236d9` (0.32.5-dev), linked through the
experiment-only path (`MLX_LIB_DIR` plus `AX_MLX_EXPERIMENT_COMMIT`). Raw tables:
`probe-mlx-0.32.4.txt`, `probe-mlx-main-56c2bf92.txt`. `vs_m1` is the cost
relative to one row; `exact` / `DIFF` compares each multi-row result's raw f32 bits
with stacking the 1-row results.

Selected rows (`vs_m1`, pinned -> main):

| Shape | 8 rows | 16 rows |
|---|---|---|
| 27B MXFP4 `ffn_down` | 1.77 -> 0.48 | 2.90 -> 0.49 |
| 27B MXFP4 `ffn_up` | 1.58 -> 1.17 | 1.40 -> 0.98 |
| 27B MXFP4 `q_proj` | 1.60 -> 1.19 (exact -> DIFF) | 2.63 -> 1.29 |
| Flash Next MXFP4 `q_proj` | 1.60 -> 1.01 (exact -> DIFF) | 3.01 -> 1.49 |
| Flash Next MXFP4 expert | 1.19 -> 1.09 | 1.40 -> 1.56 |
| Flash Next affine 6-bit `q_proj` | 1.25 -> 0.85 | 2.88 -> 1.54 |

Reading: the direction matches the M2 Ultra probe — MLX main makes MTP/DFlash
verify-window shapes much cheaper for 27B MXFP4 projections and Flash Next q_proj;
Flash Next expert shapes stay roughly neutral. Ratios below 1.0 mean the one-row
timing is itself noisy at these sub-millisecond sizes, so treat single cells as
directional. Row invariance gets worse on main (two MXFP4 `q_proj` 8-row cells
turn from exact to DIFF), which conflicts with the bit-exact posture. Production
stays on 0.32.4; a bump needs an end-to-end verify-window benchmark and an
exactness re-baseline first.
