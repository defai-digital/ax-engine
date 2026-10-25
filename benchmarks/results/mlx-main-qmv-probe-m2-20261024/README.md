# MLX main quantized-matmul probe, M2 Ultra 192 GB — 2026-10-24

`crates/ax-engine-mlx/src/bin/mlx_main_qmv_probe.rs` times quantized matmul for 1/2/6/8/16
rows at Qwen 3.8 Flash Next (hidden 2560, expert 640, q_proj 6144) and Qwen 3.8 27B (hidden
5120, FFN 17408) shapes, for MXFP4 g32 and affine 6-bit / 4-bit g64, and compares each
multi-row result's raw f32 bits with stacking the 1-row results (`exact` / `DIFF`). It ran
against the pinned MLX 0.32.4 wheel and against MLX main `56c2bf9236d9` (0.32.5-dev, built
from source; experiment-only linking via `MLX_LIB_DIR` plus `AX_MLX_EXPERIMENT_COMMIT`).
`vs_m1` is the cost relative to one row. Production stays on 0.32.4.

Main result: for 27B MXFP4 FFN projections the multi-row cost falls sharply on MLX main
(`ffn_up` 8 rows 1.95x -> 1.29x of one row, 16 rows 3.04x -> 1.68x; `ffn_down` 8 rows
2.08x -> 1.40x), which is the MTP / DFlash verify-window shape. Flash Next shapes and affine
6-bit/4-bit are roughly neutral (within about +-10%). Row invariance is unchanged except
27B MXFP4 `q_proj` at 8 rows, which becomes `DIFF` on main. Single host, no NAX GPU; the M5
Max run and an end-to-end verify-window benchmark are still needed before any bump.
