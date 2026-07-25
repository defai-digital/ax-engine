# Pure A/B: multi-token GEGLU→down C++ fuse (mbp-m5)

Opt-in `AX_MLX_DENSE_GEGLU_DOWN_FUSE=1` after dual gate_up split qmm.

| | cold median |
|--|--:|
| OFF | **9074 ms** |
| ON | **9222 ms** (1.016×) |

Default OFF. Does not move pure wall ≥7.5%.
