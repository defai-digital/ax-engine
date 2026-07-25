# Lever: GEMM-class dual gate/up Metal v3 (pure thr residual)

## Profile residual

Pure Gemma 13.8k prefill profile (mbp-m5):

| stage | wall | share |
|-------|-----:|------:|
| gate_up dual qmm | ~3.26 s | ~38% |
| down | ~2.08 s | ~24% |
| sdpa | ~1.22 s | ~14% |

Exclusive S1 thr **1.036×** (need ≥1.15×) ⇒ ~11% scenario wall cut.
gate_up is the only stage with theoretical headroom for that cut.

## mlxcel / prior AX path

mlxcel multi-token bits=8: two `quantized_matmul` + `compiled_geglu_approx`
(op-at-a-time; #705 shaped compile decode-only). AX production: same dual
MLX qmm + Metal GEGLU (already faster pure than mlxcel).

Prior custom Metal:
- v1: one OutDim row / TG, re-read X → ~8.5× pure
- v2: BM=4 / K stride TG → most threads idle → ~25× pure

## v3 design

Classical tiled dual-qmm GEMM + fused GEGLU:

- Tile **BM=8** output rows × **BN=16** tokens per threadgroup (**TG=128**)
- Reduce over **BK=128** (multiple of gs=64 and pack_factor=4 for bits=8)
- Full-TG cooperative load of X tile + dequantized gate/up W tiles
- Each thread owns one (row, token) of the BM×BN tile (no cross-TG reduce)
- Write `gelu_approx(gate) * up` at end

Dispatch: `grid = num_row_blocks * num_token_blocks * 128`.

Env: `AX_MLX_GEMMA_DUAL_GATE_UP_METAL=1` (default OFF).

## Success

Pure 13.8k cold median ≤ **0.925×** portable OFF → cool S1 thr → S0–S3 flip.
Else reject; leave default OFF; gates unchanged.

## Result (mbp-m5, 2026-07-25) — **REJECT**

| variant | cold median | text |
|---------|------------:|------|
| OFF (dual MLX qmm + Metal GEGLU) | **8570 ms** | `" The"` |
| ON (v3 tiled GEMM) | **72991 ms** | empty |

- **ratio_median = 8.52×** (worse; need ≤0.925)
- Correctness: empty completion text under ON → numerical/path failure
- decision: **reject_keep_off**

Naive BM×BN×BK tiled dual-qmm still loses ~8.5× to MLX steel qmm (similar to
v1 class). Leave `AX_MLX_GEMMA_DUAL_GATE_UP_METAL` default OFF. No thr headroom;
skip cool S1 / S0–S3 flip. Gates unchanged.
