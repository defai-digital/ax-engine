# Residual: GEMM-class dual-gate hybrid (packed steel qmm + split GEGLU)

Host: mbp-m5 / M5 Max. Date: 2026-07-25.

## mlxcel review (v0.4.2 flip package)

`src/models/gemma4.rs` dense MLP multi-token bits=8:
- Calls `compiled_gelu_approx_mlp_forward` when no NVFP4 sidecar.
- C++ `mlx_cxx_bridge.cpp` **#680**: for non-(gs64/bits=4) multi-token, falls
  back to **op-at-a-time** dual `quantized_matmul` + `gelu_tanh_approx` + down.
- There is **no** dual-output steel GEMM or dual-stream gate/up in mlxcel for
  flip Gemma MLP bits=8. Same class as AX portable dual qmm.

## AX residual

Load-time `pack_dense_ffn_gate_up_projection` concatenates gate+up weight rows
into one affine quant matrix → **one steel qmm** materializes both projections
(single X activation load). Long Gemma prefill forces **split** two qmms via
`AX_MLX_GEMMA4_SPLIT_PREFILL_FFN` default ON (prior packed A/B ~1.03× worse).

Packed path normally runs `packed_geglu_metal` on the concatenated activation.
Hypothesis: GEMM packing helps; packed GEGLU metal hurts. Hybrid kill-switches:
- `AX_MLX_GEMMA4_SPLIT_PREFILL_FFN=0` → packed qmm
- `AX_MLX_DENSE_GEGLU_PACKED_METAL=0` → slice + production split Metal GEGLU

## Pure A/B bar

Under multi-process keep_base (`CACHE_ONLY_CHUNK_EVAL=1`), keep if pure ratio
≤ **0.96** (thr≥1.15 physics). Else reject; no cool S1 / no S0–S3 flip claim.
