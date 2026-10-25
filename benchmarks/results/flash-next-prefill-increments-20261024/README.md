# Flash Next prefill increments — 2026-10-24

Follow-up to the reference review's "Flash Next prefill pipeline" item. Pinned
MLX 0.32.4, Flash Next MXFP4, 8,192 prompt tokens, chunk 2048, measured with the
ignored real-pack test `flash_next_prefill_stage_profile`.

## Where prefill time goes (M5 Max 128 GB, synchronized attribution)

Total 14.84 s for 8,192 tokens. Largest stages: `moe_compute` 4.08 s,
`qsa_d_attend` 3.89 s, `qsa_indexer` 1.77 s, `gdn_chunked` 1.11 s, `gdn` 0.69 s,
attention/MLP hyper-connection reads 0.67 + 0.67 s. Chunks after the first cost
about 2x the first because the first fits the 2,051-token selection budget and
takes dense causal attention.

Prefill already attends only to the indexer-selected tokens (up to 512 complete
4-token blocks plus the visible tail per query); it is not dense-masked over the
full context. The selected KV is gathered once per query, and that staging is
the attention cost. Removing both QSA buckets entirely would cap the gain at
about 1.62x under this profile, so a matched 4x gap to the reference claim (a
different machine and context length) is not reachable from QSA alone.

## Shipped: GPU-side QSA prefill selection (default on)

Prefill selection used to copy every chunk's block scores to the host and select
there. A device kernel now builds an order-preserving key (score descending,
block id ascending, the host's signed-zero rule) and selects on the GPU; rows with
a causal non-finite score or fewer eligible blocks than the budget fall back to
the host selector, and future blocks can never be selected.
`AX_MLX_FLASH_NEXT_QSA_PREFILL_DEVICE_SELECT=0` restores host selection.

The 8,192-token real-pack test matches indices and final-logit bits. Stage
profile: `qsa_indexer` 1.775 -> 0.045 s, unsynchronized prefill 13.31 ->
11.79 s. Server TTFT (one request per prompt after a purge, 128 generated
tokens, greedy text identical at every length):

| Prompt tokens | Host selection | GPU selection | Prefill tok/s |
|---|---:|---:|---:|
| 2,207 | 2.37 s | 2.34 s | 933 -> 945 |
| 8,976 | 13.64 s | 11.56 s | 658 -> 776 |
| 39,225 | 75.37 s | 57.76 s | 521 -> 679 |

## Tried and not merged

| Increment | Exactness | Measured (M2 Ultra, release) | Decision |
|---|---|---|---|
| Indexed sparse prompt attention: Metal kernel reading selected KV by index instead of per-query copies (`AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL`) | Tolerance tests pass; greedy text differs at 8,192 tokens. Diagnosed as rounding drift, not a bug: prefill argmax identical in all four chunks, the first greedy flip is an exact tie (margin 0.0) on the shipped path. Bit-exactness is not reachable (MLX SDPA vector 2-pass splits keys across simdgroups; the kernel walks them sequentially) | `qsa_d_attend` 4.401 -> 4.189 s (-4.8%), wall 23.42 -> 22.22 s (-5.1%), peak unchanged | Not merged |
| MoE inverse permutation by scatter instead of a second argsort (`AX_MLX_FLASH_NEXT_MOE_INVERSE_SCATTER`) | Bit-identical (integer permutation and all outputs) | Microbench 0.38 -> 0.24 ms per inverse; end-to-end `moe_compute` within noise (warm pair 10.81 -> 11.13 s) | Not merged |

Already shipped from the same item: tiled QSA scores (opt-in, ~2.6% TTFT with an
outer chunk of 2048 or more on M5 Max; see `../reference-wave-m5max-20261024/`).

## What would move the number

MoE compute is the largest bucket and is expert-matmul bound with sorted rows
already in place; the next lever there is the MLX main quantized-matmul paths
(`../mlx-main-qmv-probe-m5-20261024/`), which need an MLX bump and an exactness
re-baseline. For QSA, a kernel that removes per-query staging must first pass
greedy text identity before its ~5% is worth the risk.
