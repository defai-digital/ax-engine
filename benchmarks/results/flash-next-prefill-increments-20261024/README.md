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

## Shipped opt-in: relaxed indexed sparse prompt attention (default off)

`AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL=1` attends the selected tokens through a
Metal kernel that reads KV by index instead of copying the selected KV per query.
It is not bit-exact with the gathered SDPA path (MLX's vector 2-pass SDPA splits
keys across simdgroups; the kernel walks them sequentially): greedy text can
differ after a near-tie, but the first 80 characters match on both long prompts.
On top of GPU selection, M5 Max 128 GB, MXFP4, 128 generated tokens:

| Prompt tokens | Default TTFT | Indexed TTFT | Prefill tok/s |
|---|---:|---:|---:|
| 2,207 | 2.34 s | 2.27 s | 944 -> 971 |
| 8,976 | 11.54 s | 8.86 s | 778 -> 1,013 |
| 39,225 | 59.78 s | 42.39 s | 656 -> 925 |

Decode is unchanged (44.3-47.5 tok/s). The kernel is compiled at load only when
the switch is on. Off, the only default-path change is that the gathered path
now caps a sub-batch by its actual padded gather (outputs unchanged).

Versus the pre-wave baseline (39k TTFT 76.2 s, 515 tok/s) the default path is
now 57.8 s / 679 tok/s and the opt-in relaxed path 42.4 s / 925 tok/s.

## Tried and not merged

| Increment | Exactness | Measured (M2 Ultra, release) | Decision |
|---|---|---|---|
| MoE inverse permutation by scatter instead of a second argsort (`AX_MLX_FLASH_NEXT_MOE_INVERSE_SCATTER`) | Bit-identical (integer permutation and all outputs) | Microbench 0.38 -> 0.24 ms per inverse; end-to-end `moe_compute` within noise (warm pair 10.81 -> 11.13 s) | Not merged |

Already shipped from the same item: tiled QSA scores (opt-in, ~2.6% TTFT with an
outer chunk of 2048 or more on M5 Max; see `../reference-wave-m5max-20261024/`).

## Shared-gather exactness check (no benefit)

A bit-exact way to cut the per-query KV staging would be to run queries with an
identical score-ordered selection (and the same tail start) as rows of one SDPA
call over one gather; MLX 0.32.4's vector SDPA reduces each query independently,
so that would keep today's arithmetic. The ignored test
`flash_next_qsa_prefill_pair_rate_on_real_pack` measured it on the 8,192-token
MXFP4 prefill: 0 of 18.8 million query pairs qualify. Adjacent queries share
about 68% of their blocks, but identical block sets occur in 0.07% of adjacent
pairs and never with the same order and tail. Not pursued.

## What would move the number

MoE compute is the largest bucket and is expert-matmul bound with sorted rows
already in place; the next lever there is the MLX main quantized-matmul paths
(`../mlx-main-qmv-probe-m5-20261024/`), which need an MLX bump and an exactness
re-baseline. For QSA, a kernel that removes per-query staging must first pass
greedy text identity before its ~5% is worth the risk.
