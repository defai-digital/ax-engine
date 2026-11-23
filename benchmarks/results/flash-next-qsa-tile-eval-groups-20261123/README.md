# Flash Next QSA attention allocation with grouped tile evaluation

This synthetic MLX 0.32.4 probe keeps the selected attention arithmetic,
1 GiB per-gather budget, input seed and 255-row tile width of the
[scratch reservation study](../flash-next-qsa-prefill-scratch-20261123/README.md).
The tile width and masks are identical across arms. A group size of zero
evaluates the entire attention graph at once. Other sizes materialize one,
two or four completed SDPA tiles before continuing the graph. All 48 runs
cover 24 query heads, two K/V heads, 256 channels and 2,051 selected rows.
The MLX Metal kernels are unchanged.

Mac Studio M2 Ultra 192 GB, macOS 27.2.1. The table reports the worst
incremental MLX peak over three order-varied repetitions. Outputs are
bitwise identical across all four methods in every geometry. The
2,048-row, 8,192-history fixture also matches the earlier probe's hash.

| Physical query rows | History rows | Whole graph | One tile | Two tiles | Four tiles |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 8,192 | 2.813 GiB | 1.401 GiB | 2.796 GiB | 2.807 GiB |
| 2,048 | 8,192 | 11.208 GiB | 2.805 GiB | 2.813 GiB | 5.596 GiB |
| 512 | 32,768 | 2.813 GiB | 1.401 GiB | 2.796 GiB | 2.807 GiB |
| 2,048 | 32,768 | 11.208 GiB | 2.419 GiB | 2.813 GiB | 5.596 GiB |

Two tiles per evaluation lowers the selected-attention peak by **74.9%**
at 2,048 queries in both history lengths. The attention-only median
times are 64.72 vs 64.97 ms for the 8,192-history cell and 66.87 vs
66.81 ms for the 32,768-history cell. At 512 queries a group of two
effectively retains the two full tiles together. Single-tile evaluation
has the smaller peak there.

The pinned MLX implementation allocates selected K/V and per-block SDPA
partials. The oMLX `5eeec775` and MTPLX `eb580fce` references show why
keeping intermediates local to a tile can reduce allocation. Their
indexed prefill paths use distinct arithmetic. This probe keeps the
existing AX gather and SDPA and changes when results materialize.
Pack-wide footprint, timing, and promotion need real-model controls.
The current admission reserve still covers full gathered fallback.

The [measurements](observed.json) retain each peak, timing and output
SHA-256. The [comparison](comparison.json) rebuilds the grouped peaks
and verifies the original input and output binding.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
