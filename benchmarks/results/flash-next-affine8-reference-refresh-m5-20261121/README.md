# Flash Next bounded PLE reference refresh on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1. Both arms used
`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. The native server came from
`3d6a5442ac8b5c9bce74d5513f5159714561b3df` with Rust 1.99.1 and
MLX 0.32.4. The bounded reference used the pinned MLX-VLM graph at
`6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb`, MLX and MLX Metal 0.32.2,
and the separate affine8 PLE row reader.

The replay matched all 105 signed QA outputs and grades. All 15 fixed
matrix cases matched the signed 128-token outputs and the current native
prompt hashes. The reference peaked at 67.91 GiB of MLX tracked memory
and 59.16 GiB of process RSS. All 25 pressure samples were normal.

| Prompt tokens | Native default TTFT | Bounded reference TTFT | Native default decode | Bounded reference decode |
| --- | ---: | ---: | ---: | ---: |
| 512 | 0.546 s | 2.185 s | 49.75 tok/s | 37.76 tok/s |
| 2,048 | 1.645 s | 5.493 s | 46.28 tok/s | 36.79 tok/s |
| 8,192 | 10.575 s | 14.934 s | 44.19 tok/s | 36.80 tok/s |

Across both native arms, the observed decode rate was 1.199–1.317 times
the bounded reference. Observed native TTFT was 0.250–0.708 times the
bounded reference. CPU idle was 95.91% before the reference and 88.35%
after it, below the 90% post-run floor. These are diagnostic observations.
The ordinary MLX-VLM release reference and the delivered-copy startup
gate remain open.

From the repository root, replay the data and compare the output with
[comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-affine8-reference-refresh-m5-20261121/compare_outputs.py
```

The [bounded result](bounded-result.json.gz), [native matrix](native-matrix.json.gz),
[resource samples](memory-samples.jsonl.gz) and [source bindings](source.json)
retain the underlying evidence. The [previous bounded comparison](../flash-next-affine8-reference-m5-20261121/README.md)
includes the smoke trial and links to the historical signed ordinary result.
