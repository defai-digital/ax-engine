# Flash Next pinned reference parameter graph on M2 Ultra

Mac Studio M2 Ultra 192 GB, internal SSD, macOS 27.2.1. This metadata
probe constructed `load_model(..., lazy=True, strict=True).language_model`
using MLX and MLX Metal 0.32.2 and pinned MLX-VLM `6102cb4ad`. It checked
the reference loader and language byte hashes, the config and index hashes
and all 49 published member sizes for MXFP4 revision `ff2a2848`. The
probe enumerated shapes and dtypes before the ordinary reference's
explicit weight evaluation. MLX peak during the probe was 32,208 bytes.
The model files' payload checksums are outside this metadata-only probe.

| Parameter group | Arrays | Logical bytes |
| --- | ---: | ---: |
| PLE n-gram weights, scales and biases | 384 | 53.644 GiB |
| All other parameters | 1,942 | 63.811 GiB |
| Total | 2,326 | 117.455 GiB |

The ordinary runner evaluates this traversal in 73 batches of up to 32
arrays. N-gram entries occur in batches 2–14 with zero-based indexing.
The largest logical batch accounts for 4.587 GiB. The data describes
parameter geometry. The OS, allocator, other graph allocations and
temporary buffers have separate memory costs, and the probe does not
locate the batch that triggered the M5 warning. The [signed ordinary
M5 run](../qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/reference/result.json)
recorded 121.56 GiB of peak MLX tracked memory on an earlier macOS release.

This explains why the upstream option to keep PLE rows in separately
bounded storage is worth testing on 128 GB. The [prior bounded
benchmark](../flash-next-affine8-reference-refresh-m5-20261121/README.md)
supplies output and pressure controls under its own profile. The frozen
ordinary reference and [current M5 materialization result](../flash-next-ordinary-reference-m5-20261121/README.md#materialization-only-diagnostic)
remain the release evidence for that path.

To recalculate the graph totals and batch accounting without MLX or
model weights, run the following and compare against [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-reference-parameter-graph-m2-20261121/replay.py
```
