# Flash Next layer-boundary decode on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, MLX
0.32.4 and Rust 1.99.1. The resident release-server test checks all
49 publisher files and the delivered MXFP4 manifest before and after
running. It reuses the [8,192-token stage-profiling prompt](../flash-next-mxfp4-prefill-partition-m5-20261121/holdout-8192-prompts.jsonl.gz)
and its twenty physical widths from the [stage attribution](../flash-next-fixed-prefill-stages-m5-20261122/README.md).

The test evaluates the staged long request at layers 8, 16, 24, 32
and 40. After each barrier a distinct request performs one singleton
decode on the same weights. The two physical forwards exercised start
at token offsets zero and 7,680 and have widths of 512 and 511. Both
requests use their own cache owner. The interleaved long request keeps
its final logits and serialized state bit for bit. All 11 short-request
output IDs, all short logit rows and its serialized state also match
the solo control.

| Physical forward | Solo elapsed | With five peer decodes |
| --- | ---: | ---: |
| Offset 0, width 512 | 393.92 ms | 508.15 ms |
| Offset 7,680, width 511 | 590.91 ms | 732.04 ms |

The median explicit layer evaluation is 41.76 ms and the median peer
decode is 25.82 ms. These intervals characterize the model probe. The
explicit evaluation interval omits any prior operations in the layer
group that already synchronized on the host. Sampled pressure remains
normal. Pre-run host idle is 84.12%, below the 90% timing threshold
used by the serving protocol. The probe establishes state and output
parity at those two positions. Serving latency and throughput remain
open, and the interleave runs only in the test binary.

The [receipt](receipt.json) binds the build, input hashes and original
[test result](result.json). The replay reconstructs the exact long
prompt and checks both arms against the same physical geometry.

```bash
python3 benchmarks/results/flash-next-prefill-layer-peer-m5-20261122/replay.py
```
