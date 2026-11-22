# Flash Next page-aligned weight loading on M5 Max

The MacBook Pro M5 Max 128 GB used internal NVMe storage, macOS 27.2.1
and MLX and MLX Metal 0.32.4. The [M2 probe binary](../flash-next-page-aligned-reader-m2-20261122/README.md),
publisher inventory, selected checkpoint extents and counterbalanced
order are byte-identical. TensorFold `f0cc0a134` motivated aligning
both Darwin file ranges and the bounded staging buffer.

| Shard and initial page state | Mapped median | Aligned buffered median | Aligned no-cache median |
| --- | ---: | ---: | ---: |
| 17, none resident | 1.673 s | 0.190 s | 0.293 s |
| 18, none resident | 1.674 s | 0.186 s | 0.282 s |
| 17, all resident | 0.209 s | 0.151 s | 0.156 s |
| 18, all resident | 0.195 s | 0.146 s | 0.149 s |

The aligned no-cache mode saves 82.51% and 83.15% of component time on
the initially nonresident selections. It leaves zero of 102,404 selected
pages resident in all four clone trials. The other two modes leave all
selected pages resident. Initial residency for an APFS clone describes
its own mapping. Physical-device caches may be shared.

All 24 runs match the M2 payload, allocation and Metal checksums and
the unaligned EOF control. Memory pressure is normal at every sampled
boundary. CPU idle is 94.86% before the phase and 91.23% after it. The
scope is two 1.5625 GiB native reader selections. Whole-model validation
must measure HTTP readiness and warmed serving with the candidate in
the normal Flash Next resident path.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
