# Flash Next early server-side tail deferral on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1 and pinned
MLX and MLX Metal 0.32.4. This four-arm retry used server source
`fce5959de`, the first server-side tail implementation, with Rust
1.99.1 and the resident two-slot MXFP4 profile. The corrected
`2b905c012` implementation adds a deadline recheck before KV
allocation and requires its own primary-tier receipt.

Provenance verification caught that the staged launcher still carried
the older binary and its M5-only driver. Within the retry the same
binary and driver run all four processes. The pinned pack, independent
solo outputs, request generator and original launch flags match the
previous [M5 control](../flash-next-server-prefill-tail-m5-20261122/README.md).
The exercised M5 verifier and route trace functions also match the
later driver's functions.

All 768 generated IDs across the four arms match the frozen solo
outputs. All 563 memory samples report normal pressure and every
arm drains all five counters. Each enabled arm records one singleton
hold while the ending short request decodes.

| Order | Mode | Short tail maximum gap | Long TTFT from trigger | Burst tokens/s | CPU idle before / after |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Physical split | 860.05 ms | 10.868 s | 12.61 | 91.36% / 98.62% |
| 2 | Tail hold | 22.73 ms | 9.206 s | 14.20 | 98.37% / 98.61% |
| 3 | Tail hold | 22.89 ms | 9.246 s | 14.08 | 97.12% / 91.15% |
| 4 | Physical split | 740.67 ms | 10.401 s | 12.93 | 92.89% / 88.48% |

The first pair meets the host timing conditions. The last physical
arm falls below 90% post-run CPU idle. Its reverse timing is diagnostic.
This comparison exercises one deliberately late 8K arrival. Broader
two-client scheduling, sustained load, and the corrected runtime remain
separate acceptance gates.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
