# Flash Next 32K MTP safety control on M2 Ultra

Mac Studio M2 Ultra 192 GB, macOS 27.2.1, external SSD, Rust 1.99.1,
and MLX 0.32.4. The signed `9df17f26a` test executable reuses the
[canonical singleton verifier audit](../flash-next-expanded-mtp-audit-m2-20261122/README.md)
with two deeper workloads on the pinned resident MXFP4 pack. The
launcher verifies all 49 publisher members and the derived manifest
before and after the test.

| Workload | Input tokens | Output tokens | Verifier decisions | Accepted drafts | Rejected drafts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen lookup | 29,774 | 64 | 38 | 25 | 13 |
| Fixed workload near the context cap | 32,640 | 128 | 74 | 53 | 20 |

The second case reaches the full 32,768-token budget. The audit checks
all 190 post-first-token positions. Every accepted draft equals the
host argmax on the verifier's own state. The direct and MTP paths
return identical IDs for all 192 generated tokens on this device.

This fixed-budget greedy diagnostic disables the MTP cost controller
and ignores EOS. Primary-tier repetition, natural stopping, and
acceleration remain separate measurements.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
