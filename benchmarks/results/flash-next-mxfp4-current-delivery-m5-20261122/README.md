# Flash Next fresh delivery on the current M5 production build

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1. The
unmodified native product CLI and server were built from `4bd15681f`
with Rust 1.99.1 and MLX 0.32.4. The product CLI started with an empty
internal-SSD cache, downloaded the pinned MXFP4 snapshot in 1,733.066
seconds without a failed attempt, and verified all 49 publisher
members and 132,261,877,478 bytes. The manifest digest matches the
earlier frozen current-source receipt.

| Control mode | HTTP model-list readiness | Frozen bound | Functional checks |
| --- | ---: | ---: | ---: |
| Disabled, initial load | 122.196 s | 300 s | 7/7 |
| Required | 125.258 s | 120 s | 7/7 |
| Product default | 122.002 s | 120 s | 7/7 |

The lockout control rejects required MTP with the process kill switch.
The two later readiness controls exceed the warm limit. The earlier
fresh delivery measured 157.292 seconds for the default. The new
measurement narrows that gap while retaining the readiness failure.
The delivery gate passes. This receipt covers the delivered copy;
current lifecycle, QA, matrix and independent-reference gates have
their own evidence.

The [sealed evidence](evidence.tar.xz) preserves the original 21-control
result and the full download event stream apart from redacted local
staging paths. It retains both original and portable SHA-256 digests.
Replay the [comparison](comparison.json) using the original gate code.

```bash
python3 benchmarks/results/flash-next-mxfp4-current-delivery-m5-20261122/replay.py
```
