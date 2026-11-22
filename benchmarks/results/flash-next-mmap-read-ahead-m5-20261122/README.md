# Flash Next guarded mmap read-ahead on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, Rust 1.99.1
and MLX 0.32.4. All four processes used experiment `c2358dd81`, the
pinned MXFP4 snapshot and the single-request, resident 32K profile.
Guarded mmap copies the selected bytes into the same MLX-owned array
representation in every arm. The advised arms enable bounded hints for
pages inside the next validated tensor. Each issued 984 accepted hints
for a total of 16,534,716,416 bytes. Acceptance records the kernel call
result. Both modes use the same startup trace settings.

| First prompt | Guarded mmap HTTP ready | Advised HTTP ready | Guarded mmap resident phase | Advised resident phase |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 76.209 s | 40.111 s | 58.447 s | 24.286 s |
| 8,192 | 46.226 s | 38.144 s | 30.838 s | 24.939 s |

| First prompt | Warm prompt | Advised change in median TTFT | Advised change in median decode rate |
| ---: | ---: | ---: | ---: |
| 512 | 512 | −0.16% | −3.07% |
| 512 | 2,048 | +12.73% | +0.93% |
| 512 | 8,192 | +8.37% | −2.35% |
| 8,192 | 512 | +2.54% | −3.00% |
| 8,192 | 2,048 | +13.47% | −2.86% |
| 8,192 | 8,192 | +12.38% | −2.31% |

The replay verifies all 15 shard identities and 1,942 tensors against
the [startup trace](../flash-next-mxfp4-startup-phases-m5-20261121/README.md).
It confirms all 1,792 greedy output IDs, the 32K context boundary and
overflow error, zero MTP verifier steps, and 124 normal pressure samples.
The eligible short-first pair exceeded the existing warmed timing
bands. The long-first guarded mmap arm measured 82.84% post-run CPU
idle against the 90% floor. The terminal comparator failed, so that
pair's times remain diagnostic. Keep the existing runtime default.

The [sealed evidence](evidence.tar.xz) retains the regular receipts,
memory samples, source and executable digests, and server logs with
local listen addresses redacted. Reproduce [comparison.json](comparison.json)
from the repository root.

```bash
python3 benchmarks/results/flash-next-mmap-read-ahead-m5-20261122/replay.py
```
