# Flash Next grouped QSA evaluation on the M5 resident pack

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, Rust 1.99.1,
and MLX 0.32.4. One signed native test executable uses the verified
49-member pinned MXFP4 pack and the same 8,192-token prompt as the
[M2 isolated control](../flash-next-qsa-serial-real-pack-m2-20261123/README.md).
Each arm starts a fresh process. The four arms run lazy, grouped,
grouped, lazy with a fixed physical plan of four `2047, 1` items.
The test-only candidate evaluates at most two unchanged selected
attention tiles per fence. Stock gather, masks, and SDPA arithmetic
are retained.

| Arm | Peak MLX allocation | Peak sampled physical footprint | Model forward |
| --- | ---: | ---: | ---: |
| Lazy first | 82.25 GiB | 85.20 GiB | 14.034 s |
| Grouped first | 74.04 GiB | 76.82 GiB | 13.437 s |
| Grouped reverse | 74.04 GiB | 77.15 GiB | 13.261 s |
| Lazy reverse | 82.25 GiB | 85.52 GiB | 12.850 s |

All arms finish within 96 GiB and preserve the complete serialized
state and final logits bitwise. Each grouped arm completes 324 tiles
in 180 explicit evaluation fences. All 467 pressure samples stay
normal. Grouping reduces peak MLX allocation by **8.21 GiB** and
sampled process footprint by **8.38 GiB** in both orders.

Forward time falls 4.25% in the first order and rises 3.20% in the
reverse. One endpoint in each grouped arm misses 90% idle, at
84.60% and 70.84%. These timings do not establish a speed change.
The memory result covers this resident forward and fixed prompt.
Native HTTP, the longest supported context, recovery, and alternative
attention fallback retain separate validation. The current production
reserve still covers the full lazy fallback.

The [archive](evidence.tar.xz) binds the signed executable digest,
source patch, exact inputs, arm receipts, and raw pressure samples.
The [comparison](comparison.json) verifies physical chunk boundaries,
output and state digests, and order-specific effects.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
