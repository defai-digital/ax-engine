# Flash Next fixed-prefix extension on M5 Max

Four fresh processes on MacBook Pro M5 Max 128 GB use the same signed
`release-server` image and the pinned resident MXFP4 pack on internal
SSD with macOS 27.2.1 and MLX and MLX Metal 0.32.4. All 49 publisher
members pass both inventory checks. The [M2 admission replay](../flash-next-fixed-prefix-plain-layout-m2-20261123/README.md)
binds the same image and source patch. MTP, generic n-gram acceleration,
disk caching, adaptive prefill, and direct coalescing are off. Two live
slots and the 2,049-token batch limit stay fixed.

All four arms match 1,346 generated IDs each. Each switch-on arm has
three eligible hits of 2,048 physically reused tokens. The peer decodes
on both original scheduler items. An interrupted target stops at its
first output and an independent replacement completes. The unseeded and
outside-length controls remain cold. All 701 sampled pressure
readings stay normal. Peak physical footprint is 74.46 GiB.

| Process | CPU idle before / after | Combined tokens/s | First-case target TTFT | First-case peer gap at the prefix step |
| --- | ---: | ---: | ---: | ---: |
| off-1 | 92.37% / 92.60% | 33.66 | 2.049 s | 1.551 s |
| on-1 | 91.25% / 96.80% | 40.93 | 0.530 s | 0.030 s |
| on-2 | 96.32% / 91.24% | 40.79 | 0.530 s | 0.031 s |
| off-2 | 91.44% / 96.48% | 33.59 | 2.053 s | 1.561 s |

Combined throughput counts 576 generated IDs across the first and
replacement pairs and divides by their summed wall time. The interrupted
case retains its own latency and correctness measurements. Both process orders meet the 90% CPU-idle floor. Their paired
throughput changes are +21.59%, +21.41%, respectively.

This run evaluates seeded 2,048-token prefixes with 64-token suffixes in
the guarded two-slot profile. The direct release and concurrency
qualification have separate evidence.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
