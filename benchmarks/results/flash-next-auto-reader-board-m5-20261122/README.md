# Flash Next automatic reader release board on M5 Max

The signed `88c480c03` build ran on MacBook Pro M5 Max 128 GB,
internal NVMe SSD, macOS 27.2.1, and MLX 0.32.4. One executable
binds the [independent delivery](../flash-next-auto-reader-fresh-m5-20261122/README.md),
[frozen QA](../flash-next-auto-reader-qa-m5-20261122/README.md),
[lifecycle](../flash-next-auto-reader-lifecycle-m5-20261122/README.md),
and all four measured matrix processes. Each of the 124 output arrays
matches the signed native token oracle. The retained explicit
Required-sidecar bypass control rejects before payload startup.

CPU idle is 91.91% before the matrix and 96.10% afterward. The selected
Disabled and Default processes reach HTTP readiness in 104.413 and
104.602 seconds. Default decodes 128-token results as follows. Each
entry is the median of three measured requests after two warmups.

| Input tokens | First token | Decode after first token |
| --- | ---: | ---: |
| 512 | 0.564 s | 47.74 tokens/s |
| 2,048 | 1.668 s | 44.75 tokens/s |
| 8,192 | 10.580 s | 43.91 tokens/s |

The unchanged evaluator passes five of nine gates. The Disabled
lifecycle start takes 129.887 seconds and misses the 120-second staged
bound. Its final CPU idle sample is 87.29%. The pinned ordinary MLX-VLM
run remains incomplete, leaving integrity, independent quality, and
reference-relative timing open. The observed matrix figures describe
this executable and hardware.

Replay checks the build, publisher inventory, raw native receipts,
exact repeat arrays, and original thresholds.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
