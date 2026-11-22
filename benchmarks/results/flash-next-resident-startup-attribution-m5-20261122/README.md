# Flash Next resident startup attribution and paired-reader holdout

The [counterbalanced M5 direct starts](../flash-next-mxfp4-startup-phases-m5-20261121/README.md)
and the [M5 matching-manifest startup trace](../flash-next-delivery-bound-startup-m5-20261122/README.md)
all materialized the same 1,942 resident tensors in the same 15 shards.
The timers cover guarded mmap payload reads, copies, and allocation.
File cache state was uncontrolled in the full starts.

| M5 direct order | All shard timers, second minus first | Shards 13–18 | Original probe shards 21 and 24 | Holdout shards 17 and 18 |
| --- | ---: | ---: | ---: | ---: |
| Disabled then default | +20.683 s | +22.552 s | −1.156 s | +8.181 s |
| Default then disabled | +23.981 s | +22.117 s | +0.857 s | +7.481 s |

Every shard from 13 through 18 took longer in the second launch in both
orders. The original probe shards still account for 17.9–22.5% of
copy and allocation in the separate matching-manifest series. Their
small net shifts in the counterbalanced run motivated a second probe
on two of the consistently changing shards. The trace locates elapsed
time by shard. It leaves cache, device bandwidth, and allocation
causes open.

The pinned MLX file reader uses 32 MiB positional batches with disjoint
destination ranges and waits for all reads. The holdout reuses the
[AX diagnostic binary and driver](../flash-next-paired-resident-reads-m2-20261122/README.md).
Its paired path splits selected tensors of at least 64 MiB across two
reads. The two modes use MLX-owned storage with the same allocation
geometry and array construction order.

A Mac Studio M2 Ultra with 192 GB, an external APFS SSD, macOS 27.2.1
and MLX 0.32.4 ran all 18 arms. Both shards matched their publisher
SHA-256, selected bytes, allocator sizes and Metal computations.
The holdout covers 316 tensors and 10,288,018,560 unique published
payload bytes. Each paired arm exercises 11 divided reads. The
short-read negative control passed and sampled memory pressure was
normal. The clone probes observed zero selected resident pages.
The reused probes observed every selected page resident.

| Holdout shard | Clone serial | Clone paired | Reused serial | Reused paired |
| --- | ---: | ---: | ---: | ---: |
| 17 | 2.224 s | 2.283 s | 0.586 s | 0.373 s |
| 18 | 4.275 s | 2.297 s | 0.565 s | 0.365 s |

Read times are medians within each probe mode. The M2 functional
control has mixed cold behavior. The M5 holdout follows the staged
native validation sequence. A full-loader experiment requires M5
reader evidence across the original and holdout shards as well as
unchanged correctness and warmed serving evidence.

The [comparison](comparison.json) records the archived M5 input
digests, individual shard differences, and the checked M2 medians.
Run `shasum -a 256 -c SHA256SUMS` and `python3 -B replay.py` from this
directory to verify the package.
