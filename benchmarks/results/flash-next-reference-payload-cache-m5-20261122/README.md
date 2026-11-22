# Flash Next reference payload caching on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1. The
reference uses MLX-VLM `6102cb4a` with MLX and MLX Metal 0.32.2.
The experiment holds the [aligned no-cache checksum path](../flash-next-reference-checksum-materialization-m5-20261122/README.md)
fixed. It varies `F_NOCACHE` on the actual native payload readers
after constructing the lazy graph and before materializing weights.
The pinned MLX `v0.32.2` `ParallelFileReader` keeps the original open
descriptor for positional reads, so changing an unrelated checksum
descriptor would exercise a different mechanism.

All four processes start at normal pressure and at least 90% sampled
CPU idle. Each checksums 26 publisher files with zero observed file
pages before and after each checksum. Each opens one native descriptor
for each of the 25 trunk payloads. The diagnostic binds those live
descriptors by device and inode before applying the selected mode.

| Arm | Completed batches | Process footprint at first warning | System compressor at first warning |
| --- | ---: | ---: | ---: |
| No-cache first | 20 | 63.45 GiB | 61.13 GiB |
| Buffered first | 50 | 94.87 GiB | 75.67 GiB |
| Buffered second | 40 | 84.00 GiB | 69.74 GiB |
| No-cache second | 28 | 70.76 GiB | 70.14 GiB |

The monitor sends SIGTERM at the first sampled level-2 pressure
warning. Every process is still loading. Both no-cache arms complete
fewer of the 73 batches than either buffered control in this run.
The sampled footprint differs across arms. These endpoints record
pressure-limited progress, without a latency or total-read comparison
at an equal amount of materialization. Exact cache bypass leaves the
unchanged independent reference blocked before first prefill.

The [sealed records](evidence.tar.xz) retain the paired wrapper sources,
input hashes, original descriptor and checksum receipts, ordered phase
events, pressure samples and partial reference results. The replay
checks the publisher inventory, reference identity and terminal boundary
and reproduces [the comparison](comparison.json).

```bash
python3 benchmarks/results/flash-next-reference-payload-cache-m5-20261122/replay.py
```
