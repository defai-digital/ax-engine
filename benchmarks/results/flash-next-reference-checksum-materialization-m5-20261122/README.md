# Flash Next reference checksum caching on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1. This
diagnostic follows the [M2 file-cache control](../flash-next-reference-checksum-cache-m2-20261122/README.md)
and [M5 materialization trace](../flash-next-reference-materialization-batches-m5-20261122/README.md).
Each arm used a fresh APFS clone of the pinned publisher pack, an
identical page-aligned 4 MiB checksum buffer, and the same pinned MLX-VLM
materialization graph. The two modes varied only in whether the
checksum file descriptor received `F_NOCACHE`. The launcher checked the
software input hashes and ran the arms in the order below, one at a time,
with no wired memory override.

All arms began at normal pressure with at least 90% CPU idle and zero
observed resident pages in each payload file before its checksum. Each
verified all 26 publisher payload hashes and 132,229,386,263 bytes.
The monitor stopped each arm at the first sampled level-2 pressure
warning while the model was still loading. Each graph has 73 batches.

| Arm | Resident pages summed after individual checksums | Host file-backed after last checksum | Completed batches | Process footprint at warning | Host compressor at warning |
| --- | ---: | ---: | ---: | ---: | ---: |
| No-cache first | 0 | 105.64 GiB | 48 | 92.28 GiB | 75.55 GiB |
| Buffered first | 8,070,650 | 105.55 GiB | 41 | 85.53 GiB | 62.59 GiB |
| Buffered second | 8,070,650 | 105.85 GiB | 48 | 91.69 GiB | 73.64 GiB |
| No-cache second | 0 | 3.99 GiB | 44 | 88.05 GiB | 61.43 GiB |

The file-page sum combines observations taken after separate checksums.
Host file-backed memory is a system-wide snapshot. Cache bypass avoided
observed residency in the cloned files, while system file-backed memory
varied across the bypass arms. Even the bypass arm with 3.99 GiB of host
file-backed memory still hit the pressure guard. The materialization
endpoints overlap across modes. This experiment establishes a file-cache
effect, and cache bypass alone leaves the ordinary reference blocked
before first prefill. The ordinary qualification gates remain open.

The [sealed evidence](evidence.tar.xz) retains the input hash manifest,
diagnostic wrapper sources, raw checksum and phase streams, pressure
samples, and partial and terminal receipts. Replay against the pinned
publisher inventory and graph and compare with [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-reference-checksum-materialization-m5-20261122/replay.py
```
