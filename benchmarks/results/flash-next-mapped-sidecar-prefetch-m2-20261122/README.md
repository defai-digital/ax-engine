# Flash Next MTP sidecar read-ahead on M2 Ultra

The component comparison uses Mac Studio M2 Ultra 192 GB, an external
APFS SSD, macOS 27.2.1 and MLX/MLX Metal 0.32.4. It uses the complete
31-tensor MTP sidecar and four selected expert arrays from each of two
MXFP4 publisher shards. The runner checks the three full file hashes
against the pinned 49-file publisher inventory and independently hashes
the selected payloads.

The experiment follows TensorFold's bounded positional read-ahead in
`checkpoint_read_ahead.zig` at `f0cc0a134`. One reader stages up to 4 MiB
at a time from the sidecar's existing descriptor while the trunk arrays
are copied. It joins before the mapped sidecar copy. Both treatments
allocate the same MLX-owned storage and copy mapped tensors in the
same order. Each process uses new APFS clones with zero resident
selected virtual pages at entry. The device's physical cache is
uncontrolled.

| Shard | Order | Sequential mapped total | Overlapped total |
| --- | --- | ---: | ---: |
| 17 | Sequential then overlap | 88.957 s | 11.252 s |
| 17 | Overlap then sequential | 74.386 s | 11.265 s |
| 18 | Overlap then sequential | 38.840 s | 14.564 s |
| 18 | Sequential then overlap | 130.619 s | 10.896 s |

The overlapped reader covers all 5,214,301,696 payload bytes in
2.230–5.996 seconds. In the treatment, the mapped sidecar copy itself
takes 0.718–0.765 seconds. Across all eight arms, 280 array observations retain identical
allocation sizes, CPU payload digests and Metal probe digests for the
same selected payload. Normal pressure
holds across all arms. The synthetic case matches the payloads and
rejects an extent that runs beyond the file.

The repeated component difference supports a native disk-cache
pre-read test. These arms compare normal mapped I/O with staged
pre-reading plus overlap, so they cannot attribute the total gain to
overlap alone. Whole-pack M5 startup, first-request latency, warmed
serving, and Required MTP activation need separate evidence.

The archive contains the measured C++ reader, Python runner, input
receipts, build digest and per-arm data. The replay also binds the
earlier [trunk controls](../flash-next-page-aligned-reader-m2-20261122/README.md)
and [complete sidecar control](../flash-next-owned-positional-sidecar-m2-20261122/README.md).

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
