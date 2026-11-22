# Flash Next page-aligned weight loading on M2 Ultra

TensorFold `f0cc0a134` uses page-aligned Darwin checkpoint reads to keep
uncached I/O out of the file cache. A previous AX experiment verified
this behavior for checksum reads. This probe measures selected checkpoint
bytes into MLX-owned native arrays. It compares mapped copy, aligned
buffered reads and aligned `F_NOCACHE` reads in forward and reverse order.
Both aligned modes use one 4 MiB staging buffer and copy only the selected
tensor bytes into the same per-tensor MLX allocations as the mapped mode.

The Mac Studio M2 Ultra 192 GB uses external APFS storage, macOS 27.2.1
and MLX and MLX Metal 0.32.4. The same four resident tensors in each of
MXFP4 shards 17 and 18 provide 1.5625 GiB per selection. The driver
verifies each complete publisher shard and the independent selected-byte
digest. All 24 processes match tensor bytes, allocation sizes and Metal
checksums, with normal pressure at each measurement boundary.

| Shard and initial page state | Mapped median | Aligned buffered median | Aligned no-cache median |
| --- | ---: | ---: | ---: |
| 17, none resident | 8.979 s | 0.751 s | 1.148 s |
| 18, none resident | 8.538 s | 2.261 s | 1.124 s |
| 17, all resident | 0.270 s | 0.254 s | 0.246 s |
| 18, all resident | 0.273 s | 0.253 s | 0.246 s |

Across both initially nonresident orders, the no-cache reader cuts
component time by 86.76–87.32% and leaves zero of 102,404 selected
pages resident. The mapped and aligned buffered readers leave all
selected pages resident. Reused inputs already have all selected pages
resident. APFS clones may share underlying storage caches, so the
measurements describe initial mapping residency and component wall time.

The executable also verifies an unaligned tensor that spans the staging
boundary and ends in a partial EOF page. It rejects a declared payload
that extends past the file. Codex GPT-6.2 Sol high reviewed the range
and allocation changes. The source length and tensor geometry checks
bound offsets before the aligned read; the EOF control exercises the
short final read, and the selected bytes survive the Metal test after
staging is released.

The primary-tier follow-up uses this exact diagnostic binary with the
same selected tensors on a MacBook Pro M5 Max 128 GB internal SSD.
Whole-model startup, warmed serving and the 120-second product gate have
their own requirements.

The replay reconstructs the patched code and driver from the
[prior reader probe](../flash-next-paired-resident-reads-m2-20261122/README.md).
For a live rerun, compile the reconstructed C++20 source against the
pinned MLX headers and library and stage the frozen inventory and
selection. `run_m2.sh` takes the pack and APFS scratch paths and uses
`PYTHON`. `check_unaligned_tail.py` takes the diagnostic executable.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
