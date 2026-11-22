# Flash Next targeted release of copied mmap pages on M2 Ultra

Mac Studio M2 Ultra 192 GB, external APFS SSD, macOS 27.2.1, and MLX and
MLX Metal 0.32.4. TensorFold `513e30e3c` motivated releasing file
pages after copying weights into owned buffers. This comparison keeps
the existing private mapping and MLX allocation. The treatment opens
a shared alias on the same descriptor and requests release only for
whole pages covered by the union of selected tensor extents.

The synthetic control preserves all copied bytes and Metal results.
A shared alias releases the one fully selected page and retains all
four sampled adjacent or partially selected pages. A private alias
returns success for the same hint and leaves all five pages resident.

Both 1.5625 GiB publisher selections use a fresh private APFS clone
for each arm, either initially nonresident or primed before launch.
The `msync` step takes 0.065–0.098 seconds across the four median cells.
Each treated arm retains eight selected pages out of 102,404, compared
with all selected pages for the unchanged mmap arms.

| Shard | Initial page state | Restart after mmap | Restart after mmap and release |
| --- | --- | ---: | ---: |
| 17 | Nonresident | 0.255 s | 9.588 s |
| 17 | Primed | 0.254 s | 12.361 s |
| 18 | Nonresident | 0.258 s | 25.636 s |
| 18 | Primed | 0.255 s | 8.421 s |

The 16 initial reads and 16 immediate mapped restarts match the
[earlier M2 weight and Metal observations](../flash-next-page-aligned-reader-m2-20261122/README.md)
across 128 tensor observations. Buffer sizes match and pressure stays
normal. Initial nonresident mmap times vary with the order and shard.
APFS clones share storage, so the page observations describe virtual
memory residency and cannot prove a physical cold-storage condition.

The component results demonstrate page release and a substantial
warm-restart cost. They leave the ordinary mmap loader in place and
supply no primary-tier or delivered-copy readiness result.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
