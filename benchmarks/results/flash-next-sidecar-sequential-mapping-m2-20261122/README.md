# Flash Next sidecar sequential mapping advice on M2 Ultra

Mac Studio M2 Ultra 192 GB, external APFS SSD, macOS 27.2.1, Apple
clang 21.0.0, MLX and MLX Metal 0.32.4. The full admitted MXFP4 MTP
sidecar contains 31 BF16 tensors in contiguous file and lexical order.
llama.cpp `418fe97b5` uses sequential file advice for its checkpoint
mapping on Linux. This Darwin diagnostic measures `MADV_SEQUENTIAL`
directly against the ordinary mapping.

The same executable creates the same read-only private mapping, copies
each selected tensor into the same MLX allocator, and runs the same
Metal operation. Only the mapping advice differs. All eight processes
match the exact 5,214,301,696-byte payload, allocation sizes and Metal
digest from the [earlier positional control](../flash-next-owned-positional-sidecar-m2-20261122/README.md).
Pressure is normal at all 16 process boundaries. Each fresh clone
starts with zero resident selected pages. Reused input starts with
all 318,256 selected pages resident.

| Input | Experiment order | Ordinary | Sequential advice |
| --- | --- | ---: | ---: |
| Fresh clone | Ordinary then sequential | 42.263 s | 57.426 s |
| Fresh clone | Sequential then ordinary | 79.228 s | 49.608 s |
| Reused file | Sequential then ordinary | 1.383 s | 0.844 s |
| Reused file | Ordinary then sequential | 1.222 s | 0.850 s |

Cold timing changes direction with process order. For the reused file,
the two measured savings are 0.539 and 0.371 seconds. Keep the ordinary
sidecar mapping in the runtime. The scope is M2 file materialization
and Metal byte parity.

The [comparison](comparison.json) replays the [terminal result](result.json)
and the captured build, geometry and benchmark sources.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
