# Flash Next storage-probe follow-up on M5 and M4

The signed `26c0ccefe` release-server build exercised the pinned
resident MXFP4 pack on MacBook Pro M5 Max 128 GB, internal NVMe SSD,
macOS 27.2.1, and MLX 0.32.4. The three MTP policy modes each reproduced
seven baseline HTTP responses and their stable usage fields. With the
reader override absent, each launch prefetched 5,214,301,696 bytes.

| MTP mode | HTTP ready | Verifier steps |
| --- | ---: | ---: |
| Disabled | 89.437 s | 0 |
| Required | 112.511 s | 107 |
| Default | 112.988 s | 0 |

The explicit Required-sidecar bypass rejected before payload work. All
launch measurements use the retained verified pack. Full QA, lifecycle,
repeat matrix, and an independent empty-cache delivery remain separate
stages for this runtime.

The same signed focused test executable and pinned MLX libraries also
passed 24 loader and override checks on M4 Pro 64 GB with macOS 27.0
and M4 16 GB with macOS 26.6.1. These checks cover the policy controls,
timeout fallback, header and descriptor validation, and the explicit
`auto` override.

The replay checks the raw HTTP responses and exact pre-read byte counts
with the same frozen verifier and response oracle as the
[earlier automatic-reader campaign](../flash-next-auto-reader-fresh-m5-20261122/README.md).

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
