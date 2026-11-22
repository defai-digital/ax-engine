# Flash Next aligned resident reader with Required MTP on M5 Max

MacBook Pro M5 Max 128 GB with internal NVMe storage, macOS 27.2.1,
Rust 1.99.1 and MLX and MLX Metal 0.32.4. The crossover uses the
same `31e2b6e1f` release-server binary, frozen 49-file MXFP4 inventory,
and fresh APFS-cloned inputs as the
[default crossover](../flash-next-aligned-native-default-m5-20261122/README.md).
The four servers use Required MTP, its canonical verifier, no generic
n-gram acceleration and the same one-request resident 32K envelope.
The file-cache change affects only the audited resident trunk.

| First prompt | Mmap readiness | Aligned readiness | Mmap trunk load | Aligned trunk load |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 180.408 s | 28.114 s | 156.415 s | 11.293 s |
| 8,192 tokens | 112.313 s | 26.095 s | 85.128 s | 11.315 s |

All 448 output IDs per process, stop reasons and per-request verifier
steps match. Each arm records 277 canonical verification steps with
zero route errors. All four arms retain the 32K boundary and pass the
before-and-after health checks. All measured pressure samples are normal.

The short-first pair meets the 90% CPU-idle requirement. At 2,048 and
8,192 tokens its warmed TTFT increases by 7.32% and 10.96%. Its
2,048-token median decode rate drops by 5.95%. These values exceed the
3% guard. Both reverse-order arms fall below the CPU-idle floor after
the requests, to 82.46% and 79.94%. Keep the ordinary mmap reader.
The crossover uses fresh APFS clones and remains separate from the
official delivered-copy readiness and MTP qualification gates.

The archive recomputes the comparison from all four process receipts,
checked inputs, resource samples and reader traces. Local listener
addresses and paths are redacted.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
