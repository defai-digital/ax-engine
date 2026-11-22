# Flash Next shared cold weights on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1, Rust
1.99.1, MLX and MLX Metal 0.32.4. The [same executable as the M2
functional control](../flash-next-shared-cold-weights-m2-20261122/README.md)
runs two native Flash Next constructors on a cold shared-weight cell.
The delivered view verifies all 49 MXFP4 publisher files and the
original derived manifest. Both sessions keep experts resident.

The real-pack test passes on the primary host. It checks that both
runners hold the same weight allocation, retain the attached
drafter, and match all eight greedy output IDs with MTP disabled.
All 44 system memory-pressure samples are normal. The separate
software tests cover concurrent loader invocation and retry after
failure. This receipt covers the optional shared SDK lifecycle and
one greedy request per runner.

The [native test output](test-output.txt.gz), [receipt](receipt.json),
and [comparison](comparison.json) bind the pinned executable and pack.
The 128.71-second test duration describes this run.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
