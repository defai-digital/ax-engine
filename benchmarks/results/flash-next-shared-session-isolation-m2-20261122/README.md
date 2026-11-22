# Flash Next request isolation with cold shared weights

Mac Studio M2 Ultra 192 GB, external APFS storage, macOS 27.2.1,
Rust 1.99.1 and MLX and MLX Metal 0.32.4. The same resident MXFP4
publisher pack as the [M5 shared-load
control](../flash-next-shared-cold-weights-m5-20261122/README.md) passes
verification of all 49 files before and after execution.

Two cold native runners use the same weight cell, different prompts
and the same request ID. Two fresh solo runners on the same cell
provide independent greedy baselines. Both concurrent trajectories
match their own eight solo token IDs. The two solo outputs differ,
confirming that the prompts distinguish their request state. All four
runners hold the same weights allocation and run with MTP disabled.
All seven memory-pressure samples are normal.

This test verifies request isolation on the shared-weight SDK path.
The [comparison](comparison.json) binds the compile-time source
snapshot, pack and [native test result](test-output.txt.gz). The exact tested tree is
commit `0e1a3acee`.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
