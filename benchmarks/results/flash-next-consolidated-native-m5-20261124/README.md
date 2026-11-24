# Flash Next consolidated native release controls on M5 Max

The signed `3fc0a02cd` native CLI and server ran the pinned MXFP4 pack
on MacBook Pro M5 Max 128 GB with internal SSD, macOS 27.2.1 and MLX
0.32.4. Rust 1.99.1 built the `release-server` profile. The ordinary
profile uses one resident request with up to 32,768 prompt and output
tokens combined. The default route recorded zero MTP verifier steps.

The product downloader verified all 49 files and 132,261,877,478
bytes from an empty cache. It took 1,099.559 seconds. The first delivered
launch reached HTTP readiness in 32.209 seconds. Default readiness was
30.113 seconds. The three delivery modes passed 21 native HTTP controls.
The later workloads reused the verified snapshot and signed executables
after repairing an empty-array expansion in the campaign launcher.
The delivery receipt retains the original uninterrupted transfer.

Both ordinary modes passed all 14 lifecycle actions with drained
counters. Each mode completed the 105-item QA cohort. All 210 response
fingerprints match the signed v7.9.0 control. Default QA peak server RSS
was 66.38 GiB, and the 29,774-token lookup took 46.462 seconds.

| Default input | Median first token | Median decode |
| ---: | ---: | ---: |
| 512 tokens | 0.600 s | 47.07 tok/s |
| 2,048 tokens | 1.975 s | 45.23 tok/s |
| 8,192 tokens | 11.702 s | 42.77 tok/s |

The timing matrix recorded 90.89% CPU idle before and 92.11% after.
All 31 output sequences match across each mode's two independent
processes. All four matrix process fingerprints match the signed
v7.9.0 native control. The modes also match on all nine measured
cells and all 16 recorded trajectories.

The frozen board passes delivery, lifecycle, memory, readiness,
long lookup and MTP policy. Integrity, independent quality and
reference-relative timing remain open. The initial native lifecycle
endpoint recorded 82.31% CPU idle against the 90% floor. The
[ordinary reference](../flash-next-ordinary-reference-m5-20261121/README.md)
has three QA rows and no timing matrix. Its
[quiet-entry retry](../flash-next-reference-quiet-entry-m5-20261124/README.md)
stopped during the first QA item. The direct-default certification
record remains Candidate. The [lifecycle repeat](lifecycle-repeat/README.md)
reproduced every action and recorded 90.63% CPU idle before and 88.34%
after. The repeated endpoint still falls short of the 90% floor.

The verifier regenerates the board from the raw results using the
frozen evaluator. The archive binds the delivery stream, 210 QA rows,
all four raw matrix process results, inventory, runtime digests and
campaign input files. It records hashes for the two delivery files
whose scratch directory was normalized for portability.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -E -s -B replay.py | cmp comparison.json -
```
