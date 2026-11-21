# Flash Next fresh-cache delivery on M5 Max

On MacBook Pro M5 Max 128 GB with an internal SSD and macOS 27.2.1,
the product CLI downloaded the pinned MXFP4 pack
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` into an empty cache in
1,032.336 seconds. The transfer used two workers and a 60-second
read timeout. All 49 publisher files (132,261,877,478 bytes) matched
their expected sizes and SHA-256 digests. The generated native
manifest reported ready. The transfer ran once.

| Native mode | Controls passed | Time to readiness | MTP verification steps |
| --- | ---: | ---: | ---: |
| Disabled | 7/7 | 76.428 s | 0 |
| Required | 7/7 | 79.438 s | 107 |
| Product default | 7/7 | 78.672 s | 0 |

The same `b36b7b287` release-server build with MLX 0.32.4 and
Rust 1.99.1 served all three modes. Required mode with
`AX_NO_SPEC=1` failed with its expected activation error. Both direct
modes left the drafter inactive. The file cache was uncontrolled.

The first verification rejected the downloader's native state
marker after checking the publisher files. Commit `030ab57c6`
accepts that marker when it is a local regular file and binds the
exact repository and revision. A subsequent control attempt failed
in the scratch launcher because the MLX library path was discarded
at `caffeinate`. The final run reapplied the path in the child
launch and repeated inventory verification and all native controls
on the original snapshot. The frozen delivery and delivered
readiness checks pass for this receipt. The timing comparison,
endurance and model certification remain separate.

The [original transfer](original-transfer.json),
[verified delivery](delivery.json), and
[all native controls](native-support.json.gz) retain the hashes,
per-request checks and readiness measurements.
