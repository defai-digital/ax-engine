# Flash Next simultaneous cold shared-weight startup

Two native runners start from the same empty MLX shared-weight cell
on a Mac Studio M2 Ultra 192 GB with an external APFS SSD. Both use the
pinned Flash Next MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`, Rust 1.99.1 and MLX
0.32.4. The launcher verifies the 49 publisher files and the native
manifest before the functional test.

Both runners keep experts resident and serve an independent greedy
request with MTP disabled. They assert pointer identity of the held
weight allocations and match all eight output IDs. Every sampled
pressure level is normal. The focused tests separately hold the first
load open while the second caller arrives, and verify that a failed
first load leaves the cell available for retry.

This control covers the optional shared SDK weight lifecycle. It
provides no M5 timing, sustained load or two-slot qualification result.
The same executable is scheduled on the primary hardware.

Run `python3 -B replay.py` for the sealed test record and output
verification. Optionally pass `--source` with a worktree at the tested
commit to verify the precise runner and test source hashes.
