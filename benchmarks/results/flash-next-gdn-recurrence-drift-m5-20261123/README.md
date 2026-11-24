# Flash Next GDN decay and recurrence replay

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, MLX 0.32.4.
The signed diagnostics use Rust and Cargo 1.99.1, the `release-server`
profile and the pinned 49-member MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.

Both native scheduler runs reproduce their archived 64-token HTTP
completions. All saved layer state hashes, four decision-logit rows and
22 sampled GDN core rows match the earlier controls. The precision
comparison covers every layer-zero row in the common 2,047-token
prefix. The physical partitions are `512,512,512,511` and `2047`.

- Normalized Q/K, V and beta match bitwise at all 2,047 positions.
  Actual decay differs at zero-based positions 494 and 1040.
- The original recurrence outputs first differ at 494 and differ at
  1,030 positions in total. The final recurrent states also differ.
- For each input set, the default MLX recurrence reproduces that arm's
  entire original output and final state bitwise. A single call and
  four calls give exact results with lazy and explicitly evaluated
  boundaries. The retained input sets reproduce the between-arm
  difference even with the same partition.

All 174 memory-pressure samples are normal. Peak sampled footprints
are 76.27 GiB for the 512-row arm and 86.45 GiB for the wider arm,
within the 96 GiB guard. This is an attribution measurement. Its
scope is one prefix and layer zero. The decay branch remains the next
variable under investigation and the 512-row default continues to apply.

From this directory, `python3 -B replay.py` regenerates
`comparison.json` and verifies the sealed inputs, the earlier
HTTP and operator controls, and the bitwise observations.
