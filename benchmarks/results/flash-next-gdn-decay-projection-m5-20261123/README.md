# Flash Next GDN decay projection witness

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, MLX 0.32.4.
Both signed runs use Rust and Cargo 1.99.1, the `release-server` profile
and the pinned 49-member Flash Next MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.

The native scheduler runs reproduce all 128 archived HTTP IDs, four
saved logit rows, and both prior recurrence receipts. The layer-zero
attention input to the decay projection matches bitwise at all six
sampled positions, including 494 and 1040, where the final decay differs.
All stage comparisons cover the 2,047-token common prefix for the
physical partitions `512,512,512,511` and `2047`.

The first recorded difference is at the raw decay projection. It
differs at zero-based positions 494, 1040 and 1544. The FP32 bias add
rounds away the difference at 1544. At 494, the first affected head's
projection delta is 0.0078125 and its final decay delta is approximately
0.001858. At 1040, two heads differ in the final decay. The downstream
softplus and exponential stages propagate the projection differences.

All 154 pressure samples are normal. Peak sampled footprints are
76.26 and 86.51 GiB, within the 96 GiB guard. This is a layer-zero
attribution check. A shape-stable projection needs its own component
and full-model checks before the wider profile can be reconsidered.

`python3 -B replay.py` verifies all stage and input witnesses against
the sealed run and regenerates `comparison.json`.
