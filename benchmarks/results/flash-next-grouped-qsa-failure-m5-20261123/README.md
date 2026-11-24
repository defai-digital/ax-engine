# Flash Next grouped QSA compile-fault isolation on M5 Max

A signed release-server-profile test on MacBook Pro M5 Max 128 GB,
internal SSD, macOS 27.2.1, Rust 1.99.1, and MLX 0.32.4 uses the
pinned resident 49-member Flash Next MXFP4 pack. The test uses the
production text-prefill boundary with the experimental grouped route
enabled. The fault hook is compiled only in tests.

After committing 2,048 prompt tokens, the test starts a 2,047-token
suffix. It completes one full pair of attention tiles and the first
tile of the next pair. An invalid Metal symbol in a lazy kernel on
the second tile raises the grouped-prefill evaluation error. The
committed cursor stays at 2,048 and the serialized cache is bitwise
identical to the original checkpoint. The diagnostic retry reaches
4,095, with all logits and full serialized state equal to the control.

All 125 sampled pressure levels are normal. Peak sampled physical
footprint is 81.20 GiB, within the 96 GiB guard. The executable, source
patch, exact token input and all pack member digests are bound by the
archived receipt. The included log projection preserves the failed
stage and Metal compiler marker from the original hashed log.

The fault is a pre-dispatch Metal compilation error. Failed GPU
command buffers may poison the encoder and dependent events.
The release server retires a failed model worker on an unclassified
MLX error. This control supplies commitment and compile-fault evidence.
The experimental default and qualification board are unchanged.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
