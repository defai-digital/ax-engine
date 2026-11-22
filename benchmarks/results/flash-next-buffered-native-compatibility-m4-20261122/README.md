# Flash Next aligned cache-policy compatibility on M4 Pro

Mac mini M4 Pro 64 GB, internal SSD, macOS 27.0. The Rust 1.99.1
test binaries use source `23c5c8cb8`. The remote run verifies both
executable checksums and the pinned MLX 0.32.4 library and Metal
payload. Pressure reports normal before and after the tests.

Both FFI controls pass with the buffered and no-cache routes. They
compare the unaligned packed tensors, normalized boolean data and
empty arrays, and check short payloads and unrepresentable ranges.
The two Flash Next controls verify identical selected tensors across
the mapped and both aligned readers, and restrict the experiment to
an explicitly requested, resident MXFP4 trunk. The no-cache route
wins when both experimental switches are set. The fastpath inventory
test passes with both switches classified.

This is a software compatibility check on the supported tier. It
contains no full-model performance measurement.

```sh
shasum -a 256 -c SHA256SUMS
```
