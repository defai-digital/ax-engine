# Flash Next aligned native reader compatibility on M4 Pro

Mac mini M4 Pro 64 GB, internal SSD, macOS 27.0. The two uploaded
Rust unit-test binaries were built from `31e2b6e1f` with Rust 1.99.1.
The launcher verifies their digests and the pinned MLX and MLX Metal
0.32.4 library and Metal payload before running either binary.

Both `mlx-sys` tests pass. They cover an unaligned packed tensor,
boolean normalization, an empty tensor, premature EOF, and overflow
in the native shape and offset checks. Both `ax-engine-mlx` routing
tests pass. They cover the resident MXFP4 opt-in and matching loaded
tensor selection and values across the two readers. Memory pressure is
normal before and after the controls.

This is a supported-tier software compatibility check. Full Flash Next
serving and the startup decision use the primary M5 hardware.

```sh
shasum -a 256 -c SHA256SUMS
```
