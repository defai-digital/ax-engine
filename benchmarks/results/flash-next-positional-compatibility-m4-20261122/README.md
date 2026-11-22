# Flash Next owned-storage compatibility controls on M4 Pro

Mac mini M4 Pro 64 GB, internal SSD, macOS 27.0. The test executable
comes from experiment `8729b80c2`, built with Rust 1.99.1. The helper
records the uploaded binary digests in [inputs.sha256](inputs.sha256)
and checks the pin against the local MLX 0.32.4 library before starting.
The recorded [MLX digest](mlx.sha256) matched the build environment's
library byte for byte. Host memory pressure was normal at preflight.

All 175 `mlx-sys` tests pass, including mixed unaligned U32 and BF16
safetensors reads and rejection of a partial positional read. A separate
C ABI check confirms construction of an empty float array and
normalization of noncanonical boolean bytes. Both processes exit zero.

These are software compatibility checks on the supported hardware tier.
Full-pack latency and readiness are measured on the M5 standard host.

[Rust tests](rust.log.gz), [C ABI output](abi.log), and [phase status](phase-status.txt).
