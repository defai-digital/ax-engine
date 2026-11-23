# Flash Next fixed-prefix compatibility on M4 Pro

A Mac mini M4 Pro with 64 GB unified memory and macOS 27.0 ran five
focused `release-server` library tests from the signed experiment
source `73b769080`, using Rust 1.99.1 and MLX and MLX Metal 0.32.4.
The tests cover exact model-load ownership, token-hash collision
rejection, pressure-based reclamation, retention boundaries, and
numerics switch registration. Each test passes. The pinned runtime,
source patch, and test executable hashes are in `build.json`.

The product's primary resident Flash Next serving evidence uses the
[M5 Max crossover](../flash-next-fixed-prefix-extension-m5-20261123/README.md).

```sh
shasum -a 256 -c SHA256SUMS
```
