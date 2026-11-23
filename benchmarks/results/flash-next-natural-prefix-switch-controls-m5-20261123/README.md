# Flash Next document-prefix switch controls on M5 Max

A MacBook Pro M5 Max 128 GB with macOS 27.2.1 served the pinned resident
Flash Next MXFP4 pack from its internal SSD with Rust 1.99.1, MLX and
MLX Metal 0.32.4, two request slots, and MTP off. The controls use the
same signed binary as the [M5 repeat](../flash-next-natural-prefix-telemetry-m5-20261123/README.md).
The run verifies all 49 publisher members before and after serving.

Two fresh processes replay the same seven cases, first with both
extension and learning disabled and then with only learning requested.
Both configurations report zero physical prefix hits and match every
one of the 1,921 token IDs from the
[same-host cold oracle](../flash-next-variable-prefix-suffix-http-m5-20261123/README.md).
The replay checks the independent peer's progress, cancellation,
drainage, and host pressure. This verifies switch isolation for
the guarded workload.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
