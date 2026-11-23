# Flash Next variable-suffix prefix reuse on M2 Ultra

Mac Studio M2 Ultra 192 GB ran the pinned resident Flash Next MXFP4
pack with ordinary tensor files on external SSD, macOS 27.2.1, Rust
1.99.1, and MLX and MLX Metal 0.32.4. Both arms use the same signed
`release-server` executable and verify all 49 publisher members before
and after execution.

The switch-on arm reuses a separately seeded 2,048-token prefix for
suffixes of 1, 65, 512, 1,024 and 2,048 tokens. One target disconnects
on its first output and an independent replacement completes. Each hit
reuses 2,048 tokens and retains the original scheduler item, with its
peer making progress on the same step. The larger 2,049-token suffix
and the unseeded control take the cold route. Both processes match
1,922 generated IDs and all worker counters drain. All 527 sampled
pressure readings are normal.

The fixture reuses the tested suffix boundaries from the
[native state comparison](../flash-next-prefix-suffix-matrix-m2-20261123/README.md).
The source patch builds on the earlier [plain-layout 64-token experiment](../flash-next-fixed-prefix-plain-layout-m2-20261123/README.md).
Primary-tier performance and product admission use separate gates.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
