# Flash Next variable-suffix prefix reuse on M5 Max

A MacBook Pro M5 Max 128 GB served the pinned Flash Next MXFP4 pack from
its internal SSD on macOS 27.2.1. The four fresh processes ran in
off, on, on, off order with the same signed release-server executable,
Rust 1.99.1, and MLX and MLX Metal 0.32.4. The preflight and final
receipts cover all 49 publisher members. The
[M2 Ultra receipt](../flash-next-variable-prefix-suffix-http-m2-20261123/README.md)
binds the same executable and source patch.

Both enabled arms record exactly five 2,048-token physical hits for
suffixes of 1, 65, 512, 1,024 and 2,048 tokens. The unseeded and
2,049-token controls take the cold route. Each matching target keeps
its scheduler item and its peer progresses on the same step. The
cancellation and independent replacement controls drain successfully.
All four processes match exactly 1,922 generated IDs. All 742 sampled
host-pressure readings are normal.

The first disabled process ended at 75.18% CPU idle against the 90%
floor. The overall crossover is ineligible for a primary-tier speed
claim. The receipts retain timing observations for the next controlled
repeat. This diagnostic does not change product qualification.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
