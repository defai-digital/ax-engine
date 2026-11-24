# Flash Next shared gather indices on M2 Ultra

This synthetic Metal test ran on Apple M2 Ultra 192 GB, macOS 27.2.1,
with MLX and MLX Metal 0.32.4. The axes use the native MXFP4 pack's
two K/V heads and 256 features. A trial evaluates both BF16 gathers
with padded, repeated and out-of-order indices. The separate arm
constructs one integer tile for each take. The shared arm passes the
same array to both takes. Both gather arrays matched bit for bit in
each geometry.

| Query tile width | Logical context | Padded selection | Median peak increment, separate | Median peak increment, shared |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 4,096 | 2,051 | 194.59 MiB | 193.44 MiB |
| 128 | 8,192 | 2,051 | 518.81 MiB | 515.78 MiB |
| 255 | 16,384 | 2,051 | 1,033.50 MiB | 1,027.50 MiB |

There are eight pairs per geometry with the order reversed in
alternation. Each measured iteration synchronizes the device and
clears its allocation cache first. The live allocation before each
trial in a geometry was identical. Timings move in both directions
across the three geometries. The initial exploratory result reused
allocator state between iterations and has different peak spread.

The isolated Rust branch keeps the gathered MLX SDPA reduction
unchanged. Its 53 focused QSA tests pass, including F32/BF16 checks
with real negative padding, distinct K and V and evaluation of V
before K. The selected and dense paths keep their route eligibility.
Primary-tier full-model decisions require M5 serving measurements
and an exact real-pack comparison of the long-prompt path.

The [probe](probe.py), [controlled samples](result.json) and
[exploratory samples](allocator-uncontrolled.json) retain all
measurements. Run the probe with Python, NumPy and the pinned MLX
wheels. Verify raw inputs with `shasum -a 256 -c SHA256SUMS`.
