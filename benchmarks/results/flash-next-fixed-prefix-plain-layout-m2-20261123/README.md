# Flash Next fixed-prefix extension from a plain-weight model directory

Mac Studio M2 Ultra 192 GB ran the pinned Flash Next MXFP4 pack on
external SSD with macOS 27.2.1, Rust 1.99.1, and MLX and MLX Metal
0.32.4. All 26 tensor files use the ordinary filesystem layout, which
cannot provide a durable content address. The RAM checkpoint records
the existing unique identity of its model load. Both arms use the same
signed `release-server` executable and recheck all 49 publisher members
before and after the run. The archive carries the exact source patch
recorded by the isolated experiment commit `73b769080`.

The standalone seed uses a 2,048-token item at width 512. Each eligible
64-token suffix records one hit and 2,048 physically reused tokens,
including a disconnected request and its replacement. The independent
peer emits on both logical items for each target. The first target
output occurs in the suffix item, and every pair drains the worker and
stream counters. Unseeded input and the different-length holdout remain
cold. Switch-off records zero reuse for every pair.

Both processes match 1,346 generated IDs, including the first output
from the interrupted stream. All 366 pressure samples are normal. The
replay verifies the frozen inputs,
software build, raw seed, paired token arrays, shared scheduler steps,
pressure samples, and exact request metrics. The scope is functional
HTTP activation on M2. M5 performance and product qualification retain
their own acceptance evidence.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
