# Flash Next single-pass worker startup warmup

The experimental release-server build from `decb30fd5` runs the
pinned resident MXFP4 pack on Mac Studio M2 Ultra 192 GB, external
APFS storage, macOS 27.2.1 and MLX 0.32.4. Both arms verify all 49
publisher files before and after the run. Each advertises a one-slot,
32,768-token context and serves the ordinary MTP-off policy.

The opt-in keeps the three worker request shapes, reducing the
worker repetitions from three to one. Both arms perform all three
HTTP warmup streams. The startup trace records the number of
successful requests per worker shape separately from the elapsed
worker and HTTP phases.

| Scoped measurement | Three worker passes | One worker pass |
| --- | ---: | ---: |
| Successful worker requests | 9 | 3 |
| Worker warmup | 5.166 s | 2.415 s |
| HTTP warmup | 2.774 s | 2.746 s |
| Spawn to HTTP readiness | 30.321 s | 432.416 s |

The second process spends substantially longer in resident weight
loading. The timed worker phase still falls by 2.751 seconds.
These same-host functional arms match each other's 17 responses,
2,176 output IDs and stop reasons, with zero route errors or MTP
verification steps. All 841 pressure samples are normal. The M5
counterbalanced run measures readiness, first request and warmed
serving on primary hardware. The operator switch remains off by
default.

Four focused startup software checks also pass on Mac mini M4 Pro
64 GB, macOS 27.0, using the pinned MLX library. The [M4 receipt](m4.json)
binds the test executable and [actual output](m4-startup-tests.txt.gz).
The [M2 evidence archive](evidence.tar.xz) contains the original arms,
warmup traces and terminal receipt. Its [replay](replay.py) checks
the startup phases and exact response identity. A failed worker shape
is rejected even with a matching recomputed log checksum.

```sh
python3 -B benchmarks/results/flash-next-worker-warmup-once-m2-20261122/replay.py
```
