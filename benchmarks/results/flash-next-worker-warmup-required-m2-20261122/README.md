# Flash Next single-pass worker warmup with Required MTP

The [same experimental release-server binary and pinned MXFP4
snapshot](../flash-next-worker-warmup-once-m2-20261122/README.md) run
on Mac Studio M2 Ultra 192 GB, external APFS storage, macOS 27.2.1
and MLX 0.32.4. Both processes advertise the resident one-slot 32K
profile. Required MTP exercises the canonical verifier.

The worker warmup runs the same three shapes three times or once.
The trace confirms nine and three successful worker requests.
Each arm runs the same three HTTP warmup streams.

| Measurement | Three worker passes | One worker pass |
| --- | ---: | ---: |
| Worker warmup | 5.681 s | 2.778 s |
| HTTP warmup | 2.916 s | 2.905 s |
| Process start to HTTP readiness | 30.310 s | 343.589 s |
| Normal memory pressure samples | 240 | 542 |

The three-pass baseline runs first. The one-pass process spends
considerably longer in weight loading. The observed saving of 2.903 seconds is
scoped to the worker warmup. Primary-tier total readiness and warmed
serving have a separate counterbalanced comparison.

All 17 responses per arm have exact prompt digests, 2,176 generated
IDs, stop states, and per-request MTP reports. All requests exercise
the verifier, for 1,277 steps per arm. Both arms record zero route errors.
This is same-host Required MTP functional evidence.

The [comparison](comparison.json) replays the [raw HTTP and startup
observations](evidence.tar.xz) against the pinned executable and pack
provenance, with a fixed trial order.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
