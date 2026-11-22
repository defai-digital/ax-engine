# Flash Next persistent PLE workers over native HTTP

The same release-server build from `566cac1d6` ran with scoped
and pooled PLE reads on a Mac Studio M2 Ultra with 192 GB, an external
SSD and MLX and MLX Metal 0.32.4. Both processes used the pinned
MXFP4 pack, its ordinary default serving route, 32 configured read
threads, one admitted request at a time and the advertised 32,768-token
context. The publisher inventory was verified before and after the run.

The worker probe queries the running process at the end of each arm.
The scoped process has zero named pool workers and the pooled process
has 31. All 415 sampled pressure readings are normal.
Each arm serves one frozen greedy trajectory and five 512-token
requests, with two warmups. The [comparison](comparison.json) binds
the worker counts, all six prompt digests, all 768 generated token
IDs and all six finish reasons in each arm. Every completed request uses zero MTP verifier steps
and reports zero route errors.

The [previous native staging comparison](../flash-next-ple-reader-pool-m2-20261122/README.md)
includes the experimental source patch and the TensorFold and MLX
references. This HTTP check is functional evidence. The M5 follow-up
uses one conditioning process and an off/on/on/off order, with fixed
512-, 2,048- and 8,192-token cells. It records per-arm CPU idle, pressure,
readiness, TTFT and decode throughput.

The [build](build.json), [scoped arm](arm-1.json.gz), [pooled arm](arm-2.json.gz)
and [terminal receipt](result.json) can be replayed from this directory.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py
```
