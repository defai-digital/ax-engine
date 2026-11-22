# Flash Next admitted sidecar reader on the M5 delivered snapshot

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1. The
`838b331a3` experiment binary uses Rust 1.99.1 and MLX 0.32.4. With
its explicit selector enabled it reads only the admitted MTP sidecar
ranges from their existing descriptor into owned MLX buffers. The
trunk uses the same guarded mmap route. The harness binds the same
publisher inventory, delivered manifest and native control order as
the [previous production delivery](../flash-next-mxfp4-current-delivery-m5-20261122/README.md).

| Mode | Previous production readiness | Candidate readiness | Existing limit | Candidate sidecar stage |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 122.196 s | 107.474 s | 300 s | 0.584 s |
| Required | 125.258 s | 109.679 s | 120 s | 0.604 s |
| Product default | 122.002 s | 108.904 s | 120 s | 0.603 s |

The candidate matches all 21 frozen request signatures and output
choices. Required records 107 verifier steps and both direct arms
record zero. The required kill-switch control rejects activation.
The original inventory and manifest checks pass before and after the
observations. All three candidate readiness samples meet the existing
limits in this repeat. The guarded trunk accounts for 98.444–99.525
seconds of the three observations.

The candidate uses the already delivered snapshot. The prior report
starts from an empty download cache. Page-cache conditions can differ
between the two published samples. Treat this result as readiness and
functional evidence for the isolated branch candidate. Warm serving on
this exact build and the current production release evaluation are
separate checks.

Replay the [raw native result](native-result.json.gz) and
[startup trace records](startup-traces.json) against the previous
production receipt and [comparison](comparison.json).

```bash
python3 benchmarks/results/flash-next-admitted-sidecar-delivered-m5-20261122/replay.py
```
