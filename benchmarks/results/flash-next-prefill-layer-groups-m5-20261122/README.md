# Flash Next layer-group sweep on M5 Max

One Rust 1.99.1 release-server executable from experimental commit
`646827ee3` runs the 8-, 12- and 16-layer groups in ascending order.
The MacBook Pro uses an M5 Max, 128 GB unified memory, an internal
NVMe SSD and MLX and MLX Metal 0.32.4. The launcher verifies the same
pinned MXFP4 publisher inventory and delivered manifest as the
[preceding HTTP comparison](../flash-next-prefill-layer-http-m5-20261122/README.md).
All three arms use two request slots, resident experts, greedy direct
decoding and disabled prefix reuse.

The frozen early arrival overlaps a 512-token decode stream with an
8,192-token prefill. Every arm retains the 20 physical prefill widths
and matches 192 output IDs from the independent single-slot run.
All layer suspensions keep the long stream's prompt cursor fixed,
publish a real peer decode token on that step and resume before the
original physical boundary. Service counters drain in each arm.
All sampled memory pressure levels are normal, with a peak server
footprint of 74.59 GiB. Every arm has at least 90% CPU idle before and
after measurement.

| Layers per checkpoint | Suspended steps | Short p95 token gap | Long arrival to first token | Long first token before short completion | Total output throughput |
| ---: | ---: | ---: | ---: | :---: | ---: |
| 8 | 77 | 275.84 ms | 12.773 s | No | 12.495 tok/s |
| 12 | 48 | 326.91 ms | 12.324 s | Yes | 12.403 tok/s |
| 16 | 32 | 379.97 ms | 11.978 s | Yes | 12.437 tok/s |

The 16-layer arm gives the earliest long first token in this sweep
and raises the short-stream p95 gap by 104.12 ms from the eight-layer
arm. Total throughput spans 12.403–12.495 tok/s. The earlier eligible
physical-only control measured 748.40 ms for the short-stream p95 gap
and 11.241 s for the long first token, on a different executable.
The fixed ordering and single burst per group motivate a sustained
same-binary comparison with order reversal. The 16-layer setting is
the next candidate because it gave the earliest long first token and
restored overlap at long first-token emission in this probe.
Two-slot qualification remains open.

The [de-identified capture](evidence.json.gz) retains SSE step traces,
client timings and token IDs, memory samples, and the terminal receipt.
The [driver archive](driver.tar.xz) contains scripts, frozen burst
inputs, the verified inventory and the build receipt. The binary and
original unredacted solo output are referenced by digest in
[inputs.sha256](inputs.sha256). Replay uses the preserved single-slot
IDs in the [mixed-serving archive](../flash-next-mxfp4-mixed-serving-m5-20261121/README.md)
and reproduces [comparison.json](comparison.json).

```sh
python3 benchmarks/results/flash-next-prefill-layer-groups-m5-20261122/replay.py
```
