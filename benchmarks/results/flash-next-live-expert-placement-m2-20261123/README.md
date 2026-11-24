# Flash Next live expert placement startup control

The signed native server uses the pinned 49-member Flash Next MXFP4 pack on
Mac Studio M2 Ultra 192 GB, external USB SSD, macOS 27.2.1, Rust 1.99.1, and
MLX 0.32.4. Each arm starts a fresh server. Both query the ready model on
`GET /v1/memory` with the standard configured 32,768-token cap and one slot.

| Expert policy | Reported `expert_streaming_active` | Resident shard tensors | Normal pressure samples |
| --- | --- | ---: | ---: |
| Auto | `false` | 1,942 | 25 |
| Forced on | `true` | 1,654 | 7 |

The original wrapper exited 1 after both HTTP responses because it expected
the generic MoE expert-streaming log message. A separate postflight verified
the signed executable, rehashed all 49 pack members, and checked Flash Next's
own resident shard counts against the exact captured HTTP bodies and
pressure samples. The 15 shared shard lines report 288 fewer loaded tensors
when streaming is forced. The archive preserves the original wrapper exit,
the raw HTTP and pressure captures, and a projection of the startup log
that retains shard names and tensor counts. Sampled footprint covers
startup. This control makes no inference or qualification claim.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
