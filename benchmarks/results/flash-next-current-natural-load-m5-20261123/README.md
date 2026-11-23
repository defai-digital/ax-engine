# Flash Next guarded natural load on M5 Max

The MacBook Pro M5 Max 128 GB uses internal NVMe storage, macOS 27.2.1,
MLX and MLX Metal 0.32.4, and Rust and Cargo 1.99.1. Both arms use the same
signed release-server binary, the pinned resident AXQ MXFP4 pack
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`, mapped startup, and MTP off.
The archive binds the signed binary digest, runtime patch and candidate
software checks.

Each arm replays 220 arrivals at 0.25 requests/s. The frozen mix contains
132 prompts of 512 tokens, 66 of 2,048, and 22 of 8,192. Each request
generates 96 greedy tokens. The one-slot arm uses one client dispatch
lane. The guarded two-slot arm uses two dispatch lanes and the bounded
2,048/512-token prefill profile for eligible pairs. Both client TTFT from
dispatch and first-token time from scheduled arrival are retained.

| Order | Arm | CPU idle before / after | Timing eligible | Output IDs exact |
| ---: | --- | ---: | --- | ---: |
| 1 | Single first | 90.52% / 91.40% | Yes | 21,120 |
| 2 | Paired first | 93.27% / 92.54% | Yes | 21,120 |
| 3 | Paired second | 95.93% / 95.77% | Yes | 21,120 |
| 4 | Single second | 93.24% / 89.27% | No | 21,120 |

The first pair passes the 90% pre-run and post-run CPU idle requirement.
The second pair supplies functional evidence because its final single-slot
sample misses the timing floor. Every output ID matches the frozen
single-request oracle, all five work counters drain, and all 4,159 memory
pressure samples are normal. Peak footprint across four arms is 75.83 GiB.
Both guarded arms include actual 8,192-token and short-peer shared steps.

| First eligible pair | One slot | Guarded two slots |
| --- | ---: | ---: |
| Delivered tokens/s | 23.16 | 22.46 |
| 512-token dispatch TTFT p95 | 0.530 s | 12.598 s |
| 512-token arrival-to-first-token p95 | 34.677 s | 57.687 s |
| 512-token per-request p95 gap, p95 across requests | 24.444 ms | 60.416 ms |
| 2,048-token dispatch TTFT p95 | 2.373 s | 3.177 s |
| 8,192-token dispatch TTFT p95 | 11.640 s | 12.595 s |

The measured workload delivers 3.0% fewer tokens/s with the guarded
profile and increases short-request tail latency. The product profile
remains one resident request. The separately controlled short-first
prefill experiment audits whether a bounded change in scheduling order
can reduce this delay while preserving long-request progress.

The archive retains the measured streaming rows, step traces, pressure
samples, receipts, frozen inputs, replay, and original capture digests.
The stream JSON redactions replace only the loopback endpoint and the
absolute corpus path.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
