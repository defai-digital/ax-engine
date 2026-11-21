# Flash Next eight-hour endurance on M5 Max

The pinned MXFP4 Flash Next pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`
served **452 successful requests** in
28,844.016 measured seconds on
MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1.
The native server used the `b36b7b287`
release-server build and MLX 0.32.4.
One process served the run and exited normally. The launch used one
request slot, 32,768 context tokens, a 2,048-token scheduling budget,
expert streaming Auto, and MTP disabled.

| Workload | Four-hour baseline | Full run |
| --- | ---: | ---: |
| Short unique | 159 | 317 |
| Medium unique | 34 | 68 |
| Shared prefix | 22 | 45 |
| Long unique | 11 | 22 |

The native streams, health checks and post-request lifecycle drains
passed throughout. Both capacity-reclamation probes passed.
The measured route recorded zero MTP requests and verifier steps.
The runner finalized a passing verdict with zero recorded memory,
performance or observation-continuity alerts.

The 225-request final evaluation window compares with the four-hour
baseline by workload. The values below are final-to-baseline ratios.

| Workload | TTFT p95 | Decode throughput p05 | Prefill throughput p05 |
| --- | ---: | ---: | ---: |
| Short unique | 0.976 | 1.005 | 1.030 |
| Medium unique | 0.975 | 1.026 | 1.024 |
| Shared prefix | 0.958 | 0.992 | 1.044 |
| Long unique | 0.988 | 1.001 | 1.011 |

The recorded limits are 1.50× for TTFT and 0.75× for each lower-tail
throughput measure.

The 480 resource samples span the measured interval.
The largest wall-clock gap was 60.166 seconds.
Observed process RSS was 71.06–76.60 GiB, and system swap
use was 3.22–6.10 GiB. These are sampled process RSS
and system swap observations. The workload issues one bounded request
at a time and leaves roughly one minute idle after each request.
The same-host independent comparison and sustained multi-user load
use their own validation suites.

## Evidence and replay

[The compressed receipt](endurance.json.gz) includes all portable
observations, the terminal summary, and SHA-256 digests of the original
artifacts. The detached runner recorded its own SHA-256, which matches
the committed runner at `f74faaf1a`. Its Git
checkout fields were unavailable in this staged execution and the
independent audit records that condition. The recorded executable SHA
matches the same M5 build used for the earlier native QA. Pack identity
checks bind the model manifest, tokenizer, and config metadata.

Use the pinned server and pack with
`scripts/run_axq_endurance.py` at `f74faaf1a`.
Set the model ID to `qwen3.8-flash-next:axq`, duration to eight hours,
baseline to four hours, and request and resource intervals to 60 seconds.
Use 2,048 KV blocks of 16 tokens, 2,048 batch tokens, and one slot.
The receipt includes the remaining runner policy and launch metadata.
