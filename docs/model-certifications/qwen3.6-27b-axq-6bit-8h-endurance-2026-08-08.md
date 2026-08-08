# Qwen 3.6 27B AXQ 6-bit: 8-Hour Endurance Report

Status: **completed short-duration endurance evidence; not a 72-hour qualification**

Test date: **2026-08-07 to 2026-08-08**

## Technical Summary

AX Engine 6.13.5 served the pinned Qwen 3.6 27B AXQ 6-bit model continuously
for an **8-hour endurance test with 8.87 hours of actual measured runtime** on
a 64 GB M4 Pro Mac mini. The same owned server process handled **437/437
measured requests successfully**, with no restart,
client error, health failure, lifecycle drain timeout, observation gap, swap,
OOM, panic, or thermal/performance warning.

The four-hour baseline completed with `ready` quality. In the complete 4–8 hour
comparison window, same-shape p95 TTFT and p05 decode/effective-prefill rates
remained within every configured guardrail. At the final quiescent sample,
server RSS was 45.7 MiB below the baseline median and its post-baseline slope
was -2.49 MiB/hour; MLX active memory was flat.

The operator deliberately interrupted the run after 8.87 hours to strengthen
observability before restarting the full 72-hour qualification. The completed
interval is therefore useful short-duration evidence, but its terminal verdict
is correctly `watch`, not `pass`: it cannot establish 72-hour stability or rule
out a leak that becomes visible only after tens of hours.

## The Run Was Continuous and Error-Free Until the Planned Stop

The measured interval began only after the full warm-up and logical-KV capacity
rehearsal passed. It ran from `2026-08-07T20:02:03Z` to
`2026-08-08T04:54:22Z`, for 31,939.13 seconds. The resource sampler retained
532 samples from `20:02:03Z` through `04:53:40Z` with no continuity concern.

| Evidence | Result |
| --- | ---: |
| Measured duration | 8.87 hours |
| Successful / attempted requests | 437 / 437 |
| Baseline / post-baseline requests | 198 / 239 |
| Short / medium / shared-prefix / long requests | 306 / 66 / 43 / 22 |
| Client error rate | 0.0000% |
| Health failures | 0 |
| Lifecycle drain timeouts / inconclusive drains | 0 / 0 |
| Missing native KV reports | 0 |
| Resource samples / continuity gaps | 532 / 0 |
| Server restarts | 0 |

The final `interrupted/watch` status records the operator stop. It does not
represent a server or model failure. The runner converted the termination
signal into an immutable final checkpoint, then shut down its owned server;
neither process survived cleanup.

## Serving Performance Stayed Inside the Locked Guardrails

The table compares the four-hour baseline with the complete 4–8 hour window.
It uses the same prompt shape on both sides. The locked limits were p95 TTFT no
higher than 1.5× baseline and p05 decode/effective-prefill no lower than 0.75×
baseline. Every shape had at least the required eight samples in both windows.

| Shape | Samples, baseline → 4–8 h | p95 TTFT, ms | p05 decode, tok/s | p05 effective prefill, tok/s |
| --- | ---: | ---: | ---: | ---: |
| Short unique | 139 → 138 | 1,918.00 → 1,918.44 | 11.79 → 11.77 | 82.90 → 82.88 |
| Medium unique | 30 → 29 | 9,281.51 → 9,075.38 | 12.01 → 12.18 | 117.73 → 120.32 |
| Shared prefix | 19 → 20 | 9,887.84 → 9,886.80 | 12.18 → 12.28 | 120.56 → 120.57 |
| Long unique | 10 → 10 | 34,863.18 → 34,852.69 | 11.75 → 11.76 | 122.84 → 122.85 |

These are low-rate, concurrency-one serving results. They demonstrate stability
for the measured workload, not maximum throughput or concurrent-load capacity.

## Retained Memory Did Not Show an Eight-Hour Leak Signal

Memory comparisons use lifecycle-drained samples so request-phase Metal pinning
is not mistaken for retained growth. A warm KV or allocator cache is allowed to
change within its bounded capacity; the relevant signal is persistent
post-baseline growth correlated across process, MLX, model-KV, compressor, and
swap measurements.

| Quiescent series at the final checkpoint | Current | Growth from baseline | Post-baseline slope |
| --- | ---: | ---: | ---: |
| Server RSS | 20,159.3 MiB | -45.7 MiB | -2.49 MiB/h |
| Host wired | 2,309.1 MiB | +0.4 MiB | -99.95 MiB/h |
| Host compressor | 85.8 MiB | -0.9 MiB | -0.08 MiB/h |
| MLX active | 20,571.8 MiB | 0.0 MiB | 0.00 MiB/h |
| MLX reusable cache | 227.5 MiB | +64.0 MiB | +0.92 MiB/h |
| Model logical KV | 76.1 MiB | +60.1 MiB | +0.89 MiB/h |
| Model physical KV | 226.8 MiB | +64.0 MiB | +0.85 MiB/h |

Swap remained unused, and macOS reported no thermal or performance warning.
The bounded MLX/KV cache movement did not correlate with process RSS growth.
This supports an eight-hour no-leak observation; it does not prove that a
slower multi-day leak is absent.

## KV Reclamation Passed; Prefix Reuse Needs Layered Reporting

The post-warm-up rehearsal deliberately filled the 1,024-block logical pool
until a fresh 4,281-token request required reclamation. The fresh long probe
completed with 0.70 prefill steps per 1,000 prompt tokens at 96.8% KV
utilization; the medium probe completed with 0.92 steps per 1,000 tokens. Both
were far below the 64/1,000 guardrail and did not reproduce the former
token-by-token prefill failure.

All 43 shared-prefix requests completed. The final partial window recorded four
engine retained-cache hits, four prefix-reused requests, and 4,288 reused
tokens. The lower MLX physical-snapshot cache independently reported stores and
misses but zero physical-cache hits. These are different cache layers: the
result demonstrates engine-core exact-prompt reuse, not an MLX physical-cache
hit. Future checkpoints must show both layers explicitly.

## Scope and Reproducible Test Identity

| Item | Locked value |
| --- | --- |
| Host | `df-macmini-03`; Mac mini `Mac16,11`; Apple M4 Pro; 64 GB unified memory |
| OS | macOS 26.6, build 25G72 |
| Server | `ax-engine-server 6.13.5` |
| Server SHA-256 | `aa2ae3beae66f47e5b776481d743c0f1a338eefceed5bf6333b1ee706fd6f308` |
| Source provenance | v6.13.5 / `93b89773`, with recorded workspace changes |
| Model | `AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP` |
| Model revision | `8c37715c7b5ebca00eda6f73be47116a3e4ebc` |
| Model payload | 6 safetensors; 20,836,337,632 bytes |
| MLX | 0.32.0 |
| Server topology | Native MLX; one model; one process; concurrency 1; no restart |
| Logical KV | 1,024 blocks × 16 tokens; maximum batch tokens 2,048 |
| Workload cycle | 14 short unique, 3 medium unique, 2 shared-prefix, 1 long unique |
| Pacing / sampling | 60 seconds after each completion / 60 seconds |
| Baseline / checkpoints | 4 hours / every 4 hours |

The build was versioned as 6.13.5 but included recorded workspace changes, so
this report does not claim a byte-for-byte clean release-tag artifact. The
AXQuant artifact also retained the previously documented doctor/lineage
limitation; serving success does not resolve artifact-integrity certification.

## Why the Full 72-Hour Run Will Be Restarted

vLLM reported a host-memory problem that appeared only during multi-day
continuous testing: process working-set memory grew linearly from about 721 GiB
to 800 GiB over tens of hours while GPU metrics remained normal. The cause was
an internal list appended on every KV block allocation but drained only for a
different model condition. Time-based RSS monitoring detected the symptom;
request/KV-allocation-correlated bounded-state telemetry was needed to isolate
the producer/consumer mismatch. See the
[vLLM production validation report](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-07-23-glm-5.2-nvfp4-b300-pd.md#71-a-problem-visible-only-during-long-running-stability-tests).

AX's existing runner would detect a similar material RSS slope, but this pilot
showed that the next qualification should also explain *what grows*. The new
72-hour design will therefore add:

1. request-, token-, engine-step-, KV-allocation-, free-, and eviction-normalized
   retained-memory slopes;
2. explicit gauges for bounded internal state, including live requests,
   retained terminal snapshots/count/bytes, request-owner retention, logical KV
   tables, cached/free blocks, and cache entries;
3. trends for host-resident, MLX active/cache, attributed and unattributed
   memory rather than retaining some of them only as raw samples;
4. periodic macOS process-footprint and VM-region summaries for root-cause
   evidence without enabling high-overhead allocation tracing during the test;
5. separate engine prefix-reuse and MLX physical-cache evidence;
6. an early warning when growth is sustained and monotonic, plus the existing
   material-growth failure gate.

The next measured interval starts from zero after the revised binary passes
unit tests, the same cache-capacity preflight, and a bounded pilot. None of this
8.87-hour interval will be added to the new 72-hour duration.

## Evidence Integrity and Limitations

The source artifacts remain on the test host. Their SHA-256 receipts are:

| Artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `806c160ed84fac0d2506c7e235ae757b739538c9d97d4b2cd8876201fc1fd72b` |
| `events.jsonl` | `eec7ccd50118657201014d4ca9f745015778ffda81c523c7eb18f59684ce1a56` |
| `summary.json` | `9dbbc1174421a7311f2c44347647164ce177278fb78a71a8b2cb1769bcf34da7` |
| `server.log` | `79b375871fdcd350241301e13d97a15f5713e70a1560a579537598e197ab9f8a` |
| Final checkpoint JSON | `973c751d08a63357f4017c6c69152ac6e99ad301654b546de0297bfe892ab48b` |
| Final checkpoint Markdown | `664733c25ab185a9c5ef7fcf00a0a9291b076b54f6edb7cb7bbe594f791b4963` |

This report makes descriptive claims about one host, binary, model artifact,
and low-rate workload. It does not establish high-concurrency behavior,
maximum throughput, all-model stability, clean artifact lineage, or 72-hour
production qualification.

## Next Decision

The eight-hour evidence is accepted as a successful observability pilot. It
does not change the AXQ model's candidate certification status. Promotion
remains blocked until the redesigned test completes a fresh 72-hour interval
and the independent artifact-integrity and quality gates are satisfied.

Further questions for the redesigned run are whether retained bytes correlate
with requests, tokens, engine steps, or KV churn; whether every internal
retention gauge reaches a bounded plateau; and whether agentic-style long-lived
request/session mixes require a separate follow-on workload after the
concurrency-one endurance qualification.
