# Flash Next mixed serving on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, MLX and MLX
Metal 0.32.4. All six arms used the same Rust 1.99.1 release-server
executable from `cdc7ce2e9` and the pinned resident MXFP4
MTP pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. Drafting, n-gram acceleration,
direct coalescing and prefix reuse were disabled for this comparison.
The executable and all four physical holdouts are bound in the archive.

Each arm replayed 220 measured arrivals at a mean offered rate of
0.25 requests per second, with 132 prompts of 512 tokens, 66 of 2,048
and 22 of 8,192. The closed client lanes replayed the same offered
timeline. Two warmups per arm are excluded. All 1,320 measured replies
matched the original solo output of 96 token IDs each. Every arm drained
its request and stream counters. All 5,629 sampled system-pressure
readings were normal. The largest observed server footprint was
75.92 GiB.

| Run order | Arrival to first token p95 | Largest token gap | Gaps of at least 100 ms | Short-request gap tail | Requests with shared split |
| --- | ---: | ---: | ---: | ---: | ---: |
| Solo 1 | 19.081 s | 0.033 s | 0 | 23.5 ms | 0 |
| Two slots, split off 1 | 26.220 s | 3.371 s | 272 | 50.5 ms | 0 |
| Two slots, split on 1 | 20.841 s | 0.847 s | 420 | 349.1 ms | 57 |
| Two slots, split on 2 | 14.940 s | 0.858 s | 571 | 753.9 ms | 73 |
| Two slots, split off 2 | 13.529 s | 3.233 s | 259 | 47.8 ms | 0 |
| Solo 2 | 16.900 s | 0.031 s | 0 | 23.9 ms | 0 |

Arrival to first token includes the wait for a client lane. The replay
reports that wait and client-observed time to first token separately.
The short-request tail is the p95 across the 512-token requests' own
p95 inter-token gaps. A shared split requires the published logical
boundary and step-ID evidence of another request decoding.

Splitting reduced the largest gap in both orders and raised the short
request tail in both. The sign of the arrival-to-first-token p95 change
reversed with arm order. Matching requests individually identified 55
and 80 short requests, respectively, with a smaller maximum gap and a
higher p95. The two-slot comparison supports continued evaluation of a
selective policy. The supported product profile remains single-stream
resident.

## Replay

The [evidence archive](evidence.tar.xz) contains all six terminal
receipts, SSE step traces, memory samples, the frozen token corpus and
the exact measurement scripts. Only the client base URL and temporary
corpus path were removed from the client JSON. The original client
hashes and all archive member hashes are in [sources.json](sources.json).
The [comparison](comparison.json) was regenerated independently after
redaction and matched the replay of the original capture.

```sh
python3 benchmarks/results/flash-next-mxfp4-mixed-serving-m5-20261121/replay.py
```
