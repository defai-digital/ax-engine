# Flash Next exact prefix reuse alongside an active stream

The first pair on MacBook Pro M5 Max 128 GB uses internal SSD, macOS
27.2.1, MLX 0.32.4 and the same [signed guarded server](../flash-next-openai-lifecycle-repeat-m5-20261123/README.md)
in both processes. The resident MXFP4 pack and two-slot scheduler stay
fixed. Both arms seed two disjoint 2,048-token prompts. A target repeats
one seed while an independent 512-token peer generates 192 tokens.

| Observation | Cache disabled | In-memory L1 |
| --- | ---: | ---: |
| Target first-token latency | 1.601 s | 0.070 s |
| Peer maximum inter-token gap | 1.580 s | 0.050 s |
| Pair wall time | 8.051 s | 6.567 s |
| Scheduled tokens on first shared output step | 2,049 | 2 |

The completed target and peer match all 288 output IDs. The warm target
reports a request-owned L1 restore of 2,048 tokens in its terminal
route. Its first shared output step schedules one token per stream.
The next case disconnects on that first output. Its warm step also
schedules two tokens, and the first SSE frame contains no cache-route
event. The original comparator requires that event and terminates
with status 1 before the warm arm's closing CPU sample and publisher
recheck. The table is a diagnostic observation from this incomplete
pair. The corrected rerun adds before-and-after physical reuse counters
to the cancellation check while retaining the shared step evidence.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
