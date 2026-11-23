# Flash Next exact-prefix reuse with a live peer on M5 Max

Four processes on MacBook Pro M5 Max 128 GB ran the same signed
guarded server and pinned resident MXFP4 pack on internal SSD with
macOS 27.2.1 and MLX 0.32.4. The two-slot profile, scheduler budget,
prompts, and greedy generation limits stay fixed while the in-memory
prefix cache is disabled or enabled. All 49 publisher members were
verified at both ends of the run.

The [initial attempt](../flash-next-live-prefix-initial-m5-20261123/README.md)
stopped on an intermediate SSE frame. The new capture checks physical
cache-hit and reused-token counters around each pair. In both warm
processes a complete exact hit and a hit interrupted on its first
token each add one hit and 2,048 reused tokens, with one scheduled
token per active request on the first shared step. The independent
decoding peers report zero reused tokens. The replacement reuses
its own complete prompt. Every strict 2,112-token extension remains
cold and runs distinct 2,048- and 64-token prefill items.

All four processes match 867 output IDs each, including the first
token observed before each interruption. Their work counters drain.
All 612 memory-pressure samples are normal. Peak physical footprint
is 72.88 GiB.

| Process order | CPU idle before / after | Exact target first token | Peer maximum gap |
| --- | ---: | ---: | ---: |
| Cold 1 | 92.30% / 74.66% | 1.715 s | 1.694 s |
| Warm 1 | 91.17% / 92.18% | 0.070 s | 0.051 s |
| Warm 2 | 91.44% / 87.61% | 0.071 s | 0.050 s |
| Cold 2 | 91.40% / 85.79% | 1.601 s | 1.581 s |

Both comparison orders miss the host timing rule. The values above
retain the original diagnostics. The result supports functional
exact-prefix reuse under the guarded two-slot experiment. Strict
extension reuse and product concurrency throughput await their gates.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
