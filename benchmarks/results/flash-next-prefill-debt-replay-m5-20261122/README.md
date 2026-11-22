# Flash Next fixed-slice decode repayment replay

The [M5 Max 128 GB step traces](../flash-next-mxfp4-prefill-step-cost-m5-20261121/README.md)
capture a 512-token request producing 96 tokens while an 8,192-token
peer prefills. The peer ran four logical 2,048-token items, each using
physical forwards of 512, 512, 512, 511 and 1 tokens. In the early
decoder trace, the 95 post-first-token runner steps have a p95 cost of
736.0 ms and a maximum of 747.4 ms, with 16 steps taking at least 100 ms.

The replay retains the measured cost of each of the 20 mixed steps and
charges a fraction of that cost as decode time to repay before admitting
the next peer prefill. It uses the captured median 20.388 ms single-decoder
cost for repayment and the median 42.890 ms two-decoder cost once the peer
can decode. The JSON also varies those costs at p10 and p90.

| Repayment share | Short runner step p95 | Steps at least 100 ms | Peer slices left at short completion |
| ---: | ---: | ---: | ---: |
| 0% | 736.0 ms | 16 | 0/20 |
| 25% | 729.0 ms | 10 | 8/20 |
| 50% | 494.5 ms | 6 | 13/20 |
| 75% | 179.3 ms | 5 | 14/20 |
| 100% | 20.4 ms | 4 | 16/20 |

The half-cost scenario leaves 12 or 13 slices across the decoder cost
variants and still encounters a mixed step over 700 ms. The full-cost
scenario leaves 16 slices in every variant. The short request encounters
fewer expensive steps while more of the long peer's work remains. Each
mixed step keeps its measured cost in the projection. The replay covers
runner work through short-request completion and stops there. It has no
client transport, long-request TTFT, or sustained throughput measurement.

The next useful test holds logical ends and final singleton steps fixed
while comparing the actual cost and numerical state of smaller physical
forwards. A repayment-only scheduler change needs live latency and
throughput from both clients before admission.

Reproduce the [comparison](comparison.json) from the sealed step archive.

```bash
python3 benchmarks/results/flash-next-prefill-debt-replay-m5-20261122/replay.py \
  | diff -u benchmarks/results/flash-next-prefill-debt-replay-m5-20261122/comparison.json -
```
