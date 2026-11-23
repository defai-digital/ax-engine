# Flash Next resident one-slot and two-slot crossover on M5 Max

The same signed experimental binary ran one slot, two slots, two slots,
and one slot on MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1,
and MLX 0.32.4. Both profiles used the pinned MXFP4 pack, mapped n-gram
reader, 2,049-token scheduling budget, resident experts, and MTP off.
The two-slot profile enabled the physical split and priority short
drain from the [previous experiment](../flash-next-natural-short-drain-cpu-m5-20261123/README.md).
Both server and client slot limits followed the arm. The client retained
the same 220 offered arrivals per arm and queued requests awaiting a
client slot. Dispatch TTFT starts when the request is sent. Offered
TTFT includes the client queue.

All four arms match the independent solo oracle for **84,480 generated
token IDs**. All five lifecycle gauges return to zero. All **4,069**
memory-pressure samples are normal and the maximum sampled process
footprint is **75.84 GiB**.

| Arm | Delivered tokens/s | 512 offered TTFT p95 | 512 dispatch TTFT p95 | 512 p95 of per-request largest gap |
| --- | ---: | ---: | ---: | ---: |
| One slot, first | 22.785 | 49.99 s | 0.64 s | 42.73 ms |
| Two slots, first | 23.741 | 13.11 s | 2.65 s | 682.25 ms |
| Two slots, reverse | 22.660 | 50.45 s | 2.89 s | 692.69 ms |
| One slot, reverse | 22.982 | 40.49 s | 0.53 s | 29.68 ms |

Measured two-slot throughput moves by **+4.2%** in the first order and
**-1.4%** in the reverse. Offered TTFT moves in opposite directions
for all three prompt sizes. Two slots increase dispatch TTFT and
stream gaps in both orders. The 2,048-token dispatch TTFT p95 is
12.81–14.00 s with two slots and 2.34–2.40 s with one slot.

## Timing eligibility

| Arm | Endpoint CPU idle, before / after | Full client window idle |
| --- | ---: | ---: |
| One slot, first | 98.93% / 98.93% | 80.09% |
| Two slots, first | 99.53% / 92.31% | 90.25% |
| Two slots, reverse | 99.24% / 91.39% | 84.90% |
| One slot, reverse | 91.56% / 74.17% | 83.05% |

The first order meets the 90% endpoint threshold in both arms, with
different full-window load. The reverse one-slot arm misses the
endpoint threshold. Neither pair has 90% full-window idle in both
arms. Performance magnitudes remain diagnostic. The one-slot resident
profile remains the qualification standard.

## Source of output gaps

The replay joins every consecutive client token interval to its
measured engine steps and classifies peer work by recorded physical
prefill width. All intervals in these four arms are consecutive engine
steps. Every one of the **606** short-stream gaps of at least 100 ms
in the two-slot arms ends on a peer prefill of 511 or 512 rows.
That includes 388 intervals from 2,048-token peers, 168 from 8,192-token
peers, and 50 from other 512-token peers. Every observed 511/512-row peer
prefill in these short-stream intervals exceeds 100 ms. All 104 peer
one-row split completions stay below 100 ms, as do all intervals in the
two one-slot arms.

For 126 of 132 medium requests across the two-slot arms, the largest
gap comes from the whole 512-row prefill of another request. The 34
short streams with an explicit full priority hold have no gaps of
100 ms and a maximum of 29.48 ms. A hold limited to the last 12
output tokens would be eligible for only 158 of the 606 short gaps
and 35 of the 126 affected medium maxima in this trace. This is
observed eligibility under the fixed output budget. It does not
predict the result of rescheduling those steps.

The attribution follows the cost of one physical call. TensorFold
`f0cc0a134` bounds whole prefill calls while a decode is ready and
oMLX `5eeec775` repays measured prefill time between calls. The AX
[16-layer checkpoint study](../flash-next-prefill-layer-sustained-m5-20261122/README.md)
preserves forward shapes and reduces short-stream tails, with
order-dependent long TTFT and throughput. The
[half-rate study](../flash-next-prefill-layer-half-rate-m5-20261122/README.md)
also shows a long-request TTFT cost. The current crossover supports
retaining the bounded one-slot profile and evaluating any wider
profile with a workload-specific latency objective.

The [comparison](comparison.json) records the raw measurements and
per-width attribution. The [portable capture](evidence.tar.xz) retains
the signed source and input receipts, request IDs, output IDs,
per-request steps, memory samples and CPU windows. The executable is
identified by digest. The transport address and local workload path
are replaced in the archived client metadata.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
