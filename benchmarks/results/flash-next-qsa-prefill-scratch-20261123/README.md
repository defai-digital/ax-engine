# Flash Next QSA prefill scratch reservation

The existing plan capped the gather reservation at 1 GiB and counted
one of the two selected K/V arrays. The cap still limits each gather
tile. The lazy attention graph can retain buffers from more than one
tile through the same physical chunk.

The new reservation prices both padded K/V arrays across the chunk,
indices and masks, query and output tiles, and temporary storage for
masked vector SDPA on MLX 0.32.4. The existing 256 MiB indexer-score
reservation is separate. Gather-then-transpose includes the extra copy.
An indexed-attention opt-in can still fall back to the gathered path,
so admission retains the full fallback reservation.

## Allocation control

Mac Studio M2 Ultra 192 GB, macOS 27.2.1, MLX 0.32.4. The synthetic
shape uses the Flash Next attention geometry of 24 query heads, two
KV heads, dimension 256, and 2,051 selected rows. Measurements sample
MLX peak allocation above the pre-evaluated inputs. They exclude QSA
indexer scores, real model weights, and the rest of the forward.

| Physical queries | Previous gather estimate | Observed stock attention peak | Corrected attention estimate |
| ---: | ---: | ---: | ---: |
| 128 | 0.25 GiB | 0.70 GiB | 0.71 GiB |
| 256 | 0.50 GiB | 1.41 GiB | 1.42 GiB |
| 511 | 1.00 GiB | 2.81 GiB | 2.83 GiB |
| 512 | 1.00 GiB | 2.81 GiB | 2.83 GiB |
| 1,024 | 1.00 GiB | 5.61 GiB | 5.66 GiB |
| 2,048 | 1.00 GiB | 11.21 GiB | 11.33 GiB |

Every tested sub-batch width and serial evaluation produce identical
attention bits within each control. At 512 queries the new charge also
covers the gather-then-transpose peak (4.81 GiB) and the explicit 160
partition run (3.00 GiB). Forced serial evaluation reduces the synthetic
2,048-query peak to 2.41 GiB. Runtime evaluation order is unchanged in
this patch.

## Admission projection

The before and after planner tests use the same installed model
metadata, a 2,048-row charging budget, and the same hypothetical
MacBook Pro M5 Max 128 GB ceiling. These are plan decisions from
metadata. The after plan raises the total prefill scratch reservation
from 1.69 GiB to 12.02 GiB.

| Pack and scenario | Projected working set before | Projected working set after | Auto after |
| --- | ---: | ---: | --- |
| MXFP4, eight requested sequences | 83.76 GiB | 94.09 GiB | Resident |
| 6-bit, one requested sequence | 115.15 GiB | 125.48 GiB | Stream |

Both runs use the 115.20 GiB ceiling. MXFP4 needs no planner-imposed
sequence cap in this projection. The supported Flash Next MXFP4
profile keeps its existing one-request limit. The 6-bit forced-resident
profile uses the existing explicit `--stream-experts off` override.

## Real-pack native HTTP control

The signed before and after builds use the same verified 49-member
MXFP4 pack and prompt IDs on M2 Ultra 192 GB. Both retain Auto
resident experts and the Mapped reader with one slot, a 32,768-token
context budget, MTP off, and matched disabled prefix reuse. Four
native SSE requests at 512, 2,048 and two 8,192-token prompts match
all 256 before/after greedy token IDs. The after build drains all
five lifecycle gauges. All 77 pressure samples are normal and
peak physical footprint is 85.7 GiB. The startup log
confirms both reserve values and retained resident placement.

Three reservation tests pass on Mac mini M4 Pro 64 GB, macOS 27.0, with
the pinned MLX dylib. Real primary-tier readiness and placement retain
their own run. This result does not award a release or performance gate.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
