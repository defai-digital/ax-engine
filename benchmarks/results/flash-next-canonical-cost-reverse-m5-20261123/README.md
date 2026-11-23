# Flash Next canonical MTP reverse order on M5 Max

This order starts with Required canonical MTP and then starts Disabled.
It uses the same signed server, verified resident MXFP4 pack, 32K
one-request envelope and frozen prompt panels as the
[direct-first comparison](../flash-next-canonical-cost-m5-20261123/README.md).
Prompt order is also reversed. Both processes warm once before
serving all ten workloads with ordinary EOS.

All 8,002 output IDs match in this order. Prompt hashes, output IDs and
finish reasons also match the direct-first order across both modes.
Each measured request reports normal memory pressure.

| Mode | Before CPU idle | After CPU idle |
| --- | ---: | ---: |
| Required | 98.37% | 79.51% |
| Disabled | 92.64% | 84.18% |

Both final samples fall below the 90% timing floor. The raw comparison
retains the individual stream intervals and verifier phase times for
diagnosis. The direct-first order supplies eligible comparative timings.

~~~sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
~~~
