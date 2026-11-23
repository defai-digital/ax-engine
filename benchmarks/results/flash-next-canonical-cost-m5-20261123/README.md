# Flash Next canonical MTP cost on M5 Max

The signed `26c0ccefe` release-server build uses the same resident MXFP4
pack as the [fresh native delivery](../flash-next-probe-followup-fresh-m5-20261123/README.md).
Both arms run one 32,768-token sequence on MacBook Pro M5 Max 128 GB,
macOS 27.2.1, internal NVMe SSD, and MLX 0.32.4. The Disabled process
runs first. Required uses the canonical singleton verifier. Each process
warms once and runs all ten prompts with their original limits and
normal EOS handling. Both disable prefix reuse and generic n-gram
acceleration. CPU idle is 99.45% / 94.98% for Disabled and 97.40% /
98.33% for Required. Every request's pressure readings are normal.

| Panel | Matched output IDs | Direct decode | Required decode | Required / direct | Accepted / verified |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python module coding | 3,000 | 49.23 tok/s | 44.97 tok/s | 0.914× | 1,473 / 1,524 |
| Long-form general | 5,000 | 48.61 tok/s | 43.58 tok/s | 0.897× | 2,064 / 2,936 |
| Short arithmetic | 2 | 45.74 tok/s | 19.58 tok/s | 0.428× | 1 / 1 |

Decode rates use emitted tokens after the first event divided by their
client-observed intervals. The two-token arithmetic result has one
measured decode interval. Its end-to-end request takes 0.270 seconds in
Disabled and 0.245 seconds in Required.

All 8,002 output IDs match. Coding accepts 96.65% of verified drafts.
Its Required target correction and bonus phases total 61.941 seconds
against 60.879 seconds for the entire direct decode window. General
records 105.976 seconds of Required target work against 102.745 seconds
of direct decode. These phase totals and client intervals are measured
at different boundaries. The result supplies one timing-eligible
process order. The reverse order retains a separate receipt, and MTP-P
qualification remains open.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
