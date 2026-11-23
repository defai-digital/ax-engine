# Flash Next paired QSA gather activation in native HTTP serving

A same-binary direct control on Mac Studio M2 Ultra 192 GB, internal SSD,
macOS 27.2.1, Rust 1.99.1 and MLX 0.32.4. The two fresh servers use the
49-member MXFP4 pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
Both use one slot, a 32,768-token context budget, Auto resident
experts, the Mapped reader, MTP off and prefix reuse disabled. Only the experimental
arm sets `AX_MLX_FLASH_NEXT_QSA_PAIRED_GATHER=1`.

All eight native SSE requests emit their full 64-token budget. The
two arms match all 256 paired output token IDs. The server's QSA
metric counts K/V tile pairs added to the lazy MLX graph
during serving steps. The client reads before and after each request
and waits for the five lifecycle gauges to drain.

| Prompt | Stock tile pairs | Experimental tile pairs |
| --- | ---: | ---: |
| 512 tokens | 0 | 0 |
| 2,048 tokens | 0 | 0 |
| Original 8,192-token prompt | 0 | 432 |
| Second 8,192-token prompt | 0 | 432 |

The raw rows retain forward times, client timings, cumulative host CPU
ticks and sampled memory pressure. All 154 observations are normal. Peak physical
footprint is 85.9 GiB across both arms. This run covers route engagement,
output identity and bounded resource drainage. The M5 primary-tier
timing comparison uses separate endpoints and crossover orders.
The [signed model-forward comparison](../flash-next-paired-qsa-prefill-20261123/README.md)
retains the separate bitwise state and logits evidence.

The archive contains the hash-bound build metadata, runtime patch, input
IDs and audited inventory, exact driver modules, request rows and sampled
memory records. The verifier recomputes the comparison offline.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -E -s -B replay.py | cmp comparison.json -
```
