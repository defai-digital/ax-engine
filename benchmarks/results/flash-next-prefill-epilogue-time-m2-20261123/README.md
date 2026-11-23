# Flash Next rounded prefill SwiGLU timing on M2 Ultra

This experiment times the exact gate-activation rounding contract from
the [previous correctness probe](../flash-next-expert-epilogue-rounding-m2-20261123/README.md).
The same real routed MXFP4 layers 0 and 32 use their split gate, up and
down projections on Mac Studio M2 Ultra 192 GB with MLX 0.32.4.
The source verifies both publisher shards and the index before and
after the probe. It alternates the two arms after warmup, evaluates
each output before reading the clock, and retains ten timings per arm
and shape. CPU idle is 96.45% before and 93.89% after the probe.

| Scope at 511/512 prompt rows | Compiled time / eager time, six cases |
| --- | ---: |
| SwiGLU on pre-evaluated gate/up inputs | 0.727–0.804× |
| Gate, up, SwiGLU and down projections | 0.978–0.987× |

All eight shapes preserve bitwise output within both scopes. The full
routed component saves 1.3–2.2% at the prefill widths. This evidence
isolates one activation epilogue and does not time model-level sorting,
shared experts, attention, scheduling or serving. Primary-tier benefit
and a product default require separate evidence. The next higher-impact
candidate needs to reduce projection or memory traffic across the
routed path while retaining its BF16 rounding boundary.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```

For a hardware rerun, copy `packed_probe.py` and `mxfp4-inventory.json`
from the correctness probe beside this script, and copy that directory's
`probe.py` as `rounding_probe.py`. Provide the pinned pack with
`--pack`, the publisher inventory with `--inventory`, and an output
path with `--out`.
