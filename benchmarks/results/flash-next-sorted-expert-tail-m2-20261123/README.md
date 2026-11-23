# Flash Next sorted routed reduction on M2 Ultra

A probe fuses inverse-permutation lookup, routed weighting and reduction,
and the gated shared output add for the pinned Flash Next MXFP4 model. It
adapts the engine's historical sorted addressing from `236611eec`. In the
Flash Next BF16 path each routed product, partial addition and shared
product retains the eager rounding boundary. Ten routed slots use
the two paired partials and remaining fold order of the pinned MLX 0.32.4
small strided reduction.

The test checks two publisher shards and their index before and after
execution. Mac Studio M2 Ultra 192 GB runs real quantized gate, up, and
down projections from layers 0 and 32 with resident tensors and
deterministic synthetic activations, routing weights and shared outputs.
The six cases use 511 or 512 rows and select across 512 or 64 experts.
Both the standalone tail and the route with all three projections match
all 7,859,200 eager output elements for their respective scopes, bitwise.
The float-accumulation variant changes 2,625,243 of those elements.

| Timed scope, ten evals per arm and case | Fused / eager median time |
| --- | ---: |
| Tail using pre-evaluated down projections | 0.617–0.686× |
| Real routed gate, up, SwiGLU, down and tail | 0.984–0.994× |

CPU idle is 96.10% before and 91.00% after. All three recorded pressure
samples are normal. The complete routed component saves 0.6–1.6%.
Full-graph and M5 performance remain unverified. This probe does not
change the runtime dispatch or certify the arithmetic on a different MLX
version.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```

For a hardware rerun, copy `packed_probe.py` and the inventory from the
[epilogue correctness probe](../flash-next-expert-epilogue-rounding-m2-20261123/README.md),
and copy its Python probe as `rounding_probe.py`. Run `real_pack.py`
with the pinned pack, inventory and output paths.
