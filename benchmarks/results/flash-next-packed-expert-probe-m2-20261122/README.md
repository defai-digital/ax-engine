# Flash Next routed gate/up packing probe on M2 Ultra

This Metal probe verifies the two publisher shards holding routed
weights for layers 0 and 32 of the pinned MXFP4 pack. It uses MLX
0.32.4, real packed weights and scales, and deterministic BF16
activations. The input shapes are one decode row and 511/512 prefill
tokens. Each row has ten selections. The longer cases cover 512
or 64 distinct experts with the original sorted gather geometry.

| Variant | 511/512 wall relative to split | Additional allocation |
| --- | ---: | ---: |
| Already packed gate/up | 0.979–1.004× | One 850 MiB packed pair per measured layer |
| Concatenate inside each forward | 1.212–1.351× | About 850 MiB of extra peak workspace |

Both variants match the complete BF16 gate, up, SwiGLU and down tensors
in all eight cases. Every reported median uses six samples per route
in balanced order. The timed graph includes gate, up, activation and
down through Metal evaluation. Router work, expert reduction,
the shared expert, HTTP serving and whole-model memory are outside this
measurement's scope.

Retain split gate/up and exclude per-forward concatenation from serving
experiments. A viable already-packed route would replace the split pair,
include startup or pack conversion cost, and need actual M5 decode and
prefill measurements. A packed duplicate across all 48 layers would
consume 39.844 GiB beyond the current weights. TensorFold's fused
gate/up kernel reads separate weight buffers and also performs the
activation, which is a different experiment from this MLX packing test.

Replay the [raw observations](result.json) and the exact probe binding.

```bash
python3 benchmarks/results/flash-next-packed-expert-probe-m2-20261122/replay.py
```
