# Flash Next routed gate/up packing probe on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, MLX 0.32.4. The probe uses
the [same source and workload as the M2 control](../flash-next-packed-expert-probe-m2-20261122/README.md).
It verifies the publisher shards for two real MXFP4 expert layers,
feeds deterministic BF16 inputs and retains all six Metal evaluation
times per route and shape. All eight cases reproduce the exact gate,
up, rounded SwiGLU and down results. Host idle is 91.37% before and
90.98% after the probe. The final pressure observation is normal.

At 511 and 512 tokens, already-packed gate/up takes 0.995–0.998×
the split time. Concatenating inside every forward takes 1.791–2.603×
the split time and adds about 850 MiB of peak MLX allocation. At one
token, prepacking moves in opposite directions across the two layers.
The probe exercises the routed computation only. It supports keeping
the current split weights. A fused kernel that reads separate weight
buffers, as used by TensorFold, needs its own numerical and runtime
measurements.

Replay the [raw timings and equality observations](result.json) and
compare against [comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-packed-expert-probe-m5-20261122/replay.py
```
