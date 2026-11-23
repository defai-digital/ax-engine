# Flash Next routed expert epilogue rounding on M2 Ultra

Mac Studio M2 Ultra 192 GB, macOS 27.2.1, internal SSD and MLX and
MLX Metal 0.32.4. The probe verifies the two publisher shards and
index used by layers zero and 32 of pinned MXFP4 revision
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. It reuses the
[packed-expert probe](../flash-next-packed-expert-probe-m2-20261122/README.md)
and its deterministic BF16 inputs and selected expert layouts. The
eight cases include one decode row, 511 and 512 prefill rows and a
concentrated selection across 64 experts.

Both candidates use the exact same quantized gate, up and down
projections as the eager control. The first candidate compiles the
existing epilogue and retains BF16 rounding of the SiLU activation
before multiplying by up. The second keeps the activation and
product in F32 and rounds the product to BF16.

| Epilogue | Activation elements with different bits | Down projection elements with different bits |
| --- | ---: | ---: |
| Compile with explicit BF16 activation | 0 / 19,660,800 | 0 / 78,643,200 |
| Round the product to BF16 | 5,248,877 / 19,660,800 | 43,397,139 / 78,643,200 |

The late-rounding variant changes both stages in every case. Its
maximum absolute differences are 0.03125 in the activation and
0.015625 after down. All six pressure observations are normal.
The probe supplies a numerical requirement for any future
separate-buffer gate/up fusion. It measures tensor outputs of
two layers with synthetic activations. Full model output and
kernel speed require their own live measurements.

Reproduce the original GPU probe with the pinned pack and the
included inventory.

```sh
python3.12 -B probe.py --pack "$PACK" --inventory mxfp4-inventory.json --output observed.json
```

Recheck the recorded evidence on any Python 3.12+ host.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
