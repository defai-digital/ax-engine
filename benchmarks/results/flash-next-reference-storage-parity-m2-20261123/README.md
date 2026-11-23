# Flash Next resident and bounded PLE parity on M2 Ultra

Mac Studio M2 Ultra 192 GiB, macOS 27.2.1, and the pinned AXQ MXFP4
pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. Two independent
processes use the same MLX-VLM `6102cb4a` language graph, MLX and MLX
Metal 0.32.2, Transformers 5.18.0.dev0, Tokenizers 0.23.2, and fixed
1,024-token prefill chunks. The resident storage control runs first.
The bounded storage control opens the affine8 PLE table using checked
positional reads. Both processes use the same diagnostic runner.

The two storage modes produce identical generated IDs, text, stops
and grades on all 105 frozen QA rows. They also reproduce every
output ID across the same 15 fixed 128-token matrix rows. The runner
rechecks publisher inventory, its language and runtime environment,
and the frozen workload before each receipt completes.

| Storage mode | Materialized arrays | Peak tracked MLX memory | Peak process RSS |
| --- | ---: | ---: | ---: |
| Resident | 2,326 | 121.56 GiB | 89.51 GiB |
| Bounded PLE | 1,942 | 67.91 GiB | 64.34 GiB |

The bounded path uses 53.65 GiB less tracked MLX memory in this pair.
All sampled memory-pressure levels are normal. These observations
establish storage parity over the tested M2 workload and support the
memory attribution. They leave M5 timing, portability to other packs
and the approved ordinary-reference release gate open.

The replay binds the original prompt hashes to the
[guarded native matrix](../flash-next-guarded-reader-matrix-m5-20261123/README.md)
and compares the completed reference outputs on the same M2.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
