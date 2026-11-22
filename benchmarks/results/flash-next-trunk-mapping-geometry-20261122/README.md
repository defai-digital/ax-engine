# Flash Next MXFP4 resident tensor mapping geometry

TensorFold `f0cc0a134` keeps checkpoint mappings alive and gives its
Metal kernels buffer-relative offsets. This audit checks the geometry
of the same idea through MLX managed buffers and typed array views for
the AX MXFP4 checkpoint at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
The header scan follows the `38c90b27a` resident trunk role and sidecar
selection and reproduces the 15 shards and 1,942 resident tensors
from the [M5 startup trace](../flash-next-delivery-bound-startup-m5-20261122/README.md).
The scan reads the manifest, weight index, and shard headers.

| Property | Observed |
| --- | ---: |
| Resident weight bytes | 68,516,576,280 |
| Tensors starting on a 16 KiB page boundary | 0 / 1,942 |
| Tensors starting on a boundary for their element type | 1,289 / 1,942 |
| Selected bytes starting on a boundary for their element type | 27,039,321,880 (39.46%) |
| Excluded tensor bytes within the enclosing per-shard page ranges | 6,262,989,462 |
| Enclosing page ranges ending past the file's last byte | 13 / 15 |

The independent M2 Ultra GPU control uses MLX and MLX Metal 0.32.4.
Viewing 64 KiB of `U8` storage as BF16, U32, or I64 adds zero active MLX
bytes at an aligned offset. A one-byte displacement adds 65,536 bytes
for each type. All six GPU byte checks pass in each of three processes.
This matches the alignment check and copy fallback in the reviewed
MLX `View::eval_gpu` at `6ac6c0f4b`.

The geometry and GPU control rule out direct per-tensor page borrowing
for this checkpoint. A shared `U8` mapping would still copy more than
60% of the selected bytes on the typed path. Mapping enclosing shard
ranges also needs explicit handling of excluded payloads and terminal
pages. An alignment-aware storage or kernel design would need its own
real-weight lifetime, parity, and startup checks.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B audit_trunk_geometry.py --model-dir "$MODEL_DIR" --loader "$LOADER" | cmp geometry.json -
"$MLX_PYTHON" -B probe_mlx_view.py | cmp metal-view.json -
```
