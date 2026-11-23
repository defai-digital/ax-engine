# Flash Next oMLX PLE component replay

On an M2 Ultra 192 GB with macOS 27.2.1 and MLX 0.32.4, the original
oMLX Flash Next SSD-backed reader opened the pinned AXQ MXFP4 pack's 128
affine 8-bit/group-32 PLE shards across 12 safetensors files. The replay
uses oMLX's vendored model and its declared MLX-VLM source revision.

Synchronous and prefetched gathers each match all 62,720 values from
392 index positions. These include the first, middle, and last row of
every shard, in reverse shard order, plus eight repeated positions. Both
paths match an oracle that independently computes byte offsets from the
safetensors headers and issues positional reads for the sampled rows. The
two implementations use the same MLX affine dequantization primitive.
The replay also checks both index bounds and a read after close.

This evidence establishes same-pack PLE row access and prefetch behavior.
Full-model compatibility, M5 timing, and generation quality remain
open. The pinned ordinary-reference release gate remains applicable.

With the pack available locally, use oMLX at `5eeec7757` and its MLX-VLM
dependency at `138d7617`. Use a Python environment with MLX 0.32.4 and
the reference's import dependencies. From this directory run:

```sh
shasum -a 256 -c SHA256SUMS
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  PYTHONPATH="$OMLX:$MLX_VLM" python3 -B replay.py "$PACK" "$OMLX" "$MLX_VLM" \
  | cmp observed.json -
```
