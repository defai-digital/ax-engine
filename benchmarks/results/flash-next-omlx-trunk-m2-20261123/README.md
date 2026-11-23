# Flash Next oMLX SSD-backed trunk smoke on M2 Ultra

Mac Studio M2 Ultra 192 GiB, internal NVMe SSD, macOS 27.2.1, MLX and
MLX Metal 0.32.4. The probe uses oMLX `5eeec7757` and its declared
MLX-VLM source `138d7617`. It exercises the strict lazy model loader
with oMLX's Qwen4-Exp sanitation and SSD PLE mode, with MTP off and
remote model code disabled. The AXQ MXFP4 pack is
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.

The language trunk generates 16 greedy IDs per case with a fresh cache.
The prompt IDs and AX single-request outputs come from the
[M5 natural-load archive](../flash-next-current-natural-load-m5-20261123/README.md).
The replay checks the archived input digests. The
[component replay](../flash-next-omlx-ple-component-20261123/README.md)
verifies the reader's row addressing separately.

| Input tokens | Exact initial IDs out of 16 |
| ---: | ---: |
| 512 | 2 |
| 2,048 | 16 |
| 8,192 | 0 |

Both processes emit the same 48 greedy IDs, with identical input
digests and AX comparison sequences.
The first run peaks at 70.18 GiB process RSS and sees normal memory
pressure at the recorded checkpoints. The measured seconds describe
this smoke and include lazy materialization in the first forward.
Full quality comparison, timing controls, and the ADR-040
ordinary-reference release gates remain open.

Use a local environment with MLX 0.32.4 and the reference import
dependencies, including `openai-harmony`, `jsonschema`, and
`mlx-audio`. The recorded smoke used versions 0.0.8, 4.26.0, and
0.6.7 respectively. From this directory run:

```sh
shasum -a 256 -c SHA256SUMS
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  PYTHONPATH="$OMLX:$MLX_VLM" python3.12 -B replay.py "$PACK" "$OMLX" "$MLX_VLM"
```
