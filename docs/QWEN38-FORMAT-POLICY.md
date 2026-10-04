# Qwen 3.8 AXQ formats and migration

Qwen 3.8 27B and Qwen 3.8 Flash Next target **MLX-backed AXQ MXFP4 and
MXFP8 packs with MTP sidecars**. Standalone affine 4-bit, 6-bit and 8-bit
packs and non-MTP catalog variants are retired as of 2026-10-04.
Other model families retain their existing support.

| Model | Current selector | Pinned MXFP4 MTP revision | MXFP8 MTP |
| --- | --- | --- | --- |
| Qwen 3.8 27B | `qwen3.8-27b:axq`, `qwen3.8-27b:axq-mxfp4`, `qwen3.8-27b:mxfp4` | `b2c5354f779e430d0c1733143db848a72b71c16e` | Pack admission and SKU qualification pending |
| Qwen 3.8 Flash Next | `qwen3.8-flash-next:axq`, `qwen3.8-flash-next:mxfp4` | `0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35` | Audited pack and dedicated native-path admission pending |

The generic AXQ selectors now resolve to MXFP4 MTP. An explicitly named
retired format, such as `qwen3.8-27b:axq-6bit`, returns a migration error;
it does not silently select another precision. Explicit retired AutomatosX
Hub references are also rejected. Existing cache files are left intact.

MXFP4 and MXFP8 describe exported quantization modes, not affine integer
4-bit and 8-bit weights. Preserve per-tensor metadata: MXFP4 packs include
protected affine8 embeddings, routers or output heads, and full-precision
tensors. Those companions do not make the pack a retired affine variant.
In particular, the pinned 27B config has a top-level affine default but
its decoder projection overrides are MXFP4. Do not relabel tensor modes.

The native loader rejects retired quantization for Flash Next and for
Qwen 3.8 27B AXQuant packs identified by their source-model metadata.
Older Qwen models share the `qwen3_5` architecture, so architecture name
alone cannot identify the 27B retirement policy. Unknown Flash Next exporter
identities remain blocked. Experimental flags do not bypass retirement.

MTP sidecar availability does not promote MTP to a product default. Use
`--mlx-mtp-policy required` for explicit MTP; the normal route remains a
necessary control. MTP-S, MTP-P and MTP-D require evidence tied to each pack.
Retired 6-bit qualification, speed and safety receipts remain historical;
none qualify the replacement MXFP4 or future MXFP8 pack. SKU targets remain
Mac mini M4 Pro 64 GB for 27B and Ultra-class Mac Studio 192 GiB+ for Flash Next.

MXFP8 is in scope, but no unverified download alias, revision, throughput
claim or production promotion is published. AXQuant owns conversion and
pack publication; admission requires the immutable pack, MTP metadata and
native-path checks before SKU qualification.
