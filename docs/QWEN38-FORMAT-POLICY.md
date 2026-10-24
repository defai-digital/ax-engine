# Qwen 3.8 AXQ formats and migration

Qwen 3.8 27B and Qwen 3.8 Flash Next target **MLX-backed AXQ MXFP4 and
MXFP8 packs with MTP sidecars**. Flash Next also supports the **affine
6-bit/group64 MTP pack** as the quality/size balance point (ADR-050,
2026-10-23, superseding the MX-only rule of ADR-039). Standalone affine 4-bit
and 8-bit packs and non-MTP catalog variants stay unsupported. Other model
families retain their existing support.

| Model | Current selector | Pinned MXFP4 MTP revision | Affine 6-bit MTP | MXFP8 MTP |
| --- | --- | --- | --- | --- |
| Qwen 3.8 27B | `qwen3.8-27b:axq`, `qwen3.8-27b:axq-mxfp4`, `qwen3.8-27b:mxfp4` | `b2c5354f779e430d0c1733143db848a72b71c16e` | Loader admits affine 6-bit/group64; no pinned catalog pack | Pack admission and SKU qualification pending |
| Qwen 3.8 Flash Next | `qwen3.8-flash-next:axq`, `qwen3.8-flash-next:mxfp4` | `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` | `qwen3.8-flash-next:axq-6bit` at `3c1dd547d70cb7bf01dd0a89c41c400dde32ed39` (public; on 128 GB run with `--stream-experts off`) | Audited pack and dedicated native-path admission pending |

The generic AXQ selectors resolve to MXFP4 MTP. An explicitly named
unsupported format, such as `qwen3.8-flash-next:axq-4bit`, returns an error;
it does not silently select another precision. Existing cache files are left
intact.

MXFP4 and MXFP8 describe exported quantization modes, not affine integer
4-bit and 8-bit weights. Preserve per-tensor metadata: MXFP4 packs include
protected affine8 embeddings, routers or output heads, and full-precision
tensors. Those companions do not make the pack a standalone affine variant.
In particular, the pinned 27B config has a top-level affine default but
its decoder projection overrides are MXFP4. Do not relabel tensor modes.

The native loader admits MXFP4, MXFP8 (27B) and affine 6-bit/group64
projections for Flash Next and for Qwen 3.8 27B AXQuant packs identified by
their source-model metadata, and rejects other standalone affine widths.
Older Qwen models share the `qwen3_5` architecture, so architecture name
alone cannot identify the 27B format scope. Unknown Flash Next exporter
identities remain blocked. Experimental flags do not bypass this scope.

MTP sidecar availability does not promote MTP to a product default. Use
`--mlx-mtp-policy required` for explicit MTP; the normal route remains a
necessary control. MTP-S, MTP-P and MTP-D require evidence tied to each pack.
Receipts of one format do not qualify another: the MXFP4 release gates do
not qualify the 6-bit pack, and the September 6-bit receipts do not qualify
MXFP4 or MXFP8. SKU targets remain
Mac mini M4 Pro 64 GB for 27B and MacBook Pro M5 Max 128 GB for Flash Next.

MXFP8 is in scope, but no unverified download alias, revision, throughput
claim or production promotion is published. AXQuant owns conversion and
pack publication; admission requires the immutable pack, MTP metadata and
native-path checks before SKU qualification.
