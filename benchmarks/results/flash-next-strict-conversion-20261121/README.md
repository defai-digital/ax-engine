# Flash Next strict conversion of the published MX packs

`generate-manifest` read the original safetensors headers from isolated
linked views of the [MXFP4 pack](../qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json)
and the [MXFP8 preview pack](../qualification/2026-11-18-flash-next-mxfp8-preview/pack-inventory.json).
The converter and validator ran on Mac Studio M2 Ultra 192 GB with an
external SSD using Rust 1.99.1. These checks exercise manifest conversion
and validation without loading model weights into MLX.

| Format | Previously reported as unrecognised | Strict conversion and validation | Primary manifest specs |
| --- | ---: | --- | ---: |
| MXFP4 | 972 | Passed | 1,718 |
| MXFP8 preview | 744 | Passed with preview opt-in | 1,718 |

Every previously reported tensor was a scale or an affine group bias for
a mapped U32 weight. The native loader resolves these sidecars by name.
Its selected raw file comes from the weight index or the primary file for
colocated sidecars. The converter now counts a sidecar as recognised when
the recorded file, dtype, group geometry, and leading shape match. The two
manifests compare equal after removing the previous dropped-tensor ledgers.

Strict conversion rejects a supplemental MXFP4 view with one cross-shard
scale entry removed from its index. The generated MXFP8 manifest still
requires the explicit preview opt-in for validation. The
[result](result.json) records the pack pins, converter hash and exit codes.
