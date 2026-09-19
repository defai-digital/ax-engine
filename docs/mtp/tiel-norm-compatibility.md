# Tiel MTP norm compatibility

The initial AXQuant Tiel and Cyber-Tiel MXFP4 exports contain already converted
MLX RMSNorm multipliers in their MTP sidecars, but `mtplx_runtime.json` declares
`raw_hf_delta`. Applying the declared conversion adds one a second time. This
reduces proposal quality and can make active MTP slower than direct decoding.

All seven stored norm tensors are byte-for-byte equal to the correctly rounded
BF16 result of adding one to the original BF16 tensors from
[`ornith-ai/Ornith-1.5-35B-A3B`, revision
`10fbf86fed7ecee4a061f8b499a618f46001cac1`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/tree/10fbf86fed7ecee4a061f8b499a618f46001cac1).
Cyber's corresponding norm bytes equal Tiel's. The [tensor audit](tiel-norm-reference-diff.json)
records each digest and the raw/converted means. This conclusion does not depend
on guessing the convention from a tensor's magnitude.

## Bounded compatibility correction

AX Engine corrects the declared layout in memory only for these exact exports:

| Pack | Pack revision | Complete `mtp.safetensors` SHA-256 |
| --- | --- | --- |
| `AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` | `5ab39b24bfd7f65203be9b7823b1840486f58b6d` | `0ced87b0462269a98bd393a808e1c9c02ace7054801c0e2745dd9dd9a4077915` |
| `AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` | `fe05e871ec69ad9ae8eac01fd285555514ac7daf` | `590e87c9c3fbbaa370c8dddbcc22611f00bbc78019d28e3192f180841c030527` |

Admission also requires the exact protected-sidecar manifest schema, role,
architecture, source publisher/revision, tensor count, and size. The full sidecar
hash is recomputed; trusting the manifest's hash alone is insufficient. File
identity, length, modification time and (on Unix) change time are compared across
loading and verification; a changed file disables this compatibility path.

A warning reports `Correcting known AXQuant Tiel MTP norm metadata` with the
verified digest. Shared model files are not modified. Unknown exports retain
the normal declared-layout behavior, and genuine raw HF sidecars still receive
the required conversion. Explicit `mlx_multiplier` and automatic detection
remain unchanged. This does not change proposal sampling, verification, MTP
policy, prefix caching, or model defaults.

The extra hash read is startup work, not per-token work. Corrected future packs
should declare `mlx_multiplier`, use new immutable revisions, and undergo fresh
validation. Remove this exception once supported clients no longer pin the old
exports. Other consumers that trust the old declaration still need corrected
metadata or their own verified compatibility handling.

## Scope of validation

The source loader was checked on an M3 Max with the original pinned packs, plus
negative regression cases for changed manifest fields, altered bytes and file
replacement. The independent MTPLX generic `qwen3_5_moe` loader leaves these
already shifted norms unchanged. The oMLX and MLX-LM sources were reviewed for
normalization and source-conversion behavior; this is not an oMLX speed result.

These are development models. Throughput measurements on the M3 Max are
workstation diagnostics, not M4 Pro or M5 Max qualification. Greedy direct and
MTP graphs did not produce identical sequences in the diagnostic prompts; a
normalization fix or accepted draft counter is not a claim of cross-graph
bitwise parity, task correctness, or MTP certification.
