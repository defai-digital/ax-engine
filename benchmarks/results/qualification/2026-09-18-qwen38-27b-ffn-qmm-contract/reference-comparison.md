# Qwen 27B FFN reference and dispatch comparison

Status: **Candidate; not ship-ready**. This is selected numerical attribution
from the Mac mini M4 Pro 64 GB, not quality or release qualification.

The observer starts from `aa38f18bd4c7f6894cbd346372f42dd1717d1bf0` plus the
exact [diagnostic patch](ffn-observer.patch). Source, patch, binary, installed
wheel, loaded-library, and raw/published artifact hashes are recorded in
[the evidence JSON](ffn-observer.json). The patch is an observer, not a proposed
production implementation. No source was copied from reference projects.

## Source comparison before instrumentation

The following local reference revisions were read as design and semantic references.
These are named snapshots, not claims about latest upstream. File hashes are
included in the evidence JSON.

| Reference snapshot | Reviewed source | Relevant semantics and limit |
| --- | --- | --- |
| MTPLX `6459194642ba091ba7c17c3aba889b808340c3fc` | `mtplx/gdn_capture.py:2888-2905` | Attention residual, post-attention normalization, MLP, then residual are separate boundaries. The reviewed trace has A3B geometry; it does not qualify dense 27B. |
| mlxcel `0ef0a1a42a1203a267dfe93d743d29741dfcbdfb` | `src/models/qwen3_5.rs:36-38,1158-1186`; `src/models/qwen3_next.rs:902-914` | The Qwen 3.5 dense shell delegates to a gate/up, SiLU-product, down-projection MLP. Its recurrent implementation has a different storage contract and is not an AX state oracle. |
| Same runtime snapshot | `src/models/qwen3.rs:699-715` | Another quantized MLP path compiles the activation itself. A cached activation return must be distinguished from whole-FFN compilation; trace-time placeholders are not runtime observations. |
| MLX release tag `v0.32.2`, commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85` | `mlx/ops.cpp:4724,4803-4818` | Affine QMM combines scales/group-bias dtype and promotes with the activation dtype before projecting. This source tag is a semantic comparison, not a binary-source attestation for the installed wheel. |

AX's relevant dispatch at the observer base is in
`model/shared/utils.rs::qw_direct`: a successful verify-QMM attempt precedes
the stock MLX branch. The runner supplies the verify guard. The custom route
admits the observed S4 BF16, affine 4-bit, group-size-64 FFN gate/up geometry;
its default output-width floor excludes the 5,120-wide down projection.
Successful dispatch records, not the pack's “6-bit” name, establish what ran.

## Quantized projection arithmetic comparison

The MLX 0.32.2 short-row wide affine implementation dequantizes into float
and accumulates in float (`mlx/backend/metal/kernels/quantized.h:1005,1027-1076`).
Its eight K lanes form eight-value float subtotals within quantization groups
and use an eight-lane shuffle reduction. AX's custom S4 split-K route uses
32 lanes per partition, contiguous halves of packed words, running float
accumulators, then two reduced float partials. Both cast the final projection
to the activation dtype. These are different floating-point reduction orders.
The source comparison found no concrete nibble, group-index, row-index, or
partition-coverage defect in the observed geometry; it is not a proof of
bitwise identity or quality. Adding a BF16 cast to each dequantized weight is
not supported by the stock wide implementation.

The stock S1 route has another arithmetic order: its float helper factors
affine scale and group-bias contributions before reduction
(`quantized.h:62-69,235-243,289,779-818`). Therefore even exact stock S4 parity
would not by itself establish singleton-direct parity. Actual matched-state
measurements remain necessary. No default, kernel formula, reduction order,
or numerical tolerance was changed based on this static comparison.

## What the observer established

The unchanged compsec-079 request has 409 prompt tokens and a 192-token output
cap. The captured verifier window starts at offset 452, row 2, absolute
position 454. This historical same-input window is **not** the final clean
build's first direct/MTP divergence at output index 116.

All 192 observer production tokens match the clean installed wheel and the
previous clean native MTP trace. Modes 1 and 4 match the complete live
`[4, 248320]` logits. All **288** compiled-versus-eager leaves are exact:
hidden, conv state, recurrent state, QKV, A, and tape across 48 linear layers.
The previous observer checked five leaves per layer; this observer adds conv
state explicitly.

Layer-zero input/state, QKV/Z/A/B, normalized Q/K/V, recurrent output/state,
gated attention output, attention projection, residual, and pre-FFN input are
exact. The first differing FFN stage is the gate/up projection pair:

| Layer-zero stage | Unequal elements | Maximum absolute difference |
| --- | ---: | ---: |
| Gate projection | 5 | 0.0001220703125 |
| Up projection | 13 | 0.00048828125 |
| Activation | 15 | 0.000001907348633 |
| Down projection / complete FFN | 32 | 0.00003051757812 |
| Final layer hidden | 2 | 0.00003051757812 |

The target actually selects custom verify QMM for gate and up; the ordinary
replay selects stock MLX QMM. Down uses stock MLX in both routes. Actual
activation, scales, group-bias, and output dtypes are all BF16, with affine
4-bit group size 64 and no dense linear bias. Separate generic mixed-dtype or
dense-bias contract problems therefore do not explain this captured fixture.

Both nested compiled-SwiGLU returns match their same-input eager bodies
exactly. The observer pauses captures while nested closures compile/apply,
then records returned arrays; production closure outputs are unchanged.

The same-input verify-QMM guard-off controls have the same difference counts
and maxima as the corresponding ordinary comparisons. **That does not prove
guard-off and ordinary tensors are identical**: the observer did not directly
compare those two arrays. The two controls remain separate in the JSON and
[raw observer log](ffn-observer.log).

## Negative full-request control

A separate fresh run used the **clean installed wheel** with
`AX_MLX_MTP_VERIFY_QMM_MIN_N=2147483647`, retaining the unchanged request and
192-token cap. It remained active MTP with zero direct-fallback steps.

Its first output difference is index **109** against each of default MTP,
clean native direct, and the independent MLX reference prefix. The reference
had a 512-token budget and returned 450 tokens; only the first 192 are compared.
This selected control did not restore either direct or reference identity.
It provides no basis for changing the production default.

The floor-run provenance retained an unused earlier diagnostic-build receipt.
The executed probe arm points to the clean installed server, whose source and
binary hashes are recorded separately in the evidence JSON. That ambient
receipt must not be described as the executed binary.

The evidence locates a numerical route difference at the FFN gate/up QMM
boundary. It does not yet distinguish an arithmetic defect from differing
reduction ordering, establish causality for the later token split, or close
quality, recovery, endurance, or broader route-consistency gates.
