# Qwen GDN post-input rounding diagnostic evidence

Status: **Candidate; not ship-ready**. This report contains instrumented
attribution evidence, including an intermediate candidate **before the final
FP16 intrinsic correction**. It does not qualify the final `aa38f18b` wheel.

## Identity and source contract

The selected hardware class is Mac mini M4 Pro 64 GB (Mac16,11), using
`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` at
`3e290738e96972307c6aeb9934ab170ca0eae1c1`. The installed native-library baseline
comes from clean `4cbda8b5`; diagnostic server identities are separate.

[Parsed evidence](instrumented-evidence.json) records complete source revisions,
patch and binary hashes, input/output hashes, source-log hashes, and parsed
per-layer comparison statistics. Reference snapshot revisions and reviewed-file
hashes are included as design provenance only. The reference MLX HEAD is not
the installed published MLX 0.32.2 wheel. No reference implementation was copied.

Both diagnostic patches are based on `d0fbb3a7`. The pre-fix binary uses
`post-input-stage.patch`; the pre-FP16 fixed binary uses `stage.patch`. The latter
still spells the typed exponential/division differently from final `aa38f18b`.
The intermediate hardware observations below must not be relabeled as results
from the final commit or final installed wheel.

## Captured comparison

The unchanged compsec-079 request contains 409 prompt tokens and has a
192-token output cap. At the captured window starting at offset 452, row 2,
comparisons use the actual live cache and identical four-row input within each
run. Before repair, the observer preserves all 192 original production tokens;
target replay and same-body eager replay preserve the complete 4 x 248320
logit tensor exactly. The same observer-fidelity controls are exact in the
intermediate fixed run as well.

The AX-owned fused conv/SiLU/QK-normalization path omitted intermediate
activation-dtype stores that exist in the ordinary composition. Same-QKV
post-input comparisons locate a concrete Q/K/V difference. A/B projections
match, and tape versus standard recurrence produces identical output and final
state when given identical Q/K/V. All 240 recorded compiled-versus-same-body
functional leaves are exact in each diagnostic run; these observations do not
attribute the captured divergence to the outer compile boundary.

Layer-zero values below are **maximum absolute difference / unequal elements**:

| Comparison | Pre-fix instrumented | Pre-FP16 fixed instrumented |
| --- | --- | --- |
| post_input_q | 0.000244140625 / 4,563 | 0 / 0 |
| post_input_k | 0.00390625 / 4,642 | 0 / 0 |
| post_input_v | 0.03125 / 12,798 | 0 / 0 |
| recurrent_state | 0.07229447365 / 786,401 | 0 / 0 |
| full_layer_same_input | 0.001953125 / 11,493 | 3.051757812e-05 / 2 |

Across the 48 observed linear layers, same-input recurrent-state comparisons
change from 48 differing layers to zero. This does not make every full-layer
output exact: two layer-zero hidden elements still differ after the intermediate
fix; 62 of 64 same-input full-layer hidden comparisons still differ, with a
maximum absolute difference of 0.03125. These residuals are retained in the JSON.

The original live target ranks token 28692 at 23.125 and token 898 at 23.0.
The intermediate fixed live target ties both at 23.0. Its first difference from
the **older** direct trace is index 109; its first difference from the independent
reference prefix is index 137. These are zero-based prefix diagnostics, not
same-build direct/MTP equivalence or accuracy results. The fixed run also has
its own preceding state history, not a serialized replay of the old run's cache.

## Final code and open evidence

Final `aa38f18b` restores activation rounding in default and SIMD32 prework and
both fused verifier variants, including explicit precise float32 exponential
and division followed by activation-dtype output rounding for FP16. Final exact
BF16/FP16 post-input tests pass for sequence lengths 1 and 4, ordinary and target
scopes, and v3/SIMD32 selection. The BF16 D128 fused-verifier regression passes
for sequence lengths 2 and 4, both checkpoint variants, and their state outputs.
Those unit results are separate from the pre-FP16 model diagnostics above.

The separate [final bundled-wheel record](qualification.json) satisfies the clean matching-build,
doctor, direct/MTP QA, no-silent-fallback, and paired 64-token gates on the
selected SKU. Those final results are not attributed to these intermediate
instrumented binaries. Wheel passes remain version-bound. Unchanged long-thinking
and strict line-set quality, saved recovery, and broader route consistency
remain open. MTP Tier 2, endurance, long-context decode-at-depth, multimodal
quality, multi-model add-mode, and peer ranking are separately labeled campaign
scope in the qualification contract. No quality or performance improvement is
claimed by this diagnostic report.
