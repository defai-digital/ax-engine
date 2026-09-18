# Qwen 27B FFN attribution and QMM contract repair

Status: **Candidate; not ship-ready**.

The production repair is `d121f107`: custom verify QMM returns the unbiased
affine projection, both production callers add the separate dense Linear bias
once, and mixed affine metadata dtypes use stock MLX promotion. Matching-dtype
BF16/FP16 custom kernels retain their arithmetic and admission defaults.
These are independently reproduced generic integration defects. They do not
repair the observed bias-free, same-dtype Qwen FFN difference.

## Reproduction and source validation

The original actual `qw` caller returned 1.0 for zero input and a 0.5 bias.
BF16 input with float32 affine scale returned BF16 where stock MLX returned
float32. Both regressions fail before repair. Caller review also caught that
the intermediate helper-only change would omit bias in the direct LM-head
caller; its failing regression and final repair are preserved.

All eight focused regressions pass, including actual BF16/FP16 `qw` and
LM-head calls, independent scale/group-bias promotion cases, split-K and MSG
routes. Existing numerical tolerances are unchanged. Full Rust tests,
formatting, release-pyext development install, scripts, both Qwen dry runs,
primary-claim checks and Python tests (209 passed, 26 skipped, 140 subtests)
pass. CI-policy Clippy passes; strict `-D warnings` still fails on existing
untouched core restriction lints. The five existing CI exceptions were not
changed. Commands, source hashes, expected failures and advisory Grok/Qwen
review receipts are in [source-validation.json](source-validation.json).

## Clean wheel and selected-SKU execution

The clean `d121f107` bundled wheel has SHA256
`89d43b038bbd11950e43472e41175c15b2e8a6854d4dcbdfe908c392fb280b29`.
It installs and imports in a fresh Mac mini M4 Pro 64 GB environment with only
AX Engine and pip installed; loaded MLX/JACCL paths resolve to the wheel.
Doctor, source/model/wheel identity, 32/32 hard QA and 7/7 surfaces per route
pass, with no soft failures or skipped surfaces. Direct counters are zero;
MTP records 60 draft and 82 verify tokens. The paired 64-token probe matches.
[qualification.json](qualification.json) retains this exact artifact-bound scope.

All eight fresh-process 192-token outputs across four selected cases match
their corresponding clean aa38f18b baseline route. Direct/MTP agreement remains
2/4: 079 still splits at index 116 and 087 at 155. These are zero-based output
indices. Generic contract repairs have not eliminated these differences.
[cold-controls.json](cold-controls.json) preserves input/output hashes and
comparison results; raw external question data remains outside the repository.

## Actual-input attribution

The independent observer is based on **aa38f18b**, before the generic repair.
Its 192 production tokens match the clean wheel. Full live logits and all
288 compiled/eager leaves match. At the historical layer-zero window, the
first differing FFN cuts are gate/up projections (5/13 BF16 elements); nested
compiled SwiGLU matches its same-input eager body. Actual target gate/up use
custom QMM; ordinary gate/up use stock MLX. Both metadata and activations are
BF16, affine4/group64, without dense bias, excluding the two repaired contract
defects from this captured fixture.

The reference comparison shows different float reduction orders, without a
demonstrated index/formula defect. A separate diagnostic that excludes custom
QMM diverges earlier, at index 109, and does not restore direct/reference
identity. No QMM default, reduction formula or tolerance was changed on that
basis. This window is not the final build's first-token split and does not
establish its full causal chain. See [reference-comparison.md](reference-comparison.md)
and [ffn-observer.json](ffn-observer.json).

## Quality contract

The original selected LINE_SET prompts ask for a primary bug location, with
a small adjacent group allowed when needed. AX retains a stricter exact-gold-set
metric. The source scorer's accepted-location contract must be distinguished
from that metric; complete enumeration is not established by the original
wording. Raw grades and gold sets are unchanged. The retained aa38f18b twelve-case
replay remains 0/12 exact-set passes per route, with seven accepted-location
subset detections and five truncations per route. This is neither population
accuracy nor twelve demonstrated semantic failures.

A separate cold, two-case `complete_set_v1` experiment replaces only the
location instruction, allows noncontiguous root-cause and affected-operation
lines, and retains gold, budget and sampling. It is a new stricter task; it
does not replace the original experiment or a quality qualification suite.
Reused accepted-location keys are not thereby proven exhaustive causal sets.

All eight fresh-process requests completed without infrastructure errors.
Both original 079 requests reach the 512-token cap; both original 086
requests return an accepted location with strict-wrong status. The complete-set
variant returns `Answer: 0` for 079 and `Answer: 3` for 086 on both routes;
there are zero strict passes. This does not demonstrate a quality repair.
Exact request/response hashes, unchanged recomputed grades and counts are in
[quality-contract-controls.json](quality-contract-controls.json). The original
full twelve-case replay was not rerun on d121f107.

The pinned independent reference (MLX 0.32.2, mlx-lm 0.31.3) produces the same
`Answer: 0` and `Answer: 3` on those two stricter prompts. The installed AX
template/tokenize endpoints and reference tokenizer agree on the full prompt
bytes and every input token (436 and 408). All 22 model files were rehashed;
2,810 installed reference files match their distribution RECORDs. Both fresh
reference workers stop at EOS after four non-EOS tokens. Visible text,
completion count, finish reason and unchanged grade match both AX routes.
AX chat output token IDs were not captured, so this is not a measured output
token-ID equality claim. These selected answers also occur in the independent
reference; this does not establish correctness or explain the longer route
split. See [independent-reference.json](independent-reference.json).

## Remaining gates

Broader numerical-route consistency, a validated aligned quality protocol,
original long-thinking/recovery acceptance, long-context behavior, MTP Tier2,
and endurance remain open. No broad quality gain or release-ready promotion
is claimed. The wheel and selected-SKU execution results here are scoped
to their exact source and artifact identities. No push or public release was
performed by this tranche.
