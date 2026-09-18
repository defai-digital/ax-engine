# Qwen residual target-state investigation

Selected SKU: Mac mini M4 Pro 64 GB (Mac16,11). Production wheel/source:
`4cbda8b5`, checkpoint `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` at
`3e290738e96972307c6aeb9934ab170ca0eae1c1`.

## What is established

Four known direct/MTP differences persist with identical raw input tokens,
greedy seed 0, a 192-token cap, and a fresh server for every case and arm.
Turning off only the LA output projection fusion produces the same MTP token
arrays. The previously repaired scheduler prefix omission and LA projection
precision defect do not explain all remaining differences.

[Cold controls](cold-controls.json) preserve exact requests, output tokens,
overrides, active MTP counters, and installed artifact identity. First differing
output indices are zero-based:

| Case | Default MTP versus direct | LA projection off versus direct |
| --- | --- | --- |
| compsec-077 | 161 | 161 |
| compsec-079 | 46 | 46 |
| compsec-087 | 122 | 122 |
| compsec-092 | 182 | 182 |

On compsec-079, full checkpoint capture and disabling FA storage rebind each
produce exactly the default MTP output, still splitting at index 46. The
existing singleton-replay override delays the split to index 191 but does
not eliminate it. These are diagnostic overrides, not fixes or qualification.

## State and logits

`276bd1d4` extends the existing state oracle with an optional generated-prefix
argument. It replays each token through singleton decode and aborts if any
prediction differs from the supplied prefix. It never appends generated tokens
to the prefill prompt. It also reports top-two logits and row differences.
This protects the validity of the boundary comparison; it does not establish
production MTP correctness merely by exiting successfully.

For compsec-079 the oracle reproduces all 46 common generated tokens before
the divergence. At the resulting common state, the ordinary singleton and
ordinary two-row shared-model forwards choose token 898, but their full
logits and recurrent states differ numerically. Both report tied top logits:

| Observed path | Token 898 | Token 28692 | Selected token |
| --- | --- | --- | --- |
| Singleton from validated direct state | 23.125 | 23.125 | 898 |
| Two-row shared-model forward from that state | 23.125 | 23.125 | 898 |
| Singleton from cloned live MTP state | 23.125 | 23.125 | 898 |
| Ordinary four-row forward from that live state | 23.125 | 23.125 | 898 |
| Live production MTP verifier | 23.0 | 23.125 | 28692 |

The live verifier trace observes absolute input offset 454, row 2 of a window
starting at offset 452. Its pending drafts are `[2972, 2239, 28692]`; its target
predictions are `[2972, 2239, 28692, 92217]`. Acceptance follows the actual
target argmax. The observed difference is in target scores, not a demonstrated
acceptance-rule or argmax-selection defect.

[State observation](state-observation.json), [oracle log](state-oracle.log),
and the [observation-only patch](target-logits-observation.patch) retain the
source, patch, binary, and artifact hashes. The instrumented server reproduces
all 192 production MTP output tokens. It is a separate diagnostic binary,
not an installed release wheel. Its baseline wheel supplies native libraries;
the two binary identities are recorded separately.

A tie in one snapshot does not establish a universal BF16 tolerance, identify
every contributing operation, or justify a release waiver. The ordinary
shared-model oracle is not the live compiled MTP verifier profile. The follow-up
controls clone the actual live MTP cache before the same verification window.
Both singleton replay and ordinary four-row forwarding using identical inputs
recover the tie. Thus MTP-specific target arithmetic is sufficient to cause
this local split; prior state corruption is not required to explain it. This
does not identify the responsible operation or rule out other state drift.
Both follow-up instrumented servers preserve all 192 production tokens; their
source-bound observation patches and binary hashes are recorded separately.
The compiled whole-linear-layer cache includes the actual layer index, so
a missing layer index in that cache key is not supported as the cause.

## Independent reference and quality limits

The [independent reference](independent-reference.json) uses unchanged pinned
MLX 0.32.2 / mlx-lm 0.31.3, transformers 5.15.0 and tokenizers 0.22.2. Input
tokens are identical. For compsec-092, MTP matches the reference for all 192
tokens while direct differs at index 182. Neither AX route matches all four
reference prefixes. For compsec-079 the reference has a separate 512-token
budget; its comparison with the 192-token AX controls is explicitly prefix-only.
These selected cases do not establish overall fidelity or accuracy.

The reference's original cases with a 512-token output budget return
Answer 19 for compsec-079 (gold 18-19) and Answer 3 for compsec-086
(gold 3,13-15). Both fail unchanged strict line-set grading. The original
prompts allow a single best line, while the scorer requires the complete
gold line set.

A separate [two-case wording diagnostic](span-wording-diagnostic.json)
returns Answer 0 for compsec-079 and Answer 3 for compsec-086; both retain
their recorded failing grades. However, the altered prompt requests a
contiguous span including all intervening lines, while compsec-086 retains
the non-contiguous gold set 3,13-15. That case is therefore not a clean
control for wording effects, and these results do not resolve the
prompt/scorer mismatch. The diagnostic does not replace original grades
or acceptance results.

## Validation and reproduction

The oracle enhancement passes its prefix-validation tests, formatting, full
Rust tests, repository CI-policy Clippy, complete script tests, Python
(209 passed, 26 skipped, 140 subtests), both Qwen dry-run contracts, and primary
claim checks. [Validation](source-validation.json) binds log hashes to the
source. Clippy's existing force-warn restrictions remain unchanged.

`c9a7ff97` adds a BF16 affine4 FFN regression with exact element equality
between compiled and imperative S4 execution. It passes, as do formatting,
CI-policy Clippy, full Rust and Python suites, both dry-run contracts and
claim checks on that commit. This small deterministic fixture does not prove
production-shape FFN or whole-layer equivalence; it narrows a coverage gap.

Build the oracle with the pinned toolchain:

```sh
rustup run 1.97.1 cargo build --release -p ax-engine-mlx --bin linear_mtp_state_oracle_probe
```

Pass the pinned model directory, comma-separated prompt tokens, and
comma-separated common generated prefix from `state-observation.json`.
The probe requires native Metal and the matching MLX library. For raw controls,
replay `cold-controls.json` requests against freshly started product servers
with the recorded route flags and overrides. The diagnostic trace requires
its recorded base commit plus the observation patch, not the release wheel.

No production arithmetic, acceptance rule, gold, or fastpath default changes
in this investigation. The prior scoped qualification remains version-bound;
these diagnostics do not promote it to release readiness. Original long-thinking
QA, saved recovery, broad route consistency, quality, and endurance remain open.
Status: **Candidate; not ship-ready**.
