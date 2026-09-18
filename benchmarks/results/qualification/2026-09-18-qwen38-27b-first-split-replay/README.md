# Qwen 27B first-split replay evidence

Status: **Candidate; not ship-ready**. This retains a bounded numerical
diagnostic and a separate control-flow regression, without a release
qualification or quality improvement claim.

## Actual first-split window

Three fresh server processes on the Mac mini M4 Pro 64 GB ran the same raw
409-token input with a 192-token budget, greedy sampling and seed 0. Clean
installed direct and MTP outputs match their respective retained baselines
at all 192 positions. The observer's 192 production tokens match clean MTP.
All indices below are zero-based. Direct and MTP share the first 116 output
tokens; output 116 is token 279 for direct and token 6397 for MTP.

The admitted observation captures the actual MTP window starting at absolute
input offset 522, with materialized inputs `[279, 1906, 314, 279]`. Row 2
consumes input position `409 + 116 - 1 = 524`. Both accepted draft tokens
are retained, then the live correction is emitted: `[1906, 314, 6397]` at
output indices 114 through 116. The full selected input prefix is checked
against the common history after lazy draft IDs are materialized.

The observer clones the live pre-verification cache before attention-storage
rebinding and first replays the actual verifier under the original live
scopes. Live and replayed float32 logits have shape `[4, 248320]`; all
**993,280 elements are bit-exact**, with zero maximum absolute difference.
Both take the same non-whole-compiled route. This replay is the admission
control before ordinary routes are compared. All 128 logical arrays in each
of three cache views (saved snapshot, live source and live verifier) retain
identical shape, dtype, finite status and value hashes before and after the
diagnostic controls. Their logical positions are 522, 522 and 526,
respectively. This is immutability within each view, not equality between
the three views. The live committed cache subsequently ends at 525.

| Route on the saved live MTP history | Token 279 | Token 2849 | Token 6397 | Selected-row result |
| --- | ---: | ---: | ---: | --- |
| Live target | 21.75 | 21.75 | 21.875 | 6397 emitted |
| Faithful same-scope target replay | 21.75 | 21.75 | 21.875 | Same full logits |
| Ordinary batch, target/QMM scopes disabled | 21.75 | 21.75 | 21.875 | Unique maximum 6397 |
| Ordinary sequential singleton, same initial cache | 21.75 | 21.75 | 21.75 | Three-way maximum tie |

For singleton logits, 279 follows from the first-index tie-break rule. This
is an inference from the captured finite row, **not an independently
materialized MLX argmax result** for that control. It agrees with the actual
clean direct output ID. The ordinary batch differs from the live verifier
in 716,051 full-logit elements (maximum absolute difference 0.125), while
preserving this selected-row winner. Live versus singleton differs in
185,298 selected-row elements, also by at most 0.125.

These controls establish that current forwarding-route differences on the
same initial MTP state are sufficient to change the selected decision under
the stated tie-break. They do not observe the actual direct-server cache.
Ordinary batch forwarding also uses a different LM-head helper; this is not
a single-variable test of tensor shape or custom QMM. Disabling the custom
verify QMM scope still leaves 6397 as the unique ordinary-batch maximum, so
custom QMM is not established as the unique cause. No default is changed.

## Standalone oracle preparation: before, after and reversal

The existing standalone oracle initially seeded its cache through
`chunked_prefill_with_final_hidden`. With the requested 116-token common
prefix, it stopped at output 109: expected token 11870, observed 8240.
It correctly reported no comparable boundary and exited 1.

Changing only this diagnostic's prefill entry to `chunked_prefill` validates
all 116 supplied prefix IDs and predicts token 279 at output 116. The log
reports `replayed_generated_tokens=115`: the initial prediction is checked
separately, followed by 115 replayed steps. Re-running the original binary
again fails at output 109 with the same token pair and byte-identical log.
Case, all 22 model-file hashes and loaded native-library hashes agree across
the three runs. This isolates a probe-preparation issue; it is not a
production repair. The exact diagnostic patch is retained.

The corrected oracle already loses direct-server fidelity on its second
diagnostic singleton prediction: output 117 is 2849, while the actual direct
server returns 40278. Its ordinary batched arm returns 40278 at that position.
The corrected probe therefore validates only the requested prefix and first
next prediction. Neither matching token 279 nor the corrected prefill proves
full-logit/cache identity or sustained direct-server trajectory identity.
The standalone batched arm also does not reproduce actual MTP target scopes.

## Artifact identity and retained evidence

The installed production runtime is clean `d121f107999e3b8627f60455a79bdd0dad7ada6c`;
its bundled wheel SHA256 is
`89d43b038bbd11950e43472e41175c15b2e8a6854d4dcbdfe908c392fb280b29`.
The observer and oracle source base is the later evidence-only commit
`0f7f2d2d0ae6e66a80c7d6f746fbb058efd2a951`. The model is
`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` at
`3e290738e96972307c6aeb9934ab170ca0eae1c1`. The collector rehashed all 22
model files and verified installed native files against the retained wheel.
The observer loader trace resolves MLX/JACCL to those installed libraries;
build-library and installed-library hashes are recorded separately.

- [case.json](case.json), [request.json](request.json), the three response
  files and [runs.json](runs.json) preserve complete input/output IDs,
  HTTP-body hashes, process cleanup and route telemetry.
- [observer.log](observer.log) retains the full loader log and all 19 probe
  events, including per-array cache witnesses. [observer-analysis.json](observer-analysis.json)
  gives the selected comparisons, and [observer.patch](observer.patch)
  retains the exact observation-only source change.
- [observer-provenance.json](observer-provenance.json) and the three build
  receipts bind source, patch, binaries, model and loaded native libraries.
- [oracle-analysis.json](oracle-analysis.json), `oracle-old.*`,
  `oracle-direct-prefill.*` and `oracle-old-repeat.*` preserve the failed,
  corrected and reversed runs, including their limits.
- [publication.json](publication.json) distinguishes original raw hashes
  from published hashes. Private execution-root paths are consistently
  replaced with `<SKU_WORKSPACE>`; affected copies are explicitly labeled.
  The two patches, request and response bodies are retained byte-exact.

Full logits and cache tensors are not retained as tensor dumps. Their
complete comparisons and hashes were computed in the observer; the log and
source patch retain that measurement. The offline check validates these
recorded witnesses, without recomputing model arithmetic:

```bash
python3 benchmarks/results/qualification/2026-09-18-qwen38-27b-first-split-replay/verify_evidence.py
```

## Scoped forced-replay correction

A separate control-flow defect exists in explicitly forced greedy singleton
replay. Previously, batched acceptance chose a draft prefix and the replay
consumed it without checking intermediate singleton predictions. Recomputing
the final correction could not retract a wrongly accepted draft. The new
acceptance helper treats the provisional batched count as an upper bound,
checks each proposed token before consuming it, and stops at the first
singleton mismatch. The resulting count consistently selects cache history,
emitted prefix, proposal hidden row and accepted-token telemetry. The emitted
correction remains unconsumed by the target cache until the next decode step.

This correction is limited to explicitly forced, non-optimistic greedy replay
without logits processors. Default relaxed verification, sampled acceptance
and processor behavior retain their existing paths. The separate experimental
skip-state mechanism is discarded and its capture disabled for forced greedy
replay, including when combined with optimistic mode; optimistic acceptance
itself is outside the singleton revalidation guarantee. Replay locally disables
and restores exact, target, relaxed, whole-trace and verify-QMM scopes.

The meaningful failing-before regression injects false batched predictions
into a real tiny dense-attention fixture with token-dependent, nonzero KV.
It retains three drafts where the singleton first mismatch requires zero.
Another failing-before test demonstrates stale skip-state reuse. These prove
the bounded acceptance/cache control-flow defects; they do not numerically
reproduce the default Qwen 27B split above. Three final focused tests, 13 related
linear-MTP tests and 50 ngram tests pass. They cover first/later mismatch,
full/empty acceptance, the batched acceptance cap, scope restoration and
production helper bookkeeping. They do not execute full runner/head-refold
integration or a 27B GDN numerical qualification.

The final runtime patch SHA256 is
`90814b94b24df7ddfafcec1f144676c1081c89c92118a49e2ec8bdb9a005098e`;
this is a patch hash, not a commit ID. Meaningful negative runs, exact patches,
final test receipts and the reference comparison are in [regression](regression/README.md).
The reference review distinguishes cache rollback from acceptance revalidation;
no reference source or implementation was copied. Broader source validation
is recorded separately in [source-validation.json](source-validation.json).
Source hashes remain unchanged across all 11 checks. The strict Clippy command
fails on existing untouched core restriction lints; its first error is an
`unwrap()` in `kv_rollback_after_eviction_repro.rs:13`. Clippy under the existing
five CI warning exceptions passes. All other checks pass: focused regression,
formatting, full Rust tests, development extension install, scripts, Python
(209 passed, 26 skipped, 140 subtests), both Qwen contract dry runs and primary
claims. Dry runs do not execute hardware qualification. The complete-patch Claude
review finished and was source-adjudicated; Grok and Qwen through AX Code
complete-patch reviews and their short retries timed out. These are advisory outcomes, not test passes.
Final-wheel qualification for this correction is **pending**. The earlier
installed-wheel observer remains evidence about its recorded d121f107 artifact.

## Remaining scope

Actual direct-state capture, the longer trajectory difference and case 087
remain separate work. This evidence closes no release-quality, MTP Tier 2,
long-context or endurance gate. The forced-replay regression is separate from
the original default-route split, and its fresh-wheel validation is pending.
Existing qualification remains scoped to the artifact and
tests in the [preceding evidence](../2026-09-18-qwen38-27b-ffn-qmm-contract/README.md).
