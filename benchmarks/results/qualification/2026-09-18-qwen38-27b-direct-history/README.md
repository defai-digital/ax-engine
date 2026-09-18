# Qwen 3.8 27B direct-history diagnosis

**Candidate / not ship-ready.** The history probe now distinguishes synchronous
singleton replay from the production direct-pipeline API sequence. It rejects a
mismatching prefix before running downstream comparisons. This repairs an
ambiguous diagnostic contract; it does not repair default MTP numerical parity.

## Findings on the M4 Pro 64 GB SKU

- Both standalone modes reject the retained 192-token direct-server sequence at
  output index 117: expected 40278, predicted 2849. Pipeline replay repeats the
  same failure. Replacing the synchronous API sequence is therefore insufficient.
- One- and two-token boundaries pass with cache lengths 409 and 410. An
  118-token prefix ending in the standalone prediction 2849 passes; replacing
  that token with 0 fails. The original collector label `pipeline-reject` refers
  to the former arm; it is an accepted altered-history control, not a rejection.
- A separate observation build captures the actual direct server at input
  offset 524 (token 314). All 248,320 logits match both a faithful lazy replay
  and a synchronous singleton replay bit for bit. All three materialized
  argmax results are 279. Both replay caches equal the actual post-step cache;
  the source snapshot and live cache remain unchanged during the controls.
- Three fresh server processes (clean, observer, clean repeat) reproduce all
  192 retained direct tokens. Both loaded native libraries resolve to the
  previously qualified wheel, with matching model and native-file hashes.
- At the common cache boundary 522, all 128 logical array hashes differ between
  the observed direct history and the previously retained MTP history: 48
  convolution states, 48 recurrent states, 16 keys and 16 values. Their shape,
  dtype and cache metadata match. Equal preceding token IDs did not establish
  equal cache contents. The two observations have different recorded source
  bases; this is a comparison of retained observations, not a same-build
  intervention proving the first cause of drift.

Together with the prior same-MTP-state S4/S1 control, these results separate
history differences from execution-path differences. They do not identify one
faulty kernel or establish when each state first diverged. The standalone/live
mismatch, default MTP splits at 116 and 155, quality acceptance, Tier 2 and
endurance remain open. No arithmetic default, tolerance, gold answer or release
status changes.

## Source and validation

The diagnostic patch is based on `85d55200`; `build.json` binds its patch,
source and executable hashes. The observation-only server patch and its build
receipt are retained separately and are not installed into the product.
`reference-review.md` records the MTPLX/oMLX design comparison and the checked
DeepSeek/MiniMax review conclusions. References are not implementation sources.

Focused tests, the full Rust suite, development extension, scripts, both
qualification dry runs, primary-claim checks and Python tests pass (209 passed,
26 skipped, 140 subtests). CI-policy Clippy passes. Strict Clippy retains the
existing restriction-lint failures beginning at
`crates/ax-engine-core/tests/kv_rollback_after_eviction_repro.rs:13`.
These are diagnostic and regression checks, not new wheel qualification.

## Reproduction

Build `linear_mtp_state_oracle_probe` with pinned Rust 1.97.1 and the
`release-server` profile. On the SKU, with the pinned AXQ pack and native
libraries from the qualified wheel, pass the raw `prompt` and
`baseline_direct_output` arrays from the preceding report's `case.json` as
comma-separated arguments, followed by either:

```
--validate-history=synchronous
--validate-history=direct-pipeline
```

Both full-history commands are expected to exit 1 at index 117 for this retained
case. Omitting the option retains the synchronous two-token comparison mode.
The validation mode reports its history identity and stops before that comparison.
An exit 0 only validates the supplied prefix under the named standalone route.

Verify the retained evidence without a model or network:

```
python3 -B benchmarks/results/qualification/2026-09-18-qwen38-27b-direct-history/verify_evidence.py
```

Raw host paths and loader logs remain internal. Public request/response bodies
and structured tensor witnesses preserve the admitted evidence without host
identifiers. No push or release publication is part of this change.
