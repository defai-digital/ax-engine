# Reference contract comparison for forced greedy replay

The reference inspection preceded the correction design. The implementation
reuses AX's existing singleton `forward_argmax` and committed-prefix replay
helper; no reference source, comments, assets or implementation were copied.
`reference-identity.json` records the exact local revisions and file hashes.

## What the reference contracts establish

MTPLX revision `6459194642ba091ba7c17c3aba889b808340c3fc`:

- `.internal/reference/MTPLX/mtplx/gdn_capture.py:3126–3145` selects the
  retained convolution state for the last consumed position. Lines 3166–3184
  select the corresponding recurrent state and trim full-attention history.
  This supports using one final accepted-prefix count consistently across
  target cache and hidden-row boundaries; it does not revalidate that count
  against canonical singleton predictions.
- `.internal/reference/MTPLX/mtplx/benchmarks/runners/capture_commit_equivalence.py:95–125`
  checks a committed prefix against sequential reconstruction and applies the
  same next input to both caches. This separates a cache-commit comparison from
  a token-acceptance guarantee.
- `.internal/reference/MTPLX/mtplx/benchmarks/runners/batch_equivalence.py:69–84,132–172`
  compares batch and sequential outputs after separately prefilling the same
  prompt. It does not prove that an arbitrary live AX batch decision equals
  the canonical singleton decision from the actual live cache.

oMLX revision `b18b4b0ff85951ba16e3ea9f183114bc14842588`:

- `.internal/reference/omlx/omlx/patches/mlx_lm_mtp/batch_generator.py:3190–3205`
  derives greedy acceptance from batched target rows. Lines 3231–3303 use the
  final accepted count after further bounds for emitted positions and counters.
  This is a batched-target acceptance contract, not an independent singleton
  acceptance oracle.
- `.internal/reference/omlx/omlx/patches/mlx_lm_mtp/qwen35_model.py:567–614`
  retains the confirmed token and the accepted draft prefix, rebuilding linear
  state from saved projected inputs and trimming full-attention history.
  The supplied acceptance count is an input to rollback, not a fresh decision.
- `.internal/reference/omlx/omlx/patches/mlx_lm_mtp/batch_generator.py:1511–1517,3428–3469`
  distinguishes the pre-verification snapshot from a one-token rollback and
  delegates partial accepted-prefix restoration to the model hook.

These comparisons rule out treating successful cache rollback as proof of
singleton-equivalent acceptance. They do not establish that reference engines
promise or implement AX's explicit forced-replay guarantee.

## AX-specific correction and limits

In the base AX source at `0f7f2d2d`, `runner/mod.rs:10009–10054` computes batched
acceptance first and then replays that accepted prefix.
`ngram_accel.rs:1403–1423` recomputes the final correction but does not inspect
intermediate singleton choices. AX already has a separate singleton acceptance
loop for Gemma assistant verification at `ngram_accel.rs:1477–1535`; its zero
hidden placeholder is unsuitable as a new Qwen proposal-hidden contract.

The bounded candidate therefore adds acceptance revalidation around AX's
existing singleton replay primitive, preserving real batched proposal hidden
rows at the corrected boundary. Acceptance is capped by the provisional batch
count. It does not expand sampled rejection handling or processor guarantees,
and it does not change the default relaxed target path.

The supplied primary token and initial cache are preconditions. In particular,
matching a reconstructed prefix's IDs does not prove live direct-cache identity,
and this correction does not establish whole-request direct/MTP identity.
Explicit approximate optimistic acceptance remains excluded. Forced greedy
replay disables skip-state consumption and capture to prevent a stale batched
row from injecting the next primary through that separate experimental path.

The whole-trace marker is normally scoped to synchronous closure construction
in `model/whole_verify.rs:78,437`. Replay is called after target evaluation; the
new helper nevertheless locally disables and restores that marker alongside
the exact, target-verifier, relaxed-session and verify-QMM markers.

Final independent source audit found no blocker. The dense fixture and helper
tests prove the bounded control-flow regression, not a 27B GDN numerical result,
full runner/head-refold integration, performance, quality or release readiness.
