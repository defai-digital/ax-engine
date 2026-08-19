# Correction: the "pre-existing MTP-on drift" was a measurement-protocol error

The README in this directory (and the 08-19 probe logs) compared
**MTP-on** — which the model policy auto-runs under the exact profile —
against **MTP-off without the exact profile**. Those are two deliberately
different arithmetic contracts: since `8983bbb1` the non-exact decode
emits from the runtime 2-bit lm_head overlay while the exact profile
routes through the invariant kernels. The bisect that landed `8983bbb1`
as "first bad" was detecting that profile split, not a broken
certification property.

The fair comparison (both sides `AX_MLX_QWEN_LINEAR_MTP_EXACT=1`,
Qwen3.6-27B 6-bit MTP, fixed binary, 120-token greedy CLI):

    offexact 8 01c7e221cf2948be
    onexact  8 01c7e221cf2948be

**MTP-on-exact ≡ MTP-off-exact, token-for-token and deterministic.**
The certified exact contract is intact on HEAD. The earlier MXFP4 4-way
"clean" probe was clean because MXFP4 auto-enables exact even with
`AX_NO_SPEC`, making that comparison fair by accident.

Remaining genuinely open items, correctly scoped:
1. Server default-policy (ngram-accel stack) MTP responses varied
   run-to-run in the 08-19 probes while CLI exact streams are
   deterministic — investigate the serving-side policy stack separately.
2. 4bit-MTP acceptance ~40% on this prompt class (utility gate rightly
   bypasses MTP) — a draft-quality/economics question, not a
   correctness defect; that pack's MTP tier was never certified.

The row-by-row verify lm_head projection landed with this branch stays:
it closes the leading-dependence hole for the **non-exact** profile
(batched S>=2 verify read bf16/qmv_wide logits while non-exact singles
read the 2-bit overlay) and is arithmetic-neutral under the exact
profile, where the invariant kernels already intercept per row.

## Second correction (same day): the "fair probe" was vacuous

Route telemetry from the probe artifacts shows both sides of the "fair"
comparison ran the MTP route: `AX_MLX_QWEN_LINEAR_MTP_EXACT=1` is a
formal-harness convention that deliberately takes the certified MTP
route even under `AX_NO_SPEC` ("Env opt-in still wins", runner
mtp-policy resolution). `offexact` therefore was not an MTP-off
baseline — the token-identical hashes proved determinism, not parity.

Net honest status of the MTP-on ≡ single-step property:
- **Proven**: CLI exact MTP streams are deterministic run-to-run;
  the dense-head performance fix and the row-by-row verify projection
  are parity-neutral-or-better by construction.
- **Open**: the certified equivalence itself cannot be validated by
  improvised probes (two protocol errors in one day demonstrate why);
  it needs the formal QWEN36-LINEAR-MTP-TIER2 harness run.
- **New finding to file**: under the identical env, the Qwen3.6 6-bit
  pack reports `exact_enabled: 1` while the Qwen3.8 **8-bit** pack
  reports `exact_enabled: 0` with `exact_eligible: 1`, and its stream
  changes between selection modes — a pack-dependent exact-activation
  discrepancy in the model-policy gate worth its own investigation.
