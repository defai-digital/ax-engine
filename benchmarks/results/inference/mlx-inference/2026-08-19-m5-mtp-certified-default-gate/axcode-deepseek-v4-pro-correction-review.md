I've traced the full path through the actual code. Here's my assessment, verified against `mtp_model_policy.rs`, `fastpath.rs`, `runner/mod.rs`, and `native.rs`.

## Q1 — Agree: routes 2/3 belong in the gate; carve-out is correct

The factual error in the plan's premise is confirmed at the code level:

- `resolve_qwen_linear_certification_candidate(env_opt_in, exact_profile_enabled) = env_opt_in || exact_profile_enabled` — `mtp_model_policy.rs:46-51`.
- `resolve_qwen_linear_mtp_exact` returns `(true, Auto)` for any exact-eligible pack when the env is unset — `fastpath.rs:508-520`.
- Therefore at the construction site (`runner/mod.rs:1624-1627`), an exact-eligible non-MXFP4 3.8 pack gets `certification_candidate = false || (true && true) = true` **with no env**, landing on route 2/3.
- Routes 2/3 are `route_safe` (`mtp_model_policy.rs:232-239`) and `certified_default_on()` currently returns `true` for them (`mtp_model_policy.rs:262-268`), so MTP defaults ON. Bug confirmed.

**Carve-out semantics are correct.** Keying on the raw env flag (`qwen_linear_mtp_certification_candidate_from_env()`) is the right discriminator because it cleanly separates "formal harness explicitly opted in" from "product Auto-ON via exact eligibility". Your own risk flag is accurate: env set + pack not exact-eligible → `certification_candidate=true` but `exact_enabled=false` → falls through to route 4 (`QwenLinearUncertifiedDirectFallback`), which is `route_unsafe`, so `set_mtp_requested` is a no-op regardless of the `_ => true` arm. Moot, exactly as you say. The symmetric case also holds: env set + exact-eligible + MXFP4 → route 2/3 + carve-out `true` → stays on, preserving the harness recipe.

**Exhaustive `MtpModelPolicyKind` check — nothing missing.** Default-on-via-Auto (no env, no explicit request) routes are:
- `QwenCalibrated` (1) — gated (existing).
- `QwenLinearCertificationCandidateDepthOne/MultiDepth` (2/3) — the bug, gated by your correction.
- `GlmCalibrated` (5) and `Gemma4AssistantCalibrated` (6) — still ungated, but intentional: their packs ship no AXQuant `mtp` block, so there's nothing to gate on today. If you ever want a uniform "no ungated default-on MTP" posture, these need a *different* certification mechanism — not this one. Out of scope.
- `DeepseekV4CertificationCandidate` (8) — env-only, as you stated (`runner/mod.rs:1629` reads only the env). Confirmed true for V4.

Everything else is either route-unsafe (4, 7, 9) or has no drafter (0).

One mechanical note you'll hit: `certified_default_on()` feeds **two** entry points — the constructor's initial `mtp_requested` (`runner/mod.rs:1960-1963`) *and* the SDK `Auto` path (`native.rs:145-148, 175-181`). Your correction is uniform across both since both read `certified_default_on()`, so that's fine. But the test `certified_default_on_gates_only_qwen_calibrated` (`mtp_model_policy.rs:466-497`) and its `policy()` helper (line 438) currently **assert route 2/3 return `true`** — they must be updated, and the `certified_default_on`/`mtp_certified_default_on` doc comments rewritten, or the change won't compile/pass.

## Q2 — Sequencing: agree for affine, but MXFP4 is a false premise

There's a correctness issue in how you've grouped the winners.

- **4-bit (1.20x) and 8-bit (1.55x)** — affine, exact-eligible, `!has_mxfp4_linears` → route 2 by Auto. The correction flips them default-off. Interim regression accepted. ✓
- **MXFP4 (1.50x) is NOT on route 2 by default.** The construction site explicitly excludes it: `qwen_linear_certification_candidate = resolve(env, qwen_linear_mtp_exact_enabled && !has_mxfp4_linears)` (`runner/mod.rs:1626`). MXFP4 is "exact-capable but not auto-promoted." Without the cert env it lands on route 4 (route-unsafe, already default-off in the *current* code). If its 1.50x was measured on route 2, that run necessarily set `AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1`.

**Consequence:** stamping the MXFP4 pack's HF metadata (`optimized:true` / `measured_speedup`) will **not** turn it default-on. `certified_default_on()` is never consulted for it, because it never reaches route 2/3 without the env — the `!has_mxfp4_linears` exclusion is a *separate* gate upstream of the certification gate. MXFP4 default-on requires a distinct decision to lift that auto-promotion exclusion (the "checkpoint adopt" path the comment references), not just metadata.

So your "3 winners all go default-off, metadata restores them" framing overstates the blast radius in one direction and understates the work in another: the correction newly regresses only the **two affine** winners; MXFP4 was already off and needs its own change, which should not be folded into "metadata stamping." I'd sequence metadata for the two affine packs in the same train as the code; handle MXFP4 as a separate, explicit auto-promotion change.

## Q3 — Qwen3.6-27B-6bit-MTP default-off: correct

It's dense → route 1 `QwenCalibrated`, already gated in current code. 31.75 vs 31.80 is ~0.998x, i.e. marginally *below* 1.0x, and the gate is `enabled_by_default && (optimized || measured_speedup >= 1.0x)` (`mtp_tuning.rs:726-727`). A sub-1.0x neutral pack should not default on — the criterion correctly excludes it. Acceptable as-is.

---

**Bottom line:** your correction is sound and I agree with the gate scope and the env carve-out. The two things to act on before implementing: (1) the test + doc updates in `mtp_model_policy.rs` will be required, and (2) drop the MXFP4 winner from the "metadata restores default-on" sequencing — it needs a separate auto-promotion change, not a metadata stamp.