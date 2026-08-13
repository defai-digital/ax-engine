# ax-code scan — W0-shared

**Scope:** WAVE-0 wide scan of the shared MTP / architecture-registry surface.
**Model:** zai-coding-plan/glm-5.2
**Date:** 2026-08-11
**Verdict:** **No high-confidence defects.** All four check areas pass. Five
LOW/INFO observations recorded for follow-up; none block.

---

## Coverage map

| Check | Surface | Files read | Status |
|-------|---------|-----------|--------|
| 1 | convert `family_name` ↔ `ARCHITECTURE_REGISTRY` labels | `convert/model_family.rs` (full), `architecture_registry.rs` (full), `convert/mod.rs:148,1040-1199`, `convert/hf_config.rs:62-175`, `runner/manifest_validation.rs:69-103` | ✅ clean (1 LOW) |
| 2 | `mtp_model_policy` fail-closed defaults (qwen linear / deepseek v4) | `runner/mtp_model_policy.rs` (full), `runner/mod.rs:1554-1577,3148-3162,1761-1781` | ✅ clean |
| 3 | `smoke_compatible_models` dry-run matrix vs registry tiers | `scripts/smoke_compatible_models.py` (full), `scripts/test_smoke_compatible_models.py` (full), `support_tier.rs` (full) | ✅ clean (2 INFO) |
| 4 | dead code / drift on shared MTP policy surface | `mtp_model_policy.rs` accessors + all call sites (grep), `runner/mod.rs` consumers | ✅ clean (2 INFO) |

Forward-direction invariants are guarded by tests in all four areas. Reverse
directions and breadth are the only gaps (see findings).

---

## Findings

### DI-W0-A001 — `convert_family_names_are_registered` guard is one-directional (LOW)

**File:** `crates/ax-engine-core/src/architecture_registry.rs:470-503`

The test asserts every convert-emitted `family_name` is present in the registry
(forward). It does **not** assert the reverse. It also iterates a **hardcoded**
list (`CONVERT_FAMILY_NAMES`, 23 entries) rather than deriving the set from
`convert/model_family.rs`. This creates a three-way sync hazard:

```
convert/model_family.rs arms  ↔  test list (hardcoded)  ↔  ARCHITECTURE_REGISTRY
```

- Currently all three are in sync (verified: 23 convert arms → 23 test entries
  → all present in the 26-row registry). No defect today.
- **Risk:** adding a convert `family_name` arm and updating the test list, but
  forgetting the registry row, is caught. Adding a convert arm and forgetting
  **both** the test list and the registry row is **not** caught — the new family
  silently downgrades to `Compatible` via `support_tier_for_family`'s fallback
  (`support_tier.rs:77-81`) and to no layer-forward route.

**Evidence:**
- `architecture_registry.rs:497-502` iterates `CONVERT_FAMILY_NAMES` only.
- `support_tier.rs:77-81` returns `Compatible` for any unregistered label.

**Suggested action:** derive the expected family set from convert at test time
(e.g. a `pub(crate) const CONVERT_FAMILY_NAMES` exported from
`convert/model_family.rs` and consumed by the test), or add a reverse
"every registry label is either convert-emitted or documented as alias" check.

---

### DI-W0-A002 — registry labels `gemma3`, `deepseek_v32`, `gemma4_unified` are convert-orphaned by design (INFO, not a defect)

**File:** `crates/ax-engine-core/src/architecture_registry.rs:127,151,223`

Three registry rows are never emitted as `family_name` by
`convert/model_family.rs`:

| Label | Convert behavior | Runtime-valid? |
|-------|------------------|----------------|
| `gemma3` | no convert arm (model_type `gemma3` → `UnsupportedModelType`); `gemma3_text` → `embeddinggemma` | yes — `is_mlx_supported_model_family`, `config.rs:663`, `architecture.rs:324`, `runner/mod.rs:2000` |
| `deepseek_v32` | `"deepseek_v3" \| "deepseek_v32"` arm normalizes to `family_name: "deepseek_v3"` (`model_family.rs:169-175`) | yes — `manifest_validation.rs:89`, `model.rs:1709,1812,1986`, `architecture.rs:356` |
| `gemma4_unified` | `"gemma4" \| "gemma4_unified" \| "gemma4_unified_text"` normalizes to `family_name: "gemma4"` (`model_family.rs:58-68`) | yes — `manifest_validation.rs:98,770`, `metadata.rs:423` |

**Why this is NOT a defect:** these labels exist to support hand-authored or
externally-produced manifests (the runtime accepts both the normalized and
alias forms). I verified no production dispatch site is keyed **exclusively** on
the alias label in a way convert's normalization would skip:
- `metadata.rs:423` gates on `family == "gemma4_unified" || family.starts_with("gemma4")` — `gemma4` also matches.
- `weights.rs:6896` and `fastpath.rs:2846` references to `gemma4_unified` are in **test** code only.
- `deepseek_v3` and `deepseek_v32` map to the same `LayerForwardRoute::DeepseekV3` + `Certified` tier.

**Note:** the dual-labeling is a readability/drift smell (two labels, same
route/tier, only one produced by convert). A one-line comment on each alias row
stating "registry-only alias; convert normalizes to X" would prevent future
"why is this here" confusion. Lowest priority.

---

### DI-W0-A003 — `mtp_model_policy` fail-closed defaults are correct (NON-FINDING / verified clean)

**File:** `crates/ax-engine-mlx/src/runner/mtp_model_policy.rs`

Explicit confirmation of the check-2 contract:

| Scenario | Kind | route_code | `route_safe()` | fail-closed? |
|----------|------|-----------|----------------|--------------|
| Qwen linear, no cert-candidate env | `QwenLinearUncertifiedDirectFallback` | 4 | **false** | ✅ |
| Qwen linear, cert-candidate, depth 1 | `QwenLinearCertificationCandidateDepthOne` | 2 | true | opt-in only |
| Qwen linear, cert-candidate, depth>1 | `QwenLinearCertificationCandidateMultiDepth` | 3 | true | opt-in only |
| DeepSeek V4, no cert-candidate env | `DeepseekV4UncertifiedDirectFallback` | 9 | **false** | ✅ |
| DeepSeek V4, cert-candidate | `DeepseekV4CertificationCandidate` | 8 | true | opt-in only |
| Conflicting drafters | `ConflictingDrafters` | 7 | **false** | ✅ |

- Env opt-ins (`qwen_linear_mtp_certification_candidate_from_env`,
  `deepseek_v4_mtp_certification_candidate_from_env`) default to `false` and
  accept only strict truthy values (`mtp_model_policy.rs:20-43`).
- Wiring confirmed: `runner/mod.rs:1574-1576` feeds both env helpers into
  `MtpModelPolicyInputs`; nothing else can flip these flags.
- Unit tests `linear_exact_without_candidate_fails_closed`,
  `every_supported_mtp_family_has_an_explicit_policy`,
  `deepseek_v4_route_telemetry_exposes_fallback_and_candidate`, and
  `default_product_linear_route_is_not_active_without_candidate` lock the
  fail-closed behavior and telemetry.

**No issue.** Recorded so the next wave does not re-investigate.

---

### DI-W0-A004 — smoke matrix covers 5/26 families; no Experimental or specialized-route Certified family exercised (INFO)

**File:** `scripts/smoke_compatible_models.py:86-117`

`SMOKE_MATRIX` exercises: `qwen3` (certified), `gemma4` (certified), `llama3`
(compatible), `mistral3` (compatible). Tier assignments are correct and the
dry-run cross-check (`parse_registry_tiers` + `validate_matrix`) validates them.

**Gaps (observations, not defects):**
- No **Experimental**-tier family is in the matrix (`deepseek_v4`,
  `diffusion_gemma`). This means DI-W0-A003's DeepSeek V4 fail-closed path has
  **no end-to-end smoke coverage** — only unit tests in `mtp_model_policy.rs`.
  This is expected: the deepseek_v4 registry note states the repo-owned runtime
  graph is unimplemented, so no real checkpoint can load yet.
- No specialized-route Certified family is exercised (`glm4_moe_lite`,
  `gpt_oss`, `deepseek_v3`, `qwen3_vl`, `qwen3_5`, `qwen3_next`). The matrix
  intentionally favors small generic checkpoints; breadth is traded for cost.

**Suggested action (low priority):** when the deepseek_v4 graph lands, add a
smoke row so the fail-closed → cert-candidate promotion is covered end-to-end.

---

### DI-W0-A005 — `QwenLinearCertificationCandidateMultiDepth` has no gate default while DepthOne does (INFO)

**File:** `crates/ax-engine-mlx/src/runner/mtp_model_policy.rs:253-261`

`gate_default_for` returns `Some(CERTIFICATION_DEPTH_ONE_GATE /* 0.0 */)` only
for `(QwenLinearCertificationCandidateDepthOne, Qwen)`. The **multi-depth**
certification-candidate kind (route 3) returns `None`, so it runs with the
global resolver default rather than a pinned 0.0 gate.

- This is very likely intentional (the depth-1 formal harness is the only
  calibrated contract; multi-depth is still open). The asymmetry is just
  undocumented.
- The kind is otherwise healthy: reachable, `route_safe == true`, covered by
  `every_supported_mtp_family_has_an_explicit_policy`.

**Suggested action:** a one-line comment on `gate_default_for` noting
"multi-depth retains the global default until its harness is calibrated" would
make the asymmetry deliberate-looking rather than accidental-looking.

---

## Dead code candidates

**None.** The MTP policy surface was inspected end-to-end:

- All 10 `MtpModelPolicyKind` variants are reachable from `from_loaded`
  (`mtp_model_policy.rs:111-186`) and have route codes (`route_code`,
  `as_str`-style telemetry).
- Every `pub(super)` accessor (`is_qwen_linear_direct_fallback`,
  `is_deepseek_v4_direct_fallback`, `is_deepseek_v4_certification_candidate`,
  `is_qwen_linear_certification_candidate`, `has_conflicting_drafters`,
  `has_attached_drafter`, `max_depth`, `route_safe`, `qwen_gate_default`,
  `glm_gate_default`) is consumed in production (`runner/mod.rs:1227-1244,
  1761-1781, 3154-3162, 8261, 9912-10070`) or telemetry
  (`append_route_decisions`).
- The private `model_gate_default` is layered indirection (called once) but
  live; not dead.
- Both env helper fns are wired at `runner/mod.rs:1574,1576`.

No removal candidates identified.

---

## Completeness self-score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Check 1 coverage | 9/10 | Read full convert + registry + runtime-validation surfaces; verified forward sync live; identified the one-directional guard as the real (LOW) residual risk. Did not exhaustively diff every `model_type` alias in `hf_config.rs` (not needed — `family_name` output set is what the registry keys on). |
| Check 2 coverage | 10/10 | Full policy file + construction site + every call site + telemetry + tests. Fail-closed proven for both target families. |
| Check 3 coverage | 9/10 | Full smoke script + its test file + tier source. Confirmed tier consistency live. Breadth gap noted as INFO. Did not run `--dry-run` (regex/parser already unit-tested and assertions are source-level). |
| Check 4 coverage | 9/10 | Grepped every accessor + variant across `crates/`; confirmed all live. Did not run `cargo` (scan is read-only; no production edits). |
| Confidence calibration | high | Findings are evidence-anchored (file:line for every claim). No speculation reported as a defect. The scan's negative result (no HIGH/MEDIUM defect) is itself the deliverable. |

**Overall:** scan complete; surface is well-guarded. The only actionable item is
DI-W0-A001 (harden the convert↔registry guard to be structural rather than
list-based), and it is LOW priority because the lists are currently in sync.
