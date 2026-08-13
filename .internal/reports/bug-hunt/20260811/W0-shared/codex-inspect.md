# Codex inspect — W0-shared

## Executive summary

Wave 0 is **FAIL** with three high-severity open issues and three hardening gaps. The known Whisper registry orphan exists in tracked `HEAD` but is already fixed by the current uncommitted working-tree changes.

Current source-level checks found 23 converter-emitted families, 26 unique registry labels, and no converter output missing from the working-tree registry. `smoke_compatible_models.py --dry-run` passes. No high-confidence defect was found in the scoped MTP policy or routing logic. No Cargo build, Rust test suite, or real-weight smoke run was performed.

## Findings

### DI-W0-001

- Class: Registry consistency
- Sev: High when present; resolved in working tree
- Title: Whisper converter output was registry-orphaned
- Symbols: [`model_family_for_type`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/convert/model_family.rs:220), [`ARCHITECTURE_REGISTRY`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/architecture_registry.rs:290), [`support_tier_for_family`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/support_tier.rs:77)
- Why real: Tracked `HEAD` emits `family_name = "whisper"` but has no matching registry row, so architecture lookup returns `None`. The Compatible fallback in `support_tier_for_family` masks that omission. The current working tree adds the Whisper row, route assertion, converter-family parity test, and explicit Compatible-tier assertion; the current source scan reports no orphaned converter families.
- Fix direction: Retain the working-tree changes and strengthen the converter/registry invariant as described in DI-W0-003.
- Disposition suggestion: **Resolved pending commit and targeted test execution.**

### DI-W0-002

- Class: Support-tier correctness
- Sev: High
- Title: Converted Gemma 4 Unified artifacts inherit the Certified Gemma 4 tier
- Symbols: [`model_family_for_type`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/convert/model_family.rs:58), [`gemma4` and `gemma4_unified` registrations](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/architecture_registry.rs:135), [`support_tier_for_manifest`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/support_tier.rs:89)
- Why real: Both `gemma4_unified` and `gemma4_unified_text` are canonicalized to `family_name = "gemma4"`. The registry grades `gemma4` as Certified but explicitly grades `gemma4_unified` as Compatible. Because tier resolution uses only the emitted manifest label, normally converted Unified artifacts receive the Certified tier and bypass the explicit no-certification promise for `gemma4_unified`.
- Fix direction: Emit `gemma4_unified` for the Unified converter arms while retaining their tensor map, then add a converter-to-tier regression test. If canonicalization is intentional, remove the conflicting registry identity and explicitly document/certify Unified under the shared tier.
- Disposition suggestion: **Open; block Certified support claims for converted Unified artifacts.**

### DI-W0-003

- Class: Regression-test integrity
- Sev: Medium
- Title: The new converter/registry parity test manually duplicates converter outputs
- Symbols: [`convert_family_names_are_registered`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/architecture_registry.rs:470), [`model_family_for_type`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/convert/model_family.rs:18)
- Why real: `CONVERT_FAMILY_NAMES` is a handwritten list with no mechanical connection to the converter match arms. Adding another `family_name` arm without updating this list leaves the test green, allowing the exact Whisper orphan regression to recur.
- Fix direction: Define canonical emitted-family labels once and consume that declaration from both converter and registry tests, or add a deterministic source-level parity gate that extracts all converter outputs.
- Disposition suggestion: **Open; required follow-up to DI-W0-001.**

### DI-W0-004

- Class: Smoke-gate validity
- Sev: High
- Title: Full smoke runs silently reuse stale binaries
- Symbols: [`ensure_binary`](/Users/akiralam/code/ax-engine/scripts/smoke_compatible_models.py:293), [`main`](/Users/akiralam/code/ax-engine/scripts/smoke_compatible_models.py:624)
- Why real: `ensure_binary` immediately returns whenever a binary exists, even when `--no-build` was not supplied. It never asks Cargo to check or relink that executable against current sources. The current workspace already contains server binaries older than scoped source changes, so this is an observable path to testing old code while reporting current-tree smoke evidence.
- Fix direction: Unless `--no-build` is set, always invoke the appropriate incremental Cargo build before returning the binary. Building both required packages in one invocation would also prevent one executable from remaining stale.
- Disposition suggestion: **Open; invalidate full-smoke evidence produced without a clean or explicit rebuild.**

### DI-W0-005

- Class: Artifact provenance
- Sev: High
- Title: An explicit `--models-dir` can fall through to ambient Hugging Face cache
- Symbols: [`resolve_snapshot`](/Users/akiralam/code/ax-engine/scripts/smoke_compatible_models.py:243), [`--require-any`](/Users/akiralam/code/ax-engine/scripts/smoke_compatible_models.py:530)
- Why real: After an absent, invalid, or nonmatching `models_dir`, `resolve_snapshot` unconditionally searches the default cache. Therefore `--require-any` does not reliably catch a mis-mounted directory: any matching ambient snapshot counts as a run. Matching ambient snapshots are present on this host, making the advertised failure mode currently reproducible.
- Fix direction: When `--models-dir` is supplied, restrict resolution to that directory by default. Require a separate explicit cache-fallback option and record the resolved artifact source in summary output.
- Disposition suggestion: **Open; block release-gate reliance on `--models-dir --require-any`.**

### DI-W0-006

- Class: Tier-test completeness
- Sev: Low
- Title: The “deliberate tier decision” test defaults every unknown row to Compatible
- Symbols: [`expected_tier`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/support_tier.rs:169), [`compatible_families_have_no_cert_promise`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/support_tier.rs:245)
- Why real: The test claims a new registry row fails without a deliberate tier decision, but its wildcard arm returns Compatible. A newly added Compatible row therefore passes without appearing in any expected-tier list; the separate Compatible test also does not assert list completeness.
- Fix direction: Make the expected-tier mapping exhaustive and fail on unknown labels, or assert that the union of explicit Certified, Compatible, and Experimental sets equals the registry label set exactly.
- Disposition suggestion: **Open hardening.**

### DI-W0-007

- Class: Registry/gate parity
- Sev: Medium
- Title: Duplicate registry labels resolve differently at runtime and in the smoke parser
- Symbols: [`lookup_architecture`](/Users/akiralam/code/ax-engine/crates/ax-engine-core/src/architecture_registry.rs:301), [`parse_registry_tiers`](/Users/akiralam/code/ax-engine/scripts/smoke_compatible_models.py:124)
- Why real: Runtime lookup returns the first matching registration, while the Python dictionary silently keeps the last duplicate. A duplicate label with different metadata could therefore pass the smoke tier check using a tier different from the one runtime resolves. The current registry is unique, but neither implementation enforces that invariant.
- Fix direction: Add a registry uniqueness unit test and make `parse_registry_tiers` reject duplicate labels instead of overwriting them.
- Disposition suggestion: **Open hardening.**

## Recommended fix order

1. Fix DI-W0-002 before publishing or consuming support-tier claims.
2. Fix DI-W0-004 and DI-W0-005 before accepting new real-weight smoke evidence.
3. Retain and commit the DI-W0-001 working-tree fix.
4. Close DI-W0-003 so the Whisper class of regression is mechanically prevented.
5. Harden tier and registry invariants with DI-W0-006 and DI-W0-007.

## Wave 0 checklist (pass/fail/n/a)

- **PASS** — All six scoped files were inspected; unrelated crates were not explored.
- **PASS** — No source files were edited by this audit.
- **FAIL** — Tracked `HEAD` converter-to-registry parity: Whisper is orphaned.
- **PASS** — Current working-tree converter-to-registry parity: 23/23 emitted families registered.
- **PASS** — Current registry uniqueness: 26/26 labels unique.
- **FAIL** — Converter output preserves intended support tier for Gemma 4 Unified.
- **PASS** — Scoped MTP policies fail closed for conflicting, uncertified Qwen-linear, and default DeepSeek V4 drafters.
- **PASS** — No high-confidence correctness defect identified in scoped MTP request-routing predicates.
- **PASS** — `python3 scripts/smoke_compatible_models.py --dry-run`.
- **PASS** — Scoped working-tree diff whitespace check.
- **FAIL** — Full smoke guarantees binaries correspond to current sources.
- **FAIL** — Explicit model-directory selection has deterministic artifact provenance.
- **N/A** — Cargo builds and Rust test suites, omitted by instruction.
- **N/A** — Real-weight smoke execution, omitted by instruction.
- **FAIL** — Overall Wave 0 gate.