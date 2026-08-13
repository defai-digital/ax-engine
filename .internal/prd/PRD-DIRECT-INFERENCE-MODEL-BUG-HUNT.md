# Product Requirements: Direct-Inference Model Bug Hunt (One-by-One)

| Field | Value |
| --- | --- |
| Status | **Active — orchestration contract** |
| Owner | AX Engine maintainers |
| Last updated | 2026-08-11 |
| Scope | Every **repo-owned direct MLX** inference family (not delegated adapters) |
| Method | Multi-agent loop: **Grok CLI** orchestrates **Codex CLI (max reasoning)** + **ax-code CLI (GLM 5.2 1M)** |
| Status ledger | [DIRECT-INFERENCE-MODEL-BUG-HUNT-STATUS.md](../DIRECT-INFERENCE-MODEL-BUG-HUNT-STATUS.md) |
| Implementation plan | [direct-inference-model-bug-hunt-implementation-plan.md](../planning/direct-inference-model-bug-hunt-implementation-plan.md) |
| Agent prompts | [`.internal/planning/bug-hunt-agents/`](../planning/bug-hunt-agents/) |
| Public model matrix | [`docs/SUPPORTED-MODELS.md`](../../docs/SUPPORTED-MODELS.md) |
| Architecture registry | `crates/ax-engine-core/src/architecture_registry.rs` |
| Related packages | ADR-010 / Gemma4–Qwen3 deepening · ADR-020 / Qwen36 linear MTP Tier 2 · Sibling Tier 2 expansion · Decode dispatch efficiency |

## 1. Decision summary

AX Engine will run a **serial, evidence-gated bug-hunt program** over every
**direct-inference** model family: one family at a time, fully closed before the
next family opens.

For each family the program **must**:

1. **Inspect** the owned graph, convert/manifest path, runner/MTP policy, KV,
   sampling, server surface, and tests.
2. **Find** bugs, wrong implementations, wrong MTP mode design, bottlenecks, and
   dead code — with file/symbol anchors and severity.
3. **Fix** accepted defects (code + tests) until the family **exit gate** passes.
4. **Loop** inspect → fix → re-verify until residual risk is only documented
   intentional limits (not unfixed bugs).
5. **Orchestrate** the work through **Grok CLI** as the program conductor, with
   two specialist agents:
   - **Codex CLI** at **maximum reasoning effort** (deep architecture / MTP /
     correctness review and surgical patches).
   - **ax-code CLI** with **`zai-coding-plan/glm-5.2[1m]`** (1M-context wide
     corpus scan, dead-code maps, cross-file inconsistency sweeps).

This PRD is **not** a model-range expansion. It is a **trust and quality**
program over what already claims direct support. Public performance claims still
require existing benchmark gates; this program never invents tok/s marketing
from smoke runs.

## 2. Problem

### 2.1 User-visible gap

Direct support is the product default path. Families are registered and
documentable while residual risk still clusters in:

| Risk class | Symptom |
| --- | --- |
| Correctness bugs | Wrong tokens, silent weight drops, broken SWA/MLA/linear state, bad multimodal expansion |
| Wrong implementation | Shared `standard` path used for non-isomorphic graphs; family flags stringly mismatched |
| Wrong MTP mode design | Exactness tied to request flags; optimistic draft as “support”; short-answer forced MTP; wrong policy per family |
| Bottlenecks | Host/device serialize, unnecessary dequant, serial linear prefill, dead draft/verify paths still hot |
| Dead code / drift | Unreachable routes, obsolete env knobs, docs claiming paths the graph does not take |

Prior packages (Gemma/Qwen deepening, Qwen36 linear MTP Tier 2, sibling Tier 2)
are **vertical** on one concern. They do **not** replace a disciplined
**horizontal** pass that every direct family must survive.

### 2.2 Structural hotspots (why multi-agent + loop)

| Hotspot | Approx. size / role | Audit risk |
| --- | --- | --- |
| `crates/ax-engine-mlx/src/runner/mod.rs` | ~19k LOC | Family heuristics, MTP gates, prefix restore, route decisions |
| `crates/ax-engine-mlx/src/model/mod.rs` | ~8k LOC | Dispatch / forward variants |
| `crates/ax-engine-mlx/src/mtp.rs` | ~3.5k LOC | Draft/verify/rollback contracts |
| `runner/mtp_*.rs`, `mtp_adaptive_gate.rs`, `gemma4_assistant_mtp.rs` | policy surface | Wrong mode design per family |
| `crates/ax-engine-core/src/convert/*` | family mapping | Silent drops, wrong `model_family` |
| Shared `families/standard.rs` + divergent families | graphs | Secondary families inherit primary assumptions |

Single-pass human review cannot close this surface. The PRD therefore mandates a
**repeatable multi-agent loop** with fail-closed exit criteria per model.

### 2.3 Explicit boundary

| In scope | Out of scope |
| --- | --- |
| Direct `selected_backend=mlx` repo-owned graphs | `mlx_lm_delegated` as a “fix” path for AX graphs |
| Convert + manifest + load + generate + server/SDK smoke | NVIDIA/CUDA / AX Serving / vLLM workers |
| MTP / assistant-MTP / n-gram when the family owns them | Claiming delegated llama.cpp GGUF as direct support |
| Bottleneck diagnosis with evidence (probes, traces) | Unrelated product features (new families, remote media fetch) |
| Dead code on the family’s critical path | Full repo-wide cosmetic cleanup unrelated to the active family |

## 3. Goals

| ID | Goal |
| --- | --- |
| DI-G-001 | Every listed direct family completes a **serial audit wave** with a written finding set and disposition. |
| DI-G-002 | Every **P0/P1** finding either lands a fix + regression test or is explicitly **wontfix/limit** with owner rationale. |
| DI-G-003 | No silent capability loss: convert drops, gated modalities, and degraded routes are loud. |
| DI-G-004 | MTP modes for each MTP-capable family match design contracts (exactness, fail-closed defaults, short-budget policy). |
| DI-G-005 | Bottlenecks are measured (not guessed); only evidence-backed hot path changes land. |
| DI-G-006 | Dead code and obsolete knobs on the family path are removed or quarantined with tests. |
| DI-G-007 | Multi-agent workflow is **reproducible**: prompts, model IDs, and artifact paths are fixed in this package. |
| DI-G-008 | Status ledger is the single source of progress; no family is “done” without ledger exit. |

### 3.1 Non-goals

- Expanding the supported-model catalog or certifying new public tok/s rows.
- Using delegated backends to paper over broken direct graphs.
- Parallel “fix everything” branches that cross-contaminate family diffs.
- Lowering exactness or formal MTP gates from ADR-020 / sibling Tier 2.
- Full rewrite of `runner/mod.rs` unless a family exit gate **proves** that is the only safe fix (prefer surgical extraction).
- Public marketing language; internal ledgers only until release owners promote docs.

## 4. Model inventory (audit order)

**Rule:** One family (or tightly coupled SKU group) at a time. Do not open Wave
*N+1* until Wave *N* ledger status is `closed` or `parked-with-rationale`.

Waves prioritize product critical path first (primary productivity), then
multimodal/speech, then secondary preview, then experimental.

### Wave 0 — Shared substrate (once, before family waves)

| ID | Surface | Why first |
| --- | --- | --- |
| W0-registry | `architecture_registry.rs`, `support_tier.rs`, convert family map | Wrong labels poison every later audit |
| W0-runner-mtp | `runner/mtp_*.rs`, `mtp.rs`, adaptive gate, speculation profile | Shared MTP policy bugs multiply across families |
| W0-convert-manifest | generate-manifest, role maps, drop accounting | Silent drops are systemic |
| W0-tooling | smoke scripts, probes, QA matrix entry points | Ensures the loop has harnesses |

Wave 0 may land **shared** fixes only when they do not change family-specific
semantics without a family owner note. Prefer minimal shared patches.

### Wave 1 — Primary chat / agent (highest traffic)

| Order | Family / SKU group | Manifest labels | Graph / entry anchors | MTP mode |
| ---: | --- | --- | --- | --- |
| 1 | Qwen 3.6 dense 27B | `qwen3_5` | `qwen3_linear` + hybrid standard; AutomatosX MTP packs | Fused MTP sidecar; linear exact scope (ADR-020) |
| 2 | Qwen 3.6 MoE 35B-A3B | `qwen3_5` / MoE | Same + MoE experts | Linear MTP + MoE verify (sibling Tier 2) |
| 3 | Qwen 3.5 9B | `qwen3_5` | Hybrid GatedDelta + dense FFN | MTP packs where published |
| 4 | Qwen3-Coder-Next | `qwen3_next` | Hybrid sparse-MoE coding | Direct decode; no download-mtp head |
| 5 | Qwen 3 dense | `qwen3` | `standard` dense; batched decode cert | n-gram / optional speculation profile |
| 6 | Gemma 4 unified 12B | `gemma4` / unified roles | `gemma4_unified.rs` | Assistant-MTP package |
| 7 | Gemma 4 26B / 31B / E2B / E4B | `gemma4`, `gemma4_vl` | standard + VL towers | Assistant-MTP where packaged |
| 8 | GLM 4.7 Flash | `glm4_moe_lite` | `glm4_moe_lite.rs` | Optional GLM MTP sidecar prep |

### Wave 2 — Multimodal / OCR / speech / embed

| Order | Family / SKU group | Labels | Anchors | Notes |
| ---: | --- | --- | --- | --- |
| 9 | Qwen3-VL dense / MoE | `qwen3_vl`, `qwen3_vl_moe` | `qwen3_vl.rs` | Image/video; text decode shares qwen3 |
| 10 | MiniCPM-V 4.6 | `minicpmv4_6` | `minicpm_v.rs` | OCR / multi-image |
| 11 | Nemotron 3 Nano Omni | `nemotron_h` | `nemotron_h.rs`, omni media | Image/audio mixed |
| 12 | Unlimited-OCR | `unlimited_ocr` | `unlimited_ocr.rs` | Protected-prefix R-SWA |
| 13 | Whisper large-v3-turbo | `whisper` (audio runtime) | `whisper.rs` | Not chat generate |
| 14 | EmbeddingGemma / Qwen3-Embedding | `embeddinggemma`, embed SKUs | encoder embed path | `/v1/embeddings` |
| 15 | Nemotron 3 Embed | `nemotron_embed` | encoder embed | Compatible path; not Omni |

### Wave 3 — Secondary preview direct

| Order | Family / SKU group | Labels | Anchors |
| ---: | --- | --- | --- |
| 16 | Llama 3.x | `llama3` | `standard` + Llama3 chat |
| 17 | Llama 4 Scout | `llama4` | `llama4.rs` |
| 18 | Mistral / Ministral / Devstral | `mistral3` | `mistral3.rs` → standard |
| 19 | GPT-OSS 20B / 120B | `gpt_oss` | `gpt_oss.rs` MXFP4 packed experts |
| 20 | DeepSeek V3 / V3.2 | `deepseek_v3`, `deepseek_v32` | `deepseek_v3.rs` dense MLA contract |

### Wave 4 — Experimental (audit honesty, not certification)

| Order | Family / SKU group | Labels | Anchors | Bar |
| ---: | --- | --- | --- | --- |
| 21 | DiffusionGemma | `diffusion_gemma` | `diffusion.rs` | Experimental path correctness + docs honesty |
| 22 | DeepSeek V4 Flash | `deepseek_v4` | `deepseek_v4.rs` | Experimental only; no support claim inflation |

**SKU pinning:** Prefer AutomatosX managed packs for Wave 1 formal checks; mlx-community
aliases for secondary. Record exact `repo_id` + revision/manifest SHA in the ledger.

## 5. Defect taxonomy (must classify every finding)

Every finding uses exactly one primary class:

| Code | Class | Definition | Default severity |
| --- | --- | --- | --- |
| **BUG** | Functional bug | Incorrect tokens, crash, hang, KV corruption, wrong sampling, wrong API status | P0–P1 |
| **IMPL** | Wrong implementation | Code does not match intended architecture / paper / manifest contract | P0–P2 |
| **MTP** | Wrong MTP mode design | Policy, exactness, draft/verify, adaptive gate, family eligibility | P0–P1 |
| **PERF** | Bottleneck | Proven hot path waste; regression vs known peer/probe | P1–P2 |
| **DEAD** | Dead code / drift | Unreachable, obsolete flags, docs/code mismatch | P2–P3 |
| **DOC** | Documentation only | Misleading public/internal docs without runtime bug | P3 |
| **LIMIT** | Intentional limit | Fail-closed by design; not a defect if documented | n/a |

### 5.1 Severity

| Sev | Meaning | Exit rule |
| --- | --- | --- |
| **P0** | Wrong tokens, data corruption, crash on default path, silent weight drop of active tower | Must fix or park family as unsafe |
| **P1** | Wrong MTP exactness, major correctness edge, severe perf cliff on primary SKU | Must fix before wave close |
| **P2** | Secondary SKU issues, non-default flags, moderate perf, substantial dead code | Fix in-wave or schedule with date |
| **P3** | Nits, pure docs, optional cleanup | May carry with ticket |

## 6. Multi-agent operating model

### 6.1 Roles

```text
┌─────────────────────────────────────────────────────────────┐
│  Grok CLI (orchestrator / program manager)                  │
│  - picks active family from ledger                          │
│  - spawns specialist passes with frozen prompts             │
│  - merges findings, de-dupes, prioritizes                   │
│  - drives fix loop, runs gates, updates ledger              │
│  - refuses to open next family until exit gate              │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────┐   ┌─────────────────────────────────┐
│ Codex CLI                 │   │ ax-code CLI                     │
│ reasoning: maximum / SoL  │   │ model: zai-coding-plan/         │
│ very high                 │   │         glm-5.2[1m]             │
│                           │   │                                 │
│ Best for:                 │   │ Best for:                       │
│ - deep causal reasoning   │   │ - 1M-context full-path scans    │
│ - MTP exactness design    │   │ - dead code / stringly match    │
│ - surgical Rust patches   │   │ - cross-crate inconsistency     │
│ - test-first fixes        │   │ - large file maps (runner/mod)  │
│ - formal A/B interpretation│  │ - convert/manifest corpus       │
└───────────────────────────┘   └─────────────────────────────────┘
```

| Agent | Tool | Model / effort | Authority |
| --- | --- | --- | --- |
| **Orchestrator** | `grok` | default Grok Build model (session conductor) | Ledger ownership, wave control, merge, gate runs |
| **Deep reasoner** | `codex exec` | strongest available Codex model + **highest reasoning effort** | Architecture verdicts, MTP design, P0/P1 patches |
| **Wide scanner** | `ax-code run` | `zai-coding-plan/glm-5.2[1m]` | Broad finding sets, dead-code maps, second opinion |

Interpretation of user requirement **“codex cli with sol very high”**: run Codex at
the **highest available reasoning configuration** on this machine (max reasoning
effort / strongest Sol-class model the local Codex install exposes). Record the
exact `-m` / config in each ledger entry so runs are reproducible. If the local
Codex profile renames models, update
[`bug-hunt-agents/README.md`](../planning/bug-hunt-agents/README.md) without
changing this PRD’s intent.

### 6.2 Division of labor per family cycle

| Phase | Grok | Codex (high reasoning) | ax-code (GLM 5.2 1M) |
| --- | --- | --- | --- |
| **F1 Discover** | Assign family, freeze file list | Deep read of graph + MTP contracts | Wide scan of convert/runner/server references |
| **F2 Cross-check** | Diff both finding reports | Challenge false positives; design review | Challenge missed call sites / dead paths |
| **F3 Prioritize** | Severity matrix → work queue | Confirm P0/P1 root causes | Confirm DEAD/PERF maps |
| **F4 Fix** | One finding at a time; run tests | Primary implementer for P0/P1 | Optional implementer for DEAD/DOC or second pass |
| **F5 Verify** | Run family gate suite | Re-review the diff for regressions | Re-scan for remaining dead/inconsistency |
| **F6 Close** | Update ledger; commit notes | Sign-off on MTP/correctness items | Sign-off on scan completeness |

**Conflict rule:** If Codex and ax-code disagree on a P0/P1:

1. Grok records both arguments.
2. Codex deep-reasoner re-evaluates with the contradiction attached.
3. Prefer **fail-closed product safety** over performance speculation.
4. Never “average” exactness; tokens must be correct or the path is off.

### 6.3 Canonical CLI invocations

Exact flags may evolve; keep working copies under
`.internal/planning/bug-hunt-agents/`. Baseline contracts:

```bash
# --- Paths ---
REPO=/Users/akiralam/code/ax-engine   # or $PWD of checkout
LEDGER=$REPO/.internal/DIRECT-INFERENCE-MODEL-BUG-HUNT-STATUS.md
PROMPTS=$REPO/.internal/planning/bug-hunt-agents
OUT=$REPO/.internal/reports/bug-hunt/$(date +%Y%m%d)/$FAMILY_ID
mkdir -p "$OUT"

# --- Grok orchestrator (interactive program manager) ---
grok --cwd "$REPO" "You are the Direct-Inference Bug Hunt orchestrator. Read $LEDGER and PRD-DIRECT-INFERENCE-MODEL-BUG-HUNT.md. Active family: $FAMILY_ID. Execute phase F1–F6 using the prompts in $PROMPTS. Do not open another family."

# --- Codex: deep inspect (read-only first) ---
codex exec \
  -C "$REPO" \
  -s read-only \
  -m "$CODEX_MODEL" \
  -c "model_reasoning_effort=\"$CODEX_REASONING_EFFORT\"" \
  --output-last-message "$OUT/codex-inspect.md" \
  < "$PROMPTS/01-codex-inspect.md"

# --- Codex: implement fix (workspace write; human-approved) ---
codex exec \
  -C "$REPO" \
  -s workspace-write \
  -m "$CODEX_MODEL" \
  -c "model_reasoning_effort=\"$CODEX_REASONING_EFFORT\"" \
  --output-last-message "$OUT/codex-fix-$FINDING_ID.md" \
  < "$PROMPTS/03-codex-fix.md"

# --- ax-code: wide 1M scan ---
ax-code run --model "zai-coding-plan/glm-5.2[1m]" \
  "$(cat "$PROMPTS/02-axcode-scan.md")"
# Export/session log to $OUT/axcode-scan.md via ax-code export if available

# --- ax-code: second-opinion review of a proposed fix ---
ax-code run --model "zai-coding-plan/glm-5.2[1m]" \
  "$(cat "$PROMPTS/04-axcode-review-fix.md")"
```

**Environment contract for agents (record in every report):**

| Variable | Purpose |
| --- | --- |
| `FAMILY_ID` | Ledger key, e.g. `qwen36-27b`, `gemma4-12b-unified` |
| `MANIFEST_FAMILY` | e.g. `qwen3_5`, `gemma4` |
| `MODEL_ARTIFACTS_DIR` | Local snapshot with `model-manifest.json` |
| `CODEX_MODEL` | Installed Codex model id for max quality |
| `CODEX_REASONING_EFFORT` | Highest supported (e.g. `xhigh` / `high`) |
| `AX_CODE_MODEL` | Fixed: `zai-coding-plan/glm-5.2[1m]` |

### 6.4 Orchestration principles (Grok)

1. **Serial families** — never two family fix branches in one worktree without isolation.
2. **One finding per fix PR/commit theme** when possible (keeps reviewable history).
3. **Tests before claims** — no “looks correct” close.
4. **Prefer symbol anchors** over volatile line numbers (project convention).
5. **Respect unsafe ban / clippy / rustfmt** (`AGENTS.md` / `Agents.md`).
6. **Do not open unsolicited public PRs** without maintainer agreement (repo policy);
   keep work on local/issue-linked branches.
7. **Loop until clean:** after each fix batch, re-run F1-lite (targeted re-scan) +
   family gate; if new P0/P1 appears, stay on the family.

## 7. Per-family audit checklist

Every family must be checked against this matrix. Mark `pass` / `fail` / `n/a`
in the ledger.

### 7.1 Convert & manifest

- [ ] HF `model_type` → `model_family` mapping is correct and unique.
- [ ] All active towers/roles present; drops are counted and loud.
- [ ] `generate-manifest` is idempotent; stale manifests regenerated with `--force` story.
- [ ] Quantization / MXFP4 / OptiQ / AXQ load path matches residency design.
- [ ] `probe_mlx_model_support.py` reports `repo_owned_runtime_ready` when artifacts exist.

### 7.2 Graph & forward

- [ ] Layer route matches registry (`LayerForwardRoute`).
- [ ] Attention (full / SWA / MLA / linear / hybrid) matches config tensors.
- [ ] MoE routing (top-k, shared expert, sigmoid vs softmax) correct.
- [ ] RoPE / MRoPE / YaRN parameters match checkpoint.
- [ ] Logit softcap / QK norm / gates present when configured.
- [ ] No accidental use of another family’s fastpath assumptions.

### 7.3 KV, cache, prefix

- [ ] Prefill over window correct for SWA/rotating layouts.
- [ ] Prefix cache identity includes media when multimodal.
- [ ] Rollback after speculation / eviction is safe.
- [ ] Protected prefix (Unlimited-OCR) honored if applicable.

### 7.4 Decode & sampling

- [ ] Greedy temperature-0 stable and coherent on smoke prompt.
- [ ] Finish reasons correct; Gemma loop detection policy family-scoped.
- [ ] Batched decode only when certified/structural candidate allows.

### 7.5 MTP / speculation (if applicable)

- [ ] Eligibility policy matches family (linear vs assistant vs n-gram).
- [ ] Exact arithmetic **independent** of `mtp_requested` where required (ADR-020 class).
- [ ] Default product path fail-closed until certified.
- [ ] Short remaining budget skips MTP (no forced slowdown).
- [ ] Draft/verify/rollback telemetry honest.
- [ ] Sidecar / assistant package provenance validated (`check_mtp_sidecar_provenance` class tools).
- [ ] No optimistic-only path labeled as certified support.

### 7.6 Multimodal / audio (if applicable)

- [ ] Capability discovery matches tower tensors (fail closed if missing).
- [ ] Image/audio/video expansion order correct; remote URL fetch stays disabled.
- [ ] Soft-token budgets / placeholders match design.
- [ ] Whisper stays off chat generate routes.

### 7.7 Server / SDK / CLI

- [ ] Alias / preset resolves to correct artifacts.
- [ ] OpenAI chat/completions + native generate behave.
- [ ] Multi-model allowlist membership correct for family (add vs replace).
- [ ] Error messages suggest close aliases; no fuzzy wrong model.

### 7.8 Performance (evidence only)

- [ ] Hot path identified via existing probes/traces where available.
- [ ] No accidental host sync / double eval / dequant expand on critical path.
- [ ] PERF findings require before/after or probe artifact — no vibes.

### 7.9 Dead code & consistency

- [ ] Unreachable env flags / routes removed or documented deprecated.
- [ ] String `match model_family` sites include this family or correctly fall through.
- [ ] Docs (`SUPPORTED-MODELS`, FAQ snippets) match runtime behavior.
- [ ] Tests cover regressions for every P0/P1 fix.

## 8. Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| DI-R-001 | Program processes families **strictly one-by-one** per §4 waves. | P0 |
| DI-R-002 | Each family cycle uses **both** Codex (max reasoning) and ax-code GLM 5.2 1M for inspect, orchestrated by Grok. | P0 |
| DI-R-003 | Every finding is classified with §5 taxonomy + severity + symbol anchors. | P0 |
| DI-R-004 | P0/P1 findings must be fixed or explicit park before family close. | P0 |
| DI-R-005 | Every code fix includes regression coverage (unit/integration/smoke as appropriate). | P0 |
| DI-R-006 | MTP-capable families re-verify exactness vs direct under temperature-0 where MTP is in scope. | P0 |
| DI-R-007 | Family exit requires the gate suite in §9. | P0 |
| DI-R-008 | Ledger updated after every phase; reports stored under `.internal/reports/bug-hunt/`. | P0 |
| DI-R-009 | Shared substrate changes that affect multiple families require Wave-0 note + impact list. | P1 |
| DI-R-010 | PERF changes need probe/bench artifact paths recorded in the ledger. | P1 |
| DI-R-011 | Dead-code removal must not break feature flags still referenced by docs/tests; update both. | P1 |
| DI-R-012 | Experimental families (Wave 4) may close with “honest LIMIT” if graph is incomplete — never upgrade support tier without evidence. | P0 |
| DI-R-013 | Agents must not claim public support or performance beyond existing docs gates. | P0 |
| DI-R-014 | Prefer surgical patches; large refactors need explicit Grok+Codex design agreement first. | P1 |

## 9. Family exit gate

A family may move to **closed** only when **all** of the following hold:

1. **Inspect complete:** Codex inspect report + ax-code scan report exist under
   `.internal/reports/bug-hunt/<date>/<family>/`.
2. **Finding disposition complete:** every P0/P1 is `fixed` or `parked` with rationale.
3. **Build/lint:** targeted `cargo test` for touched crates; `cargo fmt` / clippy clean on touched code as project requires.
4. **Smoke:** where weights are available:
   - `scripts/probe_mlx_model_support.py --model-dir …` ready, and
   - short greedy generate via server or bench, and
   - family-specific script if exists (e.g. multimodal QA, MTP e2e).
5. **MTP gate (if applicable):** temperature-0 direct vs MTP identical on the
   documented trial set **or** documented fail-closed default with no false
   acceleration claim.
6. **Re-scan clean:** post-fix Codex+ax-code pass reports **no new P0/P1**.
7. **Ledger row** filled: commit SHAs, artifact digests, residual LIMIT list.

If weights are unavailable, family may be **`closed-code-only`** with explicit
hardware follow-up ticket — **not** “closed” for release claims.

## 10. Loop protocol (inspect → fix → until clean)

```text
                    ┌──────────────────────┐
                    │ Select next family   │
                    │ from ledger (serial) │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
              ┌────►│ F1 Discover          │
              │     │ Grok + Codex + axcode│
              │     └──────────┬───────────┘
              │                ▼
              │     ┌──────────────────────┐
              │     │ F2 Cross-check       │
              │     │ merge / de-dupe      │
              │     └──────────┬───────────┘
              │                ▼
              │     ┌──────────────────────┐
              │     │ F3 Prioritize queue  │
              │     └──────────┬───────────┘
              │                ▼
              │     ┌──────────────────────┐
              │     │ Open P0/P1 empty?    │──yes──► F6 Exit gate ──pass──► next family
              │     └──────────┬───────────┘              │
              │                │ no                       fail
              │                ▼                          │
              │     ┌──────────────────────┐              │
              │     │ F4 Fix one finding   │              │
              │     │ (Codex primary)      │              │
              │     └──────────┬───────────┘              │
              │                ▼                          │
              │     ┌──────────────────────┐              │
              │     │ F5 Verify tests+scan │◄─────────────┘
              │     └──────────┬───────────┘
              │                │
              └────────────────┘  (loop)
```

**Hard stop rules:**

- Infinite thrash (>3 full F1 cycles with no P0/P1 reduction) → park family,
  escalate human architecture review.
- Fix introduces new P0 on another family → stop wave, revert or shared fix via Wave 0.
- Agent proposes lowering exactness/gates → reject by policy.

## 11. Success metrics

| Metric | Target |
| --- | --- |
| Wave 1 families closed (or closed-code-only with follow-up) | 100% of §4 Wave 1 |
| Open P0 on closed families | 0 |
| Open P1 on closed primary families | 0 (or dated park ≤ 14 days) |
| MTP exactness regressions introduced by this program | 0 |
| Dual-agent inspect coverage | Every closed family has Codex + ax-code artifacts |
| Dead-code removals without test breakage | 100% of landed DEAD fixes |

Program-level exit: Waves 1–3 closed; Wave 4 honestly parked or closed with LIMIT
list; status ledger summary signed by orchestrator.

## 12. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Agent hallucinated bugs | Dual-agent cross-check; require symbol-level proof; fail reproduction first |
| Large unsafe refactors | DI-R-014; design agreement; prefer minimal diffs |
| MTP exactness regressions | Always A/B greedy before close; ADR-020 still binding |
| Weight-unavailable “fake close” | `closed-code-only` status distinct from release-ready |
| Context overflow on runner/mod.rs | ax-code 1M for maps; Codex deep on sliced symbols; Grok maintains work queue |
| Parallel family contamination | Serial rule; isolated worktrees if needed |
| Scope creep into new features | Non-goals; reject tickets that are not BUG/IMPL/MTP/PERF/DEAD/DOC |
| Tooling flag drift (Codex/ax-code) | Version prompts in repo; ledger records actual CLI versions |

## 13. Deliverables

| Deliverable | Path |
| --- | --- |
| This PRD | `.internal/prd/PRD-DIRECT-INFERENCE-MODEL-BUG-HUNT.md` |
| Status ledger | `.internal/DIRECT-INFERENCE-MODEL-BUG-HUNT-STATUS.md` |
| Implementation plan | `.internal/planning/direct-inference-model-bug-hunt-implementation-plan.md` |
| Agent prompts | `.internal/planning/bug-hunt-agents/*.md` |
| Per-run reports | `.internal/reports/bug-hunt/<YYYYMMDD>/<family-id>/` |
| Code fixes | Normal git commits on issue-linked branches |

## 14. Dependencies

- Local Apple Silicon for real-weight smoke (as per CONTRIBUTING).
- Hugging Face cache / AutomatosX packs for Wave 1.
- Working installs: `grok`, `codex`, `ax-code` (see `which` on operator host).
- Existing harnesses: `smoke_compatible_models.py`, `probe_mlx_model_support.py`,
  MTP bench/e2e scripts, family QA scripts under `scripts/` and `qa/`.
- Related ADRs remain authoritative for MTP defaults and multimodal bar.

## 15. Open decisions (resolve in ledger, not ad hoc)

| ID | Question | Default until decided |
| --- | --- | --- |
| OD-1 | Exact Codex model id on this host for “SoL very high” | Document in agents README after `codex` probe |
| OD-2 | Whether Wave 0 is a full week or a short shared pass | Short shared pass; expand only if systemic bugs found |
| OD-3 | Commit strategy: monorepo direct vs stacked branches | Small serial commits per finding |
| OD-4 | Minimum formal host for MTP exactness recheck | Prefer host with existing MTP formal evidence when available |

---

*End of PRD. Execution starts only after status ledger initialization and agent prompt freeze.*
