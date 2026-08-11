# Implementation plan: Direct-Inference Model Bug Hunt

| Field | Value |
| --- | --- |
| PRD | [PRD-DIRECT-INFERENCE-MODEL-BUG-HUNT.md](../prd/PRD-DIRECT-INFERENCE-MODEL-BUG-HUNT.md) |
| Ledger | [DIRECT-INFERENCE-MODEL-BUG-HUNT-STATUS.md](../DIRECT-INFERENCE-MODEL-BUG-HUNT-STATUS.md) |
| Agents | [bug-hunt-agents/](./bug-hunt-agents/) |
| Date | 2026-08-11 |

## 1. Operating principle

Execute **serially**: Wave 0 → Wave 1 families in order → Wave 2 → Wave 3 → Wave 4.
Each family runs the F1–F6 loop until the exit gate passes. Grok CLI owns the
loop; Codex (max reasoning) owns deep correctness/MTP fixes; ax-code GLM 5.2 1M
owns wide scans and second opinions.

## 2. Phase 0 — Bootstrap (day 0)

| Step | Action | Owner |
| --- | --- | --- |
| B1 | Confirm `grok`, `codex`, `ax-code` on PATH | Human / Grok |
| B2 | Probe Codex model + max reasoning effort; write into agents README + ledger | Grok |
| B3 | Confirm ax-code model id `zai-coding-plan/glm-5.2[1m]` resolves | Grok |
| B4 | Create `.internal/reports/bug-hunt/` | Grok |
| B5 | Freeze prompt files (no mid-wave silent rewrite) | Grok |
| B6 | Inventory local HF snapshots useful for Wave 1 | Grok |

### Suggested bootstrap commands

```bash
which grok codex ax-code
codex --version 2>/dev/null || true
ax-code models 2>/dev/null | rg -i 'glm-5.2|1m'
mkdir -p .internal/reports/bug-hunt
python3 scripts/smoke_compatible_models.py --list
```

## 3. Phase 1 — Wave 0 shared substrate

Run a **short** dual-agent pass on shared surfaces only:

1. Registry / convert family map consistency.
2. Shared MTP policy module inventory (no family-specific default flips).
3. Manifest drop-accounting path.
4. Harness dry-runs (`smoke_compatible_models.py --dry-run`).

**Exit:** tooling works; systemic P0 shared bugs fixed or parked with impact list.

## 4. Phase 2 — Wave 1 families (primary)

For each family in PRD §4 Wave 1 order:

### 4.1 Standard cycle (per family)

| Step | Command pattern | Artifact |
| --- | --- | --- |
| Set `FAMILY_ID` / paths | export env | — |
| F1 Codex inspect | `codex exec … < 01-codex-inspect.md` | `codex-inspect.md` |
| F1 ax-code scan | `ax-code run --model zai-coding-plan/glm-5.2[1m] …` | `axcode-scan.md` |
| F2 Merge | Grok merges into `findings.md` | `findings.md` |
| F3 Queue | Priority table in ledger | ledger update |
| F4 Fix loop | Codex fix prompts one finding at a time | `codex-fix-*.md` + git commits |
| F5 Verify | cargo tests + smoke + dual re-scan | `verify.md` |
| F6 Close | Exit gate checklist | ledger `closed` |

### 4.2 Recommended first SKUs / artifacts

| Family ID | Preferred pack | Why |
| --- | --- | --- |
| `qwen36-27b` | AutomatosX AXQ or OptiQ 6-bit MTP | Formal MTP surface |
| `qwen36-35b-a3b` | AutomatosX 4/6-bit MTP | MoE MTP |
| `qwen35-9b` | AutomatosX 4-bit MTP | Small host |
| `qwen3-coder-next` | AutomatosX 4/6-bit | Coding agent |
| `qwen3-dense` | mlx-community Qwen3-4B-4bit | Dense cert path |
| `gemma4-12b-unified` | AutomatosX Assistant-MTP | Multimodal + assistant |
| `gemma4-e-series-26-31` | AutomatosX 26B/31B Assistant-MTP | VL + MoE/dense |
| `glm47-flash` | mlx-community GLM-4.7-Flash-4bit | MLA MoE |

### 4.3 Family-specific gate extras

| Family ID | Extra gates |
| --- | --- |
| `qwen36-*` | Greedy direct vs MTP A/B; linear exact scope independence |
| `gemma4-*` | Assistant-MTP if package present; loop detection policy; media fail-closed |
| `qwen3-coder-next` | No false MTP packaging claims |
| `glm47-flash` | Native MLX tier default (not accidental delegated) |

## 5. Phase 3 — Waves 2–4

Same cycle; lower SKU priority. Wave 4 may close with honest `LIMIT` without
support-tier inflation.

## 6. Fix hygiene

1. Prefer minimal diffs; no drive-by refactors.
2. One logical finding per commit when practical.
3. Always add regression tests for P0/P1.
4. Run targeted tests first; full workspace suite before wave boundary.
5. Update ledger in the same session as the fix.
6. If a fix touches shared MTP/runner code, list all impacted families.

### Suggested verification commands (adjust to touch set)

```bash
cargo fmt --check
cargo test -p ax-engine-mlx --lib --quiet
cargo test -p ax-engine-core --lib --quiet
# when server surface touched:
cargo test -p ax-engine-server --lib --quiet
# weights available:
python3 scripts/probe_mlx_model_support.py --model-dir "$MODEL_ARTIFACTS_DIR"
# family smoke as applicable
```

## 7. Reporting layout

```text
.internal/reports/bug-hunt/
  YYYYMMDD/
    <family-id>/
      codex-inspect.md
      axcode-scan.md
      findings.md
      codex-fix-DI-<family>-001.md
      verify.md
      env.txt          # model ids, git sha, host
```

`env.txt` must include:

```text
git_sha=
family_id=
manifest_family=
model_artifacts_dir=
codex_model=
codex_reasoning_effort=
ax_code_model=zai-coding-plan/glm-5.2[1m]
host=
date=
```

## 8. Schedule sketch (indicative, not a commitment)

| Block | Focus | Calendar hint |
| --- | --- | --- |
| Week 0 | Bootstrap + Wave 0 | 1–2 days |
| Weeks 1–3 | Wave 1 (8 families) | ~2–4 days each depending on weight availability |
| Weeks 4–5 | Wave 2 | multimodal heavier |
| Week 6 | Wave 3 | secondary preview |
| Week 7 | Wave 4 + program close | honesty pass |

Slip is expected when weights or formal MTP hosts are unavailable; use
`closed-code-only` rather than fake closes.

## 9. Done definition (program)

- Ledger shows Waves 1–3 at least `closed` or `closed-code-only`.
- No open P0 on any closed family.
- All dual-agent artifacts archived.
- Residual LIMIT list exported for docs owners (no silent public claim).

## 10. First action after reading this plan

1. Grok: set ledger `Active family` to `W0-registry` (or full Wave 0 as one unit).
2. Run F1 with both agents using frozen prompts.
3. Do not start `qwen36-27b` until Wave 0 is closed or explicitly waived with rationale.
