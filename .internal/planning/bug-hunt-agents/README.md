# Bug-hunt multi-agent prompts

Frozen prompts for the Direct-Inference Model Bug Hunt program.

| File | Agent | Phase |
| --- | --- | --- |
| [00-grok-orchestrator.md](00-grok-orchestrator.md) | Grok CLI | All (system conduct) |
| [01-codex-inspect.md](01-codex-inspect.md) | Codex max reasoning | F1 inspect |
| [02-axcode-scan.md](02-axcode-scan.md) | ax-code GLM 5.2 1M | F1 wide scan |
| [03-codex-fix.md](03-codex-fix.md) | Codex max reasoning | F4 fix |
| [04-axcode-review-fix.md](04-axcode-review-fix.md) | ax-code GLM 5.2 1M | F5 review |
| [05-grok-merge-findings.md](05-grok-merge-findings.md) | Grok CLI | F2 merge |
| [06-family-exit-gate.md](06-family-exit-gate.md) | Grok CLI | F6 close |

## Model binding

| Role | CLI | Model / effort |
| --- | --- | --- |
| Orchestrator | `grok` | default Grok Build session model |
| Deep reasoner | `codex exec` | `gpt-5.6-sol` · `model_reasoning_effort=max` |
| Wide scanner | `ax-code run` | `zai-coding-plan/glm-5.2[1m]` |

### Probe Codex “SoL very high” (run once, record results)

```bash
# Host config (~/.codex/config.toml) already binds:
#   model = "gpt-5.6-sol"
#   model_reasoning_effort = "max"
codex exec -C /path/to/ax-engine -s read-only \
  -m gpt-5.6-sol \
  -c 'model_reasoning_effort="max"' \
  --output-last-message /path/to/out.md < prompt.md
```

Record here after probe:

| Field | Value |
| --- | --- |
| CODEX_MODEL | `gpt-5.6-sol` |
| CODEX_REASONING_EFFORT | `max` |
| Probed on | 2026-08-11 (codex-cli 0.147.0; `~/.codex/config.toml`) |
| AX_CODE_MODEL | `zai-coding-plan/glm-5.2[1m]` (confirmed via `ax-code models`) |

## Substitution variables

Before invoking a prompt, substitute:

| Variable | Example |
| --- | --- |
| `{{FAMILY_ID}}` | `qwen36-27b` |
| `{{MANIFEST_FAMILY}}` | `qwen3_5` |
| `{{MODEL_ARTIFACTS_DIR}}` | absolute path or `UNAVAILABLE` |
| `{{FINDING_ID}}` | `DI-qwen36-27b-001` |
| `{{FINDING_TITLE}}` | short title |
| `{{REPO}}` | absolute repo root |
| `{{OUT_DIR}}` | report directory for this run |

## Rules for all agents

1. Direct MLX only; do not “fix” via delegated backends.
2. Classify every issue: BUG / IMPL / MTP / PERF / DEAD / DOC / LIMIT.
3. Prefer symbol anchors over line numbers.
4. No public support or tok/s claims without existing evidence gates.
5. Fail-closed product safety beats speculative speed.
6. One family at a time; do not expand scope to unrelated families unless a shared root cause is proven.
