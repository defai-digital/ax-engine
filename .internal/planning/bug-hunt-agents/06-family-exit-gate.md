# Grok F6 — family exit gate

You are the Grok orchestrator deciding whether `{{FAMILY_ID}}` may leave
`in_progress`.

## Checklist (all required for `closed`)

| # | Gate | Evidence path / command result |
| --- | --- | --- |
| 1 | Codex inspect artifact exists | `codex-inspect.md` |
| 2 | ax-code scan artifact exists | `axcode-scan.md` |
| 3 | Merged findings disposition complete for all P0/P1 | `findings.md` + ledger |
| 4 | Post-fix re-scan shows no new P0/P1 | verify notes |
| 5 | Targeted cargo tests pass for touched crates | command log |
| 6 | fmt/clippy expectations met for touched code | command log |
| 7 | Weights smoke **or** explicit `closed-code-only` | probe/smoke or rationale |
| 8 | MTP A/B exactness **or** documented n/a / fail-closed | if applicable |
| 9 | Residual LIMIT list written | findings or ledger |
| 10 | `env.txt` complete | OUT_DIR |

## Decision

Emit exactly one:

```text
EXIT_DECISION=closed
EXIT_DECISION=closed-code-only
EXIT_DECISION=parked
EXIT_DECISION=remain_in_progress
```

With:

- Reasons
- Open P0/P1 counts
- Follow-ups
- Whether the **next** family may start (yes only if closed/closed-code-only/parked)

## Update the ledger

- Set family status
- Clear `Active family` if exiting
- Append change log row with date and summary
- Paste session handoff block
