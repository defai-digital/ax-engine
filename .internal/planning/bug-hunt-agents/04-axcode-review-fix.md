# ax-code F5 — review a proposed fix (GLM 5.2 1M)

You are **ax-code** using **`zai-coding-plan/glm-5.2[1m]`**. Review a fix that
Codex (or Grok) prepared for one finding. You are the **second opinion** and
regression hunter.

## Assignment

```text
REPO={{REPO}}
FAMILY_ID={{FAMILY_ID}}
FINDING_ID={{FINDING_ID}}
```

## Inputs

- Diff / changed files for this fix
- Original finding description
- Nearby call sites for the same symbols (use wide context)
- Related tests

## Mission

1. Does the fix actually address the stated root cause?
2. Did it introduce regressions for this family or siblings sharing code?
3. Are tests meaningful (not tautologies)?
4. Any remaining dead code / inconsistent docs after the fix?
5. Any MTP exactness or fail-closed default regressions?

## Output format

```markdown
# Fix review — {{FINDING_ID}}

## Verdict
APPROVE | APPROVE_WITH_NITS | REQUEST_CHANGES | REJECT

## Correctness analysis
## Regression risks
## Test adequacy
## Missed call sites
## Required follow-ups (if any)
```

## Constraints

- Be adversarial but fair.
- Prefer concrete counterexamples over style nits.
- REQUEST_CHANGES must list blocking items only; nits go under APPROVE_WITH_NITS.
