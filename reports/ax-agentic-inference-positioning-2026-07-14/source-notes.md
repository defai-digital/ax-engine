# Source and QA notes

## Decision framing

- Audience: AX product and engineering leadership.
- Decision: whether to reposition AX Engine and AX Serving around agentic workloads, and how far up the stack to move.
- Recommendation confidence: medium; architecture evidence is strong, but customer-demand and production-certification evidence is incomplete.

## Chart map

- `multi_turn_ttft`: grouped bar chart comparing first-turn and eleventh-turn TTFT for three model families with physical prefix snapshot reuse in the AX Engine repository benchmark. This chart was selected because it shows the existing product asset most directly related to multi-turn agent workloads.
- A market-adoption chart was intentionally omitted because McKinsey and Gartner metrics use different populations and constructs; plotting them on one axis would imply false comparability.
- Vendor-reported Dynamo, vLLM, and Autellix performance results were kept in narrative context rather than plotted, because the workloads and baselines are not comparable with AX.
- The July 14 AX Serving deployment changes were not charted because they are binary implementation/status evidence rather than comparable quantitative measures.

## Strategy score method

Options were scored from 1 to 5 using these weights:

- technical fit: 25%;
- differentiation: 20%;
- time to market: 15%;
- market pull: 20%;
- defensibility: 10%;
- execution risk: 10%, with a higher score meaning lower risk.

Underlying option scores:

- A: `[5, 2, 5, 3, 2, 5]` = 3.70;
- B: `[5, 4, 4, 5, 4, 4]` = 4.45;
- C: `[2, 2, 1, 5, 1, 1]` = 2.25.

These are decision aids, not measured market outcomes.

## Validation caveats

- The AX Engine source checkout was assessed at commit `98b9bf1a3a1325406c77d07f7c12c6fc674ce61b` on July 14, 2026.
- AX Serving was reassessed from the user-specified clean `main` working tree at `/Users/akiralam/code/ax-serving`, commit `7d3636eea9a5552ee9ea868fffa863b1761e02b0`, on July 14, 2026. The earlier read-only clone at `7257d5f4` was superseded.
- Existing uncommitted AX Engine changes were not treated as published evidence.
- AX Serving explicitly states that production certification is pending.
- The current AX Serving checkout implements a first-party Helm chart, Compose evaluation stack, control-plane readiness separate from fleet routability, and stream-aware drain admission. These strengthen the recommended control-plane positioning but do not add an agent-session contract.
- Source conflict: the AX Serving README still defines `/readyz` using routable-worker semantics, while the current runtime contract and code use control-plane readiness plus `/routablez`; the implementation ledger also still says the Helm chart and readiness split are pending even though they exist in the same commit.
- GitHub Actions run `29312964367` for commit `7d3636e` failed at the reporting snapshot. Formatting, Kubernetes/Helm YAML parsing, and Clippy failures were observed; SDK conformance, dependency audit, and code-quality workflows passed.
- No customer interviews, revenue, activation, retention, or deployment telemetry were available.
