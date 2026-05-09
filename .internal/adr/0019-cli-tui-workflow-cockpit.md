# ADR 0019: Add a CLI TUI as a Local Workflow Cockpit

Status: Proposed
Date: 2026-05-09
Deciders: AX Engine

## Context

AX Engine now exposes several command and access surfaces:

- `ax-engine-server` for local HTTP/SSE access;
- `ax-engine-bench` for readiness, manifest generation, workload contracts, and
  benchmark-support commands;
- `scripts/bench_mlx_inference_stack.py` for repo-owned MLX throughput evidence;
- Python helpers such as `ax_engine.download_model()`;
- source-tree scripts such as `scripts/download_model.py`.

These surfaces are useful, but the user journey crosses many commands and
artifact locations. A user has to know whether they are in a Homebrew install or
source checkout, which model path is valid, whether a manifest exists, which
runtime route is selected, which benchmark command answers which question, and
where the resulting artifacts were written.

Adding a terminal UI can improve the workflow, but it carries architectural risk:
a "manager" can accidentally become a new runtime layer, a new benchmark system,
or a production control plane.

## Decision

Add a Ratatui-based CLI TUI as a separate boundary crate and binary:

```text
crates/ax-engine-tui -> ax-engine-manager
```

The TUI is a local workflow cockpit. It orchestrates existing commands, polls
existing server endpoints, and reads existing artifacts. It does not implement
download, inference, server routing, benchmark measurement, scheduling, KV
management, or remote orchestration.

The dependency direction is:

```text
existing tools and endpoints
    -> ax-engine-manager

ax-engine-core
    -> ax-engine-sdk
        -> ax-engine-server
        -> ax-engine-py
        -> ax-engine-tui (only if in-process SDK use becomes justified later)
```

Initial implementation should prefer subprocess and HTTP contracts over direct
runtime embedding. If the TUI later uses Rust APIs directly, it must use
`ax-engine-sdk` or a dedicated tooling contract, not `ax-engine-core`.

## Architecture Rules

1. Ratatui and Crossterm dependencies are allowed only in `ax-engine-tui`.
2. The TUI must not make `ax-engine-server` an embedded child library.
3. The TUI must not parse fragile human text when a JSON contract can be added.
4. The TUI must not report official benchmark numbers without an artifact
   produced by the canonical benchmark command.
5. The TUI may manage only local processes it started.
6. The TUI must label runtime route identity and benchmark evidence class.
7. Saved profiles are launch recipes, not authoritative runtime config.
8. Remote orchestration, auth, tenancy, fleet control, and production serving are
   out of scope.

## Rationale

This follows ADR 0008's thin-access-layer rule: user-facing surfaces are allowed
when they sit above the runtime contract and do not redefine engine behavior.

It also follows ADR 0005 and ADR 0017's evidence discipline: benchmark results
must remain artifact-backed and route-labeled. A UI can make the artifacts easier
to produce and inspect, but it must not create a second measurement regime.

Keeping the TUI in a separate crate contains terminal dependencies and keeps the
runtime core lean. It also makes packaging explicit: Homebrew can install
`ax-engine-manager` next to `ax-engine-server` and `ax-engine-bench` without
changing the server's own contract.

## Alternatives Considered

### Do nothing

Pros:

- no new dependency or maintenance surface;
- no risk of UI drift.

Cons:

- the README workflow remains multi-step and easy to misorder;
- support requests still need manual command reconstruction;
- users can confuse benchmark evidence types.

Rejected because the current command surface is powerful but not yet easy enough
for guided local use.

### Add interactive prompts to existing CLIs

Pros:

- smaller dependency set;
- keeps each command close to its implementation.

Cons:

- fragments the workflow across several binaries;
- does not provide a shared job history or artifact browser;
- makes `ax-engine-bench` and `ax-engine-server` carry UI concerns.

Rejected because it spreads interaction logic into tools that should remain
scriptable and evidence-oriented.

### Build a web dashboard

Pros:

- richer UI and easier layout;
- potentially better artifact visualization.

Cons:

- implies a longer-lived service/control-plane shape;
- adds browser, asset, and network concerns;
- risks making the preview local server look like a production management plane.

Rejected for now. A web dashboard can be reconsidered only after the local CLI
workflow contract proves useful.

### Embed server/runtime directly into the TUI

Pros:

- fewer subprocess edges;
- potentially tighter lifecycle control.

Cons:

- creates another runtime boundary;
- blurs server, SDK, and TUI ownership;
- makes terminal UI failures more likely to affect runtime behavior.

Rejected for the initial design. The TUI should launch `ax-engine-server` as a
process and poll its HTTP endpoints.

## Consequences

Positive:

- new users get a guided local path through install checks, model readiness,
  server startup, and benchmark selection;
- maintainers get better support summaries with commands and artifact paths;
- benchmark evidence classes become more visible;
- Ratatui dependencies remain contained in one tooling crate.

Negative / risks:

- a new binary must be tested, packaged, and documented;
- terminal event handling and subprocess lifecycle add failure modes;
- structured command contracts must be kept stable;
- a polished TUI may create expectations beyond preview/local scope unless the
  product boundary is repeated in the UI.

## Implementation Direction

Phase 0: Contract cleanup

- add or confirm stable JSON output for download and manifest workflows;
- align or explicitly distinguish source and installed download destinations;
- ensure benchmark commands expose artifact paths reliably.

Phase 1: Read-only cockpit

- add `crates/ax-engine-tui`;
- render readiness, model validation, server health, and artifact browsing;
- avoid long-running job management except doctor.

Phase 2: Local job runner

- add subprocess lifecycle for download, manifest generation, server launch,
  smoke checks, and benchmark runs;
- store launch profiles and recent artifacts;
- label evidence class for every run.

Phase 3: Release integration

- install `ax-engine-manager` through Homebrew;
- add non-interactive diagnostics for CI and package tests;
- update public docs only after the binary exists.

## Review Rule

Review this ADR when one of the following happens:

- `ax-engine-manager` needs direct in-process runtime access;
- a benchmark feature is proposed that bypasses canonical artifacts;
- remote machine management or production serving is requested;
- Homebrew packaging changes the installed binary set;
- download/model acquisition moves from Python/script helpers into a Rust CLI
  contract.
