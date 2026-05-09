# CLI TUI Workflow Cockpit PRD

Status: Draft
Date: 2026-05-09
Owner: AX Engine
ADR: ../adr/0019-cli-tui-workflow-cockpit.md

## 1. Summary

AX Engine already has the core command surfaces needed by local users:

- `ax-engine-bench doctor` for readiness;
- `scripts/download_model.py` and `ax_engine.download_model()` for model download
  and manifest generation;
- `ax-engine-server` for the local HTTP adapter;
- `ax-engine-bench` and `scripts/bench_mlx_inference_stack.py` for workload and
  performance evidence.

The CLI TUI should not reimplement those surfaces. It should make them easier to
discover, run, monitor, and interpret from one local terminal workflow.

The product is a local workflow cockpit: a Ratatui-based command-line UI that
orchestrates existing tools, keeps route and benchmark labels visible, and links
users to the artifacts produced by the canonical commands.

## 2. Problem

The README path is correct but still cognitively heavy for a new or returning
user:

1. install the released tools;
2. run doctor;
3. choose a model acquisition path;
4. ensure `model-manifest.json` exists;
5. choose a runtime route;
6. start the server on a free port;
7. run smoke checks or benchmark commands;
8. inspect JSON and Markdown artifacts.

This creates practical friction:

- a user can run the right commands in the wrong order;
- source-build and Homebrew paths can differ;
- download helpers and server examples can drift in default paths;
- benchmark evidence types can be mixed up;
- failures often require reading logs and docs at the same time.

## 3. Current Surfaces To Reuse

The TUI must treat these as source-of-truth surfaces:

| Surface | Current responsibility | TUI role |
|---|---|---|
| `ax-engine-bench doctor --json` | local readiness report | display readiness and remediation |
| `ax-engine-bench generate-manifest <model-dir> --json` | AX model manifest generation, `ax.generate_manifest.v1` summary | invoke and show result |
| `scripts/download_model.py` | source-tree model download helper | invoke in source workflow |
| `ax_engine.download_model()` | Python SDK download helper | document and optionally invoke via Python path |
| `ax-engine-server` | local HTTP/SSE adapter | start, stop, poll, and show runtime state |
| `/health`, `/v1/runtime`, `/v1/models` | live server metadata | poll for health and route identity |
| `scripts/bench_mlx_inference_stack.py` | repo-owned MLX throughput evidence | launch and link artifacts |
| `ax-engine-bench scenario/replay/matrix` | workload-contract evidence | launch and link artifacts |

## 4. Pros And Cons

### Pros

- **Lower time-to-first-success**: users can follow one guided flow instead of
  jumping between README sections, docs, shell history, and artifact folders.
- **Better support quality**: the TUI can collect the exact command, exit code,
  tool version, selected backend, support tier, model path, and artifact path.
- **Less benchmark confusion**: each run can be labeled as readiness,
  workload-contract, route-contract, or MLX throughput evidence before it starts.
- **Safer model onboarding**: the UI can validate `config.json`,
  `model-manifest.json`, safetensors, and support-tier expectations before server
  startup.
- **Reusable local workflow history**: recent models, ports, commands, and
  artifacts can be stored without changing runtime semantics.
- **Product polish without runtime risk**: a terminal cockpit improves usability
  while keeping the engine, server, and benchmark contracts unchanged.

### Cons And Risks

- **New maintenance surface**: Ratatui, Crossterm, terminal event handling, job
  management, and packaging add their own tests and release obligations.
- **Duplication risk**: if the TUI starts parsing ad hoc text output or
  reconstructing CLI logic, it can drift from the canonical tools.
- **False control-plane impression**: a manager UI can make the preview server
  look like a production serving platform unless the product boundary stays
  explicit.
- **Benchmark misuse risk**: live UI timings are tempting but must not become
  performance claims outside the existing artifact contracts.
- **Terminal UX complexity**: resize handling, log panes, long command output,
  subprocess cancellation, and raw-mode cleanup need careful implementation.
- **Packaging complexity**: Homebrew users need the binary and dynamic MLX
  dependencies to behave consistently with `ax-engine-server` and
  `ax-engine-bench`.

## 5. Best-Practice Position

The best practice is to build a TUI only after the underlying command contracts
are made structured enough to automate safely.

### Rules

1. **Workflow orchestration only**: the TUI launches existing commands and calls
   existing HTTP endpoints. It does not implement download, server, inference, or
   benchmark logic.
2. **Structured contracts before UI polish**: add JSON or stable artifact outputs
   where needed before the TUI depends on a command.
3. **Fail closed on provenance gaps**: if the TUI cannot identify the model path,
   manifest path, selected backend, support tier, benchmark kind, or artifact
   output, it should not mark the workflow as successful.
4. **Keep benchmark classes separate**: readiness, workload-contract,
   route-contract, MLX throughput, startup-latency, and prefix-reuse artifacts
   remain different run types.
5. **Local-only process control**: the TUI may manage local processes it starts.
   It must not become remote orchestration, auth, fleet management, or tenancy.
6. **Profiles are launch recipes**: saved TUI profiles store command arguments
   and recent paths. They are not runtime configuration authority.
7. **Artifacts are read-only by default**: the TUI may browse, summarize, and open
   artifacts. It should not rewrite benchmark results.
8. **One access-layer boundary**: any future in-process use should go through
   `ax-engine-sdk`, not direct `ax-engine-core` calls.

## 6. Goals

G1. Provide a guided local workflow from install check to model-ready server.

G2. Make backend route identity visible: `selected_backend`, `support_tier`, and
`resolution_policy` should be shown anywhere a server or benchmark run appears.

G3. Reduce command-order mistakes in the README flow.

G4. Preserve benchmark evidence boundaries and artifact provenance.

G5. Provide a useful failure view with command, exit code, stderr tail, and next
remediation step.

G6. Keep implementation isolated from the engine core and server runtime.

## 7. Non-Goals

- Do not replace `ax-engine-server`.
- Do not replace `ax-engine-bench`.
- Do not replace `scripts/bench_mlx_inference_stack.py`.
- Do not implement a new model downloader.
- Do not infer official benchmark numbers from UI timing.
- Do not manage remote machines or production deployments.
- Do not add auth, tenancy, quotas, billing, or multi-node scheduling.
- Do not make Ratatui a dependency of `ax-engine-core`, `ax-engine-sdk`, or
  `ax-engine-server`.

## 8. User Stories

US-1. As a new Homebrew user, I can run `ax-engine-manager`, see whether the
installed runtime is healthy, and know the exact next step if `mlx-c` or another
dependency is missing.

US-2. As a local developer, I can select or download an `mlx-community` model,
generate the manifest if missing, and start `ax-engine-server` without copying
paths between terminal panes.

US-3. As a maintainer, I can run the correct benchmark command for the question I
am asking and see the artifact path immediately after completion.

US-4. As a reviewer, I can inspect recent benchmark artifacts and see whether a
result was MLX throughput evidence, workload-contract evidence, or delegated
route-contract evidence.

US-5. As a support engineer, I can ask a user for the TUI run summary and get
commands, versions, model path, route metadata, and artifact paths.

## 9. Product Shape

### Primary Tabs

#### Readiness

- Run `ax-engine-bench doctor --json`.
- Show host, toolchain, MLX runtime availability, model readiness blockers, and
  remediation hints.
- Display Homebrew/source mode where detectable.

#### Models

- List configured recent model directories.
- Validate `config.json`, safetensors, and `model-manifest.json`.
- Run `scripts/download_model.py` in source checkout mode.
- Run `ax-engine-bench generate-manifest <model-dir> --json`.
- Show the exact artifact path to pass into server profiles.

#### Server

- Build a local launch profile for `ax-engine-server`.
- Validate host/port and model artifact path before launch.
- Start, stop, restart, and tail logs for the server process started by the TUI.
- Poll `/health`, `/v1/runtime`, and `/v1/models`.
- Show `selected_backend`, `support_tier`, `resolution_policy`, model id, and
  live URL.

#### Benchmarks

- Offer run types instead of generic "benchmark":
  - readiness: `ax-engine-bench doctor --json`;
  - workload contract: `ax-engine-bench scenario/replay/matrix`;
  - MLX throughput: `scripts/bench_mlx_inference_stack.py`;
  - server smoke: `scripts/check-server-preview.sh`.
- Show command preview before launch.
- Capture stdout, stderr, exit code, duration, and artifact path.
- Link to result JSON and `summary.md` when available.

#### Artifacts

- Browse `benchmarks/results/`.
- Summarize known artifact schemas.
- Show route labels, model identity, prompt shape, repetition count, baseline
  identity, and pass/fail status.
- Never rewrite artifacts.

## 10. Required Contract Work Before Rich UI

W0 is a prerequisite for a dependable TUI:

- add or confirm stable JSON output for download helpers;
- align or explicitly label default download destinations between
  `scripts/download_model.py` and `ax_engine.download_model()`;
- use `ax-engine-bench generate-manifest <model-dir> --json` as the parseable
  success path;
- preserve `doctor --json` as the readiness source of truth;
- use `ax-engine-bench doctor --json` workflow discovery to distinguish
  installed-tools mode from source-checkout mode;
- use `ax-engine-bench doctor --json --mlx-model-artifacts-dir <path>`
  `model_artifacts` fields for model-directory readiness;
- use benchmark launcher `--json` output (`ax.benchmark_artifact.v1`) for
  discoverable artifact paths.

## 11. Implementation Phases

### Phase 0: Contract Cleanup

Scope:

- align download destination behavior or add explicit `--dest` defaults in docs;
- add `--json` to `scripts/download_model.py` if absent;
- use `ax.download_model.v1`, `ax.generate_manifest.v1`, and
  `ax.benchmark_artifact.v1` run-summary schemas for TUI subprocess jobs;
- document TUI-supported commands and their structured fields, including
  workflow-discovery command argv.

Exit criteria:

- no TUI code depends on fragile text parsing for success detection;
- model download, manifest generation, doctor, model directory readiness, server
  health, and benchmark artifact paths are machine-readable.
- `bash scripts/check-cli-tui-phase0.sh` passes as the non-interactive Phase 0
  contract gate before Ratatui implementation begins.

### Phase 1: Read-Only Cockpit

Scope:

- new `crates/ax-engine-tui` binary, shipped as `ax-engine-manager`;
- Ratatui shell with Readiness, Models, Server, Benchmarks, and Artifacts tabs;
- read-only views for doctor JSON, model directory validation, server health,
  and artifact browsing;
- no subprocess start/stop except doctor.

Exit criteria:

- `cargo test -p ax-engine-tui` passes;
- terminal render snapshot tests cover major tabs;
- unsupported or missing tools produce clear UI states.
- `bash scripts/check-cli-tui-phase1.sh` passes before Phase 2 job-runner work.

### Phase 2: Local Job Runner

Scope:

- local subprocess management for download, manifest generation, server launch,
  smoke checks, and benchmark commands;
- cancellation and log-tail views;
- profile storage under the user config directory;
- explicit labels for benchmark kind and evidence class.

Exit criteria:

- canceling a running job restores terminal state and cleans up child processes
  the TUI started;
- server start/stop is covered by a fake-process or test harness;
- no official benchmark result is displayed without an artifact path.
- `bash scripts/check-cli-tui-phase2.sh` passes before interactive job controls
  are added.

### Phase 3: Release Integration

Scope:

- install `ax-engine-manager` through Homebrew with the existing package;
- update README only after the binary is real;
- add smoke tests for `ax-engine-manager --help` and non-interactive diagnostics;
- add support-bundle export.

Exit criteria:

- Homebrew install includes `ax-engine-server`, `ax-engine-bench`, and
  `ax-engine-manager`;
- a fresh install can run `ax-engine-manager --check` without terminal raw mode;
- the support bundle contains no model weights or secrets.
- `bash scripts/check-cli-tui-phase3.sh` passes before publishing Homebrew
  artifacts.

## 12. Suggested Implementation Architecture

```text
crates/ax-engine-tui
  src/main.rs
  src/app.rs
  src/action.rs
  src/ui/
  src/jobs/
  src/profiles/
  src/contracts/
```

Responsibilities:

- `app`: state machine and selected tab;
- `action`: keyboard and job-result actions;
- `ui`: Ratatui rendering only;
- `jobs`: subprocess management and HTTP polling;
- `profiles`: user config files and recent history;
- `contracts`: parse stable JSON from existing tools and server endpoints.

The crate may depend on Ratatui, Crossterm, Tokio, Serde, Reqwest, directories,
and thiserror. It must not add dependencies to engine-core or server crates.

## 13. Testing Strategy

- Unit-test contract parsers with fixed JSON fixtures.
- Snapshot-test TUI rendering with synthetic app state.
- Test keyboard actions without a real terminal.
- Test subprocess job state with fake commands.
- Test server polling with a local test HTTP server.
- Add a non-interactive `--check` or `--doctor` mode for CI and Homebrew smoke
  tests.
- Run `bash scripts/check-cli-tui-phase0.sh` after any Phase 0 contract change.
- Run `cargo fmt --all` and `cargo test -p ax-engine-tui` for TUI-only changes.
- Run broader workspace tests only when shared contracts or dependency manifests
  change.

## 14. Success Metrics

- New-user path from installed tools to healthy local server is guided in one
  terminal workflow.
- Download, manifest, server, and benchmark failures include actionable next
  steps.
- Support reports include exact command lines and artifact paths.
- No benchmark evidence type is mislabeled in the UI.
- No TUI dependency enters the runtime core, SDK, or server.

## 15. Open Questions

- Should the released source-free TUI call a Python download helper, or should
  download orchestration eventually move into an installed Rust CLI subcommand?
- Should `ax-engine-manager` store profiles in TOML, JSON, or a plain
  command-history format?
- Should model discovery include only explicit paths and recent downloads, or
  also scan Hugging Face cache roots?
- Should benchmark artifact browsing stay local-only, or later support importing
  community result bundles?
- What is the minimum terminal size supported for a polished layout?
