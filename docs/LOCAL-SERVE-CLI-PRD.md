# Local Serve CLI PRD

## Status

Proposed.

## Background

AX Engine already has the core pieces needed for local model serving:

- `ax-engine-server` starts the OpenAI-compatible HTTP server.
- `scripts/download_model.py` downloads MLX community models and generates AX
  manifests.
- `ax-engine-bench generate-manifest` makes model directories loadable by the
  repo-owned MLX runtime.
- `scripts/prepare_mtp_sidecar.py` packages quantized Qwen3.6 serving models
  with MTP sidecars from source checkpoints.

The current user experience is still fragmented. A user must know which helper
script to run, which model artifact directory to pass, how to map a friendly
model alias to a Hugging Face repo, and how to start the server afterward.
External MLX server tools expose this as a simple product workflow such as:

```text
lightning-mlx serve qwen3.5-4b
lightning-mlx status
lightning-mlx convert-mtplx /path/to/Qwen3.6-35B-A3B-4bit --mtp-source /path/to/Qwen3.6-35B-A3B
```

AX Engine should provide the same class of low-friction workflow without
weakening its runtime provenance, manifest validation, or benchmark-claim
boundaries.

## Goals

- **G1** Provide a single installed command surface for common local serving
  workflows: model alias resolution, acquisition guidance, manifest readiness,
  and foreground server launch.
- **G2** Provide a first-class `convert-mtplx` command that wraps the existing
  MTP sidecar packaging contract and emits stable machine-readable output.
- **G3** Preserve AX's explicit backend boundaries: repo-owned MLX, delegated
  `mlx_lm.server`, and delegated `llama.cpp` remain distinguishable in runtime
  metadata.
- **G4** Make the CLI useful in both source checkouts and installed
  Homebrew/Python environments.
- **G5** Defer daemon persistence until foreground serving and conversion are
  stable, tested, and documented.

## Non-goals

- Replacing `ax-engine-server`; the new command is an orchestration layer over
  the existing server binary and runtime contract.
- Promoting new model families as direct support without manifest, smoke, and
  benchmark evidence.
- Treating `convert-mtplx` output as publication-grade benchmark evidence by
  itself. README performance claims still require checked artifacts and route
  metadata.
- Implementing TUI monitoring in the first release.
- Installing boot-persistent LaunchAgents or systemd user services in the first
  release.

## User Stories

**US-1** As a new local user, I can run one command to start a supported model
without manually discovering the HF cache path.

```text
ax-engine serve qwen3.5-9b
```

**US-2** As a power user with a local model directory, I can serve it directly
without losing explicit runtime metadata.

```text
ax-engine serve /path/to/Qwen3.6-27B-8bit --port 8010
```

**US-3** As a user preparing a quantized Qwen3.6 MTP model, I can package the
sidecar from a full source model through a stable command.

```text
ax-engine convert-mtplx /path/to/Qwen3.6-35B-A3B-4bit --mtp-source Qwen/Qwen3.6-35B-A3B
```

**US-4** As an automation caller, I can add `--json` and receive structured
paths, selected model alias, server command, and readiness state without
parsing human text.

**US-5** As an operator, I can later start a daemon, inspect status, and stop it
by alias or id once the foreground command is stable.

## Product Requirements

### P0: Foreground `serve`

The first release adds:

```text
ax-engine serve <alias-or-model-dir> [--port <port>] [--host <host>] [--dry-run] [--json]
```

The command must:

1. Resolve a known alias to a model profile.
2. Resolve a local path directly when `<alias-or-model-dir>` points to a model
   directory.
3. Validate or generate `model-manifest.json` when possible.
4. Print the exact `ax-engine-server` command before launching.
5. Launch `ax-engine-server` in the foreground.
6. Preserve `Ctrl-C` shutdown semantics.
7. Support `--dry-run` before foreground launch ships so users can inspect the
   resolved server command without starting a process.

The command must fail with remediation text when:

- the model is missing from the HF cache;
- a manifest is missing and cannot be generated;
- the alias exists but is not direct-runtime ready;
- the model family is unsupported by the selected backend.

### P0: Alias Registry

Initial alias support should be small and evidence-backed:

| Alias | Model source | Route |
|---|---|---|
| `qwen3.5-9b` | `mlx-community/Qwen3.5-9B-MLX-4bit` | repo-owned MLX if manifest validates |
| `qwen3.6-27b-4bit` | `mlx-community/Qwen3.6-27B-4bit` | repo-owned MLX |
| `qwen3.6-27b-8bit` | `mlx-community/Qwen3.6-27B-8bit` | repo-owned MLX |
| `qwen3.6-35b` | `mlx-community/Qwen3.6-35B-A3B-4bit` | repo-owned MLX |
| `glm4.7-flash-4bit` | `mlx-community/GLM-4.7-Flash-4bit` | repo-owned MLX |

Requested aliases that are not yet covered by repo support docs or model I/O
checks, such as `qwen3.5-4b`, should remain candidate entries until direct
runtime validation exists.

Aliases must not silently imply performance claims. The runtime report remains
the source of truth for selected backend, support tier, and acceleration route.

### P0: `convert-mtplx`

The first release adds:

```text
ax-engine convert-mtplx <base-model> --mtp-source <source-model> [--output <dir>] [--quantize 4|8] [--mtp-depth-max <n>] [--json]
```

The command wraps the existing `prepare_mtp_sidecar.py` behavior and must:

1. Accept a local directory or repo id for the serving base.
2. Accept a local directory or repo id for the MTP source.
3. Write `mtp.safetensors`, `mtplx_runtime.json`, patched `config.json`, and
   `ax_mtp_sidecar_manifest.json`.
4. Run sidecar provenance validation before reporting success.
5. Return the output directory in JSON mode.

The current helper accepts source checkpoints through `--hf-repo`; local
`--mtp-source` paths require wrapper-owned local shard discovery or an explicit
helper extension before the command can claim local-source support.

### P1: `status` and `kill`

After foreground serving ships, add a non-persistent daemon mode:

```text
ax-engine serve qwen3.5-9b --daemon
ax-engine status
ax-engine kill <id-or-alias>
```

Daemon mode should write process metadata under a user-scoped AX directory, not
inside the repo. The first daemon release should be session-detached only; it
should not install boot-persistent services.

### P2: Boot Persistence and TUI

Boot-persistent services and TUI monitoring are follow-up work. They require a
separate service-management decision because they touch OS-level launch
mechanisms, log retention, upgrade behavior, and cleanup semantics.

## UX Requirements

- Every failure must include a concrete next command.
- Every command that starts a server must print the URL and model label.
- JSON output must be stable and documented before automation depends on it.
- Alias resolution must be explicit in output; users should see both alias and
  resolved repo/path.
- Commands must work from an installed package without assuming a source
  checkout.

## Success Metrics

| Metric | Target |
|---|---:|
| New-user command count from cached model to server | 1 |
| New-user command count from uncached mlx-community model to server | 2 or fewer |
| `convert-mtplx` output accepted by AX MTP loader | 100% for supported Qwen3.6 profiles |
| JSON schema tests for `serve --dry-run --json` and `convert-mtplx --json` | Present |
| Existing `ax-engine-server` CLI compatibility | No regressions |

## Acceptance Criteria

- `ax-engine serve qwen3.6-35b --dry-run --json` resolves an alias and prints
  the exact server argv without starting a process.
- `ax-engine serve /tmp/model --dry-run --json` preserves the explicit path and
  selected backend metadata.
- `ax-engine convert-mtplx ... --json` writes a sidecar directory, passes
  `scripts/check_mtp_sidecar_provenance.py`, and reports the output path.
- `bash scripts/check-scripts.sh` remains green after wrapper changes.
- `bash scripts/check-bench-doctor.sh` remains green after CLI workflow changes.
- README and `docs/CLI.md` document the new commands without changing
  benchmark claims.
