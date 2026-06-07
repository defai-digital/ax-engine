# Local Serve CLI Technical Specification

## Overview

This spec describes a new installed orchestration CLI, `ax-engine`, that wraps
existing AX Engine components into a local-serving workflow:

```text
ax-engine serve <alias-or-model-dir>
ax-engine convert-mtplx <base-model> --mtp-source <source-model>
```

The first implementation should focus on foreground `serve`, `serve --dry-run`,
and `convert-mtplx`. Daemon/status/kill are specified as a follow-up phase so
the command surface can be designed without committing to boot persistence.

## Current Boundary

Relevant current files:

| File | Role |
|---|---|
| `crates/ax-engine-server/src/args.rs` | Existing server CLI and runtime flags. |
| `crates/ax-engine-server/src/args/presets.rs` | Existing server preset definitions. |
| `crates/ax-engine-server/src/args/session.rs` | Server args to SDK session config. |
| `crates/ax-engine-bench/src/main.rs` | Existing workload CLI dispatch. |
| `crates/ax-engine-bench/src/generate_manifest.rs` | Installed manifest generation. |
| `scripts/download_model.py` | MLX community model acquisition helper. |
| `scripts/prepare_mtp_sidecar.py` | Generic Qwen3.6 MTP sidecar packager. |
| `scripts/check_mtp_sidecar_provenance.py` | Sidecar provenance validator. |
| `docs/CLI.md` | Current CLI documentation. |

The new CLI must not duplicate runtime loading logic. It should compose these
surfaces and keep `ax-engine-server` as the process that owns HTTP serving.

## Package Shape

The PyPI first implementation lives in `python/ax_engine/_cli.py` as the
`ax-engine` console script, because the wheel already ships `ax-engine-server`.
Homebrew/Rust parity can either keep wrapping that script or add a Rust binary
crate later:

```text
crates/ax-engine-cli/
  Cargo.toml
  src/main.rs
  src/aliases.rs
  src/serve.rs
  src/convert_mtplx.rs
  src/doctor.rs
  src/process_registry.rs
```

Binary name:

```text
ax-engine
```

The command should be shipped by the same Homebrew and Python wheel paths that
ship `ax-engine-server` and `ax-engine-bench`. The PyPI path is already a
console script; Homebrew parity is a follow-up packaging slice.

## Command Model

Use subcommands:

```rust
#[derive(Parser)]
#[command(name = "ax-engine", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

enum Command {
    Serve(ServeArgs),
    ConvertMtplx(ConvertMtplxArgs),
    Status(StatusArgs),
    Kill(KillArgs),
}
```

P0 implements `serve` and `convert-mtplx`. `status` and `kill` are P1 commands;
they should either be omitted from the shipped parser or return a clear
not-implemented error until the process registry exists.

### `serve`

```text
ax-engine serve <alias-or-model-dir>
  [--host 127.0.0.1]
  [--port 8080]
  [--model-id <label>]
  [--backend auto|mlx|mlx-lm-delegated|llama-cpp]
  [--hf-cache-root <dir>]
  [--download]
  [--dry-run]
  [--json]
  [-- <extra ax-engine-server args>]
```

Rules:

1. A filesystem path wins over alias lookup when it exists.
2. Alias lookup returns a `ModelProfile`.
3. `--backend auto` picks the profile's preferred backend.
4. `--download` may call the existing download helper; without `--download`,
   a missing cached model fails with remediation text.
5. `--dry-run` resolves everything and prints the server argv but never starts
   a process.
6. Additional server flags after `--` are appended after generated argv, and
   conflicts should be rejected when they override managed fields such as
   `--mlx-model-artifacts-dir`.

### `convert-mtplx`

```text
ax-engine convert-mtplx <base-model>
  --mtp-source <source-model>
  [--output <dir>]
  [--mtp-depth-max <n>]
  [--quantize 4|8]
  [--quantization-group-size 64]
  [--json]
```

This is a stable wrapper around `scripts/prepare_mtp_sidecar.py` behavior. The
Rust wrapper may call a library implementation, spawn the Python script in
source-checkout mode, or initially delegate to an installed helper. The
automation contract is the `ax-engine convert-mtplx --json` output, not the
intermediate script text.

Argument mapping to the current helper:

| `ax-engine convert-mtplx` | `scripts/prepare_mtp_sidecar.py` |
|---|---|
| `<base-model>` | `--base <base-model>` |
| `--mtp-source <repo-id>` | `--hf-repo <repo-id>` |
| `--output <dir>` | `--output <dir>` |
| `--mtp-depth-max <n>` | `--mtp-depth-max <n>` |
| `--quantize 4|8` | `--quantize 4|8` |
| `--quantization-group-size <n>` | `--group-size <n>` |

The current helper's `--hf-repo` path supports Hugging Face repo ids. If
`--mtp-source` is a local directory, the wrapper must extend the helper or use a
library path that discovers local `mtp.*` shards from the source
`model.safetensors.index.json`. It must not pass a filesystem path to
`--hf-repo` and report success as if local-source support were implemented.

## Alias Registry

Represent aliases as data, not ad hoc match arms scattered through serve code:

```rust
pub struct ModelProfile {
    pub alias: &'static str,
    pub aliases: &'static [&'static str],
    pub repo_id: &'static str,
    pub model_id: &'static str,
    pub expected_model_types: &'static [&'static str],
    pub preferred_backend: BackendKind,
    pub support_tier: SupportTier,
    pub default_port: u16,
    pub default_server_args: &'static [&'static str],
    pub notes: &'static str,
}
```

Initial profiles should mirror documented support only. A profile can be
present but marked as `requires_delegation` or `preview` when direct-runtime
evidence is incomplete.

Alias matching must be case-insensitive and normalize `_`, `-`, and `.` only
for lookup. Output must preserve the canonical alias.

## Model Resolution

Resolution order:

1. Explicit local path containing `config.json`.
2. Explicit local HF-cache-style directory with `snapshots/<hash>`.
3. Alias registry.
4. Raw repo id when it contains `/`.

Resolved model directory readiness:

1. `config.json` exists.
2. Safetensors exist.
3. `model-manifest.json` exists or `generate-manifest` can run.
4. Manifest model family matches the profile's expected model types when a
   profile is used.

When `--download` is set and a repo id is missing, the CLI should call the same
workflow as `scripts/download_model.py`: acquire via `mlx-lm`, validate files,
and generate a manifest.

## Server Command Construction

For repo-owned MLX:

```text
ax-engine-server
  --host <host>
  --port <port>
  --model-id <model-id>
  --mlx
  --support-tier mlx-preview
  --mlx-model-artifacts-dir <resolved-dir>
```

For delegated `mlx_lm.server`, the CLI should not start `mlx_lm.server` in the
first phase. It should require `--mlx-lm-server-url` or fail with a concrete
example.

For delegated `llama.cpp`, the CLI should preserve current server flags and
require explicit model path or server URL.

## JSON Contracts

### `serve --dry-run --json`

```json
{
  "schema_version": "ax.local_serve_plan.v1",
  "command": "serve",
  "alias": "qwen3.6-35b",
  "resolved_model": {
    "kind": "hf_cache_snapshot",
    "repo_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "path": "/Users/me/.cache/huggingface/hub/...",
    "manifest_present": true,
    "model_family": "qwen3_next"
  },
  "backend": "mlx",
  "support_tier": "mlx-preview",
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "url": "http://127.0.0.1:8080",
    "argv": ["ax-engine-server", "--mlx", "..."]
  }
}
```

### `convert-mtplx --json`

```json
{
  "schema_version": "ax.convert_mtplx.v1",
  "command": "convert-mtplx",
  "base_model": {
    "input": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "path": "/Users/me/.cache/huggingface/hub/..."
  },
  "mtp_source": {
    "input": "Qwen/Qwen3.6-35B-A3B"
  },
  "output_dir": "/Users/me/.cache/huggingface/hub/models--ax-local--Qwen3.6-35B-MTP/snapshots/v1",
  "files": {
    "mtp": "mtp.safetensors",
    "runtime": "mtplx_runtime.json",
    "config": "config.json",
    "provenance": "ax_mtp_sidecar_manifest.json"
  },
  "validation": {
    "provenance": "ok"
  }
}
```

## Process Registry for P1

Daemon mode should write metadata outside the repo:

```text
~/Library/Application Support/ax-engine/servers/<id>.json
~/.local/share/ax-engine/servers/<id>.json
```

Record:

```json
{
  "schema_version": "ax.server_process.v1",
  "id": "qwen35-9b-8080",
  "alias": "qwen3.5-9b",
  "pid": 12345,
  "started_at": "2026-06-06T00:00:00Z",
  "url": "http://127.0.0.1:8080",
  "model_dir": "/path/to/model",
  "argv": ["ax-engine-server", "..."],
  "stdout_log": "/path/to/log",
  "stderr_log": "/path/to/log"
}
```

`status` should verify `pid` liveness and server `/health` separately. A live
PID with failed health is `degraded`, not `running`.

## Error Handling

Every user-facing failure should include:

- what was attempted;
- what was found;
- the next command to run;
- whether the failure is model acquisition, manifest generation, backend
  selection, server startup, or runtime readiness.

Examples:

```text
Model alias qwen3.5-9b resolved to mlx-community/Qwen3.5-9B-MLX-4bit,
but no cached snapshot was found.

Run:
  ax-engine serve qwen3.5-9b --download
```

```text
The model directory has config.json but no model-manifest.json.

Run:
  ax-engine-bench generate-manifest /path/to/model
```

## Validation

Unit tests:

- alias normalization and canonicalization;
- path-vs-alias precedence;
- server argv construction;
- JSON schema stability;
- conflict detection for managed server flags;
- missing model remediation text;
- `convert-mtplx` argument mapping.
- local `--mtp-source` behavior: implemented local shard discovery or a
  fail-closed error that states local source support is not available yet.

Integration checks:

- `ax-engine serve qwen3.6-35b --dry-run --json`;
- `ax-engine serve /tmp/fake-model --dry-run --json` with a fixture manifest;
- `ax-engine convert-mtplx --help`;
- `ax-engine convert-mtplx ... --json` runs the provenance checker before
  returning success;
- existing `bash scripts/check-bench-doctor.sh`;
- existing `bash scripts/check-server-preview.sh`;
- existing `bash scripts/check-scripts.sh`.

Live model checks should remain opt-in through `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR`
or explicit fixture paths.

## Rollout Plan

1. Add `ax-engine` with `serve --dry-run`, alias registry, and JSON plan output.
2. Add foreground `serve` process execution.
3. Add `convert-mtplx` wrapper and JSON output.
4. Document commands in `docs/CLI.md`, README quick-start, and release notes.
5. Add non-persistent daemon/status/kill.
6. Evaluate boot persistence and TUI in a separate ADR.
