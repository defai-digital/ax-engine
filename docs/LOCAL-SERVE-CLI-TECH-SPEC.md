# Local Serve CLI Technical Specification

## Overview

This spec describes a new installed orchestration CLI, `ax-engine`, that wraps
existing AX Engine components into a local-serving workflow:

```text
ax-engine serve <alias-or-model-dir>
ax-engine convert-mtplx <base-model> --mtp-source <source-model>
```

The first implementation should focus on foreground `serve`, explicit
`serve --download`, standalone `download`, `serve --dry-run`, and
`convert-mtplx`. Daemon/status/kill are specified as a follow-up phase so the
command surface can be designed without committing to boot persistence.

Current P0 status: the PyPI console script implements `serve`, `serve --download`,
`serve --dry-run --json`, `download`, and `convert-mtplx`; `ax-engine-server`
remains available for backward-compatible explicit server invocation.

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

Use subcommands. The current PyPI parser exposes:

```text
ax-engine serve ...
ax-engine download ...
ax-engine convert-mtplx ...
```

P0 implements `serve`, `download`, and `convert-mtplx`. `status` and `kill` are
P1 commands; they should either be omitted from the shipped parser or return a
clear not-implemented error until the process registry exists.

### `serve`

```text
ax-engine serve <alias-or-model-dir>
  [--host 127.0.0.1]
  [--port 8080]
  [--hf-cache-root <dir>]
  [--download]
  [--dry-run]
  [--json]
  [-- <extra ax-engine-server args>]
```

Rules:

1. A filesystem path wins over alias lookup when it exists.
2. Alias lookup maps to the server's preset vocabulary.
3. Missing optional parameters use model-specific defaults from the selected
   preset or model profile; explicit user flags always override defaults.
4. `--download` explicitly downloads supported aliases or raw Hugging Face repo
   ids before launch. Without `--download`, missing model artifacts fail closed
   with a concrete next command.
5. `--dry-run` resolves everything and prints the server argv but never starts
   a process.
6. Additional server flags after `--` are appended after generated argv, and
   future hardening should reject conflicts when they override managed fields such as
   `--mlx-model-artifacts-dir`.
7. Explicit backend selection remains follow-up work.

### `download`

```text
ax-engine download <alias-or-repo-id>
  [--dest <dir>]
  [--force]
  [--json]
```

Rules:

1. Alias input resolves through the Qwen3.6 and Gemma 4 MLX entries in the same
   model profile table as `serve`. Qwen3.6 27B and Gemma 4 E2B include
   4/5/6/8-bit download aliases.
2. Inputs containing `/` are treated as raw Hugging Face repo ids.
3. Unknown non-repo inputs fail with the supported alias list.
4. The command delegates to `scripts/download_model.py`, which downloads through
   `mlx-lm`, validates files, and generates `model-manifest.json` when possible.
5. JSON mode emits the helper's `ax.download_model.v1` summary plus resolved
   alias/preset metadata when applicable.
6. Bit variants without a server preset still serve through the downloaded local
   artifact directory after `serve --download`; they do not pretend to be preset
   aliases.
7. The default destination is the Hugging Face Hub cache. `HF_HUB_CACHE`,
   `HF_HOME`, and `XDG_CACHE_HOME` control the cache location. `--dest` copies
   the resolved snapshot to an explicit directory and is not the default.

### `convert-mtplx`

```text
ax-engine convert-mtplx <base-model>
  --mtp-source <source-model>
  [--output <dir>]
  [--mtp-depth-max <n>]
  [--quantize 4|8]
  [--group-size 64]
  [--fair-base-only]
  [--json]
```

This is a stable wrapper around `scripts/prepare_mtp_sidecar.py` behavior. The
Rust wrapper may call a library implementation, spawn the Python script in
source-checkout mode, or initially delegate to an installed helper. The
automation contract is the `ax-engine convert-mtplx --json` output, not the
intermediate script text.

If `--mtp-depth-max` is omitted, infer the safe default from the model identity:
Qwen3.6 27B uses depth 3; Qwen3.6 35B-A3B uses depth 1. Unknown MTP models fall
back to depth 1 until they have an explicit profile.

Argument mapping to the current helper:

| `ax-engine convert-mtplx` | `scripts/prepare_mtp_sidecar.py` |
|---|---|
| `<base-model>` | `--base <base-model>` |
| `--mtp-source <repo-id>` | `--hf-repo <repo-id>` |
| `--output <dir>` | `--output <dir>` |
| `--mtp-depth-max <n>` | `--mtp-depth-max <n>` |
| `--quantize 4|8` | `--quantize 4|8` |
| `--group-size <n>` | `--group-size <n>` |
| `--fair-base-only` | provenance checker `--fair-base-only` |

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

When `--download` is set and a repo id is missing, the CLI calls the same
workflow as `scripts/download_model.py`: acquire via `mlx-lm`, validate files,
and generate a manifest. `serve --download` requires the download summary to
report `status: "ready"` before it starts `ax-engine-server`.

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

The common user-facing path should prefer:

```text
ax-engine serve <alias-or-model-dir>
```

Direct `ax-engine-server ...` invocation remains supported for backward
compatibility, source-build debugging, and cases where callers need to spell out
every runtime flag.

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
  "input": "qwen36-35b",
  "resolved": {
    "kind": "preset",
    "model": "qwen36-35b",
    "preset": "qwen3.6-35b",
    "resolution": "hf-cache"
  },
  "server": {
    "url": "http://127.0.0.1:8080",
    "argv": ["ax-engine-server", "--host", "127.0.0.1", "--port", "8080", "--mlx", "..."]
  }
}
```

### `download --json`

```json
{
  "schema_version": "ax.download_model.v1",
  "input": "qwen36-35b",
  "alias": "qwen3.6-35b",
  "preset": "qwen3.6-35b",
  "repo_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
  "dest": "/Users/me/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/<hash>",
  "manifest_present": true,
  "safetensors_count": 5,
  "config_present": true,
  "status": "ready",
  "errors": []
}
```

### `download --list --json`

```json
{
  "schema_version": "ax.download_options.v1",
  "default_destination": {
    "kind": "huggingface_hub_cache",
    "env": ["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"],
    "dest_semantics": "--dest copies the resolved snapshot to an explicit directory"
  },
  "targets": []
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
  "mtp_depth_max": 1,
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
