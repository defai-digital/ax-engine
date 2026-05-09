# AX Engine Manager

`ax-engine-manager` is the local Ratatui cockpit for AX Engine. It gives a
terminal UI over the same contracts used by `ax-engine-bench`, `ax-engine-server`,
and benchmark artifacts.

The manager is deliberately conservative:

- it can run doctor and show model-artifact readiness;
- it can poll a local server's health, runtime metadata, and model list;
- it can show the default local host/port, server action buttons, and full
  local HTTP endpoint URLs;
- it can show benchmark artifact summaries and scan a results directory;
- it can project guarded job plans from the doctor workflow;
- it can export a redacted support bundle;
- it does not copy model weights, environment variables, API keys, or raw logs;
- server start/stop buttons are currently selectable preview controls; owned
  process launch, confirmation, and shutdown are intentionally a follow-up
  lifecycle slice;
- it does not yet perform interactive downloads or benchmark launches from
  inside the TUI.

## Install

Released macOS arm64 builds install the manager through Homebrew:

```bash
brew install defai-digital/ax-engine/ax-engine
ax-engine-manager --check
```

From a source checkout:

```bash
brew install mlx-c
cargo build --release -p ax-engine-server -p ax-engine-bench -p ax-engine-tui
target/release/ax-engine-manager --check
```

Use `cargo run` while developing the TUI:

```bash
cargo run -p ax-engine-tui --bin ax-engine-manager -- --check
```

## Quick Start

Download a supported MLX model and generate its AX manifest:

```bash
python scripts/download_model.py mlx-community/Qwen3-4B-4bit
MODEL_DIR="$HOME/.cache/ax-engine/models/mlx-community--Qwen3-4B-4bit"
```

Check local readiness without entering terminal raw mode:

```bash
ax-engine-manager --check --model-dir "$MODEL_DIR"
```

Open the interactive cockpit:

```bash
ax-engine-manager --model-dir "$MODEL_DIR"
```

If you are running from a source checkout without installed release binaries,
use the built binary instead:

```bash
target/release/ax-engine-manager --model-dir "$MODEL_DIR"
```

## Start With A Server

Start the local HTTP server in another terminal:

```bash
ax-engine-server \
  --mlx \
  --mlx-model-artifacts-dir "$MODEL_DIR" \
  --port 8080
```

Then launch the manager with the server URL:

```bash
ax-engine-manager \
  --model-dir "$MODEL_DIR" \
  --server-url http://127.0.0.1:8080
```

The manager polls:

- `/health`
- `/v1/runtime`
- `/v1/models`

Use `--server-url` to point at a different local port. Without `--server-url`,
the Server tab still shows the default local target:

```text
http://127.0.0.1:8080
```

The Server tab lists clickable preview controls for `Start`, `Stop`, and
`Restart`, plus the full local URLs for:

- `/health`
- `/v1/runtime`
- `/v1/models`
- `/v1/generate`
- `/v1/generate/stream`
- `/v1/chat/completions`
- `/v1/completions`

Selecting a server action updates the status line, but does not yet spawn or
kill a process. This keeps the current TUI safe while the next phase defines the
owned-process lifecycle, confirmation rules, and log capture contract.

## Use Existing JSON Contracts

For reproducible diagnostics or CI-style workflows, generate JSON with the
canonical tools and give those files to the manager.

Doctor readiness:

```bash
ax-engine-bench doctor \
  --mlx-model-artifacts-dir "$MODEL_DIR" \
  --json > /tmp/ax-doctor.json

ax-engine-manager \
  --doctor-json /tmp/ax-doctor.json \
  --model-dir "$MODEL_DIR"
```

Benchmark artifact summary:

```bash
ax-engine-bench scenario \
  --manifest benchmarks/manifests/scenario/chat_qwen_short.json \
  --output-root benchmarks/results \
  --json > /tmp/ax-benchmark.json

ax-engine-manager \
  --benchmark-json /tmp/ax-benchmark.json \
  --artifact-root benchmarks/results
```

`--artifact-root` scans benchmark result directories and reports whether
`summary.md` or `contract_failure.json` is present for each artifact.

## TUI Controls

Inside the interactive TUI:

| Key | Action |
|---|---|
| `Tab` / Right arrow | Next tab |
| `Shift-Tab` / Left arrow | Previous tab |
| Left mouse click on a tab | Select that tab |
| Left mouse click on a Server action | Select Start, Stop, or Restart preview |
| Left mouse click on a Server URL row | Select that endpoint URL |
| `q` / `Esc` | Exit |

Mouse support is intentionally scoped today: tab selection, Server preview
actions, and Server URL rows are clickable. Download pickers, benchmark launch,
and real server process start/stop still need explicit ownership and
confirmation contracts before they become mutating controls.

Tabs:

| Tab | Purpose |
|---|---|
| Readiness | Doctor status, workflow mode, model-artifact readiness, runtime issues |
| Models | Selected model artifact status, quantization, manifest and safetensors presence |
| Server | Default host/port, Start/Stop/Restart preview selection, local endpoint URLs, and metadata from `--server-url` |
| Jobs | Guarded job-plan projection from doctor workflow; currently non-mutating |
| Benchmarks | One benchmark artifact summary from `--benchmark-json` |
| Artifacts | Result-directory scan from `--artifact-root` |

## Support Bundle

Use a support bundle when you need to share diagnostic state without sharing
private model paths, weights, secrets, or logs:

```bash
ax-engine-manager \
  --model-dir "$MODEL_DIR" \
  --server-url http://127.0.0.1:8080 \
  --artifact-root benchmarks/results \
  --support-bundle /tmp/ax-engine-support
```

The command writes:

```text
/tmp/ax-engine-support/support-bundle.json
```

The bundle records contract state only:

- doctor status and workflow mode;
- whether a model path was present, without recording the path;
- model-artifact readiness booleans;
- server metadata load states;
- benchmark artifact status and whether result paths are present;
- artifact names and status flags.

It explicitly does not copy:

- model weights;
- environment variables;
- API keys or tokens;
- raw process logs;
- full model filesystem paths.

## Release Gate

Before publishing Homebrew artifacts that include the manager, run:

```bash
bash scripts/check-cli-tui-phase3.sh
```

This builds the release binaries, verifies the release archive contains
`ax-engine-server`, `ax-engine-bench`, and `ax-engine-manager`, runs manager
help/check smoke tests, and verifies support bundles do not contain model
weights or secret-like content.

## Troubleshooting

If `ax-engine-manager --check` reports `doctor=unavailable`, first run:

```bash
ax-engine-bench doctor --mlx-model-artifacts-dir "$MODEL_DIR" --json
```

If server panels are not loaded, pass a local server URL:

```bash
ax-engine-manager --server-url http://127.0.0.1:8080
```

If benchmark panels are empty, pass either a JSON summary from an
`ax-engine-bench ... --json` command or a result root:

```bash
ax-engine-manager \
  --benchmark-json /tmp/ax-benchmark.json \
  --artifact-root benchmarks/results
```

If you installed through Homebrew and `ax-engine-bench doctor` fails with
`Library not loaded: /opt/homebrew/opt/mlx-c/lib/libmlxc.dylib`, repair the MLX
C runtime:

```bash
brew install mlx-c
brew reinstall defai-digital/ax-engine/ax-engine
```
