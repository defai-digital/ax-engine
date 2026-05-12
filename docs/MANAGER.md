# AX Engine Manager

`ax-engine-manager` is the local web manager for AX Engine. By default it starts
a loopback-only web app at:

```text
http://127.0.0.1:8765
```

The manager uses the same contracts as `ax-engine-bench`, `ax-engine-server`,
and benchmark artifacts.

The manager can:

- run doctor and show model-artifact readiness;
- select a text model family and size, then run the existing guarded download
  helper for the resolved `mlx-community` repo id;
- start and stop an owned local `ax-engine-server` process for a selected model
  directory and port;
- show full local HTTP endpoint URLs;
- show benchmark artifact summaries and scan a results directory;
- project guarded job plans from the doctor workflow;
- export a redacted support bundle.

The manager does not copy model weights, environment variables, API keys, or raw
logs into support bundles.

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

Use `cargo run` while developing the manager:

```bash
cargo run -p ax-engine-tui --bin ax-engine-manager -- --no-open
```

## Quick Start

Open the local web manager:

```bash
ax-engine-manager
```

Use `--no-open` if you want the URL printed without opening the browser:

```bash
ax-engine-manager --no-open
```

The Models panel lets you choose a text model family and size from
browser-native dropdowns. Clicking `[Download]` runs:

```bash
python scripts/download_model.py <repo-id> --json
```

The helper uses `mlx-lm` for model acquisition, resolves the resulting cache
snapshot, and reports a destination path. The manager then refreshes doctor
state against that downloaded model directory.

`Qwen3-Coder-Next-4bit` is intentionally not exposed in the manager catalog
yet. AX Engine can run validated Qwen3 Next artifacts, but public cache
snapshots may fail the linear-attention sanitized-weight check at startup. If
you need that model, convert and validate the artifact manually with
`mlx_lm.convert` before using it outside the manager quick-pick flow.

## Start With A Server

If you already have a model directory:

```bash
MODEL_DIR="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/<hash>"
ax-engine-manager --model-dir "$MODEL_DIR"
```

In the web page, set the port and click `Start`. The manager starts:

```bash
ax-engine-server --mlx --mlx-model-artifacts-dir "$MODEL_DIR" --port 8080
```

The manager-owned launcher currently supports only the repo-owned AX Engine
runtime modes: `ax-engine n-gram` and `ax-engine direct`. `ax-engine n-gram`
is the default manager selection and leaves the server's default n-gram decode
acceleration enabled. `ax-engine direct` starts the same repo-owned MLX runtime
with `--disable-ngram-acceleration` for baseline/debug comparisons.
`mlx-lm` is a delegated backend that requires a separate `mlx_lm.server` plus
`ax-engine-server --support-tier mlx_lm_delegated --mlx-lm-server-url ...`.
`mlx-swift-lm` is treated as a benchmark reference adapter, not a
manager-startable server runtime.

Click `Stop` to kill the owned server process.

The Server panel lists:

- `/health`
- `/v1/runtime`
- `/v1/models`
- `/v1/generate`
- `/v1/generate/stream`
- `/v1/chat/completions`
- `/v1/completions`

## Existing JSON Contracts

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

## CLI Options

| Option | Purpose |
|---|---|
| `--check` | Print a script-friendly readiness summary and exit |
| `--model-dir <path>` | Run doctor against a model directory |
| `--server-url <url>` | Poll an existing local server for metadata |
| `--benchmark-json <path>` | Load one benchmark artifact summary |
| `--artifact-root <path>` | Scan benchmark result directories |
| `--support-bundle <dir>` | Write a redacted support bundle and exit |
| `--web-host <host>` | Bind the web manager host, default `127.0.0.1` |
| `--web-port <port>` | Bind the web manager port, default `8765` |
| `--no-open` | Print the URL without opening a browser |

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
