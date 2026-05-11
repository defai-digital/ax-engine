# Mojo SDK

`sdk/mojo/ax_engine.mojo` is the Mojo SDK for AX Engine v4.

It delegates to the Python `ax_engine` package via Mojo's `PythonObject`
interop. This keeps the Mojo surface thin while reusing the full Python SDK
contract, including backend resolution, in-process session management, and
all inference routes.

## Prerequisites

1. Install the Mojo toolchain (Magic package manager or manual install)
2. Build the Python extension: `maturin develop`
3. Ensure `ax_engine` is on `PYTHONPATH` (the `maturin develop` step handles this)

## Current Scope

- `Session` struct wrapping `ax_engine.Session`
  - `generate(input_text, max_output_tokens)` — blocking text generation
  - `generate_tokens(input_tokens, max_output_tokens)` — pre-tokenized input
  - `close()` — releases the underlying Python session
- `GenerateResult` struct with `output_text`, `output_tokens`, `finish_reason`
- `download_model(repo_id, dest, force)` — download an mlx-community model

## Install

From the repository root:

```bash
maturin develop          # build and install the Python extension
```

To reference the Mojo SDK from your project, add the repo root to your Mojo
source path:

```bash
magic run mojo -I /path/to/ax-engine-v4 your_script.mojo
```

## Example

```mojo
from sdk.mojo.ax_engine import Session, download_model

fn main() raises:
    # Download model if not already present
    # var model_dir = download_model("mlx-community/Qwen3-4B-4bit")

    var session = Session(
        "qwen3_dense",
        mlx=True,
        mlx_model_artifacts_dir="/path/to/mlx-model-artifacts",
    )

    var result = session.generate("Hello from Mojo!", max_output_tokens=64)
    print(result.output_text)

    session.close()
```

Run the checked-in example:

```bash
magic run mojo -I . examples/mojo/basic.mojo
```

## All Session Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `String` | Session label (required) |
| `mlx` | `Bool` | Enable repo-owned MLX runtime |
| `mlx_model_artifacts_dir` | `String` | Path to MLX model weights directory |
| `llama_model_path` | `String` | Path to a local GGUF model |
| `llama_server_url` | `String` | URL of a running llama.cpp server |
| `mlx_lm_server_url` | `String` | URL of a running mlx_lm.server |
| `support_tier` | `String` | e.g. `"mlx_lm_delegated"` for explicit delegation |

Validation and error handling follow the Python SDK contract: missing required
configuration raises a `ValueError` before touching the Rust extension.

## Downloading a Model

```mojo
from sdk.mojo.ax_engine import download_model

fn main() raises:
    var path = download_model("mlx-community/Qwen3-4B-4bit")
    print(path)
```

Requires `pip install mlx-lm`. This helper is for LLM models only; embedding
model artifacts must be downloaded manually. The returned string is the local
directory path, ready to pass to `Session(mlx_model_artifacts_dir=path)`.

## Design Notes

The Mojo SDK is a thin interop wrapper, not a reimplementation. The
`ax_engine.Session` Python object manages all backend resolution, lifecycle,
and inference logic. Mojo provides ergonomic typed structs at call sites while
the Python extension does the actual work.

For full session control (streaming, stepwise lifecycle, chat API), drop down
to the Python interop directly:

```mojo
from python import Python

fn main() raises:
    var ax = Python.import_module("ax_engine")
    with ax.Session(model_id="qwen3_dense", mlx=True,
                    mlx_model_artifacts_dir="/path/to/model") as session:
        for event in session.stream_generate([1, 2, 3], max_output_tokens=32):
            print(event.event, event.delta_text)
```
