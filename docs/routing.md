# Inference Routing

AX Engine now supports deterministic inference routing:

- use native `ax-engine` inference when the model is natively supported
- fall back to `llama-server` when the model is unsupported or explicitly routed

This routing behavior is shared by:

- `ax-engine` CLI
- `ax-engine-sdk`
- `ax-engine-server`
- `ax-engine-py`

## Why

The main goal is coverage without hiding what happened.

- native AX remains the preferred path for supported architectures
- unsupported GGUF models no longer need to hard-fail if `llama.cpp` is available
- routing is transparent, deterministic, and overrideable

## Support Detection

Before model construction, AX inspects the GGUF metadata and classifies native support as one of:

- `Full`
- `PartialQuant`
- `UnsupportedArch`

`AX_ROUTING=auto` uses that result to decide whether to stay native or fall back to `llama.cpp`.

## Routing Priority

Routing resolves in this order:

1. per-model override
2. per-architecture override
3. global default

## Environment Variables

Global default:

```bash
AX_ROUTING=auto
```

Supported values:

- `off`: native only; unsupported models fail
- `native`: force native path; unsupported models fail
- `llama_cpp`: force `llama.cpp`
- `auto`: use native when fully supported, otherwise fall back

Per-architecture overrides:

```bash
AX_ROUTING_ARCH="mistral=llama_cpp,deepseek=llama_cpp,qwen3=native"
```

Per-model overrides:

```bash
AX_ROUTING_MODEL="/absolute/path/to/Mistral-7B.gguf=llama_cpp"
```

`llama-server` location and startup timeout:

```bash
AX_LLAMA_SERVER_PATH=/opt/homebrew/bin/llama-server
AX_LLAMA_SERVER_TIMEOUT=120
```

`AX_LLAMA_SERVER_TIMEOUT` is in seconds.

## CLI Behavior

The CLI now prints a one-line routing summary before inference, for example:

```text
Routing: native | arch=qwen3 | source=global
Routing: llama_cpp | arch=mistral | source=global | reason=architecture 'mistral' is not natively supported by ax-engine
```

Example:

```bash
AX_ROUTING=auto \
./target/release/ax-engine \
  --model ./models/Mistral-7B-Instruct.Q4_K_M.gguf \
  --prompt "Explain unified memory."
```

When routed to `llama.cpp`, the CLI supports the basic generation path and interactive mode. Some native-only options remain unavailable on the routed path, including:

- `--reuse-bench`
- `--reuse-bench-json`
- `--speculative-draft`
- `--top-logprobs`
- `--allow-token-id`
- `--ban-token-id`
- `--logit-bias`
- `--stop-token-id`
- `--min-keep`

## Rust SDK

Use `preview_routing(...)` when you want to inspect the decision before loading:

```rust
use ax_engine_sdk::preview_routing;

let preview = preview_routing("./models/model.gguf")?;
println!("backend={}", preview.backend.as_str());
```

`Model::info()` now reports:

- `backend`
- `routing`

So callers can see whether a loaded model is native or routed.

## Server Behavior

`ax-engine-server` inherits the same routing logic through `ax-engine-sdk`.

- `GET /healthz` now includes `backend` and optional `routing`
- `GET /v1/models` now includes `backend` and optional `routing`

This makes it easy for external software to verify whether it is currently talking to native AX or a routed `llama.cpp` subprocess.

## Python Behavior

`ax-engine-py` loads through the same routed SDK path. Python callers can inspect:

- `Model.backend`
- `Model.routing`

## llama.cpp Requirement

Fallback requires `llama-server`.

Resolution order:

1. `AX_LLAMA_SERVER_PATH`
2. `llama-server` on `PATH`
3. `/opt/homebrew/bin/llama-server`
4. `/usr/local/bin/llama-server`

If it is not found, AX returns an error with the install hint:

```text
llama-server not found. Install via: brew install llama.cpp
```
