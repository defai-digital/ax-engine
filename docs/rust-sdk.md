# Rust SDK

`ax-engine-sdk` is the high-level Rust integration facade for AX Engine.

Use it when you want:

- a stable Rust API instead of depending directly on `ax-engine-core` internals
- model loading, tokenization, sessions, chat rendering, and streaming through one crate
- a better foundation for higher-level integrations such as Python bindings and `ax-engine-server`

## Scope

`ax-engine-core` remains the runtime and internal engine crate.

`ax-engine-sdk` is the intended Rust application-facing surface.

It also owns inference routing, so the same SDK surface can load a model
either natively or through `llama.cpp` fallback when routing is enabled.

## Current Surface

- `preview_routing(...)`
- `Model::load(...)`
- `Model::info()`
- `Model::tokenize(...)`
- `Model::decode(...)`
- `Model::session(...)`
- `Session::generate(...)`
- `Session::stream(...)`
- `Session::chat(...)`
- `Session::stream_chat(...)`
- `Session::reset()`

## Example

```rust
use ax_engine_sdk::{
    GenerationOptions, LoadOptions, Model, SessionOptions, preview_routing,
};

fn main() -> anyhow::Result<()> {
    let preview = preview_routing("./models/Qwen3-8B-Q4_K_M.gguf")?;
    println!("backend={}", preview.backend.as_str());

    let model = Model::load(
        "./models/Qwen3-8B-Q4_K_M.gguf",
        LoadOptions::default(),
    )?;

    let session = model.session(SessionOptions::default())?;
    let output = session.generate(
        "Explain KV cache in one short sentence.",
        GenerationOptions::default(),
    )?;

    println!("{}", output.text);
    Ok(())
}
```

`Model::info()` now includes `backend` and optional `routing` metadata so
callers can inspect whether the model is running natively or through
`llama.cpp`.

## Relationship to Python

`ax-engine-py` now builds on `ax-engine-sdk` instead of re-implementing model
load and session glue directly over `ax-engine-core`.

## Relationship to Server

`ax-engine-server` now also builds on `ax-engine-sdk`, so the repo's basic HTTP
surface, Rust integration surface, and Python binding share the same
application-facing model/session path.
