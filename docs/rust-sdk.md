# Rust SDK

`ax-engine-sdk` is the high-level Rust integration facade for AX Engine.

Use it when you want:

- a stable Rust API instead of depending directly on `ax-engine-core` internals
- model loading, tokenization, sessions, chat rendering, and streaming through one crate
- a foundation for higher-level integrations such as Python bindings

## Scope

`ax-engine-core` remains the runtime and internal engine crate.

`ax-engine-sdk` is the intended Rust application-facing surface.

## Current Surface

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
use ax_engine_sdk::{GenerationOptions, LoadOptions, Model, SessionOptions};

fn main() -> anyhow::Result<()> {
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

## Relationship to Python

`ax-engine-py` builds on `ax-engine-sdk` instead of re-implementing model
load and session glue directly over `ax-engine-core`.
