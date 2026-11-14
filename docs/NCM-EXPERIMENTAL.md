# Experimental native NCM inference

AX provides a separate native Rust/MLX unconditional TinyStories CLI and Rust
SDK. The model selects its initial concept from the pinned unigram, predicts
11 further levels and decodes 128 raw BPE positions to text. The first mode
is deterministic greedy; repeated requests produce the same story.

The supported research artifact is
`janmusic/NCM-TinyStories-89M-Upstream-MLX` at
`2d5f137713fe047abb6fcffe780aad419486f3dd`. Its F32 generator, codec,
inference distribution, byte tokenizer, spec, config and license are checked
against exact size/hash pins. Quantized and other NCM checkpoints are rejected.

## Run locally

Use macOS 26+ on M2 or later Apple Silicon with the repository's pinned MLX
and Rust versions. The SDK applies the existing AX host validation policy.
Build the experimental binary:

```bash
rustup run 1.99.1 cargo build -p ax-engine-sdk --features mlx-native --bin ncm-infer
./target/debug/ncm-infer "$MODEL_DIR" --sampling greedy
./target/debug/ncm-infer "$MODEL_DIR" --sampling greedy --json
```

`MODEL_DIR` contains the seven pinned files: `LICENSE`, `config.json`,
`spec.json`, `generator.safetensors`, `codec.safetensors`,
`inference.safetensors` and `bpe.json`. Documentation files are allowed;
unlisted weights, code, configs and directories are rejected. Supported Hub
snapshot links must satisfy AX's existing artifact-path policy. If a shared
Hub blob-store layout is rejected, materialize regular copies of those seven
files into a separate directory. Preserve their bytes; no conversion is needed.
Do not place generated reports inside the model directory.

The optional `--memory-budget-mib` defaults to 2048 and limits conservative
per-model NCM load reservation. This includes resident weights, runtime
headroom and reserved request scratch capacity. Aggregate NCM owners also
respect the physical-memory ceiling after OS headroom. It is admission accounting, not a hard MLX
allocator limit. `--timeout-ms` defaults to 60000 and must be positive and no
more than one hour. Cancellation/deadline checks occur between bounded work
units; already submitted GPU commands are allowed to finish.

## Rust SDK

Enable the SDK's `mlx-native` feature:

```rust,ignore
use std::{path::Path, time::Duration};
use ax_engine_sdk::ncm::{
    NcmGenerateRequest, NcmOutputMode, NcmSamplingMode, NcmSession,
};

let session = NcmSession::load_experimental(
    Path::new("/path/to/model"),
    2 * 1024 * 1024 * 1024,
    Duration::from_secs(60),
)?;
let request = NcmGenerateRequest {
    sampling: NcmSamplingMode::Greedy,
    output_mode: NcmOutputMode::FullSequence128,
};
let response = session.generate_unconditional(request)?.finish()?;
print!("{}", response.text);
```

The session loads once and admits one active generation with no waiting queue.
A second generation returns `Busy`. For progress/cancellation, retain the
handle, call `step()`, or obtain its `cancellation()` token. Text is returned
only after final commit. Dropping an unfinished handle cancels it; closing
or dropping the session cancels pending work and releases request state at a
GPU-safe boundary. The same loaded session can be used for later requests.

JSON requests require `sampling: "greedy"`; only `output_mode` may default
(to `full_sequence128`). Unknown fields are rejected, including default-valued
seed, temperature, top-p/top-k, numeric output budgets, prompt and streaming.
The raw output has 128 BPE positions. Presentation stops before EOS 0, skips
PAD 1, joins byte fragments before UTF-8 decoding and preserves whitespace.
The response distinguishes raw IDs from visible IDs, permits empty text and
reports zero user prompt tokens. The fixed internal prefix is not a user prompt.

## Supported scope

This is experimental unconditional inference, separate from ordinary
`ax-engine serve` and generic SDK `GenerateRequest`. No chat, arbitrary prompt,
seeded sampling, tools, grammar, stop strings, logprobs, speculative decoding,
prefix cache or partial text streaming is admitted. HTTP model registration,
official-source parity and M5/M4 admission evidence remain pending. M2 Ultra
functional controls do not establish M5 qualification, quality or speed.

## Reproduce the functional controls

From the repository root, with the pinned files already present locally:

```bash
AX_NCM_TEST_MODEL="$MODEL_DIR" \
AX_NCM_EXPECTED_TRACE="$PWD/benchmarks/ncm-native-experimental-20261114/reference/greedy-0.json" \
rustup run 1.99.1 cargo test -p ax-engine-sdk --features mlx-native --test ncm_native -- --ignored
```

The committed compact references contain expected text and BPE IDs only.
This opt-in test does not download weights or run an external inference
backend. Ordinary CI keeps the real-weight control ignored.
