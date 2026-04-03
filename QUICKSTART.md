# Quickstart

This guide gets AX Engine from a clean checkout to a working local prompt on Apple Silicon.

## 1. Prerequisites

You need:

- a Mac with Apple Silicon M3 or newer
- macOS with Xcode installed
- Rust 1.88 or newer

Check the basics:

```bash
xcode-select -p
rustc --version
cargo --version
```

If `xcode-select -p` fails, install Xcode and open it once so the Metal toolchain is available.

## 2. Build the Workspace

From the repository root:

```bash
cargo build --workspace --release
```

For day-to-day iteration:

```bash
cargo check --workspace
```

The main CLI binary will be:

```text
./target/release/ax-engine
```

## 3. Get a GGUF Model

Place at least one GGUF model in `./models/`.

Example layout:

```text
models/
  Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
```

AX Engine expects a direct path to the GGUF file:

```bash
./target/release/ax-engine --model ./models/<model>.gguf --prompt "Hello"
```

## 4. First Successful Run

For an instruction-tuned model, start with `--chat`:

```bash
./target/release/ax-engine \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --chat \
  --prompt "Explain what AX Engine does in two sentences."
```

If the model is a plain base model instead of an instruct/chat model, omit `--chat`.

## 5. Interactive Chat

Run the REPL:

```bash
./target/release/ax-engine \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --interactive \
  --chat
```

Useful REPL commands:

- `/reset`
- `/clear`
- `/quit`
- `/exit`

## 6. Useful Sampling Controls

Example with explicit sampling settings:

```bash
./target/release/ax-engine \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --chat \
  --prompt "Write a short paragraph about on-device inference." \
  --temp 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  --min-p 0.05 \
  --min-keep 2 \
  --repeat-penalty 1.1 \
  --repeat-last-n 128
```

What these do:

- `--temp`: higher means more randomness
- `--top-k`: limit candidate tokens to the top K logits
- `--top-p`: nucleus filtering
- `--min-p`: drop very low-probability tail tokens relative to the best token
- `--min-keep`: keep at least this many candidates after filtering
- `--logit-bias TOKEN=BIAS`: add an explicit bias to a token before sampling
- `--allow-token-id TOKEN`: hard-restrict sampling to specific token IDs
- `--ban-token-id TOKEN`: hard-block specific token IDs
- `--repeat-penalty`: discourage repetition
- `--repeat-last-n`: limit repetition penalty to the last N tokens; `0` disables the repetition window, `-1` means full history

## 7. Stop Generation Cleanly

AX Engine can stop on specific output strings or token IDs without leaking the stop text:

```bash
./target/release/ax-engine \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --chat \
  --prompt "Write one line and then END." \
  --stop "END"
```

You can repeat the flags:

- `--stop "END" --stop "###"`
- `--stop-token-id 2`

`--stop` matches rendered output text. `--stop-token-id` stops before printing the matching token at all.

## 8. Inspect Top Logprobs

AX Engine can print top candidate logprobs for each emitted token:

```bash
./target/release/ax-engine \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --chat \
  --prompt "Write one sentence about Rust." \
  --top-logprobs 5
```

The generated text still streams normally. The logprob payload is printed afterward as structured JSON on stderr.

## 9. Verbose Mode

Use `--verbose` for metrics and decode mode details:

```bash
./target/release/ax-engine \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --prompt "Hello" \
  --verbose
```

This prints:

- prefill throughput
- decode throughput
- decode mode selection
- latency percentiles when available
- peak RSS

## 10. Speculative Decoding

Speculative decoding is available behind `--experimental` and requires a draft model:

```bash
./target/release/ax-engine \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --speculative-draft ./models/small-draft.gguf \
  --speculative-k 4 \
  --experimental \
  --prompt "Summarize why draft-model verification works."
```

Current limitation:

- `--top-logprobs` is not supported together with speculative decoding
- `--stop` and `--stop-token-id` are not supported together with speculative decoding
- `--allow-token-id` is not supported together with speculative decoding
- `--ban-token-id` is not supported together with speculative decoding
- `--logit-bias` is not supported together with speculative decoding

## 11. Benchmarking

Run the benchmark tool after a release build:

```bash
./target/release/ax-engine-bench bench --model ./models/<model>.gguf
./target/release/ax-engine-bench profile --model ./models/<model>.gguf
./target/release/ax-engine-bench soak --model ./models/<model>.gguf --smoke
```

Use these to compare models, inspect decode mode behavior, and catch stability regressions.

If you need machine-readable artifacts, the same commands support JSON output:

```bash
./target/release/ax-engine-bench bench \
  --model ./models/<model>.gguf \
  --json \
  --json-output /tmp/ax-engine-bench.json

./target/release/ax-engine-bench speculative \
  --model ./models/<target>.gguf \
  --draft-model ./models/<draft>.gguf \
  --json \
  --json-output /tmp/ax-spec.json
```

For experimental `Q5_K` prefill runs, these artifacts now also carry the
prefill route explicitly:

- `prefill_plan`
- `q5k_prefill_mode` when present

This applies to:

- `ax-engine-bench bench`
- `ax-engine-bench profile`
- `ax-engine-bench soak`

For apples-to-apples AX vs `llama.cpp` comparisons, repeated-run guidance, and reporting rules, see [BENCHMARKING.md](./BENCHMARKING.md).

## 12. Troubleshooting

### Build fails because Metal tooling is missing

Check:

```bash
xcode-select -p
```

If needed, install Xcode and open it once.

### Bench or CLI shows `PrefillPlan: mode=serial reason=unsupported_quant:...`

This means AX found an active layer tensor that does not yet have a GPU
batch/prefill kernel path, so prefill fell back to the serial path on purpose.

Example:

```text
PrefillPlan: mode=serial reason=unsupported_quant:<tensor>:<dtype>
```

Interpretation:

- prefill is not using the normal GPU fast path for this model
- the tensor and dtype after `unsupported_quant:` tell you exactly what blocked it

Mixed-quant models with active `Q5_K` layer tensors no longer use this fallback by default.
Those models should now report a conservative GPU prefill route instead, for example:

```text
Support: Mixed-quant Q5_K layers use AX's conservative GPU prefill route...
PrefillPlan: mode=gpu_batch ... q5k_prefill=base
```

That route is conservative and not profile-tuned yet.

Current `Q5_K` auto-routing behavior:

- AX only auto-selects the small-`N` `Q5_K` prefill route when the model is
  predominantly `Q5_K` and the prompt batch is small (`<= 32` tokens)
- mixed-quant files can still stay on the base route

If you are doing validation A/B runs, you can force the route:

```bash
AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT=base \
./target/release/ax-engine-bench bench --model ./models/<model>.gguf

AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT=small \
./target/release/ax-engine-bench bench --model ./models/<model>.gguf
```

The variant override is for validation only. It is not a recommended permanent
user tuning surface.

### The model runs badly or incorrectly on M1 or M2

That is outside the supported target range. AX Engine is tuned for M3 and newer.

### An instruction model gives poor output

Try `--chat`. Many instruct models expect a model-family prompt wrapper.

### No room left to generate

Increase the context size:

```bash
./target/release/ax-engine \
  --model ./models/<model>.gguf \
  --ctx-size 8192 \
  --prompt "..."
```

### You want to see exactly which flags exist

Run:

```bash
./target/release/ax-engine --help
```

## 13. Next Steps

After the first successful run, the usual next checks are:

1. `cargo test --workspace`
2. `cargo clippy --workspace --tests -- -D warnings`
3. `./target/release/ax-engine-bench bench --model ./models/<model>.gguf`
