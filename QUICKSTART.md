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
./target/release/ax-llama
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
./target/release/ax-llama --model ./models/<model>.gguf --prompt "Hello"
```

## 4. First Successful Run

For an instruction-tuned model, start with `--chat`:

```bash
./target/release/ax-llama \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --chat \
  --prompt "Explain what AX Engine does in two sentences."
```

If the model is a plain base model instead of an instruct/chat model, omit `--chat`.

## 5. Interactive Chat

Run the REPL:

```bash
./target/release/ax-llama \
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
./target/release/ax-llama \
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
./target/release/ax-llama \
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
./target/release/ax-llama \
  --model ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --chat \
  --prompt "Write one sentence about Rust." \
  --top-logprobs 5
```

The generated text still streams normally. The logprob payload is printed afterward as structured JSON on stderr.

## 9. Verbose Mode

Use `--verbose` for metrics and decode mode details:

```bash
./target/release/ax-llama \
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
./target/release/ax-llama \
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
./target/release/ax-bench bench --model ./models/<model>.gguf
./target/release/ax-bench profile --model ./models/<model>.gguf
./target/release/ax-bench soak --model ./models/<model>.gguf --smoke
```

Use these to compare models, inspect decode mode behavior, and catch stability regressions.

If you need machine-readable artifacts, the same commands support JSON output:

```bash
./target/release/ax-bench bench \
  --model ./models/<model>.gguf \
  --json \
  --json-output /tmp/ax-bench.json

./target/release/ax-bench speculative \
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

- `ax-bench bench`
- `ax-bench profile`
- `ax-bench soak`

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
PrefillPlan: mode=serial reason=unsupported_quant:blk.10.attn_v.weight:Q5K
Support: Q5_K support defaults to decode-only baseline...
```

Interpretation:

- decode can still use the landed `Q5_K` GPU path
- prefill is not using the normal GPU fast path for this model
- the tensor and dtype after `unsupported_quant:` tell you exactly what blocked it

Today this is expected for mixed-quant models with active `Q5_K` layer tensors,
because `Q5_K` support defaults to decode-only baseline, not full GPU prefill support.

If you are deliberately testing the experimental AX-native path, set:

```bash
AX_METAL_EXPERIMENTAL_Q5K_PREFILL=1
```

That route is conservative, opt-in, and not profile-tuned yet.

Current experimental auto-routing behavior:

- AX only auto-selects the small-`N` `Q5_K` prefill route when the model is
  predominantly `Q5_K` and the prompt batch is small (`<= 32` tokens)
- mixed-quant files can still stay on the base experimental route even when the
  env var is set

If you are doing validation A/B runs, you can force the route:

```bash
AX_METAL_EXPERIMENTAL_Q5K_PREFILL=1 \
AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT=base \
./target/release/ax-bench bench --model ./models/<model>.gguf

AX_METAL_EXPERIMENTAL_Q5K_PREFILL=1 \
AX_METAL_EXPERIMENTAL_Q5K_PREFILL_VARIANT=small \
./target/release/ax-bench bench --model ./models/<model>.gguf
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
./target/release/ax-llama \
  --model ./models/<model>.gguf \
  --ctx-size 8192 \
  --prompt "..."
```

### You want to see exactly which flags exist

Run:

```bash
./target/release/ax-llama --help
```

## 13. Next Steps

After the first successful run, the usual next checks are:

1. `cargo test --workspace`
2. `cargo clippy --workspace --tests -- -D warnings`
3. `./target/release/ax-bench bench --model ./models/<model>.gguf`
