# Supported Models

AX Engine routes inference through two supported shipping paths:

- repo-owned MLX mode for MLX inference
- `llama.cpp` bypass for non-MLX inference

## Support Strategy

AX does not treat model support as a simple yes-or-no list.

Instead, support should be understood through:

- support tier
- selected backend
- capability set

This matters because a future model may be:

- available through AX-owned MLX mode
- available through the `llama.cpp` bypass path
- not supported yet

The current default route is:

- explicit MLX mode routes to `mlx`
- all non-MLX inference routes to `llama.cpp`
- local `.gguf` model paths route to `llama.cpp`
- AX native mode is no longer a supported user-facing inference mode

The current routing decision is recorded in the private ADR set.

## MLX Platform Baseline

AX Engine MLX mode targets Apple Silicon Macs where the MLX runtime and AX's
MLX integration are available.

This means:

- MLX mode is the repo-owned local Mac inference path
- non-MLX local inference should use `llama.cpp`
- support-tier language should not be read as implying broad MLX Metal model support
- retired AX native mode should not be exposed as a shipping runtime
- model-inference benchmark claims for MLX mode must come from
  `scripts/bench_mlx_inference_stack.py` with a matching required
  `mlx_lm.benchmark` primary baseline and, optionally, an explicit
  `mlx-swift-lm` `BenchmarkHelpers` / `MLXLMCommon` secondary baseline adapter
- `ax-engine-bench` scenario, replay, matrix, compare, and autotune artifacts
  describe workload-contract evidence; delegated llama.cpp manifests describe
  non-MLX route-contract evidence only

## Support Tiers

### `mlx_certified`

Meaning:

- reserved for a future certified repo-owned runtime
- not currently used for AX MLX Metal mode
- must not be assigned without benchmark and correctness evidence

### `mlx_preview`

Meaning:

- repo-owned MLX mode is selected
- MLX runtime behavior is still preview-grade unless explicitly certified
- not a claim that AX MLX Metal mode is supported

### `llama_cpp`

Meaning:

- request is handled through `llama.cpp`
- this should not be read as equivalent to AX-owned MLX-mode feature coverage or
  performance
- `vLLM`, `mistral.rs`, and MLX adapters are not part of the
  current shipping inference route

### `unsupported`

Meaning:

- AX does not currently have a credible path for that model request

## Current Runtime Direction

The current runtime direction is:

- use `mlx` for explicit MLX mode
- use `llama_cpp` for all non-MLX inference
- promote MLX preview models only after reference-runtime comparison,
  correctness smoke coverage, and public benchmark artifacts are available

## Current AX MLX Preview Models

| Family | Model | Evidence |
|---|---|---|
| Gemma 4 | gemma-4-e2b-it, gemma-4-e4b-it, gemma-4-26b-a4b-it, gemma-4-31b-it | MLX stack benchmark + workload-contract scenario; E2B affine 4/5/6/8-bit, 26B A4B MoE, and 31B dense have MLX stack benchmark + server smoke |
| Qwen 3 | Qwen3-4B | MLX stack benchmark + workload-contract scenario |
| Qwen 3.5 | Qwen3.5-9B | MLX stack benchmark + workload-contract scenario |
| Qwen 3.6 | Qwen3.6-35B-A3B 4/5/6/8-bit MLX | MLX stack benchmark, server smoke, Qwen3.5-MoE manifest regression test |
| Qwen 3 Coder Next | Qwen3-Coder-Next-4bit | MLX stack benchmark, server smoke, Qwen3Next MoE/linear-attention regression tests |

## Current Limitations And Problems

No current public benchmark row is blocked for the AX MLX preview models above.
The Gemma 4 26B A4B MoE row intentionally omits `mlx_swift_lm` because the
local Swift reference does not currently implement Gemma4 MoE Router/Experts.
Gemma 4 E2B 5/6/8-bit rows include both `mlx_lm` and `mlx_swift_lm`
reference rows.
Speculative rows remain effective-throughput measurements from AX's n-gram
policy and must not be described as raw model-kernel speedups.

## Future Model Generations

If a newer model appears in the future, for example a later Qwen or Gemma
generation, AX should not force an all-or-nothing answer.

Depending on readiness, that model may resolve to:

- `mlx_preview`
- `llama_cpp`
- `unsupported`

where `mlx_preview` means repo-owned MLX mode, not AX MLX Metal mode.

## Not a Broad LlamaCpp Matrix

The v4 rewrite is not trying to support every architecture early.

Deferred from the main path:

- multimodal models
- hybrid-only architectures
- MoE-first support
- broad llama.cpp exceptions during core bring-up

## Important Note

This document describes product direction and support strategy, not a final
llama.cpp guarantee.
The implementation is still in progress, and support claims must be earned by
actual benchmark and validation evidence. For MLX support claims, that evidence
must name the MLX reference runtime, AX decode mode, model identity, prompt
shape, host readiness state, and whether the row came from the MLX inference
stack or from an `ax-engine-bench` workload-contract artifact.
