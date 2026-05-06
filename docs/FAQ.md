# FAQ

## Why does the repo-owned MLX runtime require M4 or newer?

The repo-owned MLX runtime is a supported performance contract, not only a
best-effort code path. AX fails closed on M3 and older Apple Silicon because
current runtime, benchmark, and support claims are scoped to macOS/aarch64 on
Apple M4 or newer hosts. `AX_ALLOW_UNSUPPORTED_HOST=1` is only for internal
development or CI bring-up; it does not make the host supported and should not
be used for published benchmark numbers.

Delegated routes are separate. A non-MLX or GGUF workflow can use `llama.cpp`,
and unsupported MLX text models can use an explicitly configured
`mlx_lm.server`, but those routes are compatibility contracts rather than
repo-owned MLX throughput claims.

## Why can N-gram acceleration have little or no effect?

N-gram acceleration improves effective throughput only when the generated token
stream has repeated local patterns that the draft table can predict and the
model accepts. It can have little effect when prompts and outputs are random,
high entropy, very short, or intentionally diverse. It can also back off when
acceptance falls below the runtime gate, which is expected behavior rather than
a failure.

Benchmark rows labeled with N-gram acceleration are effective-throughput
measurements. They should not be described as raw model-kernel speedups. Use
the MLX inference-stack harness with `--ax-compare-policies` to compare the
direct path and the N-gram path on the same prompt/decode shape.

## Which runtime path should I choose first?

Use the repo-owned MLX runtime when you have a supported Qwen/Gemma MLX artifact
and need AX-owned runtime behavior or performance evidence. Use
`mlx_lm_delegated` when upstream `mlx-lm` can serve an MLX text model that AX
does not yet own. Use `llama_cpp` for GGUF and non-MLX local inference.

If you are not sure, start with `docs/GETTING-STARTED.md` and then check
`docs/SUPPORTED-MODELS.md` before interpreting benchmark numbers.

## Does `mlx_lm_delegated` count as repo-owned MLX performance?

No. `mlx_lm_delegated` keeps AX server, SDK, and OpenAI-shaped surfaces while
delegating model execution to a user-provided `mlx_lm.server`. It is useful for
compatibility, but performance claims for the repo-owned MLX runtime must come
from the MLX inference-stack harness with a matching `mlx_lm.benchmark`
baseline.

## Should AX automatically scan the Hugging Face cache for model files?

No silent cache scan is part of the default path. Prefer
`--mlx-model-artifacts-dir` or `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR`. Hugging Face
cache discovery is opt-in through the server resolver and must find exactly one
AX-ready artifact directory with `config.json`, `model-manifest.json`, and
safetensors.
