# FAQ

## What hardware does AX Engine support?

AX Engine targets high-memory Apple Silicon Macs running macOS 26 (Tahoe) or
later.

| Machine | Minimum spec | Suggested spec |
| --- | --- | --- |
| Mac Mini | M4 Pro, 32 GB | M4 Pro, 64 GB |
| MacBook Pro 14" / 16" | M2 Max, 32 GB | M3 Max, 96 GB |
| Mac Studio | M2 Max / M2 Ultra, 32 GB | M4 Max, 96 GB |

Later Pro, Max, and Ultra chip variants are supported when the machine has at
least 32 GB unified memory. M1, base M2, and smaller-memory machines are outside
the supported repo-owned MLX runtime contract.

For a typical local model stack, start with one of these higher-memory
configurations:

| Hardware | Recommended memory | Best fit |
| --- | ---: | --- |
| Mac mini M4 Pro | 64 GB RAM | Compact always-on local chatbot and agent server |
| MacBook Pro M5 Max | 128 GB RAM | Portable high-throughput chatbot, agent, and coding stack |
| Mac Studio M3 Ultra | 256 GB RAM | Larger local model portfolio, longer contexts, and heavier parallel workloads |

## What model stack should I run on high-memory Apple Silicon?

Use a three-model stack when memory allows it. One model can serve every task,
but the better local setup is to match the model to the workflow.

| Role | Recommended model | Setup | App | Why |
| --- | --- | --- | --- | --- |
| Default chatbot | Gemma 4 26B-A4B / 31B | 4-bit or 6-bit, 16K-32K | [ax-studio](https://github.com/defai-digital/ax-studio) | General assistant path for reasoning, chat, JSON/function calling, and on-device agent workflows |
| General agentic model | Qwen3.6-35B-A3B / Qwen3.6-27B | 35B A3B 4-bit; 27B 4/5/6-bit, 16K-32K | AX server / SDK | Strong general agent and coding balance; sparse MoE keeps active compute low |
| Coding specialist | Qwen3-Coder-Next | 6-bit + 16K default; 4-bit/5-bit + 32K when needed | [ax-code](https://github.com/defai-digital/ax-code) | Dedicated local coding-agent path for repo editing, tool use, and long coding sessions |
| Embedding / RAG ingest | Qwen3-Embedding or EmbeddingGemma | 0.6B / 4B / 8B (Qwen3); 300M (EmbeddingGemma) | AX server `/v1/embeddings` | Sustained ingest-scale throughput; AX last-token pooling (Qwen3) or mean pooling + Dense head (EmbeddingGemma) |

Suggested default stack (primary productivity):

```text
Chatbot:
Gemma 4 31B 4-bit/6-bit + 16K

General agent:
Qwen3.6-35B-A3B 4-bit + 32K

Coding agent:
Qwen3-Coder-Next 6-bit + 16K
```

Optional secondary stacks (same 128 GB class; one large model at a time):

```text
Research / enterprise Llama:
ax-engine download llama3.3-70b   # or llama3.1-8b for smoke/draft
ax-engine download llama4-scout  # next-gen MoE when memory allows

European market:
ax-engine download mistral-small
ax-engine download ministral-8b
ax-engine download devstral-small

Open reasoner:
ax-engine download gpt-oss-20b
ax-engine download gpt-oss-120b  # prefer 128 GB+; experts stay MXFP4-packed
```

| Use case | Best pick |
| --- | --- |
| Daily chatbot | Gemma 4 |
| Business/technical assistant | Qwen3.6-35B-A3B |
| Repo editing / coding agent | Qwen3-Coder-Next |
| Fast lightweight coding | Qwen3.6-27B when available through a direct or delegated route |
| Long-context coding | Qwen3-Coder-Next 4-bit + 32K |
| Research / Llama baseline | Llama 3.3 70B 4-bit (or 3.1 8B) |
| EU enterprise chat | Mistral Small 24B 4-bit |
| EU coding | Devstral Small 4-bit |
| Open reasoner | GPT-OSS 20B MXFP4-Q4 (120B when memory allows) |

Client positioning:

- **Safe / high-ROI path**: Gemma 4 in
  [ax-studio](https://github.com/defai-digital/ax-studio) for chatbot +
  Qwen3.6 for agent tasks.
- **Higher-return path**: add Qwen3-Coder-Next as the dedicated local coding
  specialist in [ax-code](https://github.com/defai-digital/ax-code).
- **Research / EU / open reasoner**: Llama, Mistral, and GPT-OSS download
  aliases are first-class preview direct paths; keep Qwen/Gemma as the
  performance hero stack.

Model references: [Gemma 4](https://deepmind.google/models/gemma/gemma-4/),
[Qwen3.6-35B-A3B](https://qwen.ai/blog?id=qwen3.6-35b-a3b),
[Qwen3-Coder-Next](https://unsloth.ai/docs/models/qwen3-coder-next), and
[SUPPORTED-MODELS](SUPPORTED-MODELS.md).

## Where is the serving roadmap?

AX Engine v6 is the current serving-oriented runtime line. The active serving
roadmap and evidence gates live in
[`docs/ROADMAP.md`](ROADMAP.md).

## Why does the repo-owned MLX runtime require M2 Max or newer?

The repo-owned MLX runtime is a supported performance contract, not only a
best-effort code path. AX fails closed on M1 and base M2 Apple Silicon because
current runtime, benchmark, and support claims are scoped to macOS 26
(Tahoe) or later on Apple M2 Max or newer hosts with 32 GB RAM minimum.
`AX_ALLOW_UNSUPPORTED_HOST=1` is only for internal development or CI bring-up;
it does not make the host supported and should not be used for published
benchmark numbers.

Delegated routes are separate. A non-MLX or GGUF workflow can use `llama.cpp`,
and unsupported MLX text models can use an explicitly configured
`mlx_lm.server`, but those routes are compatibility contracts rather than
repo-owned MLX throughput claims.

## Is AX faster because it replaces MLX kernels?

No. The repo-owned MLX path uses MLX directly for tensor operations through the
repo-owned `ax_shim` C ABI over MLX C++. Matrix multiply, quantized matmul, attention, RMSNorm,
and RoPE go through MLX's Apple-maintained Metal kernels. AX owns the runtime
behavior above that graph.

AX improves the runtime behavior above MLX: how tokens are speculated, how
requests are scheduled, how KV state is materialized, and how benchmark evidence
is recorded. That runtime layer is what produces higher effective throughput on
supported workloads.

Important pieces:

- **N-gram acceleration**: a bigram/trigram table built at runtime predicts up
  to 4 draft tokens per step. The target model verifies them in one forward pass
  over `[last_token, D1, ..., D_n]`. No second draft model is required.
- **Scheduler and KV manager**: request lifecycle, batching, memory-blocked
  recovery, and execution planning live in `ax-engine-core`.
- **Chunked KV cache**: keys and values grow in pre-allocated backing buffers,
  and draft rollback only moves the sequence-length pointer.
- **Graph compilation**: `mlx_enable_compile()` is called once at startup so
  Metal shader compilation and dispatch tables are reused across matching
  shapes.
- **Memory policy**: `mlx_set_wired_limit(recommendedMaxWorkingSetSize)` wires
  model weights into GPU memory at startup, reducing paging between requests.

See `docs/SCHEDULER.md`, `docs/KV-CACHE.md`, and `docs/LONG-CONTEXT.md` for
the deeper design contract.

## Why can N-gram acceleration have little or no effect?

N-gram acceleration improves effective throughput only when the generated token
stream has repeated local patterns that the draft table can predict and the
model accepts. It can have little effect when prompts and outputs are random,
high entropy, very short, or intentionally diverse. It can also back off when
acceptance falls below the runtime gate, which is expected behavior rather than
a failure.

Coding workflows are the clearest user-facing fit when they contain repeated
local structure. Code completion, edit application, structured diffs, imports,
indentation, repeated identifiers, boilerplate, JSON/tool output, and test or
config generation often reuse local token patterns. In those cases, n-gram
acceleration can draft short spans that the target model verifies in one forward
pass, improving effective decode throughput without a second draft model.

That is not a promise that every coding request speeds up. Novel algorithms,
open-ended explanations, high-entropy generated text, short answers, or outputs
with low draft acceptance can see little benefit. AX treats n-gram acceleration
as opportunistic: when the draft table stops helping, the gate backs off so the
request behaves closer to the direct decode path.

Benchmark rows labeled with N-gram acceleration are effective-throughput
measurements. They should not be described as raw model-kernel speedups. Use
the MLX inference-stack harness with `--ax-compare-policies` to compare the
direct path and the N-gram path on the same prompt/decode shape.

## What are the current limitations?

- **GatedDelta prefill (Qwen3.5 / Qwen 3.6)**: The recurrent state update in
  GatedDelta linear-attention layers serializes over time steps and cannot be
  parallelized. AX now uses a 2048-token long-prefill specialization to match
  `mlx_lm` chunk geometry; remaining gaps should be investigated with
  `AX_MLX_LINEAR_ATTENTION_PROFILE=1` before changing runtime defaults.
- **Raw HuggingFace weights**: ax-engine loads MLX community pre-sanitized
  weights. For hybrid architectures such as Qwen3.5 and Qwen3-Next, loading an
  unsanitized checkpoint raises a hard error at load time. Convert first with
  `mlx_lm.convert`, or download a pre-sanitized model from mlx-community.
- **N-gram acceleration rows**: effective-throughput measurements, not raw
  model-kernel speedups. The hit rate is prompt- and output-pattern dependent.
  Coding-shaped workloads with repeated local structure are the intended
  high-value case; random, high-entropy, very short, or deliberately diverse
  outputs may see little benefit.
- **KV compression**: the experimental compressed-KV runtime path was retired
  in favor of the durable tiered prefix cache (ADR-002). Native uncompressed
  KV is the only decode behavior.

## Which runtime path should I choose first?

Use the repo-owned MLX runtime first when you have a supported Qwen/Gemma/GLM
MLX artifact and need AX-owned runtime behavior or performance evidence. Use
`mlx_lm_delegated` or `llama_cpp` only when you explicitly want a compatibility
adapter or external reference path; unsupported model families fail closed by
default.

If you are not sure, start with `docs/GETTING-STARTED.md` and then check
`docs/SUPPORTED-MODELS.md` and `docs/MODEL-SUPPORT-POLICY.md` before
interpreting benchmark numbers.

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
