# OpenClaw

AX Engine supports OpenClaw through both OpenAI Chat Completions and the native
Ollama chat transport. The OpenAI-compatible route is the recommended baseline;
use the Ollama route when native Ollama discovery is important to the
installation.

This contract was checked against OpenClaw
[`d4ed57bc`](https://github.com/openclaw/openclaw/commit/d4ed57bc8847b2b0a40dc99189bd4d94acbb2fc7)
from 2026-07-27.

## Supported Qwen Targets

| AX target | Chat and tools | Thinking | Image input | OpenClaw role |
| --- | --- | --- | --- | --- |
| Qwen3.5 9B | Yes | On/off | Yes when the loaded manifest advertises it | Lower-memory local agent |
| Qwen3.6 27B | Yes | On/off | Yes when the loaded manifest advertises it | Strong dense general agent |
| Qwen3.6 35B-A3B | Yes | On/off | Yes when the loaded manifest advertises it | Larger MoE general agent |
| Qwen3-VL Instruct | Yes | No | Yes | Dedicated vision agent |
| Qwen3-VL Thinking | Yes | On/off | Yes | Vision plus reasoning |

There is no Qwen3.6 25B target in the AX catalog. Use Qwen3.6 27B or the
35B-A3B target; do not put a made-up 25B model ID in OpenClaw.

Image support is checkpoint-authoritative. AX reports
`capabilities.input.image=true` only when the loaded native manifest contains
the required vision tower. A text-only conversion of the same family remains a
text model.

## Start AX With Honest Limits

AX defaults to a 16,384-token context and a 2,048-token per-request output
budget. That meets OpenClaw's 16K guided-setup minimum, but a 32K context and an
8K output budget are more practical for tool-heavy agent sessions:

```bash
ax-engine serve ax-qwen3.6-27b --download --port 31418 -- \
  --total-blocks 2048 \
  --max-batch-tokens 8192
```

Equivalent managed aliases are `ax-qwen3.5-9b` and `ax-qwen3.6-35b`.
Qwen3-VL can be served from a prepared local model directory:

```bash
ax-engine serve /absolute/path/to/Qwen3-VL-8B-Instruct --port 31418 -- \
  --total-blocks 2048 \
  --max-batch-tokens 8192
```

Increasing these values consumes more memory. The source of truth is the
running server, not the checkpoint's theoretical maximum:

```bash
curl -s http://127.0.0.1:31418/v1/models |
  jq '.data[] | {
    id,
    context_length,
    max_output_tokens,
    capabilities
  }'
```

Copy the returned `id`, `context_length`, and `max_output_tokens` into
OpenClaw. Never configure OpenClaw above those values. Native Ollama requests
whose `options.num_ctx` exceeds the AX session are rejected with
`context_length_exceeded` instead of being silently truncated.

## Recommended OpenAI Configuration

Use `api: "openai-completions"`, not `openai-responses`. AX supports the
stateless Responses subset, but OpenClaw's local agent path depends on the
streaming Chat Completions tool contract.

The following example matches the 32K/8K server command above. Replace the
model ID and limits with the values returned by `/v1/models`:

```json5
{
  models: {
    mode: "merge",
    providers: {
      ax: {
        baseUrl: "http://127.0.0.1:31418/v1",
        apiKey: "ax-local",
        api: "openai-completions",
        timeoutSeconds: 300,
        models: [
          {
            id: "qwen3.6-27b-mtp",
            name: "AX Qwen3.6 27B",
            reasoning: true,
            input: ["text", "image"],
            cost: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
            },
            contextWindow: 32768,
            maxTokens: 8192,
            compat: {
              thinkingFormat: "qwen-chat-template",
              supportsStore: false,
              supportsDeveloperRole: false,
              supportsUsageInStreaming: true,
              supportsStrictMode: false,
              maxTokensField: "max_completion_tokens",
            },
          },
        ],
      },
    },
  },
  agents: {
    defaults: {
      model: { primary: "ax/qwen3.6-27b-mtp" },
    },
  },
}
```

Use the same reasoning configuration for Qwen3.5 9B, Qwen3.6 35B-A3B, and
Qwen3-VL Thinking. For Qwen3-VL Instruct, set `reasoning: false` and omit
`thinkingFormat`. Change `input` to `["text"]` whenever
`/v1/models` reports `capabilities.input.image=false`.

With `thinkingFormat: "qwen-chat-template"`, OpenClaw sends
`chat_template_kwargs.enable_thinking` and
`chat_template_kwargs.preserve_thinking`. AX uses those values to select the
Qwen generation prompt, return `reasoning_content`, and replay prior reasoning
without mixing it into visible assistant content.

## Native Ollama Configuration

AX also satisfies OpenClaw's current native Ollama contract:

- `/api/tags`, `/api/show`, `/api/ps`, and `/api/version` discovery;
- `/api/show` requests using `{ "name": "<model>" }`;
- NDJSON `/api/chat` streaming;
- `think`, `options.num_ctx`, and the common sampling options;
- function tools, assistant tool-call IDs, and `tool_name` on results;
- raw base64 `messages[].images`; and
- `completion`, `tools`, `vision`, and `thinking` capabilities when supported.

The native base URL must not include `/v1`:

```json5
{
  models: {
    mode: "merge",
    providers: {
      "ax-ollama": {
        baseUrl: "http://127.0.0.1:31418",
        apiKey: "ollama-local",
        api: "ollama",
        timeoutSeconds: 300,
        models: [
          {
            id: "qwen3.6-27b-mtp",
            name: "AX Qwen3.6 27B",
            reasoning: true,
            input: ["text", "image"],
            cost: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
            },
            contextWindow: 32768,
            maxTokens: 8192,
            params: {
              num_ctx: 32768,
              thinking: false,
            },
          },
        ],
      },
    },
  },
  agents: {
    defaults: {
      model: { primary: "ax-ollama/qwen3.6-27b-mtp" },
    },
  },
}
```

OpenClaw removes the custom provider prefix before sending the model ID to the
Ollama endpoint. Keep the model entry's `id` identical to AX's advertised ID.

## Vision and Tool Use Together

OpenClaw sends OpenAI-route images as data URLs and native Ollama images as raw
base64. AX normalizes both forms into the same Qwen3-VL runtime input. Tool
schemas, assistant `tool_calls`, tool results, and images are preserved in one
prompt, including multi-turn image/tool sessions.

AX intentionally rejects remote `http(s)` media URLs. OpenClaw must upload the
image bytes. If a visual checkpoint reports no image capability, regenerate
its manifest and restart the server:

```bash
ax-engine-bench generate-manifest --force /absolute/path/to/model
```

Do not declare `input: ["text", "image"]` until `/v1/models` confirms image
support.

## Verification

Run AX's standard live surface probes against the exact model ID:

```bash
python3 qa/surface_probes.py \
  --base-url http://127.0.0.1:31418 \
  --model qwen3.6-27b-mtp \
  --timeout 300
```

The streaming probe uses OpenClaw's `max_completion_tokens`,
`stream_options.include_usage`, and Qwen chat-template switch, and requires
both the terminal usage chunk and `[DONE]`.

For an artifact-gated end-to-end check that starts AX itself, set any relevant
paths and run:

```bash
AX_ENGINE_QWEN35_9B_ARTIFACTS_DIR=/path/to/qwen3.5-9b \
AX_ENGINE_QWEN36_27B_ARTIFACTS_DIR=/path/to/qwen3.6-27b \
AX_ENGINE_QWEN36_35B_ARTIFACTS_DIR=/path/to/qwen3.6-35b \
AX_ENGINE_QWEN3_VL_ARTIFACTS_DIR=/path/to/qwen3-vl \
python3 scripts/check_direct_model_compat_smoke.py
```

The check verifies native MLX selection, `/v1/models`, current OpenClaw
`/api/show` discovery, and tool-bearing requests through both OpenAI and Ollama
surfaces. Omit paths for models that are not installed.

## Boundaries

AX provides local inference, prompt rendering, streaming, reasoning separation,
vision preprocessing, and structured tool-call parsing. OpenClaw remains
responsible for tool execution, permissions, memory, planning, approvals, and
side effects. Model protocol compatibility also does not guarantee equal agent
quality: Qwen3.5 9B is useful under tighter memory constraints, while the 27B
and 35B-A3B targets generally provide more headroom for long, tool-heavy turns.
