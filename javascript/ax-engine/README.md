# JavaScript Preview Client

`@ax-engine/preview-client` is a thin JavaScript client for the checked-in
`ax-engine-server` preview API.

It is intentionally narrow:

- zero runtime dependencies
- built on standard `fetch`
- targets the existing preview HTTP and OpenAI-compatible endpoints
- does not pull architecture ownership away from `ax-engine-sdk` or the server

## Install

From the repository root:

```text
npm install ./javascript/ax-engine
```

## Example

```js
import { AxEngineClient } from "@ax-engine/preview-client";

const client = new AxEngineClient({
  baseUrl: "http://127.0.0.1:8080",
});

const runtime = await client.runtime();
const result = await client.generate({
  model: "qwen3_dense",
  input_tokens: [1, 2, 3],
  max_output_tokens: 2,
});

console.log(runtime.runtime.selected_backend);
console.log(result.output_tokens);
console.log(result.output_token_logprobs);
```

For compatibility-backed text requests through the server:

```js
import { AxEngineClient } from "@ax-engine/preview-client";

const client = new AxEngineClient({
  baseUrl: "http://127.0.0.1:8080",
});

const response = await client.completion({
  model: "qwen3_dense",
  prompt: "Hello from AX",
  max_tokens: 32,
});

console.log(response.choices[0].text);
```

Streaming helpers yield parsed SSE events:

```js
import { AxEngineClient } from "@ax-engine/preview-client";

const client = new AxEngineClient({
  baseUrl: "http://127.0.0.1:8080",
});

for await (const event of client.streamGenerate({
  model: "qwen3_dense",
  input_tokens: [1, 2, 3],
  max_output_tokens: 2,
})) {
  console.log(event.event, event.data);
  if (event.event === "step") {
    console.log(event.data.delta_tokens, event.data.delta_token_logprobs);
  }
}
```

Preview generate and stream payloads mirror the SDK/server contract, including:

- per-token `output_token_logprobs` on final responses
- per-step `delta_token_logprobs` on streaming step events
- request-level `finish_reason` and `terminal_stop_reason` on stepwise lifecycle payloads
- typed `step.metal_dispatch` metadata for AX-native bring-up runs, including
  model-conditioned / real-model-tensor execution flags, complete-model-forward
  support markers, step-level decode-logit / real-forward completion markers,
  multilayer prefix native-vs-CPU dispatch counts, model-side RMSNorm+QKV /
  o_proj+FFN token coverage, final logits projection + vocab-scan counts, and
  compact command-buffer / checksum summaries
