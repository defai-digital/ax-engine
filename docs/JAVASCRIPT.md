# JavaScript

AX Engine now includes a thin repo-local JavaScript preview client at
`javascript/ax-engine`.

It is intentionally small:

- it speaks to `ax-engine-server`, not directly to `ax-engine-core`
- it wraps the checked-in preview HTTP and OpenAI-compatible endpoints
- it keeps transport ownership with the server and SDK layers
- it stays dependency-free at runtime

## Current Scope

The current preview package provides:

- `AxEngineClient`
- JSON helpers for:
  `health()`, `runtime()`, `models()`, `generate()`, `submit()`,
  `requestSnapshot()`, `cancel()`, `step()`, and `embeddings()`
- OpenAI-compatible helpers through `completion()` and `chatCompletion()`
- SSE helpers through `streamGenerate()`, `streamCompletion()`, and
  `streamChatCompletion()`

It does not yet provide:

- tokenizer utilities
- model-aware chat templating
- a published npm release workflow
- tool/function calling helpers
- browser-specific packaging beyond standards-based `fetch`

## Install

From the repository root:

```text
npm install ./javascript/ax-engine
```

This installs the checked-in preview package directly from the local repo.

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
```

For a llama.cpp-backed text request through the preview
OpenAI-compatible surface:

```js
import { AxEngineClient } from "@ax-engine/preview-client";

const client = new AxEngineClient({
  baseUrl: "http://127.0.0.1:8080",
});

const response = await client.completion({
  model: "qwen3_dense",
  prompt: "Hello from AX JavaScript",
  max_tokens: 32,
});

console.log(response.choices[0].text);
```

For an embedding-capable local MLX server:

```js
import { AxEngineClient } from "@ax-engine/preview-client";

const client = new AxEngineClient({
  baseUrl: "http://127.0.0.1:8080",
});

const response = await client.embeddings({
  model: "qwen3_embedding",
  input: [1, 2, 3],
  pooling: "last",
  normalize: true,
});

console.log(response.data[0].embedding.length);
```

Streaming helpers yield parsed SSE messages:

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
}
```
