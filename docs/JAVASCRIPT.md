# JavaScript / TypeScript SDK

`javascript/ax-engine` is the JavaScript and TypeScript SDK for AX Engine,
published as `@ax-engine/sdk`.

It is intentionally thin:

- it speaks to `ax-engine-server`, not directly to `ax-engine-core`
- zero runtime dependencies (uses standard `fetch`)
- targets the preview HTTP and OpenAI-compatible endpoints
- keeps transport ownership with the server and SDK layers

## Current Scope

The current package provides:

- `AxEngineClient` with full TypeScript types
- JSON helpers for:
  `health()`, `runtime()`, `models()`, `generate()`, `submit()`,
  `requestSnapshot()`, `cancel()`, `step()`, and `embeddings()`
- OpenAI-compatible helpers through `completion()` and `chatCompletion()`
- SSE streaming through `streamGenerate()`, `streamCompletion()`, and
  `streamChatCompletion()`
- LangChain integration via `@ax-engine/sdk/langchain` — `ChatAXEngine` and
  `AXEngineLLM` backed by the OpenAI-compatible server endpoints

It does not yet provide:

- tokenizer utilities
- model-aware chat templating
- a published npm release workflow
- tool/function calling helpers

## Install

From the repository root:

```text
npm install ./javascript/ax-engine
```

For LangChain integration also install the peer dependency:

```text
npm install @langchain/core
```

## Example

```js
import AxEngineClient from "@ax-engine/sdk";

const client = new AxEngineClient({ baseUrl: "http://127.0.0.1:8080" });

const runtime = await client.runtime();
const result = await client.generate({
  model: "qwen3_dense",
  input_tokens: [1, 2, 3],
  max_output_tokens: 2,
});

console.log(runtime.runtime.selected_backend);
console.log(result.output_tokens);
```

OpenAI-compatible chat completion:

```js
const response = await client.chatCompletion({
  messages: [{ role: "user", content: "Hello!" }],
  max_tokens: 128,
  temperature: 0.7,
});
console.log(response.choices[0].message.content);
```

OpenAI-compatible text completion:

```js
const response = await client.completion({
  prompt: "Hello from AX JavaScript",
  max_tokens: 32,
});
console.log(response.choices[0].text);
```

Embeddings:

```js
const response = await client.embeddings({
  input: [1, 2, 3],
  pooling: "last",
  normalize: true,
});
console.log(response.data[0].embedding.length);
```

Streaming (native ax-engine SSE):

```js
for await (const event of client.streamGenerate({
  input_tokens: [1, 2, 3],
  max_output_tokens: 32,
})) {
  if (event.event === "step") {
    process.stdout.write(event.data.delta_text ?? "");
  }
}
```

Streaming chat completion:

```js
for await (const event of client.streamChatCompletion({
  messages: [{ role: "user", content: "Count from 1 to 5." }],
  max_tokens: 64,
})) {
  process.stdout.write(event.data.choices[0]?.delta?.content ?? "");
}
```

## Runnable Examples

```text
node examples/javascript/basic.js      # health, chat, completion, streaming, native generate
node examples/javascript/streaming.js  # all three streaming methods
node examples/javascript/langchain_chat.js  # LangChain ChatAXEngine + AXEngineLLM
```

All examples require `ax-engine-server` running on `http://127.0.0.1:8080`.

## LangChain Integration

`@ax-engine/sdk/langchain` exposes two LangChain-compatible model classes that
delegate to the ax-engine-server OpenAI-compatible endpoints. Requires
`@langchain/core >= 0.2`.

### `ChatAXEngine`

```js
import { ChatAXEngine } from "@ax-engine/sdk/langchain";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const chat = new ChatAXEngine({
  baseUrl: "http://127.0.0.1:8080",
  maxTokens: 256,
  temperature: 0.7,
});

// Blocking
const response = await chat.invoke([
  new SystemMessage("You are AX Engine."),
  new HumanMessage("Say hello in one sentence."),
]);
console.log(response.content);

// Streaming
const stream = await chat.stream([new HumanMessage("Count from 1 to 5.")]);
for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}
```

### `AXEngineLLM`

```js
import { AXEngineLLM } from "@ax-engine/sdk/langchain";

const llm = new AXEngineLLM({ baseUrl: "http://127.0.0.1:8080", maxTokens: 128 });
const text = await llm.invoke("Once upon a time");
console.log(text);
```

Both classes accept: `baseUrl`, `model`, `maxTokens`, `temperature`, `topP`,
`topK`, `minP`, `repetitionPenalty`, `stop`, `seed`, plus any standard
LangChain base params (`callbacks`, `tags`, `metadata`, etc.).

A runnable example is at `examples/javascript/langchain_chat.js`.
