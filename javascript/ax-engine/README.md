# @ax-engine/sdk

JavaScript and TypeScript SDK for AX Engine v4 — local HTTP inference server.

- zero runtime dependencies
- built on standard `fetch` (Node.js 20+, browser, Deno, Bun)
- full TypeScript types for all request/response shapes
- OpenAI-compatible endpoints
- LangChain integration via `@ax-engine/sdk/langchain`

## Install

From the repository root:

```text
npm install ./javascript/ax-engine
```

For LangChain support also install the peer dependency:

```text
npm install @langchain/core
```

## Quick Start

```js
import AxEngineClient from "@ax-engine/sdk";

const client = new AxEngineClient({ baseUrl: "http://127.0.0.1:8080" });

// Chat completion
const resp = await client.chatCompletion({
  messages: [{ role: "user", content: "Hello!" }],
  max_tokens: 128,
});
console.log(resp.choices[0].message.content);

// Streaming chat
for await (const event of client.streamChatCompletion({
  messages: [{ role: "user", content: "Count from 1 to 5." }],
  max_tokens: 64,
})) {
  process.stdout.write(event.data.choices[0]?.delta?.content ?? "");
}
```

## Native ax-engine API

```js
// Token-based generate
const result = await client.generate({
  input_tokens: [1, 2, 3],
  max_output_tokens: 32,
});
console.log(result.output_tokens);

// Streaming native events
for await (const event of client.streamGenerate({ input_tokens: [1, 2, 3], max_output_tokens: 32 })) {
  if (event.event === "step") process.stdout.write(event.data.delta_text ?? "");
}

// Stepwise lifecycle
const report = await client.submit({ input_tokens: [1, 2, 3], max_output_tokens: 32 });
const step   = await client.step();
const snap   = await client.requestSnapshot(report.request_id);
```

## LangChain Integration

```js
import { ChatAXEngine, AXEngineLLM } from "@ax-engine/sdk/langchain";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// Chat model
const chat = new ChatAXEngine({ maxTokens: 256, temperature: 0.7 });
const response = await chat.invoke([
  new SystemMessage("You are AX Engine."),
  new HumanMessage("Say hello."),
]);
console.log(response.content);

// Streaming
for await (const chunk of await chat.stream([new HumanMessage("Count from 1 to 5.")])) {
  process.stdout.write(chunk.content);
}

// Text LLM
const llm = new AXEngineLLM({ maxTokens: 64 });
console.log(await llm.invoke("Once upon a time"));
```

Both `ChatAXEngine` and `AXEngineLLM` accept: `baseUrl`, `model`, `maxTokens`,
`temperature`, `topP`, `topK`, `minP`, `repetitionPenalty`, `stop`, `seed`.

## Embeddings

```js
const resp = await client.embeddings({
  input: [1, 2, 3],
  pooling: "last",
  normalize: true,
});
console.log(resp.data[0].embedding.length);
```

## Custom fetch / headers

```js
const client = new AxEngineClient({
  baseUrl: "http://127.0.0.1:8080",
  headers: { Authorization: "Bearer token" },
  fetch: globalThis.fetch,
});
```

## Error handling

```js
import AxEngineClient, { AxEngineHttpError } from "@ax-engine/sdk";

try {
  await client.generate({ input_tokens: [], max_output_tokens: 1 });
} catch (err) {
  if (err instanceof AxEngineHttpError) {
    console.error(err.status, err.message);
  }
}
```
