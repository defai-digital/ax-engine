# ax-engine-js

`ax-engine-js` is the JavaScript client SDK for AX-compatible HTTP endpoints.

It is intended for:

- Node.js services
- Next.js route handlers and server actions
- local tools that already know how to speak to an OpenAI-style HTTP endpoint

It is intentionally an HTTP client, not a native Node addon.

## Scope

This SDK targets the shipped AX-compatible HTTP surface:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`

It supports both JSON responses and SSE streaming.

It also exposes a client-side `responses` compatibility surface for JavaScript
apps that prefer that shape, even when they are targeting chat-completions-only
providers.

This workspace now ships `ax-engine-server` for local and single-node use;
production orchestration still belongs in AX Serving.

## Status

The package directory is publishable, but this repository does not publish it
automatically. Until you publish it, install from the local path or workspace.

GitHub Actions workflows are included:

- `.github/workflows/js-sdk.yml` for CI on push and pull request
- `.github/workflows/publish-js-sdk.yml` for npm publish on manual dispatch or a release tag named `ax-engine-js-v<package-version>`

Publishing requires an `NPM_TOKEN` repository secret.

## Install

From this repository:

```bash
npm install ./packages/ax-engine-js
```

Planned publishable package name:

```text
@defai.digital/ax-engine-js
```

## Example

```js
import { AxEngineClient } from "@defai.digital/ax-engine-js";

const client = new AxEngineClient({
  baseURL: "http://127.0.0.1:3000",
  defaultModel: "Qwen3-8B-Q4_K_M",
});

const health = await client.health();
console.log(health.model);

const chat = await client.chat.completions.create({
  messages: [
    { role: "system", content: "Answer concisely." },
    { role: "user", content: "Summarize AX Engine in one sentence." },
  ],
});

console.log(chat.choices[0].message.content);
```

## Responses Compatibility

```js
const response = await client.responses.create({
  instructions: "Answer concisely.",
  input: "Summarize AX Engine in one sentence.",
});

console.log(response.output_text);
```

## Streaming

```js
for await (const chunk of client.chat.completions.streamText({
  messages: [{ role: "user", content: "List three uses for local inference." }],
})) {
  process.stdout.write(chunk);
}
```

Streaming with the `responses` surface:

```js
for await (const text of client.responses.streamText({
  input: "List three uses for local inference.",
})) {
  process.stdout.write(text);
}
```
