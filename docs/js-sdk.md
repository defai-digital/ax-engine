# JavaScript SDK

`packages/ax-engine-js` is the JavaScript client SDK for `ax-engine-server`.

It is designed for:

- Node.js applications
- Next.js route handlers, server actions, and backend code
- local tools that want an OpenAI-style HTTP client without pulling in the full OpenAI SDK

It is intentionally an HTTP client, not a native binding.

It also provides a client-side `responses` compatibility layer on top of
`/v1/chat/completions`, so Node.js and Next.js code can use a more modern
response shape without waiting for a server-side `/v1/responses` transport.

## Why This Shape

For Node.js and Next.js users, the operationally correct integration point is
the built-in AX HTTP surface:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

That keeps AX's Rust and Metal runtime inside the server process while giving
JavaScript software a stable, lightweight client API.

## Package

Current package path inside this repository:

```text
packages/ax-engine-js
```

Planned publishable package name:

```text
@defai.digital/ax-engine-js
```

## CI and Publish Workflows

This repository now includes:

- `.github/workflows/js-sdk.yml`
- `.github/workflows/publish-js-sdk.yml`

The publish workflow supports:

- manual dispatch
- GitHub release publish with a tag named `ax-engine-js-v<package-version>`

Publishing requires an `NPM_TOKEN` repository secret.

## Local Install

From this repository root:

```bash
npm install ./packages/ax-engine-js
```

## Node.js Example

```js
import { AxEngineClient } from "@defai.digital/ax-engine-js";

const client = new AxEngineClient({
  baseURL: "http://127.0.0.1:3000",
  defaultModel: "Qwen3-8B-Q4_K_M",
});

const response = await client.chat.completions.create({
  messages: [
    { role: "system", content: "Answer concisely." },
    { role: "user", content: "Explain unified memory in one sentence." },
  ],
});

console.log(response.choices[0].message.content);
```

## Responses Example

```js
const response = await client.responses.create({
  instructions: "Answer concisely.",
  input: "Explain unified memory in one sentence.",
});

console.log(response.output_text);
```

## Streaming Example

```js
for await (const text of client.chat.completions.streamText({
  messages: [{ role: "user", content: "List three uses for local inference." }],
})) {
  process.stdout.write(text);
}
```

## Next.js Example

```ts
import { AxEngineClient } from "@defai.digital/ax-engine-js";

const client = new AxEngineClient({
  baseURL: process.env.AX_ENGINE_BASE_URL ?? "http://127.0.0.1:3000",
  defaultModel: process.env.AX_ENGINE_MODEL,
});

export async function POST(req: Request) {
  const { prompt } = await req.json();
  const response = await client.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
  });

  return Response.json({
    text: response.choices[0].message.content,
  });
}
```

## Current Surface

- `client.health()`
- `client.models.list()`
- `client.completions.create(...)`
- `client.completions.stream(...)`
- `client.completions.streamText(...)`
- `client.chat.completions.create(...)`
- `client.chat.completions.stream(...)`
- `client.chat.completions.streamText(...)`
- `client.responses.create(...)`
- `client.responses.stream(...)`
- `client.responses.streamText(...)`
