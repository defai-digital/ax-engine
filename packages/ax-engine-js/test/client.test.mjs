import test from "node:test";
import assert from "node:assert/strict";

import AxEngineClient, {
  AxEngineHttpError,
  AxEngineStreamError,
  chatCompletionChunkText,
  completionChunkText,
  responseStreamEventText,
} from "../index.js";

test("health uses normalized baseURL and parses JSON", async () => {
  const calls = [];
  const client = new AxEngineClient({
    baseURL: "http://127.0.0.1:3000/",
    fetch: async (url, init) => {
      calls.push({ url: String(url), init });
      return Response.json({
        status: "ok",
        model: "Qwen3-8B-Q4_K_M",
        architecture: "qwen3",
        context_length: 4096,
        vocab_size: 151936,
        support_note: null,
      });
    },
  });

  const health = await client.health();

  assert.equal(calls[0].url, "http://127.0.0.1:3000/healthz");
  assert.equal(health.model, "Qwen3-8B-Q4_K_M");
});

test("completion request applies default model", async () => {
  const client = new AxEngineClient({
    baseURL: "http://127.0.0.1:3000",
    defaultModel: "Qwen3-8B-Q4_K_M",
    fetch: async (_url, init) => {
      const payload = JSON.parse(init.body);
      assert.equal(payload.model, "Qwen3-8B-Q4_K_M");
      assert.equal(payload.stream, false);
      return Response.json({
        id: "cmpl-1",
        object: "text_completion",
        created: 1,
        model: payload.model,
        choices: [{ index: 0, text: "hello", finish_reason: "stop" }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      });
    },
  });

  const response = await client.completions.create({
    prompt: "Hello",
  });

  assert.equal(response.choices[0].text, "hello");
});

test("chat stream yields parsed chunks and text helper", async () => {
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"Qwen3","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
        ),
      );
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"Qwen3","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        ),
      );
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"Qwen3","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
        ),
      );
      controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  const client = new AxEngineClient({
    fetch: async () =>
      new Response(stream, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
  });

  const chunks = [];
  for await (const chunk of client.chat.completions.stream({
    messages: [{ role: "user", content: "Hello" }],
  })) {
    chunks.push(chunk);
  }

  assert.equal(chunks.length, 3);
  assert.equal(chatCompletionChunkText(chunks[1]), "Hello");
  assert.equal(chatCompletionChunkText(chunks[2]), " world");
});

test("completion text stream yields only text deltas", async () => {
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"id":"cmpl-1","object":"text_completion","created":1,"model":"Qwen3","choices":[{"index":0,"text":"Hello","finish_reason":null}]}\n\n',
        ),
      );
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"id":"cmpl-1","object":"text_completion","created":1,"model":"Qwen3","choices":[{"index":0,"text":" world","finish_reason":null}]}\n\n',
        ),
      );
      controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  const client = new AxEngineClient({
    fetch: async () =>
      new Response(stream, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
  });

  const parts = [];
  for await (const part of client.completions.streamText({ prompt: "Hello" })) {
    parts.push(part);
  }

  assert.deepEqual(parts, ["Hello", " world"]);
  assert.equal(completionChunkText({ choices: [{ text: "x" }] }), "x");
});

test("non-2xx response becomes AxEngineHttpError", async () => {
  const client = new AxEngineClient({
    fetch: async () =>
      new Response(
        JSON.stringify({
          error: { message: "bad request", type: "invalid_request_error" },
        }),
        {
          status: 400,
          headers: { "content-type": "application/json" },
        },
      ),
  });

  await assert.rejects(() => client.health(), (error) => {
    assert.ok(error instanceof AxEngineHttpError);
    assert.equal(error.status, 400);
    assert.equal(error.message, "bad request");
    return true;
  });
});

test("stream error payload becomes AxEngineStreamError", async () => {
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"error":{"message":"stream failed","type":"server_error"}}\n\n',
        ),
      );
      controller.close();
    },
  });

  const client = new AxEngineClient({
    fetch: async () =>
      new Response(stream, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
  });

  await assert.rejects(
    async () => {
      for await (const _chunk of client.completions.stream({ prompt: "Hello" })) {
        // no-op
      }
    },
    (error) => {
      assert.ok(error instanceof AxEngineStreamError);
      assert.equal(error.message, "stream failed");
      return true;
    },
  );
});

test("responses create maps input and instructions onto chat completions", async () => {
  const client = new AxEngineClient({
    defaultModel: "Qwen3-8B-Q4_K_M",
    fetch: async (_url, init) => {
      const payload = JSON.parse(init.body);
      assert.equal(payload.model, "Qwen3-8B-Q4_K_M");
      assert.equal(payload.stream, false);
      assert.deepEqual(payload.messages, [
        { role: "system", content: "Answer concisely." },
        { role: "user", content: "Summarize AX Engine in one sentence." },
      ]);
      return Response.json({
        id: "chatcmpl-1",
        object: "chat.completion",
        created: 1,
        model: payload.model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "AX Engine is a local Apple Silicon inference runtime.",
            },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 9, completion_tokens: 10, total_tokens: 19 },
      });
    },
  });

  const response = await client.responses.create({
    instructions: "Answer concisely.",
    input: "Summarize AX Engine in one sentence.",
  });

  assert.equal(response.object, "response");
  assert.equal(response.output_text, "AX Engine is a local Apple Silicon inference runtime.");
  assert.equal(response.usage.input_tokens, 9);
});

test("responses stream emits compatibility events and text helper", async () => {
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"id":"chatcmpl-9","object":"chat.completion.chunk","created":9,"model":"Qwen3","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
        ),
      );
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"id":"chatcmpl-9","object":"chat.completion.chunk","created":9,"model":"Qwen3","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        ),
      );
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"id":"chatcmpl-9","object":"chat.completion.chunk","created":9,"model":"Qwen3","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
        ),
      );
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"id":"chatcmpl-9","object":"chat.completion.chunk","created":9,"model":"Qwen3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ),
      );
      controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  const client = new AxEngineClient({
    fetch: async () =>
      new Response(stream, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
  });

  const events = [];
  for await (const event of client.responses.stream({
    input: "Hello",
  })) {
    events.push(event);
  }

  assert.equal(events[0].type, "response.created");
  assert.equal(events[1].type, "response.output_text.delta");
  assert.equal(events[2].type, "response.output_text.delta");
  assert.equal(events[3].type, "response.output_text.done");
  assert.equal(events[4].type, "response.completed");
  assert.equal(responseStreamEventText(events[1]), "Hello");
  assert.equal(responseStreamEventText(events[2]), " world");
  assert.equal(events[4].response.output_text, "Hello world");
});
