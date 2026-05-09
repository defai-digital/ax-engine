import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";

import { AxEngineClient, AxEngineHttpError } from "../index.js";

async function withServer(handler, run) {
  const server = http.createServer(handler);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const baseUrl = `http://127.0.0.1:${address.port}`;
  try {
    await run(baseUrl);
  } finally {
    await new Promise((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
  }
}

test("runtime fetches preview runtime metadata", async () => {
  await withServer((req, res) => {
    assert.equal(req.method, "GET");
    assert.equal(req.url, "/v1/runtime");
    res.setHeader("content-type", "application/json");
    res.end(
      JSON.stringify({
        service: "ax-engine-server",
        model_id: "qwen3_dense",
        runtime: {
          selected_backend: "ax_native",
        },
      }),
    );
  }, async (baseUrl) => {
    const client = new AxEngineClient({ baseUrl });
    const runtime = await client.runtime();
    assert.equal(runtime.model_id, "qwen3_dense");
    assert.equal(runtime.runtime.selected_backend, "ax_native");
  });
});

test("generate posts JSON to preview endpoint", async () => {
  await withServer((req, res) => {
    assert.equal(req.method, "POST");
    assert.equal(req.url, "/v1/generate");
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      const payload = JSON.parse(body);
      assert.deepEqual(payload.input_tokens, [1, 2, 3]);
      assert.equal(payload.max_output_tokens, 2);
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          request_id: 1,
          output_tokens: [4, 5],
        }),
      );
    });
  }, async (baseUrl) => {
    const client = new AxEngineClient({ baseUrl });
    const response = await client.generate({
      input_tokens: [1, 2, 3],
      max_output_tokens: 2,
    });
    assert.deepEqual(response.output_tokens, [4, 5]);
  });
});

test("streamGenerate parses named SSE events", async () => {
  await withServer((req, res) => {
    assert.equal(req.method, "POST");
    assert.equal(req.url, "/v1/generate/stream");
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
    res.write('event: request\ndata: {"request_id":1}\n\n');
    res.write(
      'event: step\ndata: {"delta_tokens":[42],"delta_token_logprobs":[-0.25],"step":{"metal_dispatch":{"runtime_model_conditioned_inputs":true,"runtime_real_model_tensor_inputs":true,"runtime_complete_model_forward_supported":true,"runtime_model_family":"qwen3_dense","execution_direct_decode_token_count":1,"execution_direct_decode_checksum_lo":4660,"execution_logits_output_count":1,"execution_remaining_logits_handle_count":0,"execution_model_bound_ffn_decode":true,"execution_real_model_forward_completed":true,"execution_prefix_native_dispatch_count":35,"execution_prefix_cpu_reference_dispatch_count":1,"execution_qkv_projection_token_count":72,"execution_layer_continuation_token_count":37,"execution_logits_projection_token_count":1,"execution_logits_vocab_scan_row_count":151936}}}\n\n',
    );
    res.end('event: response\ndata: {"finish_reason":"max_tokens"}\n\n');
  }, async (baseUrl) => {
    const client = new AxEngineClient({ baseUrl });
    const events = [];
    for await (const event of client.streamGenerate({
      input_tokens: [1, 2, 3],
      max_output_tokens: 1,
    })) {
      events.push(event);
    }

    assert.deepEqual(events, [
      { event: "request", data: { request_id: 1 } },
      {
        event: "step",
        data: {
          delta_tokens: [42],
          delta_token_logprobs: [-0.25],
          step: {
            metal_dispatch: {
              runtime_model_conditioned_inputs: true,
              runtime_real_model_tensor_inputs: true,
              runtime_complete_model_forward_supported: true,
              runtime_model_family: "qwen3_dense",
              execution_direct_decode_token_count: 1,
              execution_direct_decode_checksum_lo: 4660,
              execution_logits_output_count: 1,
              execution_remaining_logits_handle_count: 0,
              execution_model_bound_ffn_decode: true,
              execution_real_model_forward_completed: true,
              execution_prefix_native_dispatch_count: 35,
              execution_prefix_cpu_reference_dispatch_count: 1,
              execution_qkv_projection_token_count: 72,
              execution_layer_continuation_token_count: 37,
              execution_logits_projection_token_count: 1,
              execution_logits_vocab_scan_row_count: 151936,
            },
          },
        },
      },
      { event: "response", data: { finish_reason: "max_tokens" } },
    ]);
  });
});

test("streamCompletion stops on OpenAI [DONE] sentinel", async () => {
  await withServer((req, res) => {
    assert.equal(req.method, "POST");
    assert.equal(req.url, "/v1/completions");
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
    res.write('data: {"id":"cmpl-1","choices":[{"text":"Hello"}]}\n\n');
    res.end("data: [DONE]\n\n");
  }, async (baseUrl) => {
    const client = new AxEngineClient({ baseUrl });
    const events = [];
    for await (const event of client.streamCompletion({
      prompt: "Hello",
      model: "qwen3_dense",
    })) {
      events.push(event);
    }

    assert.deepEqual(events, [
      {
        event: "message",
        data: {
          id: "cmpl-1",
          choices: [{ text: "Hello" }],
        },
      },
    ]);
  });
});

test("embeddings posts token array to OpenAI-shaped endpoint", async () => {
  await withServer((req, res) => {
    assert.equal(req.method, "POST");
    assert.equal(req.url, "/v1/embeddings");
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      const payload = JSON.parse(body);
      assert.deepEqual(payload.input, [1, 2, 3]);
      assert.equal(payload.pooling, "last");
      assert.equal(payload.normalize, true);
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          data: [{ object: "embedding", embedding: [0.1, 0.2], index: 0 }],
          model: "qwen3_embedding",
          usage: { prompt_tokens: 3, total_tokens: 3 },
        }),
      );
    });
  }, async (baseUrl) => {
    const client = new AxEngineClient({ baseUrl });
    const response = await client.embeddings({
      model: "qwen3_embedding",
      input: [1, 2, 3],
      pooling: "last",
      normalize: true,
    });
    assert.deepEqual(response.data[0].embedding, [0.1, 0.2]);
    assert.equal(response.usage.total_tokens, 3);
  });
});

test("completion returns OpenAI-shaped response", async () => {
  await withServer((req, res) => {
    assert.equal(req.method, "POST");
    assert.equal(req.url, "/v1/completions");
    res.setHeader("content-type", "application/json");
    res.end(
      JSON.stringify({
        id: "cmpl-1",
        object: "text_completion",
        created: 1234567890,
        model: "qwen3_dense",
        choices: [{ index: 0, text: "Hello world", finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
      }),
    );
  }, async (baseUrl) => {
    const client = new AxEngineClient({ baseUrl });
    const response = await client.completion({ prompt: "Hello", max_tokens: 32 });
    assert.equal(response.object, "text_completion");
    assert.equal(response.choices[0].text, "Hello world");
    assert.equal(response.usage.total_tokens, 5);
  });
});

test("chatCompletion returns OpenAI-shaped response", async () => {
  await withServer((req, res) => {
    assert.equal(req.method, "POST");
    assert.equal(req.url, "/v1/chat/completions");
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => { body += chunk; });
    req.on("end", () => {
      const payload = JSON.parse(body);
      assert.deepEqual(payload.messages[0], { role: "user", content: "Hello!" });
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          id: "chatcmpl-1",
          object: "chat.completion",
          created: 1234567890,
          model: "qwen3_dense",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Hi there!" },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
        }),
      );
    });
  }, async (baseUrl) => {
    const client = new AxEngineClient({ baseUrl });
    const response = await client.chatCompletion({
      messages: [{ role: "user", content: "Hello!" }],
      max_tokens: 32,
    });
    assert.equal(response.object, "chat.completion");
    assert.equal(response.choices[0].message.content, "Hi there!");
    assert.equal(response.usage.completion_tokens, 3);
  });
});

test("non-2xx responses raise AxEngineHttpError with payload", async () => {
  await withServer((_req, res) => {
    res.writeHead(400, {
      "content-type": "application/json",
    });
    res.end(
      JSON.stringify({
        error: {
          code: "invalid_request",
          message: "bad request",
        },
      }),
    );
  }, async (baseUrl) => {
    const client = new AxEngineClient({ baseUrl });
    await assert.rejects(
      () => client.generate({ input_tokens: [1] }),
      (error) => {
        assert.ok(error instanceof AxEngineHttpError);
        assert.equal(error.status, 400);
        assert.equal(error.message, "bad request");
        assert.deepEqual(error.payload, {
          error: {
            code: "invalid_request",
            message: "bad request",
          },
        });
        return true;
      },
    );
  });
});
