import test from "node:test";
import assert from "node:assert/strict";

let langchainAvailable = true;
let ChatAXEngine;
let AXEngineLLM;
let ChatMessage;
let HumanMessage;
let AIMessage;
let FunctionMessage;
let ToolMessage;
let InMemoryCache;
let RunnableLambda;

try {
  ({ ChatAXEngine, AXEngineLLM } = await import("../langchain.js"));
  ({ AIMessage, ChatMessage, FunctionMessage, HumanMessage, ToolMessage } = await import("@langchain/core/messages"));
  ({ InMemoryCache } = await import("@langchain/core/caches"));
  ({ RunnableLambda } = await import("@langchain/core/runnables"));
} catch (error) {
  if (error?.code === "ERR_MODULE_NOT_FOUND") {
    langchainAvailable = false;
  } else {
    throw error;
  }
}

const skipIfNoLangChain = { skip: !langchainAvailable && "@langchain/core not installed" };

for (const kind of ["chat", "completion"]) {
  test(`${kind} cache separates endpoint, model, and generation defaults`, skipIfNoLangChain, async () => {
    const cache = new InMemoryCache();
    const configurations = [
      {},
      { baseUrl: "http://127.0.0.1:31418/other" },
      { model: "another-model" },
      { maxTokens: 12 },
      { temperature: 0 },
      { topP: 0.8 },
      { topK: 20 },
      { minP: 0.1 },
      { repetitionPenalty: 1.1 },
      { stop: ["END"] },
      { seed: 42 },
      { requestMetadata: "request-1" },
    ];
    let calls = 0;
    for (const [index, configuration] of configurations.entries()) {
      const Model = kind === "chat" ? ChatAXEngine : AXEngineLLM;
      const model = new Model({ cache, ...configuration });
      const text = `answer-${index}`;
      model.client.chatCompletion = async () => {
        calls++;
        return { choices: [{ message: { role: "assistant", content: text } }] };
      };
      model.client.completion = async () => {
        calls++;
        return { choices: [{ text }] };
      };
      for (let repeat = 0; repeat < 2; repeat++) {
        const result = await model.invoke("hello");
        assert.equal(typeof result === "string" ? result : result.content, text);
      }
    }
    assert.equal(calls, configurations.length);
  });
}

function textOf(result) {
  return typeof result === "string" ? result : result.content;
}

// A model sharing `cache` that records the requests it actually sends, keyed by
// the completion text it answers with.
function cachedModel(kind, { cache, answer, ...fields }) {
  const Model = kind === "chat" ? ChatAXEngine : AXEngineLLM;
  const model = new Model({ cache, ...fields });
  model.requests = [];
  model.client = {
    async chatCompletion(request) {
      model.requests.push(request);
      return { choices: [{ message: { role: "assistant", content: answer } }] };
    },
    async completion(request) {
      model.requests.push(request);
      return { choices: [{ text: answer }] };
    },
  };
  return model;
}

for (const kind of ["chat", "completion"]) {
  for (const [label, perCall] of [["undefined", undefined], ["null", null]]) {
    test(`${kind} cache separates constructor temperature defaults for ${label} per-call temperature`, skipIfNoLangChain, async () => {
      const cache = new InMemoryCache();
      const cold = cachedModel(kind, { cache, answer: "cold", model: "temperature-model", temperature: 0.1 });
      const hot = cachedModel(kind, { cache, answer: "hot", model: "temperature-model", temperature: 0.9 });

      const first = await cold.invoke("same prompt", { temperature: perCall });
      const second = await hot.invoke("same prompt", { temperature: perCall });

      assert.equal(textOf(first), "cold");
      assert.equal(textOf(second), "hot");
      assert.equal(cold.requests.length, 1);
      assert.equal(hot.requests.length, 1);
      assert.equal(cold.requests[0].temperature, 0.1);
      assert.equal(hot.requests[0].temperature, 0.9);
    });
  }

  test(`${kind} cache treats an equivalent per-call stop as the constructor default`, skipIfNoLangChain, async () => {
    const cache = new InMemoryCache();
    const viaDefault = cachedModel(kind, { cache, answer: "shared", model: "stop-model", stop: ["END"] });
    const viaCall = cachedModel(kind, { cache, answer: "unexpected", model: "stop-model" });

    const first = await viaDefault.invoke("prompt", { stop: undefined });
    const second = await viaCall.invoke("prompt", { stop: ["END"] });

    assert.equal(textOf(first), "shared");
    assert.equal(textOf(second), "shared");
    assert.equal(viaDefault.requests.length, 1);
    assert.equal(viaCall.requests.length, 0);
  });

  test(`${kind} cache keys an empty stop array as an explicit override`, skipIfNoLangChain, async () => {
    const cache = new InMemoryCache();
    const defaulted = cachedModel(kind, { cache, answer: "defaulted", model: "stop-model", stop: ["END"] });
    const overridden = cachedModel(kind, { cache, answer: "overridden", model: "stop-model", stop: ["END"] });

    const first = await defaulted.invoke("prompt", {});
    const second = await overridden.invoke("prompt", { stop: [] });

    assert.equal(textOf(first), "defaulted");
    assert.equal(textOf(second), "overridden");
    assert.equal(overridden.requests.length, 1);
    assert.deepEqual(overridden.requests[0].stop, []);
  });
}

for (const kind of ["chat", "completion"]) {
  for (const [label, perCall] of [["undefined", undefined], ["null", null]]) {
    test(`${kind} cache separates constructor request metadata defaults for ${label} per-call requestMetadata`, skipIfNoLangChain, async () => {
      const cache = new InMemoryCache();
      const requestA = cachedModel(kind, {
        cache,
        answer: "request-a",
        model: "metadata-model",
        requestMetadata: "request-a",
      });
      const requestB = cachedModel(kind, {
        cache,
        answer: "request-b",
        model: "metadata-model",
        requestMetadata: "request-b",
      });

      const first = await requestA.invoke("same prompt", { requestMetadata: perCall });
      const second = await requestB.invoke("same prompt", { requestMetadata: perCall });

      assert.equal(textOf(first), "request-a");
      assert.equal(textOf(second), "request-b");
      assert.equal(requestA.requests.length, 1);
      assert.equal(requestB.requests.length, 1);
      assert.equal(requestA.requests[0].metadata, "request-a");
      assert.equal(requestB.requests[0].metadata, "request-b");
    });
  }
}

// A model that records each request's metadata and answers with that value, so
// a cache hit (or a wrong request field) is visible as a missing request.
function metadataEchoModel(kind, fields = {}) {
  const Model = kind === "chat" ? ChatAXEngine : AXEngineLLM;
  const model = new Model({ model: "metadata-echo-model", ...fields });
  model.requests = [];
  model.client = {
    async chatCompletion(request) {
      model.requests.push(request.metadata);
      return { choices: [{ message: { role: "assistant", content: `answer-${request.metadata}` } }] };
    },
    async completion(request) {
      model.requests.push(request.metadata);
      return { choices: [{ text: `answer-${request.metadata}` }] };
    },
  };
  return model;
}

function echoInvoke(model, kind, options) {
  return kind === "chat"
    ? model.invoke([new HumanMessage("hi")], options)
    : model.invoke("hi", options);
}

for (const kind of ["chat", "completion"]) {
  test(`${kind} public invoke sends distinct legacy per-call metadata strings`, skipIfNoLangChain, async () => {
    const cache = new InMemoryCache();
    const model = metadataEchoModel(kind, { cache, requestMetadata: "ctor" });

    const first = await echoInvoke(model, kind, { metadata: "legacy-a" });
    const second = await echoInvoke(model, kind, { metadata: "legacy-b" });

    assert.equal(textOf(first), "answer-legacy-a");
    assert.equal(textOf(second), "answer-legacy-b");
    assert.deepEqual(model.requests, ["legacy-a", "legacy-b"]);
  });

  test(`${kind} public invoke prefers explicit requestMetadata over legacy metadata`, skipIfNoLangChain, async () => {
    const model = metadataEchoModel(kind, { requestMetadata: "ctor" });

    const explicit = await echoInvoke(model, kind, { metadata: "legacy", requestMetadata: "explicit" });
    const nullish = await echoInvoke(model, kind, { metadata: "legacy-nullish", requestMetadata: null });

    assert.equal(textOf(explicit), "answer-explicit");
    assert.equal(textOf(nullish), "answer-legacy-nullish");
    assert.deepEqual(model.requests, ["explicit", "legacy-nullish"]);
  });

  test(`${kind} public stream forwards legacy per-call metadata`, skipIfNoLangChain, async () => {
    const model = kind === "chat" ? new ChatAXEngine({ requestMetadata: "ctor" }) : new AXEngineLLM({ requestMetadata: "ctor" });
    const sent = [];
    model.client = {
      async *[kind === "chat" ? "streamChatCompletion" : "streamCompletion"](request) {
        sent.push(request.metadata);
        yield { data: { choices: [kind === "chat" ? { delta: { content: "ok" } } : { text: "ok" }] } };
        yield {
          data: {
            choices: [kind === "chat"
              ? { delta: {}, finish_reason: "stop" }
              : { text: "", finish_reason: "stop" }],
          },
        };
      },
    };

    const stream = await model.stream("hi", { metadata: "legacy-stream" });
    const parts = [];
    for await (const chunk of stream) {
      parts.push(kind === "chat" ? chunk.content : chunk);
    }

    assert.equal(parts.join(""), "ok");
    assert.deepEqual(sent, ["legacy-stream"]);
  });
}

// A fake streaming transport that records the request metadata it is handed.
function streamingModel(kind) {
  const Model = kind === "chat" ? ChatAXEngine : AXEngineLLM;
  const model = new Model({ requestMetadata: "ctor" });
  model.sent = [];
  model.client = {
    async *[kind === "chat" ? "streamChatCompletion" : "streamCompletion"](request) {
      model.sent.push(request.metadata);
      yield { data: { choices: [kind === "chat" ? { delta: { content: "ok" } } : { text: "ok" }] } };
      yield {
        data: {
          choices: [kind === "chat"
            ? { delta: {}, finish_reason: "stop" }
            : { text: "", finish_reason: "stop" }],
        },
      };
    },
  };
  return model;
}

// `ensureConfig` writes `metadata.timeoutMs` whenever `timeout` is set, so a
// legacy string `metadata` threw there before this model could promote it to
// `requestMetadata`. Capture the metadata LangChain hands to callbacks so a
// leaked string is visible as a non-object.
function tracingMetadataCapture() {
  const seen = [];
  function record(_model, _input, _runId, _parentRunId, _extraParams, _tags, metadata) {
    seen.push(metadata);
  }
  return { seen, handler: { handleChatModelStart: record, handleLLMStart: record } };
}

function assertNoLeakedLegacyMetadata(tracing, legacyValue) {
  assert.ok(tracing.seen.length > 0);
  for (const metadata of tracing.seen) {
    assert.equal(typeof metadata, "object");
    assert.ok(!JSON.stringify(metadata).includes(legacyValue));
  }
}

for (const kind of ["chat", "completion"]) {
  test(`${kind} public stream accepts legacy metadata with a timeout`, skipIfNoLangChain, async () => {
    const model = streamingModel(kind);
    const tracing = tracingMetadataCapture();
    const options = { metadata: "legacy-timeout", timeout: 5000, callbacks: [tracing.handler] };

    const parts = [];
    for await (const chunk of await model.stream("hi", options)) {
      parts.push(kind === "chat" ? chunk.content : chunk);
    }

    assert.equal(parts.join(""), "ok");
    assert.deepEqual(model.sent, ["legacy-timeout"]);
    assertNoLeakedLegacyMetadata(tracing, "legacy-timeout");
    assert.equal(tracing.seen[0].timeoutMs, 5000);
    // The caller's options object is not rewritten.
    assert.equal(options.metadata, "legacy-timeout");
    assert.equal(options.requestMetadata, undefined);
  });

  test(`${kind} public stream keeps explicit requestMetadata above a legacy string`, skipIfNoLangChain, async () => {
    const model = streamingModel(kind);
    const parts = [];

    const stream = await model.stream("hi", {
      metadata: "legacy-timeout",
      requestMetadata: "explicit",
      timeout: 5000,
    });
    for await (const chunk of stream) {
      parts.push(kind === "chat" ? chunk.content : chunk);
    }

    assert.equal(parts.join(""), "ok");
    assert.deepEqual(model.sent, ["explicit"]);
  });

  test(`${kind} public streamEvents accepts legacy metadata with a timeout`, skipIfNoLangChain, async () => {
    const model = streamingModel(kind);
    const tracing = tracingMetadataCapture();
    const options = {
      version: "v2",
      metadata: "legacy-events",
      timeout: 5000,
      callbacks: [tracing.handler],
    };

    // Chat and completion runs report a different run type and event prefix.
    const runType = kind === "chat" ? "chat_model" : "llm";
    const events = [];
    for await (const event of model.streamEvents("hi", options, { includeTypes: [runType] })) {
      events.push(event);
    }

    assert.deepEqual(model.sent, ["legacy-events"]);
    assert.ok(events.some((event) => event.event.endsWith("_stream")));
    assertNoLeakedLegacyMetadata(tracing, "legacy-events");
  });

  test(`${kind} public streamLog accepts legacy metadata with a timeout`, skipIfNoLangChain, async () => {
    const model = streamingModel(kind);
    const tracing = tracingMetadataCapture();
    const options = { metadata: "legacy-log", timeout: 5000, callbacks: [tracing.handler] };

    const runType = kind === "chat" ? "chat_model" : "llm";
    const logs = [];
    for await (const log of model.streamLog("hi", options, { includeTypes: [runType] })) {
      logs.push(log);
    }

    assert.deepEqual(model.sent, ["legacy-log"]);
    assert.ok(logs.some((log) => JSON.stringify(log.ops).includes("streamed_output")));
    assertNoLeakedLegacyMetadata(tracing, "legacy-log");
  });

  test(`${kind} public batch accepts legacy metadata with a timeout`, skipIfNoLangChain, async () => {
    const model = metadataEchoModel(kind);

    await model.batch(["hi"], { metadata: "legacy-batch", timeout: 5000 });
    await model.batch(
      ["hi", "hi"],
      [
        { metadata: "legacy-array-a", timeout: 5000 },
        { metadata: "legacy-array-b", timeout: 5000 },
      ],
    );

    // Batched calls run concurrently, so compare without relying on order.
    assert.deepEqual(model.requests.sort(), ["legacy-array-a", "legacy-array-b", "legacy-batch"]);
  });

  test(`${kind} public transform accepts legacy metadata with a timeout`, skipIfNoLangChain, async () => {
    const model = streamingModel(kind);
    async function* inputs() {
      yield "hi";
    }

    const parts = [];
    const stream = model.transform(inputs(), { metadata: "legacy-transform", timeout: 5000 });
    for await (const chunk of stream) {
      parts.push(kind === "chat" ? chunk.content : chunk);
    }

    assert.equal(parts.join(""), "ok");
    assert.deepEqual(model.sent, ["legacy-transform"]);
  });
}

test("ChatAXEngine preserves legacy function calls when replaying assistant history", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine();
  const functionCall = { name: "weather", arguments: '{"city":"Toronto"}' };
  chat.client = fakeClient({
    choices: [{ message: { role: "assistant", content: null, function_call: functionCall } }],
  });
  const reply = await chat.invoke("What is the weather?");
  const request = chat._buildChatRequest([
    new HumanMessage("What is the weather?"),
    reply,
    new FunctionMessage({ content: "sunny", name: "weather" }),
  ], {});
  assert.deepEqual(request.messages[1].function_call, functionCall);
});

function fakeClient(response) {
  return {
    async chatCompletion() {
      return response;
    },
    async completion() {
      return response;
    },
  };
}

test("ChatAXEngine rejects empty choices with a clear error", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine();
  chat.client = fakeClient({ choices: [] });

  await assert.rejects(
    () => chat._generate([new HumanMessage("hello")], {}),
    /chat completions response contained no choices/,
  );
});

test("AXEngineLLM rejects empty choices with a clear error", skipIfNoLangChain, async () => {
  const llm = new AXEngineLLM();
  llm.client = fakeClient({ choices: [] });

  await assert.rejects(
    () => llm._call("hello", {}),
    /completions response contained no choices/,
  );
});

test("ChatAXEngine preserves tool calls in responses", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine();
  chat.client = fakeClient({
    choices: [{
      message: {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "call-1",
          type: "function",
          function: { name: "weather", arguments: '{"city":"Toronto"}' },
        }],
      },
      finish_reason: "tool_calls",
    }],
  });

  const result = await chat._generate([new HumanMessage("What is the weather?")], {});
  const message = result.generations[0].message;
  assert.equal(message.content, "");
  assert.deepEqual(message.tool_calls, [{
    name: "weather",
    args: { city: "Toronto" },
    id: "call-1",
  }]);
  assert.equal(message.additional_kwargs.tool_calls[0].id, "call-1");
});

test("ChatAXEngine reports token usage in LangChain's shape", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine();
  chat.client = fakeClient({
    choices: [{ message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
    usage: { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12 },
  });

  const result = await chat._generate([new HumanMessage("hello")], {});
  assert.deepEqual(result.llmOutput.tokenUsage, {
    promptTokens: 5,
    completionTokens: 7,
    totalTokens: 12,
  });
  assert.deepEqual(result.generations[0].message.usage_metadata, {
    input_tokens: 5,
    output_tokens: 7,
    total_tokens: 12,
  });
});

test("ChatAXEngine preserves fragmented streaming tool calls", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine();
  chat.client = {
    async *streamChatCompletion() {
      yield {
        data: {
          choices: [{
            delta: {
              tool_calls: [{
                index: 0,
                id: "call-1",
                type: "function",
                function: { name: "weather", arguments: '{"city":' },
              }],
            },
            finish_reason: null,
          }],
        },
      };
      yield {
        data: {
          choices: [{
            delta: {
              tool_calls: [{ index: 0, function: { arguments: '"Toronto"}' } }],
            },
            finish_reason: null,
          }],
        },
      };
      yield {
        data: { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
      };
    },
  };

  const chunks = [];
  for await (const chunk of chat._streamResponseChunks(
    [new HumanMessage("What is the weather?")],
    {},
  )) {
    chunks.push(chunk.message);
  }
  const combined = chunks.slice(1).reduce((message, chunk) => message.concat(chunk), chunks[0]);
  assert.equal(combined.tool_calls.length, 1);
  assert.equal(combined.tool_calls[0].name, "weather");
  assert.deepEqual(combined.tool_calls[0].args, { city: "Toronto" });
  assert.equal(combined.tool_calls[0].id, "call-1");
});

test("ChatAXEngine forwards seed and request metadata from per-call options", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine({ seed: 1, requestMetadata: "default" });
  const req = chat._buildChatRequest(
    [new HumanMessage("hi")],
    { seed: 42, requestMetadata: "override" },
  );
  assert.equal(req.seed, 42);
  assert.equal(req.metadata, "override");
});

test("ChatAXEngine falls back to constructor seed and request metadata", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine({ seed: 7, requestMetadata: "ctor" });
  const req = chat._buildChatRequest([new HumanMessage("hi")], {});
  assert.equal(req.seed, 7);
  assert.equal(req.metadata, "ctor");
});

test("LangChain tracing metadata is not sent as AX request metadata", skipIfNoLangChain, async () => {
  const tracingMetadata = { traceId: "trace-1" };
  const chat = new ChatAXEngine({ metadata: tracingMetadata });
  const req = chat._buildChatRequest(
    [new HumanMessage("hi")],
    { metadata: { invocation: "call-1" } },
  );
  assert.equal(req.metadata, undefined);
  assert.equal(chat.metadata.traceId, tracingMetadata.traceId);
});

test("historical string metadata remains a runtime alias", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine({ metadata: "legacy" });
  let capturedRequest;
  chat.client = {
    async chatCompletion(request) {
      capturedRequest = request;
      return { choices: [{ message: { role: "assistant", content: "ok" } }] };
    },
  };
  await chat.invoke([new HumanMessage("hi")]);
  assert.equal(capturedRequest.metadata, "legacy");
  assert.notEqual(typeof chat.metadata, "string");
});

test("AXEngineLLM keeps tracing and request metadata separate", skipIfNoLangChain, () => {
  const llm = new AXEngineLLM({
    metadata: { traceId: "trace-1" },
    requestMetadata: "request-1",
  });
  const request = llm._buildCompletionRequest("hi", {});
  assert.equal(request.metadata, "request-1");
  assert.equal(llm.metadata.traceId, "trace-1");
});

test("ChatAXEngine preserves assistant, tool, and function history", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine();
  const req = chat._buildChatRequest([
    new HumanMessage("What is the weather?"),
    new AIMessage({
      content: "",
      tool_calls: [{
        name: "weather",
        args: { city: "Toronto" },
        id: "call-1",
        type: "tool_call",
      }],
    }),
    new ToolMessage({ content: "sunny", tool_call_id: "call-1", name: "weather" }),
    new FunctionMessage({ content: "legacy result", name: "legacy_weather" }),
  ], {});

  assert.deepEqual(req.messages[1], {
    role: "assistant",
    content: "",
    tool_calls: [{
      id: "call-1",
      type: "function",
      function: { name: "weather", arguments: '{"city":"Toronto"}' },
    }],
  });
  assert.deepEqual(req.messages[2], {
    role: "tool",
    content: "sunny",
    name: "weather",
    tool_call_id: "call-1",
  });
  assert.deepEqual(req.messages[3], {
    role: "function",
    content: "legacy result",
    name: "legacy_weather",
  });
});

test("ChatAXEngine forwards the ChatMessage role through a public invoke", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine();
  let capturedRequest;
  chat.client = {
    async chatCompletion(request) {
      capturedRequest = request;
      return { choices: [{ message: { role: "assistant", content: "ok" } }] };
    },
  };

  await chat.invoke([
    new ChatMessage({ role: "system", content: "Be terse." }),
    new HumanMessage("hi"),
  ]);

  assert.deepEqual(capturedRequest.messages[0], { role: "system", content: "Be terse." });
  assert.deepEqual(capturedRequest.messages[1], { role: "user", content: "hi" });
});

test("ChatAXEngine bindTools forwards tools and a specific choice", skipIfNoLangChain, async () => {
  const chat = new ChatAXEngine();
  let capturedRequest;
  chat.client = {
    async chatCompletion(request) {
      capturedRequest = request;
      return { choices: [{ message: { role: "assistant", content: "ok" } }] };
    },
  };
  const bound = chat.bindTools(
    [{
      type: "function",
      function: {
        name: "weather",
        description: "Get the weather",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"],
        },
      },
    }],
    { tool_choice: "weather" },
  );
  await bound.invoke([new HumanMessage("What is the weather?")]);

  assert.equal(capturedRequest.tools[0].function.name, "weather");
  assert.deepEqual(capturedRequest.tool_choice, {
    type: "function",
    function: { name: "weather" },
  });
});

test("empty per-call stop overrides the constructor default", skipIfNoLangChain, () => {
  const chat = new ChatAXEngine({ stop: ["<|end|>"] });
  const request = chat._buildChatRequest([new HumanMessage("hi")], { stop: [] });
  assert.deepEqual(request.stop, []);
});

for (const kind of ["chat", "completion"]) {
  test(`${kind} public LangChain stream aggregates multiple chunks`, skipIfNoLangChain, async () => {
    const model = kind === "chat" ? new ChatAXEngine() : new AXEngineLLM();
    const method = kind === "chat" ? "streamChatCompletion" : "streamCompletion";
    model.client = {
      async *[method]() {
        for (const text of ["Hello", " world"]) {
          yield { data: { choices: [kind === "chat"
            ? { delta: { content: text } }
            : { text }] } };
        }
        yield { data: { choices: [kind === "chat"
          ? { delta: {}, finish_reason: "stop" }
          : { text: "", finish_reason: "stop" }] } };
      },
    };
    let result;
    const stream = await model.stream("hello", {
      callbacks: [{ handleLLMEnd(output) { result = output; } }],
    });
    const parts = [];
    for await (const chunk of stream) parts.push(kind === "chat" ? chunk.content : chunk);
    assert.equal(parts.join(""), "Hello world");
    assert.equal(result.generations[0][0].text, "Hello world");
    assert.equal(result.generations[0][0].generationInfo.finishReason, "stop");
  });
}

test("AXEngineLLM streams aggregate inside a RunnableLambda event stream", skipIfNoLangChain, async () => {
  const llm = new AXEngineLLM();
  llm.client = {
    async *streamCompletion() {
      for (const text of ["Hello", " world"]) {
        yield { data: { choices: [{ text, finish_reason: null }] } };
      }
      yield { data: { choices: [{ text: "", finish_reason: "stop" }] } };
    },
  };
  const runnable = RunnableLambda.from((input, config) => llm.invoke(input, config));

  const events = [];
  for await (const event of runnable.streamEvents("hello", { version: "v2" })) {
    events.push(event);
  }

  const tokens = events
    .filter((event) => event.event === "on_llm_stream")
    .map((event) => event.data.chunk.text)
    .join("");
  const endEvent = events.find((event) => event.event === "on_llm_end");
  assert.equal(tokens, "Hello world");
  assert.equal(endEvent.data.output.generations[0][0].text, "Hello world");
  assert.equal(endEvent.data.output.generations[0][0].generationInfo.finishReason, "stop");
});

test("ChatAXEngine bindings preserve legacy metadata and tools", skipIfNoLangChain, async () => {
  const model = streamingModel("chat");
  model.client.chatCompletion = async (request) => {
    model.sent.push(request.metadata);
    return { choices: [{ message: { role: "assistant", content: "ok" } }] };
  };
  const tracing = tracingMetadataCapture();
  const bound = model.bindTools([{
    type: "function",
    function: { name: "weather", description: "Get weather", parameters: { type: "object" } },
  }], { metadata: "legacy-bound", tool_choice: "weather" });

  const result = await bound.invoke("hi");
  assert.equal(result.content, "ok");
  assert.deepEqual(model.sent, ["legacy-bound"]);
  assert.equal(bound.kwargs.tools[0].function.name, "weather");
  assert.equal(bound.kwargs.tool_choice.function.name, "weather");

  const options = { metadata: "legacy-call", timeout: 5000, callbacks: [tracing.handler] };
  const parts = [];
  for await (const chunk of await bound.stream("hi", options)) {
    parts.push(chunk.content);
  }
  assert.equal(parts.join(""), "ok");
  assert.deepEqual(model.sent, ["legacy-bound", "legacy-bound"]);
  assertNoLeakedLegacyMetadata(tracing, "legacy-call");
  assert.equal(options.metadata, "legacy-call");
});

for (const kind of ["chat", "completion"]) {
  test(`${kind} withConfig chains preserve the latest legacy metadata`, skipIfNoLangChain, async () => {
    const model = metadataEchoModel(kind, { requestMetadata: "ctor" });
    const bound = model
      .withConfig({ metadata: "first" })
      .withConfig({ metadata: "second" });

    const result = await echoInvoke(bound, kind, {});
    assert.equal(textOf(result), "answer-second");
    assert.deepEqual(model.requests, ["second"]);

    const explicit = await echoInvoke(bound, kind, { requestMetadata: "call" });
    assert.equal(textOf(explicit), "answer-call");
    assert.deepEqual(model.requests, ["second", "call"]);
  });

  test(`${kind} withRetry normalizes legacy metadata before retry config`, skipIfNoLangChain, async () => {
    const model = metadataEchoModel(kind, { requestMetadata: "ctor" });
    const bound = model.withRetry({ stopAfterAttempt: 1 });

    const result = await echoInvoke(bound, kind, { metadata: "legacy-retry" });
    assert.equal(textOf(result), "answer-legacy-retry");
    assert.deepEqual(model.requests, ["legacy-retry"]);

    const streamed = streamingModel(kind);
    const retryStream = streamed.withRetry({ stopAfterAttempt: 1 });
    const parts = [];
    for await (const chunk of await retryStream.stream("hi", {
      metadata: "legacy-retry-stream",
      timeout: 5000,
    })) {
      parts.push(kind === "chat" ? chunk.content : chunk);
    }
    assert.equal(parts.join(""), "ok");
    assert.deepEqual(streamed.sent, ["legacy-retry-stream"]);
  });
}
