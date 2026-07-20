import test from "node:test";
import assert from "node:assert/strict";

let langchainAvailable = true;
let ChatAXEngine;
let AXEngineLLM;
let HumanMessage;
let AIMessage;
let FunctionMessage;
let ToolMessage;

try {
  ({ ChatAXEngine, AXEngineLLM } = await import("../langchain.js"));
  ({ AIMessage, FunctionMessage, HumanMessage, ToolMessage } = await import("@langchain/core/messages"));
} catch (error) {
  if (error?.code === "ERR_MODULE_NOT_FOUND") {
    langchainAvailable = false;
  } else {
    throw error;
  }
}

const skipIfNoLangChain = { skip: !langchainAvailable && "@langchain/core not installed" };

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
