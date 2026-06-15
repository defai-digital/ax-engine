import test from "node:test";
import assert from "node:assert/strict";

let langchainAvailable = true;
let ChatAXEngine;
let AXEngineLLM;
let HumanMessage;

try {
  ({ ChatAXEngine, AXEngineLLM } = await import("../langchain.js"));
  ({ HumanMessage } = await import("@langchain/core/messages"));
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
