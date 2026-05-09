/**
 * LangChain chat example using AX Engine.
 *
 * Requires:
 *   npm install @langchain/core @ax-engine/sdk
 *   ax-engine-server running on http://127.0.0.1:8080
 *
 * Run:
 *   node examples/javascript/langchain_chat.js
 */

import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatAXEngine } from "@ax-engine/sdk/langchain";

const chat = new ChatAXEngine({
  baseUrl: "http://127.0.0.1:8080",
  maxTokens: 256,
  temperature: 0.7,
});

// Blocking call
const response = await chat.invoke([
  new SystemMessage("You are AX Engine."),
  new HumanMessage("Say hello in one sentence."),
]);
console.log(response.content);

// Streaming
console.log("\n--- streaming ---");
const stream = await chat.stream([new HumanMessage("Count from 1 to 5.")]);
for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}
console.log();
