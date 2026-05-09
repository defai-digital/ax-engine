/**
 * Streaming examples for the AX Engine JavaScript SDK.
 *
 * Requires:
 *   npm install @ax-engine/sdk
 *   ax-engine-server running on http://127.0.0.1:8080
 *
 * Run:
 *   node examples/javascript/streaming.js
 */

import AxEngineClient from "@ax-engine/sdk";

const client = new AxEngineClient({ baseUrl: "http://127.0.0.1:8080" });

// ── Streaming chat completion (OpenAI-compat) ─────────────────────────────────
console.log("--- stream_chat_completion ---");
for await (const event of client.streamChatCompletion({
  messages:   [{ role: "user", content: "Count from 1 to 5, one number per token." }],
  max_tokens: 32,
})) {
  const content = event.data?.choices?.[0]?.delta?.content;
  if (content) process.stdout.write(content);
}
console.log();

// ── Streaming text completion (OpenAI-compat) ─────────────────────────────────
console.log("--- stream_completion ---");
for await (const event of client.streamCompletion({
  prompt:     "The quick brown fox",
  max_tokens: 16,
})) {
  process.stdout.write(event.data?.choices?.[0]?.text ?? "");
}
console.log();

// ── Native ax-engine streaming (request / step / response events) ─────────────
console.log("--- stream_generate ---");
for await (const event of client.streamGenerate({
  input_tokens:      [1, 2, 3],
  max_output_tokens: 4,
})) {
  if (event.event === "step") {
    const tokens = event.data?.delta_tokens ?? [];
    if (tokens.length) console.log("  step tokens:", tokens);
  }
  if (event.event === "response") {
    console.log("  finish_reason:", event.data?.response?.finish_reason);
  }
}
