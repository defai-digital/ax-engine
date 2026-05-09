/**
 * Basic AX Engine JavaScript SDK example.
 *
 * Requires:
 *   npm install @ax-engine/sdk   (or: npm install ./javascript/ax-engine)
 *   ax-engine-server running on http://127.0.0.1:8080
 *
 * Run:
 *   node examples/javascript/basic.js
 */

import AxEngineClient from "@ax-engine/sdk";

const client = new AxEngineClient({ baseUrl: "http://127.0.0.1:8080" });

// ── Health check ──────────────────────────────────────────────────────────────
const health = await client.health();
console.log("status:", health.status);

// ── Chat completion ───────────────────────────────────────────────────────────
const chat = await client.chatCompletion({
  messages: [
    { role: "system", content: "You are AX Engine." },
    { role: "user",   content: "Say hello in one sentence." },
  ],
  max_tokens: 64,
  temperature: 0.7,
});
console.log("chat:", chat.choices[0].message.content);

// ── Text completion ───────────────────────────────────────────────────────────
const completion = await client.completion({
  prompt:     "Once upon a time",
  max_tokens: 32,
});
console.log("completion:", completion.choices[0].text);

// ── Streaming chat ────────────────────────────────────────────────────────────
console.log("\n--- streaming chat ---");
for await (const event of client.streamChatCompletion({
  messages:   [{ role: "user", content: "Count from 1 to 5." }],
  max_tokens: 64,
})) {
  const delta = event.data?.choices?.[0]?.delta?.content;
  if (delta) process.stdout.write(delta);
}
console.log();

// ── Native generate (token IDs) ───────────────────────────────────────────────
const result = await client.generate({
  input_tokens:     [1, 2, 3],
  max_output_tokens: 4,
});
console.log("output_tokens:", result.output_tokens);
