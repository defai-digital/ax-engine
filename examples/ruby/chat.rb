# Example: chat completion with the AX Engine Ruby SDK.
#
# Requires a running ax-engine-server on http://127.0.0.1:8080.
#
# Run:
#   ruby examples/ruby/chat.rb

$LOAD_PATH.unshift File.expand_path("../../sdk/ruby/lib", __dir__)
require "ax_engine"

client = AxEngine::Client.new

# Blocking chat completion
resp = client.chat_completion(
  messages: [
    { role: "system", content: "You are AX Engine." },
    { role: "user",   content: "Say hello in one sentence." }
  ],
  max_tokens: 64,
  temperature: 0.7
)
puts resp.dig("choices", 0, "message", "content")

# Streaming chat completion
puts "\n--- streaming ---"
client.stream_chat_completion(
  messages: [{ role: "user", content: "Count from 1 to 5." }],
  max_tokens: 64
) do |event|
  delta = event.dig("data", "choices", 0, "delta", "content")
  print delta if delta
end
puts
