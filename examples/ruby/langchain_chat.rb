# LangChain chat example using AX Engine Ruby SDK.
#
# Requires:
#   gem install langchain-rb
#   ax-engine-server running on http://127.0.0.1:8080
#
# Run:
#   ruby examples/ruby/langchain_chat.rb

$LOAD_PATH.unshift File.expand_path("../../sdk/ruby/lib", __dir__)
require "ax_engine/langchain"

chat = AxEngine::Langchain::ChatModel.new(
  base_url:    "http://127.0.0.1:8080",
  max_tokens:  256,
  temperature: 0.7,
)

# Blocking
response = chat.chat(
  messages: [
    { role: "system", content: "You are AX Engine." },
    { role: "user",   content: "Say hello in one sentence." },
  ]
)
puts response.chat_completion

# Streaming
puts "\n--- streaming ---"
chat.chat(
  messages: [{ role: "user", content: "Count from 1 to 5." }],
  stream:   true
) { |delta| print delta }
puts
