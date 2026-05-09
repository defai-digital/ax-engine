# Ruby SDK

`sdk/ruby/` is the Ruby SDK for AX Engine v4, packaged as the `ax-engine-sdk` gem.

It is intentionally thin:

- it speaks to `ax-engine-server`, not directly to `ax-engine-core`
- zero runtime dependencies (stdlib `net/http` and `json` only)
- Ruby 2.7+
- block-based SSE streaming

## Install

From the repository root (local development):

```ruby
# In your Gemfile
gem "ax-engine-sdk", path: "sdk/ruby"
```

Or load directly:

```ruby
$LOAD_PATH.unshift "sdk/ruby/lib"
require "ax_engine"
```

Once the gem is published:

```bash
gem install ax-engine-sdk
```

## Quick Start

```ruby
require "ax_engine"

client = AxEngine::Client.new   # default: http://127.0.0.1:8080

resp = client.chat_completion(
  messages: [{ role: "user", content: "Hello!" }],
  max_tokens: 128
)
puts resp.dig("choices", 0, "message", "content")
```

## Client Configuration

```ruby
client = AxEngine::Client.new(
  base_url: "http://127.0.0.1:8080",   # default
  headers:  { "Authorization" => "Bearer token" },
  timeout:  300                          # seconds, default 300
)
```

## API Reference

### Native ax-engine endpoints

```ruby
# GET /health
client.health

# POST /v1/generate  (token-based)
client.generate(input_tokens: [1, 2, 3], max_output_tokens: 32)

# Stepwise lifecycle
report = client.submit(input_tokens: [1, 2, 3], max_output_tokens: 32)
snap   = client.request_snapshot(report["request_id"])
step   = client.step
snap   = client.cancel(report["request_id"])
```

### OpenAI-compatible endpoints

```ruby
# Text completion
resp = client.completion(prompt: "Hello", max_tokens: 64)
puts resp.dig("choices", 0, "text")

# Chat completion
resp = client.chat_completion(
  messages: [
    { role: "system", content: "You are AX Engine." },
    { role: "user",   content: "Hello!" }
  ],
  max_tokens: 128,
  temperature: 0.7,
  top_p: 0.9,
  seed: 42
)
puts resp.dig("choices", 0, "message", "content")

# Embeddings
resp = client.embeddings(input: [1, 2, 3], pooling: "last", normalize: true)
puts resp.dig("data", 0, "embedding").length
```

### Streaming

Streaming methods accept a block that receives one parsed SSE event hash per
call. Each hash has `"event"` (string) and `"data"` (parsed JSON hash or string).

```ruby
# Streaming chat
client.stream_chat_completion(
  messages: [{ role: "user", content: "Count from 1 to 5." }],
  max_tokens: 64
) do |event|
  delta = event.dig("data", "choices", 0, "delta", "content")
  print delta if delta
end
puts

# Streaming text completion
client.stream_completion(prompt: "Once upon a time", max_tokens: 64) do |event|
  print event.dig("data", "choices", 0, "text").to_s
end
puts

# Native ax-engine streaming
client.stream_generate(input_tokens: [1, 2, 3], max_output_tokens: 32) do |event|
  if event["event"] == "step"
    print event.dig("data", "delta_text").to_s
  end
end
puts
```

### Error handling

```ruby
begin
  client.chat_completion(messages: [])
rescue AxEngine::HttpError => e
  puts e.status   # HTTP status code
  puts e.message  # error message from server
  puts e.payload  # parsed response body
end
```

## LangChain Integration

`sdk/ruby/lib/ax_engine/langchain.rb` provides two classes compatible with
the [langchain-rb](https://github.com/patterns-ai-core/langchain) gem:

- `AxEngine::Langchain::ChatModel` — backed by `/v1/chat/completions`
- `AxEngine::Langchain::LLM` — backed by `/v1/completions`

Requires `gem install langchain-rb`. Uses `AxEngine::Client` (stdlib HTTP)
internally — no extra HTTP gems needed.

```ruby
require "ax_engine/langchain"

chat = AxEngine::Langchain::ChatModel.new(
  base_url:    "http://127.0.0.1:8080",
  max_tokens:  256,
  temperature: 0.7,
)

# Blocking
response = chat.chat(messages: [{ role: "user", content: "Hello!" }])
puts response.chat_completion

# Streaming — yields delta strings
chat.chat(messages: [{ role: "user", content: "Count from 1 to 5." }], stream: true) do |delta|
  print delta
end
puts
```

Text LLM:

```ruby
llm = AxEngine::Langchain::LLM.new(base_url: "http://127.0.0.1:8080", max_tokens: 128)
puts llm.complete(prompt: "Once upon a time").completion
```

Run the example:

```bash
ruby examples/ruby/langchain_chat.rb
```

Both classes accept: `base_url`, `headers`, `timeout`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `stop`, `seed`.

## Running Examples

```bash
# Requires ax-engine-server on http://127.0.0.1:8080
ruby examples/ruby/chat.rb
ruby examples/ruby/langchain_chat.rb  # requires: gem install langchain-rb
```

## Running Tests

```bash
cd sdk/ruby
rake test
```

Or run individual files:

```bash
ruby -Ilib -Itest test/test_sse_reader.rb
ruby -Ilib -Itest test/test_client.rb
```

Tests run fully offline — no server required (18 tests, 53 assertions).
