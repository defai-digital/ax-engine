require "minitest/autorun"
require "rbconfig"
require_relative "../lib/ax_engine/langchain"

# langchainrb is an optional dependency of the adapter. Tests that exercise
# response objects or model construction skip when it is absent; the load-order
# test below runs either way.
LANGCHAIN_AVAILABLE =
  begin
    require "langchain"
    true
  rescue LoadError
    false
  end

# Records the params the adapter would serialize and replays canned payloads.
class FakeClient
  attr_reader :calls

  def initialize(unary_response: {}, stream_events: [])
    @unary_response = unary_response
    @stream_events  = stream_events
    @calls          = []
  end

  def chat_completion(params)
    @calls << [:chat_completion, params]
    @unary_response
  end

  def completion(params)
    @calls << [:completion, params]
    @unary_response
  end

  def stream_chat_completion(params, &block)
    @calls << [:stream_chat_completion, params]
    @stream_events.each(&block) if block
    nil
  end

  def stream_completion(params, &block)
    @calls << [:stream_completion, params]
    @stream_events.each(&block) if block
    nil
  end
end

module LangchainAdapterHelpers
  LIB_DIR = File.expand_path("../lib", __dir__)

  # One-shot SSE server for the subprocess load-order tests. Kept in the child
  # so the parent never loads the library before the require under test.
  SSE_SERVER_PREAMBLE = <<~'RUBY'
    require "socket"

    server = TCPServer.new("127.0.0.1", 0)
    port = server.addr[1]
    Thread.new do
      socket = server.accept
      headers = {}
      while (line = socket.gets) && line != "\r\n"
        key, value = line.split(": ", 2)
        headers[key.to_s.downcase] = value.to_s.strip
      end
      length = headers["content-length"].to_i
      socket.read(length) if length.positive?
      socket.write "HTTP/1.1 200 OK\r\n"
      socket.write "Content-Type: text/event-stream\r\n"
      socket.write "Connection: close\r\n\r\n"
      socket.write "data: {\"choices\":[{\"text\":\"hi\"}]}\n\n"
      socket.write "data: [DONE]\n\n"
      socket.close
    end
  RUBY

  # Runs a fresh Ruby process that only performs `require_statement` (plus the
  # stdlib socket server above), streams one completion, and prints the
  # constants it resolved plus the text it received.
  def stream_in_isolated_process(require_statement)
    script = [
      SSE_SERVER_PREAMBLE,
      require_statement,
      <<~'CLIENT'
        texts = []
        AxEngine::Client.new(base_url: "http://127.0.0.1:#{port}", timeout: 5)
                       .stream_completion(prompt: "hello") do |event|
          text = event["data"].is_a?(Hash) ? event["data"].dig("choices", 0, "text") : nil
          texts << text if text
        end

        puts [AxEngine::SseReader.name, AxEngine::StreamError.name, texts.join].join("|")
      CLIENT
    ].join("\n")

    output = IO.popen([RbConfig.ruby, "-I", LIB_DIR, "-e", script],
                      err: [:child, :out], &:read)
    [output, $?]
  end

  def assert_isolated_streaming(require_statement, label:)
    output, status = stream_in_isolated_process(require_statement)

    assert status.success?, "#{label}: child process failed (#{status.exitstatus}):\n#{output}"
    assert_includes output.lines.map(&:strip),
                    "AxEngine::SseReader|AxEngine::StreamError|hi",
                    "#{label}: unexpected child output:\n#{output}"
  end

  CHAT_PAYLOAD = {
    "model" => "chat-model",
    "choices" => [
      { "message" => { "role" => "assistant", "content" => "hello" } }
    ],
    "usage" => { "prompt_tokens" => 2, "completion_tokens" => 3, "total_tokens" => 5 }
  }.freeze

  COMPLETION_PAYLOAD = {
    "model" => "text-model",
    "created" => 1_700_000_000,
    "choices" => [{ "text" => "Once upon a time", "index" => 0, "finish_reason" => "stop" }],
    "usage" => { "prompt_tokens" => 3, "completion_tokens" => 4, "total_tokens" => 7 }
  }.freeze

  def skip_without_langchain
    skip "langchainrb is not installed" unless LANGCHAIN_AVAILABLE
  end

  def chat_model(client, **defaults)
    attach_client(AxEngine::Langchain::ChatModel.new(**defaults), client)
  end

  def text_model(client, **defaults)
    attach_client(AxEngine::Langchain::LLM.new(**defaults), client)
  end

  # The adapter owns its transport internally, so tests swap in a fake after
  # construction to keep every exercised path public (chat/complete).
  def attach_client(model, client)
    model.instance_variable_set(:@client, client)
    model
  end

  def messages
    [{ role: "user", content: "hi" }]
  end
end

class TestLangchainLoadOrder < Minitest::Test
  include LangchainAdapterHelpers

  # Regression: requiring only "ax_engine/client" left AxEngine::SseReader and
  # AxEngine::StreamError undefined, so the first streaming call raised
  # NameError (uninitialized constant) instead of streaming.
  def test_direct_client_require_supports_streaming
    assert_isolated_streaming('require "ax_engine/client"', label: "ax_engine/client")
  end

  # Regression: requiring only "ax_engine/langchain" used to leave the same
  # constants undefined, so the adapter raised NameError on its first stream.
  def test_direct_require_supports_streaming_without_ax_engine_umbrella
    assert_isolated_streaming('require "ax_engine/langchain"', label: "ax_engine/langchain")
  end
end

class TestLangchainChatModel < Minitest::Test
  include LangchainAdapterHelpers

  def setup
    skip_without_langchain
  end

  def test_constructor_model_is_sent_and_per_call_model_overrides_it
    client = FakeClient.new(unary_response: CHAT_PAYLOAD)
    chat   = chat_model(client, model: "constructor-model", max_tokens: 8)

    chat.chat(messages: messages)
    params = client.calls.last[1]
    assert_equal "constructor-model", params[:model]
    assert_equal 8, params[:max_tokens]

    chat.chat(messages: messages, model: "call-model", max_tokens: 16)
    params = client.calls.last[1]
    assert_equal "call-model", params[:model]
    assert_equal 16, params[:max_tokens]
  end

  def test_unary_request_options_reach_the_body
    client = FakeClient.new(unary_response: CHAT_PAYLOAD)
    chat   = chat_model(client, response_format: { type: "json_object" },
                                tools: [{ type: "function" }],
                                tool_choice: "auto",
                                frequency_penalty: 0.1,
                                presence_penalty: 0.2)

    chat.chat(messages: messages, tool_choice: "required")
    params = client.calls.last[1]
    assert_equal :chat_completion, client.calls.last[0]
    assert_equal({ type: "json_object" }, params[:response_format])
    assert_equal [{ type: "function" }], params[:tools]
    assert_equal "required", params[:tool_choice]
    assert_in_delta 0.1, params[:frequency_penalty]
    assert_in_delta 0.2, params[:presence_penalty]
    assert_equal messages, params[:messages]
  end

  def test_streaming_request_options_reach_the_body
    client = FakeClient.new(stream_events: [
      { "event" => "message",
        "data" => { "choices" => [{ "delta" => { "content" => "hi" } }] } }
    ])
    chat   = chat_model(client, model: "constructor-model",
                                response_format: { type: "json_object" },
                                tools: [{ type: "function" }],
                                tool_choice: "auto",
                                frequency_penalty: 0.3,
                                presence_penalty: 0.4)
    deltas = []

    chat.chat(messages: messages, stream: true, model: "call-model") { |delta| deltas << delta }

    assert_equal :stream_chat_completion, client.calls.last[0]
    params = client.calls.last[1]
    assert_equal "call-model", params[:model]
    assert_equal({ type: "json_object" }, params[:response_format])
    assert_equal [{ type: "function" }], params[:tools]
    assert_equal "auto", params[:tool_choice]
    assert_in_delta 0.3, params[:frequency_penalty]
    assert_in_delta 0.4, params[:presence_penalty]
    refute params.key?(:stream), "the adapter must not add stream to the request body"
    assert_equal ["hi"], deltas
  end

  def test_unknown_options_are_not_forwarded
    client = FakeClient.new(unary_response: CHAT_PAYLOAD)
    chat   = chat_model(client)

    chat.chat(messages: messages, not_a_request_key: "ignored")
    params = client.calls.last[1]
    refute params.key?(:not_a_request_key)
    refute params.key?(:stream)
  end

  def test_unary_chat_returns_the_released_openai_response
    client = FakeClient.new(unary_response: CHAT_PAYLOAD)

    response = chat_model(client).chat(messages: messages)

    assert_instance_of ::Langchain::LLM::OpenAIResponse, response
    assert_equal "hello", response.chat_completion
    assert_equal "hello", response.completion
    assert_equal "chat-model", response.model
    assert_equal 5, response.total_tokens
  end

  def test_stream_without_block_raises_before_transport
    client = FakeClient.new
    chat   = chat_model(client)

    error = assert_raises(ArgumentError) { chat.chat(messages: messages, stream: true) }

    assert_match(/stream/, error.message)
    assert_empty client.calls
  end

  def test_streaming_yields_deltas_and_returns_nil
    client = FakeClient.new(stream_events: [
      { "event" => "message",
        "data" => { "choices" => [{ "delta" => { "content" => "he" } }] } },
      { "event" => "message",
        "data" => { "choices" => [{ "delta" => { "content" => "llo" } }] } },
      { "event" => "message", "data" => { "choices" => [{ "delta" => {} }] } }
    ])
    deltas = []

    result = chat_model(client).chat(messages: messages, stream: true) { |delta| deltas << delta }

    assert_nil result
    assert_equal %w[he llo], deltas
  end

  def test_streaming_surfaces_server_error_events
    client = FakeClient.new(stream_events: [
      { "event" => "error", "data" => { "error" => { "message" => "boom" } } }
    ])
    chat = chat_model(client)

    error = assert_raises(AxEngine::StreamError) do
      chat.chat(messages: messages, stream: true) { |_delta| flunk "no delta expected" }
    end

    assert_equal "boom", error.message
    assert_equal({ "error" => { "message" => "boom" } }, error.payload)
  end
end

class TestLangchainTextLLM < Minitest::Test
  include LangchainAdapterHelpers

  def setup
    skip_without_langchain
  end

  # Regression: OpenAIResponse#completion reads choices[0].message.content,
  # so text completions returned nil from #completion.
  def test_complete_returns_completion_text_with_model_and_usage
    client = FakeClient.new(unary_response: COMPLETION_PAYLOAD)

    response = text_model(client).complete(prompt: "Once upon a time")

    assert_kind_of ::Langchain::LLM::OpenAIResponse, response
    assert_equal "Once upon a time", response.completion
    assert_equal "text-model", response.model
    assert_equal 3, response.prompt_tokens
    assert_equal 4, response.completion_tokens
    assert_equal 7, response.total_tokens
    assert_equal Time.at(1_700_000_000), response.created_at
  end

  def test_complete_forwards_prompt_model_and_request_options
    client = FakeClient.new(unary_response: COMPLETION_PAYLOAD)
    model  = text_model(client, model: "constructor-model", max_tokens: 8)

    model.complete(prompt: "hi", model: "call-model", stop: ["\n"], not_a_request_key: 1)
    params = client.calls.last[1]

    assert_equal :completion, client.calls.last[0]
    assert_equal "hi", params[:prompt]
    assert_equal "call-model", params[:model]
    assert_equal ["\n"], params[:stop]
    assert_equal 8, params[:max_tokens]
    refute params.key?(:not_a_request_key)
    refute params.key?(:stream)
  end

  def test_stream_without_block_raises_before_transport
    client = FakeClient.new
    model  = text_model(client)

    error = assert_raises(ArgumentError) { model.complete(prompt: "hi", stream: true) }

    assert_match(/stream/, error.message)
    assert_empty client.calls
  end

  def test_streaming_yields_text_chunks_and_skips_empty_ones
    client = FakeClient.new(stream_events: [
      { "event" => "message", "data" => { "choices" => [{ "text" => "" }] } },
      { "event" => "message", "data" => { "choices" => [{ "text" => "Once" }] } },
      { "event" => "message", "data" => { "choices" => [{ "text" => " upon" }] } }
    ])
    chunks = []

    result = text_model(client).complete(prompt: "Once", stream: true) { |text| chunks << text }

    assert_nil result
    assert_equal ["Once", " upon"], chunks
    assert_equal :stream_completion, client.calls.last[0]
    refute client.calls.last[1].key?(:stream)
  end
end
