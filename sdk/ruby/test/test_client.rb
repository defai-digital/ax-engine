require "minitest/autorun"
require "json"
require "socket"
require "thread"
require_relative "../lib/ax_engine"

# Minimal single-threaded TCP server for testing.
class MockServer
  attr_reader :last_method, :last_path, :last_body

  def initialize
    @server      = TCPServer.new("127.0.0.1", 0)
    @response    = nil
    @last_method = nil
    @last_path   = nil
    @last_body   = nil
    @thread      = Thread.new { serve_loop }
    @thread.abort_on_exception = true
  end

  def port
    @server.addr[1]
  end

  def base_url
    "http://127.0.0.1:#{port}"
  end

  def set_response(status: 200, content_type: "application/json", body:)
    @response = { status: status, content_type: content_type, body: body }
  end

  def close
    @server.close rescue nil
    @thread.kill
  end

  private

  def serve_loop
    loop do
      client = @server.accept rescue break
      handle(client)
      client.close rescue nil
    end
  end

  def handle(client)
    request_line = client.gets&.chomp || ""
    parts  = request_line.split(" ")
    method = parts[0]
    path   = parts[1]

    headers = {}
    loop do
      line = client.gets&.chomp || ""
      break if line.empty?
      k, v = line.split(": ", 2)
      headers[k.downcase] = v
    end

    body = ""
    if (len = headers["content-length"]&.to_i) && len > 0
      body = client.read(len)
    end

    @last_method = method
    @last_path   = path
    @last_body   = (JSON.parse(body) rescue body) unless body.empty?

    resp = @response
    if resp
      raw_body = resp[:body].is_a?(String) ? resp[:body] : JSON.generate(resp[:body])
      client.write "HTTP/1.1 #{resp[:status]} OK\r\n"
      client.write "Content-Type: #{resp[:content_type]}\r\n"
      client.write "Content-Length: #{raw_body.bytesize}\r\n"
      client.write "Connection: close\r\n"
      client.write "\r\n"
      client.write raw_body
    else
      client.write "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    end
  end
end

class TestClient < Minitest::Test
  def setup
    @srv = MockServer.new
    @client = AxEngine::Client.new(base_url: @srv.base_url, timeout: 5)
  end

  def teardown
    @srv.close
  end

  # --- health ---

  def test_health
    @srv.set_response(body: { status: "ok", service: "ax-engine-server", model_id: "qwen3_dense" })
    resp = @client.health
    assert_equal "ok", resp["status"]
    assert_equal "GET",   @srv.last_method
    assert_equal "/health", @srv.last_path
  end

  # --- completion ---

  def test_completion
    @srv.set_response(body: {
      id: "cmpl-1", object: "text_completion", model: "qwen3_dense",
      choices: [{ index: 0, text: "Hello world", finish_reason: "stop" }],
      usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 }
    })
    resp = @client.completion(prompt: "Hello", max_tokens: 32)
    assert_equal "text_completion", resp["object"]
    assert_equal "Hello world", resp.dig("choices", 0, "text")
    assert_equal 5, resp.dig("usage", "total_tokens")
    assert_equal "POST",          @srv.last_method
    assert_equal "/v1/completions", @srv.last_path
  end

  def test_completion_request_body
    @srv.set_response(body: { choices: [{ text: "ok" }] })
    @client.completion(prompt: "test", max_tokens: 64, temperature: 0.5, seed: 7)
    body = @srv.last_body
    assert_equal "test", body["prompt"]
    assert_equal 64,     body["max_tokens"]
    assert_in_delta 0.5, body["temperature"]
    assert_equal 7,      body["seed"]
  end

  # --- chat_completion ---

  def test_chat_completion
    @srv.set_response(body: {
      id: "chatcmpl-1", object: "chat.completion", model: "qwen3_dense",
      choices: [{
        index: 0,
        message: { role: "assistant", content: "Hi there!" },
        finish_reason: "stop"
      }],
      usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 }
    })
    resp = @client.chat_completion(
      messages: [{ role: "user", content: "Hello!" }],
      max_tokens: 64
    )
    assert_equal "chat.completion", resp["object"]
    assert_equal "Hi there!", resp.dig("choices", 0, "message", "content")
    assert_equal 3, resp.dig("usage", "completion_tokens")
    assert_equal "POST",                 @srv.last_method
    assert_equal "/v1/chat/completions", @srv.last_path
  end

  def test_chat_completion_messages_forwarded
    @srv.set_response(body: { choices: [{ message: { content: "ok" } }] })
    @client.chat_completion(
      messages: [
        { role: "system",    content: "You are AX." },
        { role: "user",      content: "Hello" },
      ],
      max_tokens: 32
    )
    msgs = @srv.last_body["messages"]
    assert_equal 2,        msgs.length
    assert_equal "system", msgs[0]["role"]
    assert_equal "user",   msgs[1]["role"]
  end

  # --- embeddings ---

  def test_embeddings
    @srv.set_response(body: {
      object: "list",
      data: [{ object: "embedding", embedding: [0.1, 0.2, 0.3], index: 0 }],
      model: "qwen3_embedding",
      usage: { prompt_tokens: 3, total_tokens: 3 }
    })
    resp = @client.embeddings(input: [1, 2, 3], pooling: "last", normalize: true)
    assert_equal [0.1, 0.2, 0.3], resp.dig("data", 0, "embedding")
    assert_equal 3, resp.dig("usage", "total_tokens")
    assert_equal "/v1/embeddings", @srv.last_path
  end

  # --- HTTP error ---

  def test_http_error_raised
    @srv.set_response(
      status: 400,
      body: JSON.generate({ error: { message: "bad request" } })
    )
    err = assert_raises(AxEngine::HttpError) do
      @client.completion(prompt: "x")
    end
    assert_equal 400,           err.status
    assert_equal "bad request", err.message
  end

  def test_http_error_fallback_message
    @srv.set_response(status: 500, body: "internal error")
    err = assert_raises(AxEngine::HttpError) { @client.health }
    assert_equal 500, err.status
    assert_match(/HTTP 500/, err.message)
  end

  # --- streaming chat ---

  def test_stream_chat_completion
    chunks = [
      %[data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}],
      %[data: {"id":"c1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}],
      "data: [DONE]"
    ]
    sse_body = chunks.map { |c| "#{c}\n\n" }.join
    @srv.set_response(content_type: "text/event-stream", body: sse_body)

    collected = []
    @client.stream_chat_completion(
      messages: [{ role: "user", content: "Hi" }],
      max_tokens: 16
    ) { |event| collected << event }

    texts = collected.map { |e| e.dig("data", "choices", 0, "delta", "content") }.compact
    assert_equal ["Hello", " world"], texts
    assert_equal "POST",                 @srv.last_method
    assert_equal "/v1/chat/completions", @srv.last_path
    assert_equal true, @srv.last_body["stream"]
  end

  # --- streaming completion ---

  def test_stream_completion
    sse_body = "data: {\"choices\":[{\"text\":\"Once\"}]}\n\ndata: [DONE]\n\n"
    @srv.set_response(content_type: "text/event-stream", body: sse_body)

    collected = []
    @client.stream_completion(prompt: "Once upon", max_tokens: 8) { |e| collected << e }
    text = collected.map { |e| e.dig("data", "choices", 0, "text") }.join
    assert_equal "Once", text
  end
end
