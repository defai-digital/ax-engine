require "json"
require "net/http"
require "uri"

# SseReader and the error types are only referenced at call time, so load them
# here: `require "ax_engine/client"` on its own must be streamable.
require_relative "errors"
require_relative "sse_reader"

module AxEngine
  # HTTP client for ax-engine-server.
  #
  # Example:
  #   client = AxEngine::Client.new
  #   resp = client.chat_completion(
  #     messages: [{ role: "user", content: "Hello!" }],
  #     max_tokens: 128
  #   )
  #   puts resp.dig("choices", 0, "message", "content")
  class Client
    DEFAULT_BASE_URL = "http://127.0.0.1:31418"

    # @param base_url [String]
    # @param headers  [Hash]  default headers added to every request
    # @param timeout  [Integer] open/read timeout in seconds (default 300)
    def initialize(base_url: DEFAULT_BASE_URL, headers: {}, timeout: 300)
      @base_url = normalize_base_url(base_url)
      @headers  = headers
      @timeout  = timeout
    end

    # GET /health
    def health
      get("/health")
    end

    # GET /v1/runtime
    def runtime
      get("/v1/runtime")
    end

    # GET /v1/models
    def models
      get("/v1/models")
    end

    # POST /v1/generate  (ax-engine native token-based API)
    def generate(request = nil, **kwargs)
      post("/v1/generate", request_body(request, kwargs))
    end

    # POST /v1/requests  (submit without blocking for completion)
    def submit(request = nil, **kwargs)
      post("/v1/requests", request_body(request, kwargs))
    end

    # GET /v1/requests/:id
    def request_snapshot(request_id)
      get("/v1/requests/#{request_id_segment(request_id)}")
    end

    # POST /v1/requests/:id/cancel
    def cancel(request_id)
      post("/v1/requests/#{request_id_segment(request_id)}/cancel", {})
    end

    # POST /v1/step
    def step(model: nil)
      path = model.nil? ? "/v1/step" : "/v1/step?model=#{URI.encode_www_form_component(model)}"
      post(path, {})
    end

    # POST /v1/completions  (OpenAI-compat text completion)
    def completion(request = nil, **kwargs)
      post("/v1/completions", request_body(request, kwargs))
    end

    # POST /v1/chat/completions  (OpenAI-compat chat completion)
    def chat_completion(request = nil, **kwargs)
      post("/v1/chat/completions", request_body(request, kwargs))
    end

    # POST /v1/embeddings
    def embeddings(request = nil, **kwargs)
      post("/v1/embeddings", request_body(request, kwargs))
    end

    # POST /v1/model/load — hot-swap or add a served model without server
    # restart. All request fields pass through (load_mode, load_policy,
    # make_default, ...); the response includes default_model_id on 6.9+.
    # request: { model_id:, model_path:, load_policy: nil, load_mode: nil }
    def load_model(request = nil, **kwargs)
      post("/v1/model/load", request_body(request, kwargs))
    end

    # POST /v1/model/unload — remove a retained model.
    def unload_model(request = nil, **kwargs)
      post("/v1/model/unload", request_body(request, kwargs))
    end

    # Stream POST /v1/generate/stream  — yields SSE event hashes.
    #
    #   client.stream_generate(input_tokens: [1, 2, 3], max_output_tokens: 32) do |event|
    #     puts event["data"]["delta_text"] if event["event"] == "step"
    #   end
    def stream_generate(request = nil, **kwargs, &block)
      stream("/v1/generate/stream", request_body(request, kwargs), &block)
    end

    # Stream POST /v1/completions (stream: true) — yields SSE event hashes.
    def stream_completion(request = nil, **kwargs, &block)
      stream("/v1/completions", streaming_body(request, kwargs), requires_done: true, &block)
    end

    # Stream POST /v1/chat/completions (stream: true) — yields SSE event hashes.
    def stream_chat_completion(request = nil, **kwargs, &block)
      stream("/v1/chat/completions", streaming_body(request, kwargs), requires_done: true, &block)
    end

    private

    # Force `stream: true` regardless of the caller's key style: merging a
    # symbol key does not override a string "stream" key in the request hash,
    # which would serialize duplicate JSON keys with an unpredictable winner.
    # Dup first: for a positional request with no kwargs, `request_body`
    # returns the caller's own hash, and deleting keys would mutate it (or
    # raise FrozenError on a frozen input).
    def streaming_body(request, kwargs)
      body = request_body(request, kwargs).dup
      body.delete("stream")
      body.delete(:stream)
      body.merge(stream: true)
    end

    # Accept both a positional Hash (Ruby 2.7 style) and true kwargs (Ruby 3+).
    def request_body(request, kwargs)
      if request.nil?
        kwargs
      elsif kwargs.empty?
        request
      else
        request.merge(kwargs)
      end
    end

    # Trim only pathname slashes; a trailing slash inside a query value is data.
    def normalize_base_url(base_url)
      uri = URI(base_url)
      uri.path = uri.path.sub(%r{/+\z}, "")
      uri.to_s
    rescue URI::InvalidURIError
      base_url.chomp("/")
    end

    # Append an API path while preserving the base path prefix and query.
    def build_uri(path)
      base = URI(@base_url)
      rel = URI(path)
      base.path = "#{base.path.sub(%r{/+\z}, "")}#{rel.path}"
      if rel.query && !rel.query.empty?
        base.query = base.query.nil? || base.query.empty? ? rel.query : "#{base.query}&#{rel.query}"
      end
      base
    rescue URI::InvalidURIError
      URI("#{@base_url}#{path}")
    end

    def request_id_segment(request_id)
      if request_id.is_a?(Integer) && request_id >= 0
        request_id.to_s
      elsif request_id.is_a?(String) && request_id.match?(/\A[0-9]+\z/)
        request_id
      else
        raise ArgumentError, "request_id must be a non-negative integer or a decimal string"
      end
    end

    def get(path)
      uri = build_uri(path)
      req = Net::HTTP::Get.new(uri)
      apply_headers(req)
      execute(uri, req)
    end

    def post(path, body)
      uri = build_uri(path)
      req = Net::HTTP::Post.new(uri)
      apply_headers(req)
      req["Content-Type"] = "application/json"
      req.body = JSON.generate(body)
      execute(uri, req)
    end

    def stream(path, body, requires_done: false)
      uri = build_uri(path)
      req = Net::HTTP::Post.new(uri)
      apply_headers(req)
      req["Content-Type"] = "application/json"
      req["Accept"]        = "text/event-stream"
      req.body             = JSON.generate(body)

      reader = SseReader.new
      saw_response = false
      use_ssl = uri.scheme == "https"

      # Catch outside Net::HTTP.start so its ensure closes the connection
      # immediately, even when a peer keeps sending after the sentinel.
      done_tag = Object.new
      catch(done_tag) do
        Net::HTTP.start(
          uri.host,
          uri.port,
          open_timeout: @timeout,
          read_timeout: @timeout,
          use_ssl: use_ssl
        ) do |http|
          http.request(req) do |response|
            unless (200..299).cover?(response.code.to_i)
              # `response.body` is nil inside the streaming block until the
              # body is read; read it explicitly so the HttpError keeps the
              # server's error envelope (mirrors the non-streaming `execute`).
              raise_on_error(response, path, body: response.read_body)
            end
            response.read_body do |chunk|
              reader.feed(chunk) do |event|
                if event["event"] == "error"
                  message = extract_error_message(event["data"])
                  raise StreamError.new(message, payload: event["data"])
                end
                saw_response = true if event["event"] == "response"
                yield event
              end
              if reader.done?
                unless requires_done || saw_response
                  raise StreamError.new("stream ended without terminal response")
                end
                throw done_tag
              end
            end
            reader.flush
            if requires_done
              raise StreamError.new("stream ended without [DONE]")
            elsif !saw_response
              raise StreamError.new("stream ended without terminal response")
            end
          end
        end
      end
    end

    def execute(uri, req)
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = (uri.scheme == "https")
      http.open_timeout = @timeout
      http.read_timeout = @timeout
      response = http.request(req)
      raise_on_error(response, req.path)
      JSON.parse(response.body)
    end

    def apply_headers(req)
      @headers.each { |k, v| req[k.to_s] = v }
    end

    def extract_error_message(data)
      return "unknown error" if data.nil?

      if data.is_a?(Hash) && data["error"].is_a?(Hash) && data["error"]["message"]
        data["error"]["message"]
      elsif data.is_a?(String)
        data
      else
        "unknown error"
      end
    end

    def raise_on_error(response, path, body: nil)
      code = response.code.to_i
      return if code >= 200 && code < 300

      # Streaming callers pass the body they already read; `response.body` is
      # nil inside Net::HTTP's streaming block until the body is read.
      body = response.body if body.nil?
      payload = begin
        JSON.parse(body)
      rescue StandardError
        body
      end

      message = if payload.is_a?(Hash)
        payload.dig("error", "message") || "HTTP #{code}"
      else
        "HTTP #{code}"
      end

      raise HttpError.new(message, status: code, payload: payload)
    end
  end
end
