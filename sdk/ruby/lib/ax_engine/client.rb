require "json"
require "net/http"
require "uri"

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
    DEFAULT_BASE_URL = "http://127.0.0.1:8080"

    # @param base_url [String]
    # @param headers  [Hash]  default headers added to every request
    # @param timeout  [Integer] open/read timeout in seconds (default 300)
    def initialize(base_url: DEFAULT_BASE_URL, headers: {}, timeout: 300)
      @base_url = base_url.chomp("/")
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
    def generate(request)
      post("/v1/generate", request)
    end

    # POST /v1/requests  (submit without blocking for completion)
    def submit(request)
      post("/v1/requests", request)
    end

    # GET /v1/requests/:id
    def request_snapshot(request_id)
      get("/v1/requests/#{request_id}")
    end

    # POST /v1/requests/:id/cancel
    def cancel(request_id)
      post("/v1/requests/#{request_id}/cancel", {})
    end

    # POST /v1/step
    def step
      post("/v1/step", {})
    end

    # POST /v1/completions  (OpenAI-compat text completion)
    def completion(request)
      post("/v1/completions", request)
    end

    # POST /v1/chat/completions  (OpenAI-compat chat completion)
    def chat_completion(request)
      post("/v1/chat/completions", request)
    end

    # POST /v1/embeddings
    def embeddings(request)
      post("/v1/embeddings", request)
    end

    # Stream POST /v1/generate/stream  — yields SSE event hashes.
    #
    #   client.stream_generate(input_tokens: [1, 2, 3], max_output_tokens: 32) do |event|
    #     puts event["data"]["delta_text"] if event["event"] == "step"
    #   end
    def stream_generate(request, &block)
      stream("/v1/generate/stream", request, &block)
    end

    # Stream POST /v1/completions (stream: true) — yields SSE event hashes.
    def stream_completion(request, &block)
      stream("/v1/completions", request.merge(stream: true), &block)
    end

    # Stream POST /v1/chat/completions (stream: true) — yields SSE event hashes.
    def stream_chat_completion(request, &block)
      stream("/v1/chat/completions", request.merge(stream: true), &block)
    end

    private

    def get(path)
      uri = URI("#{@base_url}#{path}")
      req = Net::HTTP::Get.new(uri)
      apply_headers(req)
      execute(uri, req)
    end

    def post(path, body)
      uri = URI("#{@base_url}#{path}")
      req = Net::HTTP::Post.new(uri)
      apply_headers(req)
      req["Content-Type"] = "application/json"
      req.body = JSON.generate(body)
      execute(uri, req)
    end

    def stream(path, body)
      uri = URI("#{@base_url}#{path}")
      req = Net::HTTP::Post.new(uri)
      apply_headers(req)
      req["Content-Type"] = "application/json"
      req["Accept"]        = "text/event-stream"
      req.body             = JSON.generate(body)

      reader = SseReader.new

      Net::HTTP.start(uri.host, uri.port, open_timeout: @timeout, read_timeout: @timeout) do |http|
        http.request(req) do |response|
          raise_on_error(response, path)
          response.read_body do |chunk|
            reader.feed(chunk) { |event| yield event }
          end
        end
      end
    end

    def execute(uri, req)
      http = Net::HTTP.new(uri.host, uri.port)
      http.open_timeout = @timeout
      http.read_timeout = @timeout
      response = http.request(req)
      raise_on_error(response, req.path)
      JSON.parse(response.body)
    end

    def apply_headers(req)
      @headers.each { |k, v| req[k.to_s] = v }
    end

    def raise_on_error(response, path)
      code = response.code.to_i
      return if code >= 200 && code < 300

      payload = begin
        JSON.parse(response.body)
      rescue StandardError
        response.body
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
