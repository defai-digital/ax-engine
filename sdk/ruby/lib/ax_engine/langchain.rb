# AxEngine::Client loads its own dependencies, so the adapter only needs it.
require_relative "client"

# LangChain integration for AX Engine.
#
# Provides two classes compatible with the langchainrb gem:
#
#   AxEngine::Langchain::ChatModel  — wraps /v1/chat/completions
#   AxEngine::Langchain::LLM        — wraps /v1/completions
#
# Requires the langchainrb gem (gem install langchainrb).
# Uses stdlib HTTP only for inference — no extra HTTP libraries needed.
#
# Example:
#   require "ax_engine/langchain"
#
#   chat = AxEngine::Langchain::ChatModel.new(max_tokens: 256)
#   response = chat.chat(messages: [{ role: "user", content: "Hello!" }])
#   puts response.chat_completion
#
module AxEngine
  module Langchain
    def self._require_langchain!
      require "langchain"
      _define_text_completion_response!
    rescue LoadError
      raise LoadError,
        "ax_engine/langchain requires the langchainrb gem. " \
        "Install it with: gem install langchainrb"
    end

    # Response type for /v1/completions.
    #
    # The released langchainrb OpenAIResponse#completion reads
    # choices[0].message.content, which a text completion never carries, so it
    # returns nil. This subclass reads choices[0].text and inherits model,
    # created_at and token usage unchanged.
    #
    # Built lazily because langchainrb is only loaded when a model is
    # constructed; this file must be requirable without the gem.
    def self._define_text_completion_response!
      return if const_defined?(:TextCompletionResponse, false)

      klass = Class.new(::Langchain::LLM::OpenAIResponse) do
        def completion
          completions&.dig(0, "text")
        end
      end
      const_set(:TextCompletionResponse, klass)
    end

    def self.text_completion_response_class
      return TextCompletionResponse if const_defined?(:TextCompletionResponse, false)

      _require_langchain!
      TextCompletionResponse
    end

    # Surfaces server `error` SSE events as exceptions instead of silently
    # skipping them (their payload has no choices, so the delta digs miss).
    def self.raise_on_stream_error!(event)
      return unless event["event"] == "error"

      data = event["data"]
      message = data.is_a?(Hash) ? data.dig("error", "message") : data.to_s
      raise AxEngine::StreamError.new(message || "stream error", payload: data)
    end

    # Shared parameter helpers.
    module Params
      SAMPLING_KEYS = %i[
        max_tokens temperature top_p top_k min_p
        repetition_penalty frequency_penalty presence_penalty stop seed
      ].freeze

      # Request body options that must survive both the constructor and a
      # single call. Kept explicit on purpose: arbitrary kwargs are never
      # forwarded, and `stream` belongs to AxEngine::Client's transport, not
      # to a request body.
      REQUEST_KEYS = (SAMPLING_KEYS + %i[model metadata response_format tools tool_choice]).freeze

      def initialize(base_url: AxEngine::Client::DEFAULT_BASE_URL,
                     headers: {}, timeout: 300, **defaults)
        @client   = AxEngine::Client.new(base_url: base_url, headers: headers, timeout: timeout)
        @defaults = defaults.slice(*self.class::REQUEST_KEYS)
      end

      private

      # Per-call options override constructor defaults.
      def build_params(overrides)
        @defaults.merge(overrides.slice(*self.class::REQUEST_KEYS))
      end
    end

    # LangChain-compatible chat model backed by /v1/chat/completions.
    #
    # Usage:
    #   chat = AxEngine::Langchain::ChatModel.new(
    #     base_url: "http://127.0.0.1:31418",
    #     max_tokens: 256,
    #     temperature: 0.7,
    #   )
    #
    #   # Blocking
    #   resp = chat.chat(messages: [{ role: "user", content: "Hello!" }])
    #   puts resp.chat_completion
    #
    #   # Streaming (yields delta strings; returns the assembled response,
    #   # including any tool calls, which never arrive as content deltas)
    #   resp = chat.chat(messages: [...], stream: true) { |delta| print delta }
    #   resp.tool_calls
    class ChatModel
      include Params

      # /v1/completions does not define parallel_tool_calls.
      REQUEST_KEYS = (Params::REQUEST_KEYS + %i[parallel_tool_calls]).freeze

      def initialize(**kwargs)
        AxEngine::Langchain._require_langchain!
        super(**kwargs)
      end

      # Compatible with langchainrb's Langchain::LLM::Base#chat interface.
      #
      # @param messages [Array<Hash>]  e.g. [{role: "user", content: "Hi"}]
      # @param stream   [Boolean]      if true, yields content delta strings and
      #                                returns the response assembled from the
      #                                stream (content, tool_calls, finish_reason)
      # @param kwargs   [Hash]         request overrides (max_tokens, model, metadata,
      #                                response_format, tools, tool_choice,
      #                                parallel_tool_calls, …)
      def chat(messages:, stream: false, **kwargs, &block)
        if stream && block.nil?
          raise ArgumentError, "#{self.class}#chat requires a block when stream: true"
        end

        params = build_params(kwargs).merge(messages: messages)

        if stream
          assembled = StreamedChat.new
          @client.stream_chat_completion(params) do |event|
            Langchain.raise_on_stream_error!(event)
            data = event["data"]
            next unless data.is_a?(Hash)

            assembled.add(data)
            delta = data.dig("choices", 0, "delta", "content")
            block.call(delta) if delta
          end
          raw = assembled.to_completion
          ::Langchain::LLM::OpenAIResponse.new(raw, model: raw["model"])
        else
          raw = @client.chat_completion(params)
          ::Langchain::LLM::OpenAIResponse.new(raw, model: raw["model"])
        end
      end
    end

    # Folds chat completion chunks back into one non-streaming response body.
    # Tool calls stream as `delta.tool_calls` fragments keyed by index (id and
    # name on the first fragment, arguments concatenated), never as content.
    class StreamedChat
      def initialize
        @model = nil
        @content = +""
        @tool_calls = {}
        @finish_reason = nil
        @usage = nil
      end

      def add(chunk)
        @model ||= chunk["model"]
        @usage = chunk["usage"] if chunk["usage"]
        choice = chunk.dig("choices", 0)
        return unless choice.is_a?(Hash)

        @finish_reason = choice["finish_reason"] if choice["finish_reason"]
        delta = choice["delta"] || {}
        @content << delta["content"] if delta["content"]
        Array(delta["tool_calls"]).each do |fragment|
          call = (@tool_calls[fragment["index"] || 0] ||= {
            "id" => nil, "type" => "function", "function" => { "name" => +"", "arguments" => +"" }
          })
          call["id"] ||= fragment["id"]
          call["type"] = fragment["type"] if fragment["type"]
          function = fragment["function"] || {}
          call["function"]["name"] << function["name"] if function["name"]
          call["function"]["arguments"] << function["arguments"] if function["arguments"]
        end
      end

      def to_completion
        message = { "role" => "assistant", "content" => @content }
        message["tool_calls"] = @tool_calls.sort.map(&:last) unless @tool_calls.empty?
        body = {
          "object" => "chat.completion",
          "model" => @model,
          "choices" => [{ "index" => 0, "message" => message, "finish_reason" => @finish_reason }]
        }
        body["usage"] = @usage if @usage
        body
      end
    end

    # LangChain-compatible text LLM backed by /v1/completions.
    #
    # Usage:
    #   llm = AxEngine::Langchain::LLM.new(
    #     base_url: "http://127.0.0.1:31418",
    #     max_tokens: 128,
    #   )
    #
    #   puts llm.complete(prompt: "Once upon a time").completion
    #
    #   llm.complete(prompt: "Once upon a time", stream: true) { |t| print t }
    class LLM
      include Params

      def initialize(**kwargs)
        AxEngine::Langchain._require_langchain!
        super(**kwargs)
      end

      # Compatible with langchainrb's Langchain::LLM::Base#complete interface.
      #
      # @param prompt  [String]
      # @param stream  [Boolean]  if true, yields delta strings and returns nil
      # @param kwargs  [Hash]     request overrides (max_tokens, model, …)
      def complete(prompt:, stream: false, **kwargs, &block)
        if stream && block.nil?
          raise ArgumentError, "#{self.class}#complete requires a block when stream: true"
        end

        params = build_params(kwargs).merge(prompt: prompt)

        if stream
          @client.stream_completion(params) do |event|
            Langchain.raise_on_stream_error!(event)
            data = event["data"]
            next unless data.is_a?(Hash)

            text = data.dig("choices", 0, "text")
            block.call(text) if text && !text.empty?
          end
          nil
        else
          raw = @client.completion(params)
          AxEngine::Langchain.text_completion_response_class.new(raw, model: raw["model"])
        end
      end
    end
  end
end
