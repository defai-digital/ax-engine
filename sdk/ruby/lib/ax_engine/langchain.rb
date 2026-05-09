require "json"
require_relative "client"

# LangChain integration for AX Engine.
#
# Provides two classes compatible with the langchain-rb gem:
#
#   AxEngine::Langchain::ChatModel  — wraps /v1/chat/completions
#   AxEngine::Langchain::LLM        — wraps /v1/completions
#
# Requires the langchain-rb gem (gem install langchain-rb).
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
    rescue LoadError
      raise LoadError,
        "ax_engine/langchain requires the langchain-rb gem. " \
        "Install it with: gem install langchain-rb"
    end

    # Shared parameter helpers.
    module Params
      SAMPLING_KEYS = %i[
        max_tokens temperature top_p top_k min_p
        repetition_penalty stop seed
      ].freeze

      def initialize(base_url: AxEngine::Client::DEFAULT_BASE_URL,
                     headers: {}, timeout: 300, **defaults)
        @client   = AxEngine::Client.new(base_url: base_url, headers: headers, timeout: timeout)
        @defaults = defaults.slice(*SAMPLING_KEYS)
      end

      private

      def build_params(overrides)
        @defaults.merge(overrides.slice(*SAMPLING_KEYS))
      end
    end

    # LangChain-compatible chat model backed by /v1/chat/completions.
    #
    # Usage:
    #   chat = AxEngine::Langchain::ChatModel.new(
    #     base_url: "http://127.0.0.1:8080",
    #     max_tokens: 256,
    #     temperature: 0.7,
    #   )
    #
    #   # Blocking
    #   resp = chat.chat(messages: [{ role: "user", content: "Hello!" }])
    #   puts resp.chat_completion
    #
    #   # Streaming (yields delta strings)
    #   chat.chat(messages: [...], stream: true) { |delta| print delta }
    class ChatModel
      include Params

      def initialize(**kwargs)
        AxEngine::Langchain._require_langchain!
        super(**kwargs)
      end

      # Compatible with langchain-rb's Langchain::LLM::Base#chat interface.
      #
      # @param messages [Array<Hash>]  e.g. [{role: "user", content: "Hi"}]
      # @param stream   [Boolean]      if true, yields delta strings and returns nil
      # @param kwargs   [Hash]         sampling overrides (max_tokens, temperature, …)
      def chat(messages:, stream: false, **kwargs, &block)
        params = build_params(kwargs).merge(messages: messages)

        if stream && block_given?
          @client.stream_chat_completion(params) do |event|
            delta = event.dig("data", "choices", 0, "delta", "content")
            block.call(delta) if delta
          end
          nil
        else
          raw = @client.chat_completion(params)
          ::Langchain::LLM::OpenAIResponse.new(raw, model: raw["model"])
        end
      end
    end

    # LangChain-compatible text LLM backed by /v1/completions.
    #
    # Usage:
    #   llm = AxEngine::Langchain::LLM.new(
    #     base_url: "http://127.0.0.1:8080",
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

      # Compatible with langchain-rb's Langchain::LLM::Base#complete interface.
      #
      # @param prompt  [String]
      # @param stream  [Boolean]  if true, yields delta strings and returns nil
      # @param kwargs  [Hash]     sampling overrides
      def complete(prompt:, stream: false, **kwargs, &block)
        params = build_params(kwargs).merge(prompt: prompt)

        if stream && block_given?
          @client.stream_completion(params) do |event|
            text = event.dig("data", "choices", 0, "text")
            block.call(text) if text && !text.empty?
          end
          nil
        else
          raw = @client.completion(params)
          ::Langchain::LLM::OpenAIResponse.new(raw, model: raw["model"])
        end
      end
    end
  end
end
