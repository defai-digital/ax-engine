require_relative "lib/ax_engine/version"

Gem::Specification.new do |spec|
  spec.name        = "ax-engine-sdk"
  spec.version     = AxEngine::VERSION
  spec.summary     = "Ruby SDK for AX Engine v6 — local HTTP inference server"
  spec.description = "Zero-dependency Ruby client for ax-engine-server. " \
                     "Supports native generate, OpenAI-compatible completions, " \
                     "chat, embeddings, and SSE streaming."
  spec.license     = "Apache-2.0"

  spec.required_ruby_version = ">= 2.7"

  spec.files = Dir["lib/**/*.rb", "ax_engine.gemspec"]

  spec.metadata["source_code_uri"] = "https://github.com/defai-digital/ax-engine"
end
