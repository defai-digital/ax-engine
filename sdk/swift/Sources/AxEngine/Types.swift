import Foundation

// MARK: - Sampling

public struct GenerateSampling: Encodable, Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var repetitionPenalty: Double?
    public var seed: Int?

    public init(
        temperature: Double? = nil, topP: Double? = nil, topK: Int? = nil,
        repetitionPenalty: Double? = nil, seed: Int? = nil
    ) {
        self.temperature = temperature; self.topP = topP; self.topK = topK
        self.repetitionPenalty = repetitionPenalty; self.seed = seed
    }
}

// MARK: - Native ax-engine generate API

public enum Gemma4UnifiedModality: String, Encodable, Sendable {
    case image
    case audio
    case video
}

public struct Gemma4UnifiedTokenSpan: Encodable, Sendable {
    public var modality: Gemma4UnifiedModality
    public var placeholderIndex: Int
    public var replacementStart: Int
    public var softTokenCount: Int
    public var replacementTokenCount: Int

    public init(
        modality: Gemma4UnifiedModality, placeholderIndex: Int, replacementStart: Int,
        softTokenCount: Int, replacementTokenCount: Int
    ) {
        self.modality = modality; self.placeholderIndex = placeholderIndex
        self.replacementStart = replacementStart; self.softTokenCount = softTokenCount
        self.replacementTokenCount = replacementTokenCount
    }
}

public struct Gemma4UnifiedSoftTokenRange: Encodable, Sendable {
    public var start: Int
    public var softTokenCount: Int

    public init(start: Int, softTokenCount: Int) {
        self.start = start; self.softTokenCount = softTokenCount
    }
}

public struct Gemma4UnifiedImageRuntimeInput: Encodable, Sendable {
    public var span: Gemma4UnifiedTokenSpan
    public var pixelValues: [Double]
    public var pixelPositionIds: [[Int]]

    public init(span: Gemma4UnifiedTokenSpan, pixelValues: [Double], pixelPositionIds: [[Int]]) {
        self.span = span; self.pixelValues = pixelValues; self.pixelPositionIds = pixelPositionIds
    }
}

public struct Gemma4UnifiedAudioRuntimeInput: Encodable, Sendable {
    public var span: Gemma4UnifiedTokenSpan
    public var inputFeatures: [Double]
    public var frameCount: Int
    public var featureCount: Int

    public init(
        span: Gemma4UnifiedTokenSpan, inputFeatures: [Double], frameCount: Int, featureCount: Int
    ) {
        self.span = span; self.inputFeatures = inputFeatures
        self.frameCount = frameCount; self.featureCount = featureCount
    }
}

public struct Gemma4UnifiedVideoRuntimeInput: Encodable, Sendable {
    public var span: Gemma4UnifiedTokenSpan
    public var softTokenRanges: [Gemma4UnifiedSoftTokenRange]?
    public var pixelValues: [Double]
    public var pixelPositionIds: [[Int]]
    public var frameCount: Int

    public init(
        span: Gemma4UnifiedTokenSpan, softTokenRanges: [Gemma4UnifiedSoftTokenRange]? = nil,
        pixelValues: [Double], pixelPositionIds: [[Int]], frameCount: Int
    ) {
        self.span = span; self.softTokenRanges = softTokenRanges
        self.pixelValues = pixelValues; self.pixelPositionIds = pixelPositionIds
        self.frameCount = frameCount
    }
}

public struct Gemma4UnifiedRuntimeInputs: Encodable, Sendable {
    public var images: [Gemma4UnifiedImageRuntimeInput]?
    public var audios: [Gemma4UnifiedAudioRuntimeInput]?
    public var videos: [Gemma4UnifiedVideoRuntimeInput]?

    public init(
        images: [Gemma4UnifiedImageRuntimeInput]? = nil,
        audios: [Gemma4UnifiedAudioRuntimeInput]? = nil,
        videos: [Gemma4UnifiedVideoRuntimeInput]? = nil
    ) {
        self.images = images; self.audios = audios; self.videos = videos
    }
}

public struct RequestMultimodalInputs: Encodable, Sendable {
    public var gemma4Unified: Gemma4UnifiedRuntimeInputs?

    public init(gemma4Unified: Gemma4UnifiedRuntimeInputs? = nil) {
        self.gemma4Unified = gemma4Unified
    }
}

public struct PreviewGenerateRequest: Encodable, Sendable {
    public var model: String?
    public var inputTokens: [Int]?
    public var inputText: String?
    public var multimodalInputs: RequestMultimodalInputs?
    public var maxOutputTokens: Int?
    public var sampling: GenerateSampling?
    public var metadata: String?

    public init(
        model: String? = nil, inputTokens: [Int]? = nil, inputText: String? = nil,
        multimodalInputs: RequestMultimodalInputs? = nil, maxOutputTokens: Int? = nil,
        sampling: GenerateSampling? = nil, metadata: String? = nil
    ) {
        self.model = model; self.inputTokens = inputTokens; self.inputText = inputText
        self.multimodalInputs = multimodalInputs; self.maxOutputTokens = maxOutputTokens
        self.sampling = sampling; self.metadata = metadata
    }
}

public struct GenerateRoute: Decodable, Sendable {
    public var executionPlan: String?
    public var attentionRoute: String?
    public var kvMode: String?
    public var prefixCachePath: String?
    public var barrierMode: String?
}

public struct GenerateResponse: Decodable, Sendable {
    public var requestId: Int
    public var modelId: String
    public var promptTokens: [Int]
    public var promptText: String?
    public var outputTokens: [Int]
    public var outputText: String?
    public var status: String
    public var finishReason: String?
    public var stepCount: Int
    public var ttftStep: Int?
    public var route: GenerateRoute
}

public struct RequestReport: Decodable, Sendable {
    public var requestId: Int
    public var modelId: String
    public var state: String
    public var promptTokens: [Int]
    public var processedPromptTokens: Int
    public var outputTokens: [Int]
    public var promptLen: Int
    public var outputLen: Int
    public var maxOutputTokens: Int
    public var cancelRequested: Bool
    public var route: GenerateRoute
    public var finishReason: String?
    public var terminalStopReason: String?
}

public struct StepReport: Decodable, Sendable {
    public var stepId: Int?
    public var scheduledRequests: Int
    public var scheduledTokens: Int
    public var ttftEvents: Int
    public var prefixHits: Int
    public var kvUsageBlocks: Int
    public var evictions: Int
    public var cpuTimeUs: Int
    public var runnerTimeUs: Int
}

// MARK: - OpenAI-compatible chat

public struct OpenAiChatMessage: Codable, Sendable {
    public var role: String
    /// Nil is valid for assistant messages that only carry `toolCalls`.
    public var content: String?
    /// Assistant tool calls echoed back into the conversation after a tool turn.
    public var toolCalls: [OpenAiToolCall]?
    /// Required on `role: "tool"` result messages.
    public var toolCallId: String?
    public var name: String?

    public init(
        role: String, content: String? = nil,
        toolCalls: [OpenAiToolCall]? = nil,
        toolCallId: String? = nil, name: String? = nil
    ) {
        self.role = role; self.content = content
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId; self.name = name
    }
}

/// A completed tool call (assistant echo and response messages).
public struct OpenAiToolCall: Codable, Sendable {
    public var id: String
    public var type: String
    public var function: OpenAiFunctionCall

    public init(id: String, type: String = "function", function: OpenAiFunctionCall) {
        self.id = id; self.type = type; self.function = function
    }
}

public struct OpenAiFunctionCall: Codable, Sendable {
    public var name: String
    /// JSON-encoded arguments string, exactly as the model produced it.
    public var arguments: String

    public init(name: String, arguments: String) {
        self.name = name; self.arguments = arguments
    }
}

/// A streamed tool-call fragment with a stream-wide index.
public struct OpenAiToolCallDelta: Decodable, Sendable {
    public var index: Int
    public var id: String?
    public var type: String?
    public var function: OpenAiFunctionCallDelta?
}

public struct OpenAiFunctionCallDelta: Decodable, Sendable {
    public var name: String?
    public var arguments: String?
}

public struct OpenAiChatCompletionRequest: Encodable, Sendable {
    public var model: String?
    public var messages: [OpenAiChatMessage]
    public var inputTokens: [Int]?
    public var maxTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?
    public var stop: [String]?
    public var seed: Int?
    public var stream: Bool?
    public var metadata: String?
    public var multimodalInputs: RequestMultimodalInputs?
    public var logprobs: Bool?
    public var topLogprobs: Int?
    // Note: free-form `tools` / `tool_choice` / `response_format` /
    // `reasoning` request fields are intentionally not typed yet:
    // Foundation's .convertToSnakeCase also rewrites *dictionary* keys, so
    // arbitrary user schema keys (e.g. "userId") would be silently
    // corrupted. Typed support needs the explicit-CodingKeys refactor
    // tracked in docs/sdk/swift.md.

    public init(
        model: String? = nil, messages: [OpenAiChatMessage],
        inputTokens: [Int]? = nil,
        maxTokens: Int? = nil, temperature: Double? = nil,
        topP: Double? = nil, topK: Int? = nil, minP: Double? = nil,
        repetitionPenalty: Double? = nil, stop: [String]? = nil,
        seed: Int? = nil, metadata: String? = nil,
        multimodalInputs: RequestMultimodalInputs? = nil,
        logprobs: Bool? = nil, topLogprobs: Int? = nil
    ) {
        self.model = model; self.messages = messages; self.maxTokens = maxTokens
        self.inputTokens = inputTokens
        self.temperature = temperature; self.topP = topP; self.topK = topK
        self.minP = minP; self.repetitionPenalty = repetitionPenalty
        self.stop = stop; self.seed = seed; self.metadata = metadata
        self.multimodalInputs = multimodalInputs
        self.logprobs = logprobs; self.topLogprobs = topLogprobs
    }
}

public struct OpenAiUsage: Decodable, Sendable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int
    /// Prefix-cache reuse in the OpenAI prompt-caching shape.
    public var promptTokensDetails: OpenAiPromptTokensDetails?
}

public struct OpenAiPromptTokensDetails: Decodable, Sendable {
    public var cachedTokens: Int
}

public struct OpenAiChatMessageResponse: Decodable, Sendable {
    public var role: String
    public var content: String?
    /// Separated reasoning when the request opted in.
    public var reasoningContent: String?
    public var toolCalls: [OpenAiToolCall]?
}

public struct OpenAiChatCompletionChoice: Decodable, Sendable {
    public var index: Int
    public var message: OpenAiChatMessageResponse
    public var finishReason: String?
}

public struct OpenAiChatCompletionResponse: Decodable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [OpenAiChatCompletionChoice]
    public var usage: OpenAiUsage?
}

public struct OpenAiChatDelta: Decodable, Sendable {
    public var role: String?
    public var content: String?
    public var reasoningContent: String?
    public var toolCalls: [OpenAiToolCallDelta]?
}

public struct OpenAiChatCompletionChunkChoice: Decodable, Sendable {
    public var index: Int
    public var delta: OpenAiChatDelta
    public var finishReason: String?
}

public struct OpenAiChatCompletionChunk: Decodable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [OpenAiChatCompletionChunkChoice]
}

// MARK: - OpenAI-compatible text completion

public enum OpenAiCompletionPrompt: Encodable, Sendable {
    case text(String)
    case textBatch([String])
    case tokens([Int])

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let value):
            try container.encode(value)
        case .textBatch(let value):
            try container.encode(value)
        case .tokens(let value):
            try container.encode(value)
        }
    }
}

public struct OpenAiCompletionRequest: Encodable, Sendable {
    public var model: String?
    public var prompt: OpenAiCompletionPrompt
    public var maxTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?
    public var stop: [String]?
    public var seed: Int?
    public var stream: Bool?
    public var metadata: String?
    public var multimodalInputs: RequestMultimodalInputs?

    public init(
        model: String? = nil, prompt: String,
        maxTokens: Int? = nil, temperature: Double? = nil,
        topP: Double? = nil, topK: Int? = nil, minP: Double? = nil,
        repetitionPenalty: Double? = nil, stop: [String]? = nil,
        seed: Int? = nil, metadata: String? = nil,
        multimodalInputs: RequestMultimodalInputs? = nil
    ) {
        self.model = model; self.prompt = .text(prompt); self.maxTokens = maxTokens
        self.temperature = temperature; self.topP = topP; self.topK = topK
        self.minP = minP; self.repetitionPenalty = repetitionPenalty
        self.stop = stop; self.seed = seed; self.metadata = metadata
        self.multimodalInputs = multimodalInputs
    }

    public init(
        model: String? = nil, promptTokens: [Int],
        maxTokens: Int? = nil, temperature: Double? = nil,
        topP: Double? = nil, topK: Int? = nil, minP: Double? = nil,
        repetitionPenalty: Double? = nil, stop: [String]? = nil,
        seed: Int? = nil, metadata: String? = nil,
        multimodalInputs: RequestMultimodalInputs? = nil
    ) {
        self.model = model; self.prompt = .tokens(promptTokens); self.maxTokens = maxTokens
        self.temperature = temperature; self.topP = topP; self.topK = topK
        self.minP = minP; self.repetitionPenalty = repetitionPenalty
        self.stop = stop; self.seed = seed; self.metadata = metadata
        self.multimodalInputs = multimodalInputs
    }
}

public struct OpenAiCompletionChoice: Decodable, Sendable {
    public var index: Int
    public var text: String
    public var finishReason: String?
}

public struct OpenAiCompletionResponse: Decodable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [OpenAiCompletionChoice]
    public var usage: OpenAiUsage?
}

public struct OpenAiCompletionChunkChoice: Decodable, Sendable {
    public var index: Int
    public var text: String
    public var finishReason: String?
}

public struct OpenAiCompletionChunk: Decodable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [OpenAiCompletionChunkChoice]
}

// MARK: - Embeddings

public enum OpenAiEmbeddingInput: Encodable, Sendable {
    case tokens([Int])
    case batch([[Int]])

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .tokens(let tokens):
            try container.encode(tokens)
        case .batch(let batch):
            try container.encode(batch)
        }
    }
}

public struct OpenAiEmbeddingRequest: Encodable, Sendable {
    public var model: String?
    public var input: OpenAiEmbeddingInput
    public var encodingFormat: String?
    public var pooling: String?
    public var normalize: Bool?

    public init(
        model: String? = nil, input: [Int],
        encodingFormat: String? = nil, pooling: String? = nil, normalize: Bool? = nil
    ) {
        self.model = model; self.input = .tokens(input); self.encodingFormat = encodingFormat
        self.pooling = pooling; self.normalize = normalize
    }

    public init(
        model: String? = nil, input: [[Int]],
        encodingFormat: String? = nil, pooling: String? = nil, normalize: Bool? = nil
    ) {
        self.model = model; self.input = .batch(input); self.encodingFormat = encodingFormat
        self.pooling = pooling; self.normalize = normalize
    }
}

public struct OpenAiEmbeddingObject: Decodable, Sendable {
    public var object: String
    public var embedding: [Double]
    public var index: Int
}

public struct OpenAiEmbeddingUsage: Decodable, Sendable {
    public var promptTokens: Int
    public var totalTokens: Int
}

public struct OpenAiEmbeddingResponse: Decodable, Sendable {
    public var object: String
    public var data: [OpenAiEmbeddingObject]
    public var model: String
    public var usage: OpenAiEmbeddingUsage
}

// MARK: - Health / models

public struct HealthResponse: Decodable, Sendable {
    public var status: String
    public var service: String?
    public var modelId: String?
    /// Every loaded model id (multi-model serving); nil on pre-6.9 servers.
    public var models: [String]?
    public var runtime: RuntimeInfo?
}

public struct ModelCard: Decodable, Sendable {
    public var id: String
    public var object: String
    public var ownedBy: String
    public var capabilities: ModelCapabilities?
    public var limit: ModelLimit?
    public var contextLength: UInt32?
    public var maxOutputTokens: UInt32?
    public var axEngine: AxEngineModelMetadata?
    public var runtime: RuntimeInfo?
}

public struct ModelCapabilities: Decodable, Sendable {
    public var temperature: Bool
    public var reasoning: Bool
    public var attachment: Bool
    public var toolcall: Bool
    public var input: ModelModalities
    public var output: ModelModalities
    public var interleaved: Bool
}

public struct ModelModalities: Decodable, Sendable {
    public var text: Bool
    public var audio: Bool
    public var image: Bool
    public var video: Bool
    public var pdf: Bool
}

public struct ModelLimit: Decodable, Sendable {
    public var context: UInt32
    public var output: UInt32
}

public struct AxEngineModelMetadata: Decodable, Sendable {
    public var nativeGenerateSupported: Bool
    public var openaiCompletionsSupported: Bool
    public var openaiChatCompletionsSupported: Bool
    public var openaiToolCallingSupported: Bool
    public var openaiTextInputSupported: Bool
    public var nativeMultimodalInputSupported: Bool
    public var gemma4UnifiedMultimodalInputSupported: Bool
    public var openaiTokenizedMultimodalInputSupported: Bool
    public var primaryUse: String
    public var chatDefault: Bool
    public var codingSupported: Bool
    public var codingOnly: Bool
}

// MARK: - Runtime info

public struct RuntimeInfo: Decodable, Sendable {
    public var selectedBackend: String
    public var supportTier: String
    public var resolutionPolicy: String
    public var capabilities: CapabilityReport
    public var fallbackReason: String?
    public var host: HostInfo
    public var metalToolchain: MetalToolchainInfo
    public var mlxRuntime: MlxRuntimeInfo?
    public var mlxModel: MlxModelInfo?
}

public struct CapabilityReport: Decodable, Sendable {
    public var textGeneration: Bool
    public var tokenStreaming: Bool
    public var deterministicMode: Bool
    public var prefixReuse: Bool
    public var longContextValidation: String
    public var benchmarkMetrics: String
}

public struct HostInfo: Decodable, Sendable {
    public var os: String
    public var arch: String
    public var detectedSoc: String?
    public var supportedMlxRuntime: Bool
    public var unsupportedHostOverrideActive: Bool
}

public struct ToolStatusInfo: Decodable, Sendable {
    public var available: Bool
    public var version: String?
}

public struct MetalToolchainInfo: Decodable, Sendable {
    public var fullyAvailable: Bool
    public var metal: ToolStatusInfo
    public var metallib: ToolStatusInfo
    public var metalAr: ToolStatusInfo
}

public struct MlxRuntimeInfo: Decodable, Sendable {
    public var runner: String
    public var artifactsSource: String?
}

public struct MlxModelInfo: Decodable, Sendable {
    public var artifactsSource: String?
    public var modelFamily: String
    public var tensorFormat: String
    public var layerCount: Int
    public var tensorCount: Int
    public var tieWordEmbeddings: Bool
    public var bindingsPrepared: Bool
    public var buffersBound: Bool
    public var bufferCount: Int
    public var bufferBytes: Int64
}

/// GET /v1/runtime response.
public struct ServerInfoResponse: Decodable, Sendable {
    public var service: String
    public var modelId: String
    public var deterministic: Bool
    public var maxBatchTokens: UInt32
    public var blockSizeTokens: UInt32
    public var runtime: RuntimeInfo
}

public struct ModelsResponse: Decodable, Sendable {
    public var object: String
    public var data: [ModelCard]
}

// MARK: - Native generate stream events

public struct GenerateStreamRequestEvent: Decodable, Sendable {
    public var request: RequestReport
}

public struct GenerateStreamStepEvent: Decodable, Sendable {
    public var request: RequestReport
    public var step: StepReport
    public var deltaTokens: [Int]?
    public var deltaText: String?
}

public struct GenerateStreamResponseEvent: Decodable, Sendable {
    public var response: GenerateResponse
}

/// A typed event from the native ``/v1/generate/stream`` SSE endpoint.
///
/// Check ``event`` then read the corresponding non-nil field.
public struct GenerateStreamEvent: Sendable {
    public let event: String
    public let request: GenerateStreamRequestEvent?
    public let step: GenerateStreamStepEvent?
    public let response: GenerateStreamResponseEvent?
}

// MARK: - Model Load

public struct LoadModelRequest: Encodable, Sendable {
    public var modelId: String
    public var modelPath: String
    public var loadPolicy: String?
    public var loadMode: String?
    /// Whether the loaded model becomes the default for requests that omit
    /// `model`. Server default is true; only meaningful for `loadMode`
    /// `"add"` (a `"replace"` load rejects `false`).
    public var makeDefault: Bool?

    public init(
        modelId: String, modelPath: String,
        loadPolicy: String? = nil, loadMode: String? = nil,
        makeDefault: Bool? = nil
    ) {
        self.modelId = modelId
        self.modelPath = modelPath
        self.loadPolicy = loadPolicy
        self.loadMode = loadMode
        self.makeDefault = makeDefault
    }
}

public struct LoadModelResponse: Decodable, Sendable {
    public var modelId: String
    public var state: String
    public var contextLength: UInt32
    public var loadPolicy: String
    public var loadMode: String
    /// Default model after the load; `nil` when the server predates 6.9.
    public var defaultModelId: String?
}

public struct UnloadModelRequest: Encodable, Sendable {
    public var modelId: String

    public init(modelId: String) {
        self.modelId = modelId
    }
}

public struct UnloadModelResponse: Decodable, Sendable {
    public var modelId: String
    public var state: String
    /// Default model after the unload (reports the reassignment when the
    /// unloaded model was the default); `nil` when the server predates 6.9.
    public var defaultModelId: String?
}
