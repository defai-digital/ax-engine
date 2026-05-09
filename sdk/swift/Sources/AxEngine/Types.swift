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

public struct PreviewGenerateRequest: Encodable, Sendable {
    public var model: String?
    public var inputTokens: [Int]?
    public var inputText: String?
    public var maxOutputTokens: Int?
    public var sampling: GenerateSampling?
    public var metadata: String?

    public init(
        model: String? = nil, inputTokens: [Int]? = nil, inputText: String? = nil,
        maxOutputTokens: Int? = nil, sampling: GenerateSampling? = nil, metadata: String? = nil
    ) {
        self.model = model; self.inputTokens = inputTokens; self.inputText = inputText
        self.maxOutputTokens = maxOutputTokens; self.sampling = sampling; self.metadata = metadata
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
    public var content: String

    public init(role: String, content: String) {
        self.role = role; self.content = content
    }
}

public struct OpenAiChatCompletionRequest: Encodable, Sendable {
    public var model: String?
    public var messages: [OpenAiChatMessage]
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

    public init(
        model: String? = nil, messages: [OpenAiChatMessage],
        maxTokens: Int? = nil, temperature: Double? = nil,
        topP: Double? = nil, topK: Int? = nil, minP: Double? = nil,
        repetitionPenalty: Double? = nil, stop: [String]? = nil,
        seed: Int? = nil, metadata: String? = nil
    ) {
        self.model = model; self.messages = messages; self.maxTokens = maxTokens
        self.temperature = temperature; self.topP = topP; self.topK = topK
        self.minP = minP; self.repetitionPenalty = repetitionPenalty
        self.stop = stop; self.seed = seed; self.metadata = metadata
    }
}

public struct OpenAiUsage: Decodable, Sendable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int
}

public struct OpenAiChatMessageResponse: Decodable, Sendable {
    public var role: String
    public var content: String
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

public struct OpenAiCompletionRequest: Encodable, Sendable {
    public var model: String?
    public var prompt: String
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

    public init(
        model: String? = nil, prompt: String,
        maxTokens: Int? = nil, temperature: Double? = nil,
        topP: Double? = nil, topK: Int? = nil, minP: Double? = nil,
        repetitionPenalty: Double? = nil, stop: [String]? = nil,
        seed: Int? = nil, metadata: String? = nil
    ) {
        self.model = model; self.prompt = prompt; self.maxTokens = maxTokens
        self.temperature = temperature; self.topP = topP; self.topK = topK
        self.minP = minP; self.repetitionPenalty = repetitionPenalty
        self.stop = stop; self.seed = seed; self.metadata = metadata
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

public struct OpenAiEmbeddingRequest: Encodable, Sendable {
    public var model: String?
    public var input: [Int]
    public var encodingFormat: String?
    public var pooling: String?
    public var normalize: Bool?

    public init(
        model: String? = nil, input: [Int],
        encodingFormat: String? = nil, pooling: String? = nil, normalize: Bool? = nil
    ) {
        self.model = model; self.input = input; self.encodingFormat = encodingFormat
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
    public var service: String
    public var modelId: String
}

public struct ModelCard: Decodable, Sendable {
    public var id: String
    public var object: String
    public var ownedBy: String
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
