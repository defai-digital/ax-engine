import Foundation

/// HTTP client for ax-engine-server.
///
/// Connects to a running ``ax-engine-server`` instance (default: `http://127.0.0.1:8080`).
///
/// ```swift
/// let client = AxEngineClient()
///
/// let resp = try await client.chatCompletion(.init(
///     messages: [.init(role: "user", content: "Hello!")],
///     maxTokens: 128
/// ))
/// print(resp.choices[0].message.content)
/// ```
public final class AxEngineClient: @unchecked Sendable {
    private let baseURL: URL
    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    public init(
        baseURL: URL = URL(string: "http://127.0.0.1:8080")!,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.session = session

        encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase

        decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
    }

    // MARK: - Health / info

    /// GET /health
    public func health() async throws -> HealthResponse {
        try await get("/health")
    }

    /// GET /v1/models
    public func models() async throws -> ModelsResponse {
        try await get("/v1/models")
    }

    // MARK: - Native ax-engine API

    /// POST /v1/generate — blocking native generate.
    public func generate(_ request: PreviewGenerateRequest) async throws -> GenerateResponse {
        try await post("/v1/generate", body: request)
    }

    /// POST /v1/requests — submit a request without blocking for completion.
    public func submit(_ request: PreviewGenerateRequest) async throws -> RequestReport {
        try await post("/v1/requests", body: request)
    }

    /// GET /v1/requests/{id} — fetch a live request snapshot.
    public func requestSnapshot(id: Int) async throws -> RequestReport {
        try await get("/v1/requests/\(id)")
    }

    /// POST /v1/requests/{id}/cancel
    public func cancel(id: Int) async throws -> RequestReport {
        try await post("/v1/requests/\(id)/cancel", body: Empty())
    }

    /// POST /v1/step — advance the scheduler by one step.
    public func step() async throws -> StepReport {
        try await post("/v1/step", body: Empty())
    }

    // MARK: - OpenAI-compatible

    /// POST /v1/completions
    public func completion(_ request: OpenAiCompletionRequest) async throws -> OpenAiCompletionResponse {
        try await post("/v1/completions", body: request)
    }

    /// POST /v1/chat/completions
    public func chatCompletion(_ request: OpenAiChatCompletionRequest) async throws -> OpenAiChatCompletionResponse {
        try await post("/v1/chat/completions", body: request)
    }

    /// POST /v1/embeddings
    public func embeddings(_ request: OpenAiEmbeddingRequest) async throws -> OpenAiEmbeddingResponse {
        try await post("/v1/embeddings", body: request)
    }

    // MARK: - Streaming

    /// Stream POST /v1/generate/stream — native ax-engine SSE events.
    ///
    /// Each yielded ``GenerateStreamEvent`` carries one of `request`, `step`,
    /// or `response` depending on its ``GenerateStreamEvent/event`` field.
    ///
    /// ```swift
    /// for try await event in client.streamGenerate(.init(inputTokens: [1, 2, 3], maxOutputTokens: 32)) {
    ///     if event.event == "step", let text = event.step?.deltaText {
    ///         print(text, terminator: "")
    ///     }
    /// }
    /// ```
    public func streamGenerate(_ request: PreviewGenerateRequest) -> AsyncThrowingStream<GenerateStreamEvent, Error> {
        stream("/v1/generate/stream", body: request) { [decoder] event in
            let data = Data(event.data.utf8)
            switch event.event {
            case "request":
                let r = try decoder.decode(GenerateStreamRequestEvent.self, from: data)
                return GenerateStreamEvent(event: event.event, request: r, step: nil, response: nil)
            case "step":
                let s = try decoder.decode(GenerateStreamStepEvent.self, from: data)
                return GenerateStreamEvent(event: event.event, request: nil, step: s, response: nil)
            case "response":
                let r = try decoder.decode(GenerateStreamResponseEvent.self, from: data)
                return GenerateStreamEvent(event: event.event, request: nil, step: nil, response: r)
            default:
                return GenerateStreamEvent(event: event.event, request: nil, step: nil, response: nil)
            }
        }
    }

    /// Stream POST /v1/chat/completions (stream: true) — yields SSE chunks.
    ///
    /// ```swift
    /// for try await chunk in client.streamChatCompletion(.init(messages: messages)) {
    ///     print(chunk.choices[0].delta.content ?? "", terminator: "")
    /// }
    /// ```
    public func streamChatCompletion(_ request: OpenAiChatCompletionRequest) -> AsyncThrowingStream<OpenAiChatCompletionChunk, Error> {
        var req = request; req.stream = true
        return stream("/v1/chat/completions", body: req) { [decoder] event in
            try decoder.decode(OpenAiChatCompletionChunk.self, from: Data(event.data.utf8))
        }
    }

    /// Stream POST /v1/completions (stream: true) — yields SSE chunks.
    public func streamCompletion(_ request: OpenAiCompletionRequest) -> AsyncThrowingStream<OpenAiCompletionChunk, Error> {
        var req = request; req.stream = true
        return stream("/v1/completions", body: req) { [decoder] event in
            try decoder.decode(OpenAiCompletionChunk.self, from: Data(event.data.utf8))
        }
    }

    // MARK: - Private helpers

    private func get<R: Decodable>(_ path: String) async throws -> R {
        let urlRequest = buildRequest(method: "GET", path: path, body: nil as Data?)
        return try await execute(urlRequest)
    }

    private func post<B: Encodable, R: Decodable>(_ path: String, body: B) async throws -> R {
        var urlRequest = buildRequest(method: "POST", path: path, body: try encoder.encode(body))
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return try await execute(urlRequest)
    }

    private func execute<R: Decodable>(_ urlRequest: URLRequest) async throws -> R {
        let (data, response) = try await session.data(for: urlRequest)
        try validate(response: response, data: data)
        return try decoder.decode(R.self, from: data)
    }

    private func stream<T: Sendable>(
        _ path: String,
        body: some Encodable,
        decode: @escaping @Sendable (SSEEvent) throws -> T
    ) -> AsyncThrowingStream<T, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [encoder, session] in
                do {
                    var urlRequest = self.buildRequest(method: "POST", path: path, body: try encoder.encode(body))
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    urlRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    let (asyncBytes, response) = try await session.bytes(for: urlRequest)
                    try self.validate(response: response, data: nil)
                    for try await event in SSEParser(bytes: asyncBytes) {
                        let value = try decode(event)
                        continuation.yield(value)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func buildRequest(method: String, path: String, body: Data?) -> URLRequest {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = method
        request.httpBody = body
        return request
    }

    private func validate(response: URLResponse, data: Data?) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard http.statusCode >= 200, http.statusCode < 300 else {
            var message = "HTTP \(http.statusCode)"
            if let data, let body = try? JSONDecoder().decode(ErrorBody.self, from: data) {
                message = body.error.message
            }
            throw AxEngineHTTPError(statusCode: http.statusCode, message: message, payload: data)
        }
    }

    private struct Empty: Encodable {}

    private struct ErrorBody: Decodable {
        struct Inner: Decodable { let message: String }
        let error: Inner
    }
}
