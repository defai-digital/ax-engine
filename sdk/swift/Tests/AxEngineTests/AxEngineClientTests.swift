import XCTest
import Foundation
@testable import AxEngine

// MARK: - URLProtocol mock

final class MockURLProtocol: URLProtocol {
    nonisolated(unsafe) static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.unknown))
            return
        }
        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

// MARK: - Helpers

private func makeClient() -> AxEngineClient {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [MockURLProtocol.self]
    let session = URLSession(configuration: config)
    return AxEngineClient(session: session)
}

private func jsonResponse(_ object: Any, statusCode: Int = 200) -> (HTTPURLResponse, Data) {
    let data = try! JSONSerialization.data(withJSONObject: object)
    let resp = HTTPURLResponse(
        url: URL(string: "http://127.0.0.1:8080")!,
        statusCode: statusCode,
        httpVersion: nil,
        headerFields: ["Content-Type": "application/json"]
    )!
    return (resp, data)
}

private func sseResponse(_ text: String) -> (HTTPURLResponse, Data) {
    let resp = HTTPURLResponse(
        url: URL(string: "http://127.0.0.1:8080")!,
        statusCode: 200,
        httpVersion: nil,
        headerFields: ["Content-Type": "text/event-stream"]
    )!
    return (resp, Data(text.utf8))
}

// MARK: - Tests

final class AxEngineClientTests: XCTestCase {

    // MARK: health

    func testHealth() async throws {
        MockURLProtocol.handler = { req in
            XCTAssertEqual(req.httpMethod, "GET")
            XCTAssertTrue(req.url?.path == "/health")
            return jsonResponse(["status": "ok", "service": "ax-engine-server", "model_id": "qwen3_dense"])
        }
        let resp = try await makeClient().health()
        XCTAssertEqual(resp.status, "ok")
        XCTAssertEqual(resp.modelId, "qwen3_dense")
    }

    // MARK: chatCompletion

    func testChatCompletion() async throws {
        MockURLProtocol.handler = { req in
            XCTAssertEqual(req.httpMethod, "POST")
            XCTAssertEqual(req.url?.path, "/v1/chat/completions")
            return jsonResponse([
                "id": "chatcmpl-1", "object": "chat.completion", "created": 1_234_567_890,
                "model": "qwen3_dense",
                "choices": [[
                    "index": 0,
                    "message": ["role": "assistant", "content": "Hi there!"],
                    "finish_reason": "stop"
                ]],
                "usage": ["prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8]
            ])
        }
        let resp = try await makeClient().chatCompletion(.init(
            messages: [.init(role: "user", content: "Hello!")], maxTokens: 32
        ))
        XCTAssertEqual(resp.object, "chat.completion")
        XCTAssertEqual(resp.choices.first?.message.content, "Hi there!")
        XCTAssertEqual(resp.usage?.totalTokens, 8)
    }

    func testChatCompletionEncodesMessages() async throws {
        var capturedBody: [String: Any] = [:]
        MockURLProtocol.handler = { req in
            capturedBody = try! JSONSerialization.jsonObject(with: req.httpBody!) as! [String: Any]
            return jsonResponse([
                "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [["index": 0, "message": ["role": "assistant", "content": "ok"], "finish_reason": "stop"]]
            ])
        }
        try await makeClient().chatCompletion(.init(
            messages: [
                .init(role: "system", content: "You are AX."),
                .init(role: "user", content: "Hello"),
            ],
            maxTokens: 64, temperature: 0.7, seed: 42
        ))
        let msgs = capturedBody["messages"] as! [[String: Any]]
        XCTAssertEqual(msgs.count, 2)
        XCTAssertEqual(msgs[0]["role"] as? String, "system")
        XCTAssertEqual(msgs[1]["role"] as? String, "user")
        XCTAssertEqual(capturedBody["max_tokens"] as? Int, 64)
        XCTAssertEqual(capturedBody["seed"] as? Int, 42)
    }

    // MARK: completion

    func testCompletion() async throws {
        MockURLProtocol.handler = { _ in
            jsonResponse([
                "id": "cmpl-1", "object": "text_completion", "created": 0, "model": "qwen3_dense",
                "choices": [["index": 0, "text": "Hello world", "finish_reason": "stop"]],
                "usage": ["prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5]
            ])
        }
        let resp = try await makeClient().completion(.init(prompt: "Hello", maxTokens: 32))
        XCTAssertEqual(resp.object, "text_completion")
        XCTAssertEqual(resp.choices.first?.text, "Hello world")
        XCTAssertEqual(resp.usage?.totalTokens, 5)
    }

    // MARK: embeddings

    func testEmbeddings() async throws {
        MockURLProtocol.handler = { _ in
            jsonResponse([
                "object": "list",
                "data": [["object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0]],
                "model": "qwen3_embedding",
                "usage": ["prompt_tokens": 3, "total_tokens": 3]
            ])
        }
        let resp = try await makeClient().embeddings(.init(input: [1, 2, 3], pooling: "last", normalize: true))
        XCTAssertEqual(resp.data.first?.embedding, [0.1, 0.2, 0.3])
        XCTAssertEqual(resp.usage.totalTokens, 3)
    }

    // MARK: HTTP errors

    func testHTTPErrorThrown() async throws {
        MockURLProtocol.handler = { _ in
            jsonResponse(["error": ["message": "bad request"]], statusCode: 400)
        }
        do {
            try await makeClient().completion(.init(prompt: "x"))
            XCTFail("Expected error")
        } catch let err as AxEngineHTTPError {
            XCTAssertEqual(err.statusCode, 400)
            XCTAssertEqual(err.message, "bad request")
        }
    }

    // MARK: streaming chat

    func testStreamChatCompletion() async throws {
        let sse = """
        data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"qwen3_dense","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

        data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"qwen3_dense","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}

        data: [DONE]

        """
        MockURLProtocol.handler = { _ in sseResponse(sse) }

        var chunks: [OpenAiChatCompletionChunk] = []
        for try await chunk in makeClient().streamChatCompletion(.init(
            messages: [.init(role: "user", content: "Hi")]
        )) {
            chunks.append(chunk)
        }
        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].choices.first?.delta.content, "Hello")
        XCTAssertEqual(chunks[1].choices.first?.delta.content, " world")
        XCTAssertEqual(chunks[1].choices.first?.finishReason, "stop")
    }

    func testStreamChatCompletionSetsStreamTrue() async throws {
        var capturedBody: [String: Any] = [:]
        let sse = "data: [DONE]\n\n"
        MockURLProtocol.handler = { req in
            capturedBody = (try? JSONSerialization.jsonObject(with: req.httpBody!) as? [String: Any]) ?? [:]
            return sseResponse(sse)
        }
        for try await _ in makeClient().streamChatCompletion(.init(
            messages: [.init(role: "user", content: "x")]
        )) {}
        XCTAssertEqual(capturedBody["stream"] as? Bool, true)
    }

    // MARK: streaming completion

    func testStreamCompletion() async throws {
        let sse = """
        data: {"id":"cmpl-1","object":"text_completion.chunk","created":0,"model":"qwen3_dense","choices":[{"index":0,"text":"Once","finish_reason":null}]}

        data: [DONE]

        """
        MockURLProtocol.handler = { _ in sseResponse(sse) }

        var texts: [String] = []
        for try await chunk in makeClient().streamCompletion(.init(prompt: "Once upon")) {
            texts.append(contentsOf: chunk.choices.map(\.text))
        }
        XCTAssertEqual(texts, ["Once"])
    }

    // MARK: streamGenerate

    func testStreamGenerate() async throws {
        let sse = """
        event: request
        data: {"request":{"request_id":1,"model_id":"qwen3_dense","state":"active","prompt_tokens":[1,2,3],"processed_prompt_tokens":3,"output_tokens":[],"prompt_len":3,"output_len":0,"max_output_tokens":4,"cancel_requested":false,"route":{}}}

        event: step
        data: {"request":{"request_id":1,"model_id":"qwen3_dense","state":"active","prompt_tokens":[1,2,3],"processed_prompt_tokens":3,"output_tokens":[42],"prompt_len":3,"output_len":1,"max_output_tokens":4,"cancel_requested":false,"route":{}},"step":{"scheduled_requests":1,"scheduled_tokens":1,"ttft_events":1,"prefix_hits":0,"kv_usage_blocks":1,"evictions":0,"cpu_time_us":100,"runner_time_us":200},"delta_tokens":[42],"delta_text":"hello"}

        event: response
        data: {"response":{"request_id":1,"model_id":"qwen3_dense","prompt_tokens":[1,2,3],"output_tokens":[42],"status":"finished","finish_reason":"stop","step_count":1,"route":{}}}

        """
        MockURLProtocol.handler = { _ in sseResponse(sse) }

        var events: [GenerateStreamEvent] = []
        for try await event in makeClient().streamGenerate(.init(inputTokens: [1, 2, 3], maxOutputTokens: 4)) {
            events.append(event)
        }

        XCTAssertEqual(events.count, 3)
        XCTAssertEqual(events[0].event, "request")
        XCTAssertNotNil(events[0].request)
        XCTAssertEqual(events[1].event, "step")
        XCTAssertEqual(events[1].step?.deltaTokens, [42])
        XCTAssertEqual(events[1].step?.deltaText, "hello")
        XCTAssertEqual(events[2].event, "response")
        XCTAssertEqual(events[2].response?.response.finishReason, "stop")
    }
}
