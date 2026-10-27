#if canImport(XCTest)
import XCTest
import Foundation
@testable import AxEngine

private final class ReadCount: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
    func increment() { lock.lock(); defer { lock.unlock() }; count += 1 }
}

private struct CountedBytes: AsyncSequence {
    typealias Element = UInt8
    let data: [UInt8]
    let reads: ReadCount
    func makeAsyncIterator() -> AsyncIterator { .init(data: data, reads: reads) }
    struct AsyncIterator: AsyncIteratorProtocol {
        let data: [UInt8]
        let reads: ReadCount
        var index = 0
        mutating func next() async -> UInt8? {
            guard index < data.count else { return nil }
            reads.increment()
            defer { index += 1 }
            return data[index]
        }
    }
}

private final class StreamProbe: @unchecked Sendable {
    let body: Data
    let status: Int
    let finish: Bool
    let sendHeaders: Bool
    let starts = ReadCount()
    let started = XCTestExpectation(description: "transport started")
    let stopped = XCTestExpectation(description: "transport stopped")
    init(_ text: String, status: Int = 200, finish: Bool = false, sendHeaders: Bool = true) {
        body = Data(text.utf8); self.status = status; self.finish = finish
        self.sendHeaders = sendHeaders
    }
}

private final class ProbedURLProtocol: URLProtocol {
    nonisolated(unsafe) static var probe: StreamProbe?
    private var activeProbe: StreamProbe?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let probe = Self.probe else { return }
        activeProbe = probe
        probe.starts.increment()
        probe.started.fulfill()
        guard probe.sendHeaders else { return }
        let response = HTTPURLResponse(url: request.url!, statusCode: probe.status,
            httpVersion: nil, headerFields: ["Content-Type": "text/event-stream"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: probe.body)
        if probe.finish { client?.urlProtocolDidFinishLoading(self) }
    }
    override func stopLoading() { activeProbe?.stopped.fulfill() }
}

final class StreamingMemoryTests: XCTestCase {
    private let url = URL(string: "http://localhost/stream-fixture")!

    private func session(_ probe: StreamProbe) -> URLSession {
        ProbedURLProtocol.probe = probe
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ProbedURLProtocol.self]
        return URLSession(configuration: config)
    }

    private func integers(session: URLSession, decoded: ReadCount) -> AsyncThrowingStream<Int, Error> {
        let reader = StreamingResponse(request: .success(URLRequest(url: url)), session: session,
            limits: .init(), requiresDoneSentinel: true) { event in
                decoded.increment()
                return Int(event.data)!
            }
        return AsyncThrowingStream(unfolding: { try await reader.next() })
    }

    func testParserAcceptsExactUTF8LimitsAndAllLineEndings() async throws {
        for ending in ["\n", "\r", "\r\n"] {
            let input = "data: 😀" + ending + ending
            let reads = ReadCount()
            let bytes = CountedBytes(data: Array(input.utf8), reads: reads)
            var parser = SSEParser(bytes: bytes, limits: .init(lineBytes: 10, eventBytes: 12)).makeAsyncIterator()
            let event = try await parser.next()
            XCTAssertEqual(event?.data, "😀")
            let end = try await parser.next()
            XCTAssertNil(end)
        }
    }

    func testParserRejectsLongLineBeforeReadingTheRemainder() async throws {
        let reads = ReadCount()
        let bytes = CountedBytes(data: Array(String(repeating: "x", count: 1000).utf8), reads: reads)
        var parser = SSEParser(bytes: bytes, limits: .init(lineBytes: 8, eventBytes: 100)).makeAsyncIterator()
        do { _ = try await parser.next(); XCTFail("Expected a line limit error") }
        catch let error as AxEngineStreamLimitError { XCTAssertEqual(error, .lineTooLarge(limit: 8)) }
        XCTAssertEqual(reads.value, 9)
    }

    func testEventBudgetIncludesEmptyDataCommentsAndUnknownFields() async throws {
        for line in ["data\n", "data: x\n", ": ping\n", "ignored: x\n", "event: x\n"] {
            let reads = ReadCount()
            let bytes = CountedBytes(data: Array(String(repeating: line, count: 100).utf8), reads: reads)
            var parser = SSEParser(bytes: bytes, limits: .init(lineBytes: 16, eventBytes: 32)).makeAsyncIterator()
            do { _ = try await parser.next(); XCTFail("Expected an event limit error: \(line)") }
            catch let error as AxEngineStreamLimitError { XCTAssertEqual(error, .eventTooLarge(limit: 32)) }
            XCTAssertEqual(reads.value, 33)
        }
    }

    func testEventBudgetResetsAndPreservesMultilineData() async throws {
        let input = String(repeating: "data\ndata: x\n\n", count: 100)
        let bytes = CountedBytes(data: Array(input.utf8), reads: ReadCount())
        var parser = SSEParser(bytes: bytes, limits: .init(lineBytes: 7, eventBytes: 14)).makeAsyncIterator()
        var count = 0
        while let event = try await parser.next() { XCTAssertEqual(event.data, "\nx"); count += 1 }
        XCTAssertEqual(count, 100)
    }

    func testSlowConsumerDoesNotCreateADecodedEventQueueOrLoseOutput() async throws {
        let probe = StreamProbe((0..<300).map { "data: \($0)\n\n" }.joined() + "data: [DONE]\n\n")
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        let decoded = ReadCount()
        let stream = integers(session: session, decoded: decoded)
        try await Task.sleep(for: .milliseconds(20))
        XCTAssertEqual(probe.starts.value, 0, "An unconsumed stream must not start HTTP")
        var iterator = stream.makeAsyncIterator()
        let first = try await iterator.next()
        XCTAssertEqual(first, 0)
        try await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(decoded.value, 1, "Only consumer demand may decode events")
        let tasks = await session.allTasks
        XCTAssertEqual(tasks.count, 1)
        XCTAssertEqual(tasks.first?.state, .suspended, "Pause transport while the consumer is idle")
        var received = [first!]
        while let value = try await iterator.next() { received.append(value) }
        XCTAssertEqual(received, Array(0..<300))
        await fulfillment(of: [probe.stopped], timeout: 2)
    }

    func testPausedTransportResumesBeforeTheIdleTimeoutExpires() async throws {
        // A suspended task reads no keep-alives while URLSession's idle timer
        // runs, so a consumer pausing longer than the timeout used to fail
        // with NSURLErrorTimedOut. The pause must end before half the timeout.
        let probe = StreamProbe("data: 1\n\ndata: 2\n\ndata: [DONE]\n\n")
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: url)
        request.timeoutInterval = 0.4
        let reader = StreamingResponse(request: .success(request), session: session,
            limits: .init(), requiresDoneSentinel: true) { Int($0.data)! }
        let first = try await reader.next()
        XCTAssertEqual(first, 1)
        let paused = await session.allTasks.first?.state
        XCTAssertEqual(paused, .suspended)
        try await Task.sleep(for: .milliseconds(400))
        let resumed = await session.allTasks.first?.state
        XCTAssertEqual(resumed, .running,
                       "an idle consumer must not hold the transport past the idle timeout")
        let second = try await reader.next()
        XCTAssertEqual(second, 2)
    }

    func testOversizedHTTPErrorCancelsBeforeEOFAndPreservesStatus() async throws {
        let probe = StreamProbe(String(repeating: "x", count: 65), status: 503)
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        let client = AxEngineClient(session: session, streamLimits: .init(errorBodyBytes: 64))
        do {
            for try await _ in client.streamCompletion(.init(prompt: "x")) {}
            XCTFail("Expected a bounded error body")
        } catch let error as AxEngineStreamLimitError {
            XCTAssertEqual(error, .errorBodyTooLarge(statusCode: 503, limit: 64))
        }
        await fulfillment(of: [probe.stopped], timeout: 2)
    }

    func testExactHTTPErrorLimitKeepsNormalServerError() async throws {
        let body = "{\"error\":{\"message\":\"try later\"}}"
        let probe = StreamProbe(body, status: 429, finish: true)
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        let client = AxEngineClient(session: session, streamLimits: .init(errorBodyBytes: body.utf8.count))
        do {
            for try await _ in client.streamCompletion(.init(prompt: "x")) {}
            XCTFail("Expected HTTP error")
        } catch let error as AxEngineHTTPError {
            XCTAssertEqual(error.statusCode, 429)
            XCTAssertEqual(error.message, "try later")
            XCTAssertEqual(error.payload?.count, body.utf8.count)
        }
    }

    func testParserLimitAndDecodeFailureCancelOpenTransport() async throws {
        for (body, limit, isLimit) in [("data: " + String(repeating: "x", count: 100), 16, true),
                                      ("data: invalid-json\n\n", 100, false)] {
            let probe = StreamProbe(body)
            let session = session(probe)
            defer { session.invalidateAndCancel() }
            let client = AxEngineClient(session: session, streamLimits: .init(lineBytes: limit))
            do {
                for try await _ in client.streamCompletion(.init(prompt: "x")) {}
                XCTFail("Expected a stream failure")
            } catch {
                if isLimit { XCTAssertEqual(error as? AxEngineStreamLimitError, .lineTooLarge(limit: limit)) }
                else { XCTAssertTrue(error is DecodingError) }
            }
            await fulfillment(of: [probe.stopped], timeout: 2)
        }
    }

    func testCancellationWhileWaitingForNextEventClosesTransport() async throws {
        let probe = StreamProbe("data: 1\n\n")
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        let first = expectation(description: "first event")
        let finished = expectation(description: "cancelled consumer")
        let stream = integers(session: session, decoded: ReadCount())
        let task = Task {
            defer { finished.fulfill() }
            do {
                for try await _ in stream { first.fulfill() }
                XCTFail("Cancellation must throw")
            } catch { /* Cancellation may surface as CancellationError or URLSession cancellation. */ }
        }
        await fulfillment(of: [first], timeout: 2)
        task.cancel()
        await fulfillment(of: [finished, probe.stopped], timeout: 2)
        await task.value
    }

    func testAbandonedStreamReleasesTransportEvenWhenSessionLives() async throws {
        let probe = StreamProbe("data: 1\n\n")
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        var stream: AsyncThrowingStream<Int, Error>? = integers(session: session, decoded: ReadCount())
        var iterator = stream?.makeAsyncIterator()
        let value = try await iterator?.next()
        XCTAssertEqual(value, 1)
        iterator = nil
        stream = nil
        await fulfillment(of: [probe.stopped], timeout: 2)
    }

    func testCancellationBeforeResponseHeadersClosesTransport() async throws {
        let probe = StreamProbe("", sendHeaders: false)
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        let finished = expectation(description: "cancelled before headers")
        let stream = integers(session: session, decoded: ReadCount())
        let task = Task {
            defer { finished.fulfill() }
            do {
                for try await _ in stream {}
                XCTFail("Cancellation must throw")
            } catch { }
        }
        await fulfillment(of: [probe.started], timeout: 2)
        task.cancel()
        await fulfillment(of: [finished, probe.stopped], timeout: 2)
        await task.value
    }

    func testAlreadyCancelledConsumerDoesNotStartHTTP() async throws {
        let probe = StreamProbe("")
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        let stream = integers(session: session, decoded: ReadCount())
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                // AsyncThrowingStream may finish before invoking unfolding
                // when the consumer was already cancelled.
                for try await _ in stream { XCTFail("Cancelled stream must not emit an event") }
            } catch { XCTAssertTrue(error is CancellationError) }
        }
        await task.value
        XCTAssertEqual(probe.starts.value, 0)
    }

    func testCancellationBetweenDemandsClosesRetainedStream() async throws {
        let probe = StreamProbe("data: 1\n\n")
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        let stream = integers(session: session, decoded: ReadCount())
        let task = Task {
            var iterator = stream.makeAsyncIterator()
            let first = try await iterator.next()
            XCTAssertEqual(first, 1)
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try await iterator.next()
        }
        _ = await task.result
        await fulfillment(of: [probe.stopped], timeout: 2)
        // Keep the stream alive across the cancellation assertion.
        withExtendedLifetime(stream) {}
    }

    func testCancelledStreamDoesNotLaterReadAsCleanEnd() async throws {
        let probe = StreamProbe("data: 1\n\n")
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        let reader = StreamingResponse(request: .success(URLRequest(url: url)), session: session,
            limits: .init(), requiresDoneSentinel: true) { Int($0.data)! }
        let first = try await reader.next()
        XCTAssertEqual(first, 1)
        let blocked = Task { try await reader.next() }
        try await Task.sleep(for: .milliseconds(30))
        blocked.cancel()
        _ = await blocked.result
        await fulfillment(of: [probe.stopped], timeout: 2)
        do {
            let value = try await reader.next()
            XCTFail("A cancelled stream must not end cleanly, got \(String(describing: value))")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
    }

    func testTerminalResponseEventReleasesTransportImmediately() async throws {
        // No EOF from the transport: only the terminal event may release it.
        let probe = StreamProbe("event: step\ndata: 1\n\nevent: response\ndata: 2\n\n")
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        let reader = StreamingResponse(request: .success(URLRequest(url: url)), session: session,
            limits: .init(), requiresDoneSentinel: false) { Int($0.data)! }
        let step = try await reader.next()
        XCTAssertEqual(step, 1)
        let response = try await reader.next()
        XCTAssertEqual(response, 2)
        await fulfillment(of: [probe.stopped], timeout: 2)
        let end = try await reader.next()
        XCTAssertNil(end)
        withExtendedLifetime(reader) {}
    }

    func testConcurrentReaderMisuseClosesThePendingRead() async throws {
        let probe = StreamProbe("", sendHeaders: false)
        let session = session(probe)
        defer { session.invalidateAndCancel() }
        let reader = StreamingResponse(request: .success(URLRequest(url: url)), session: session,
            limits: .init(), requiresDoneSentinel: true) { $0.data }
        let finished = expectation(description: "first read terminated")
        let first = Task {
            defer { finished.fulfill() }
            do {
                _ = try await reader.next()
                XCTFail("The interrupted read must fail")
            } catch { }
        }
        await fulfillment(of: [probe.started], timeout: 2)
        do {
            _ = try await reader.next()
            XCTFail("Concurrent reads must be rejected")
        } catch let error as AxEngineStreamError {
            XCTAssertTrue(error.message.contains("Concurrent"))
        }
        await fulfillment(of: [finished, probe.stopped], timeout: 2)
        await first.value
    }

    func testInvalidLimitsFailBeforeSendingHTTP() async throws {
        for limits in [AxEngineStreamLimits(lineBytes: 0), .init(eventBytes: -1), .init(errorBodyBytes: 0)] {
            let probe = StreamProbe("")
            let session = session(probe)
            defer { session.invalidateAndCancel() }
            let client = AxEngineClient(session: session, streamLimits: limits)
            do {
                for try await _ in client.streamCompletion(.init(prompt: "x")) {}
                XCTFail("Expected invalid limits")
            } catch let error as AxEngineStreamLimitError {
                XCTAssertEqual(error, .invalidConfiguration)
            }
            XCTAssertEqual(probe.starts.value, 0)
        }
    }
}
#endif
