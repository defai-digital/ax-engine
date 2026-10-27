import Foundation

/// Owns one pull-driven response. There is no producer task or decoded-event
/// queue: each next() reads at most one SSE event on the consumer's task.
actor StreamingResponse<Element: Sendable> {
    private var request: Result<URLRequest, Error>?
    private let session: URLSession
    private let limits: AxEngineStreamLimits
    private let requiresDoneSentinel: Bool
    private let decode: @Sendable (SSEEvent) throws -> Element
    private var iterator: SSEParser<URLSession.AsyncBytes>.AsyncIterator?
    private var transport: URLSessionTask?
    // bytes(for:) exposes the URLSession task only after headers arrive.
    // Keep a cancellable setup task for that earlier phase; it parses no data.
    private var opening: Task<(URLSession.AsyncBytes, URLResponse), Error>?
    private var sawResponse = false
    private var finished = false
    // A cancelled stream must not later read as a clean end of stream.
    private var cancelled = false
    private var reading = false
    private var paused = false
    // Bumped on every pause/resume so a stale watchdog cannot resume a later
    // pause early.
    private var pauseGeneration = 0
    // A suspended task reads nothing, keep-alives included, while URLSession's
    // idle timer keeps running. Resume a pause before that timer can expire.
    private let maxPause: TimeInterval

    init(request: Result<URLRequest, Error>, session: URLSession,
         limits: AxEngineStreamLimits, requiresDoneSentinel: Bool,
         decode: @escaping @Sendable (SSEEvent) throws -> Element) {
        self.request = request
        let timeout = (try? request.get().timeoutInterval) ?? session.configuration.timeoutIntervalForRequest
        self.maxPause = max(min(timeout, session.configuration.timeoutIntervalForRequest) / 2, 0.05)
        self.session = session
        self.limits = limits
        self.requiresDoneSentinel = requiresDoneSentinel
        self.decode = decode
    }

    deinit { opening?.cancel(); transport?.cancel() }

    func next() async throws -> Element? {
        guard !finished else {
            if cancelled { throw CancellationError() }
            return nil
        }
        guard !reading else {
            close()
            throw AxEngineStreamError(message: "Concurrent stream iteration is not supported.", payload: "")
        }
        reading = true
        defer { reading = false }
        return try await withTaskCancellationHandler {
            do {
                return try await readNext()
            } catch {
                close()
                throw error
            }
        } onCancel: {
            // Actor reentrancy lets cancellation close a blocked byte read.
            Task { await self.cancel() }
        }
    }

    private func readNext() async throws -> Element? {
        try Task.checkCancellation()
        resumeTransport()
        if iterator == nil {
            try limits.validate()
            guard let pending = request else {
                throw AxEngineStreamError(message: "Stream request is no longer available.", payload: "")
            }
            request = nil
            let opening = Task { [session] in try await session.bytes(for: pending.get()) }
            self.opening = opening
            let (bytes, response) = try await opening.value
            self.opening = nil
            transport = bytes.task
            try Task.checkCancellation()
            guard !finished else { throw CancellationError() }
            if let http = response as? HTTPURLResponse,
               !(200..<300).contains(http.statusCode) {
                var data = Data()
                for try await byte in bytes {
                    try Task.checkCancellation()
                    guard data.count < limits.errorBodyBytes else {
                        throw AxEngineStreamLimitError.errorBodyTooLarge(
                            statusCode: http.statusCode, limit: limits.errorBodyBytes)
                    }
                    data.append(byte)
                }
                try AxEngineClient.validate(response: response, data: data)
            }
            try AxEngineClient.validate(response: response, data: nil)
            iterator = SSEParser(bytes: bytes, limits: limits).makeAsyncIterator()
        }
        guard var current = iterator else { return nil }
        // Move the buffered parser state out while awaiting. No second caller
        // may consume it, and close() can independently cancel the transport.
        iterator = nil
        let event = try await current.next()
        try Task.checkCancellation()
        guard !finished else { throw CancellationError() }
        if let event {
            if event.event == "error" { throw AxEngineClient.streamError(from: event.data) }
            let value = try decode(event)
            if event.event == "response" {
                sawResponse = true
                // The response event is terminal here; release the transport
                // now instead of on the next pull or deallocation.
                if !requiresDoneSentinel {
                    close()
                    return value
                }
            }
            iterator = current
            // Also pause transport between demands. Foundation may already
            // have buffered some bytes; this is not a whole-process byte cap.
            pauseTransport()
            return value
        }
        if requiresDoneSentinel && !current.sawDone {
            throw AxEngineStreamError(message: "stream ended without [DONE]", payload: "")
        }
        if !requiresDoneSentinel && !sawResponse {
            throw AxEngineStreamError(message: "stream ended without terminal response", payload: "")
        }
        close()
        return nil
    }

    private func pauseTransport() {
        guard let transport else { return }
        transport.suspend()
        paused = true
        pauseGeneration += 1
        let generation = pauseGeneration
        let delay = UInt64(maxPause * 1_000_000_000)
        // Weak: an abandoned stream must still deallocate and cancel.
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: delay)
            await self?.endPause(generation)
        }
    }

    private func endPause(_ generation: Int) {
        // A consumer slower than the idle timeout keeps the connection alive
        // (server keep-alives are read again); Foundation buffers what arrives.
        guard paused, generation == pauseGeneration else { return }
        resumeTransport()
    }

    private func resumeTransport() {
        guard paused else { return }
        transport?.resume()
        paused = false
        pauseGeneration += 1
    }

    private func cancel() {
        if !finished { cancelled = true }
        close()
    }

    private func close() {
        finished = true
        request = nil
        iterator = nil
        opening?.cancel()
        opening = nil
        transport?.cancel()
        transport = nil
        paused = false
    }
}
