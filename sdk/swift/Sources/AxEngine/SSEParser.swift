import Foundation

/// A single server-sent event.
struct SSEEvent: Sendable {
    let event: String
    let data: String
}

/// Async sequence that parses SSE events from ``URLSession.AsyncBytes``.
///
/// Yields one ``SSEEvent`` per complete event block (blank-line terminated).
/// Stops when the `[DONE]` sentinel is encountered or the byte stream ends.
struct SSEParser<Bytes: AsyncSequence>: AsyncSequence where Bytes.Element == UInt8 {
    typealias Element = SSEEvent

    let bytes: Bytes
    var limits = AxEngineStreamLimits()

    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(bytes: bytes.makeAsyncIterator(), limits: limits)
    }

    struct AsyncIterator: AsyncIteratorProtocol {
        var bytes: Bytes.AsyncIterator
        let limits: AxEngineStreamLimits
        var eventName = "message"
        var eventData = ""
        var hasData = false
        var done = false
        /// True once the `[DONE]` sentinel has been consumed, so callers can
        /// tell a completed OpenAI stream from a connection cut at EOF.
        private(set) var sawDone = false
        var lineBuffer = [UInt8]()
        var skipLF = false
        var firstLine = true
        var eventBytes = 0

        mutating func next() async throws -> SSEEvent? {
            guard !done else { return nil }
            try limits.validate()

            while let byte = try await bytes.next() {
                if skipLF {
                    skipLF = false
                    if byte == UInt8(ascii: "\n") { continue }
                }
                guard eventBytes < limits.eventBytes else {
                    throw AxEngineStreamLimitError.eventTooLarge(limit: limits.eventBytes)
                }
                eventBytes += 1
                if byte != UInt8(ascii: "\n") && byte != UInt8(ascii: "\r") {
                    guard lineBuffer.count < limits.lineBytes else {
                        throw AxEngineStreamLimitError.lineTooLarge(limit: limits.lineBytes)
                    }
                    lineBuffer.append(byte)
                    continue
                }
                // CR is itself a line ending; suppress only its optional LF.
                skipLF = byte == UInt8(ascii: "\r")
                var line = String(decoding: lineBuffer, as: UTF8.self)
                lineBuffer.removeAll(keepingCapacity: true)
                if firstLine {
                    if line.hasPrefix("\u{FEFF}") { line.removeFirst() }
                    firstLine = false
                }
                if let event = processLine(line) { return event }
                if done { return nil }
            }

            // An event requires a blank-line terminator, even at EOF.
            eventData = ""
            hasData = false
            lineBuffer = []
            done = true
            return nil
        }

        private mutating func processLine(_ line: String) -> SSEEvent? {
            if line.isEmpty {
                eventBytes = 0
                let name = eventName.isEmpty ? "message" : eventName
                eventName = "message"
                guard hasData else { return nil }
                let data = eventData
                eventData = ""
                hasData = false
                if name != "error" && data == "[DONE]" { done = true; sawDone = true; return nil }
                return SSEEvent(event: name, data: data)
            }
            if line.hasPrefix(":") { return nil }
            let fields = line.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
            let field = fields[0]
            var value = fields.count > 1 ? String(fields[1]) : ""
            if value.hasPrefix(" ") { value.removeFirst() }
            if field == "event" {
                eventName = value
            } else if field == "data" {
                if hasData { eventData.append("\n") }
                eventData.append(contentsOf: value)
                hasData = true
            }
            return nil
        }
    }
}

// The network iterator is Sendable. Preserve that checked conformance when
// moving parser state out of the reader actor for a single awaited read.
extension SSEParser.AsyncIterator: Sendable where Bytes.AsyncIterator: Sendable {}
