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
struct SSEParser: AsyncSequence {
    typealias Element = SSEEvent
    typealias LinesIterator = AsyncLineSequence<URLSession.AsyncBytes>.AsyncIterator

    let bytes: URLSession.AsyncBytes

    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(lines: bytes.lines.makeAsyncIterator())
    }

    struct AsyncIterator: AsyncIteratorProtocol {
        var lines: LinesIterator
        var eventName = "message"
        var dataLines: [String] = []
        var done = false

        mutating func next() async throws -> SSEEvent? {
            guard !done else { return nil }

            while let line = try await lines.next() {
                if line.isEmpty {
                    guard !dataLines.isEmpty else { continue }
                    let data = dataLines.joined(separator: "\n")
                    let name = eventName
                    eventName = "message"
                    dataLines = []
                    if data == "[DONE]" { done = true; return nil }
                    return SSEEvent(event: name, data: data)
                }

                if line.hasPrefix(":") { continue }

                if let rest = line.stripPrefix("event:") {
                    eventName = rest.trimmingCharacters(in: CharacterSet.whitespaces)
                } else if let rest = line.stripPrefix("data:") {
                    dataLines.append(rest.hasPrefix(" ") ? String(rest.dropFirst()) : rest)
                }
            }

            // Flush any trailing event not ended by a blank line.
            if !dataLines.isEmpty {
                let data = dataLines.joined(separator: "\n")
                dataLines = []
                done = true
                if data == "[DONE]" { return nil }
                return SSEEvent(event: eventName, data: data)
            }

            done = true
            return nil
        }
    }
}

private extension String {
    func stripPrefix(_ prefix: String) -> String? {
        hasPrefix(prefix) ? String(dropFirst(prefix.count)) : nil
    }
}
