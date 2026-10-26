import Foundation

/// Byte limits for streaming responses. Limits must be positive; invalid
/// configuration fails when iteration starts, before a request is sent.
public struct AxEngineStreamLimits: Sendable {
    public let lineBytes: Int
    public let eventBytes: Int
    public let errorBodyBytes: Int

    /// SSE limits count UTF-8 bytes (including field names and framing).
    /// The line limit excludes its CR/LF terminator. The event limit includes
    /// comments, unknown fields, and framing up to the blank-line terminator;
    /// CRLF is counted as one line-ending byte, just like CR or LF.
    public init(lineBytes: Int = 1_048_576, eventBytes: Int = 4_194_304,
                errorBodyBytes: Int = 65_536) {
        self.lineBytes = lineBytes
        self.eventBytes = eventBytes
        self.errorBodyBytes = errorBodyBytes
    }

    func validate() throws {
        guard lineBytes > 0, eventBytes > 0, errorBodyBytes > 0 else {
            throw AxEngineStreamLimitError.invalidConfiguration
        }
    }
}

/// A streaming response exceeded a configured byte limit. No oversized body
/// is retained in the error; HTTP status is preserved for error responses.
public enum AxEngineStreamLimitError: Error, LocalizedError, Equatable, Sendable {
    case invalidConfiguration
    case lineTooLarge(limit: Int)
    case eventTooLarge(limit: Int)
    case errorBodyTooLarge(statusCode: Int, limit: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration:
            return "Streaming byte limits must be positive."
        case .lineTooLarge(let limit):
            return "SSE line exceeds the \(limit)-byte limit."
        case .eventTooLarge(let limit):
            return "SSE event exceeds the \(limit)-byte limit."
        case .errorBodyTooLarge(let status, let limit):
            return "HTTP \(status) error body exceeds the \(limit)-byte limit."
        }
    }
}
