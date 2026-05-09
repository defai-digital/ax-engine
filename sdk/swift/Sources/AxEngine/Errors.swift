import Foundation

/// Thrown when ax-engine-server responds with a non-2xx status code.
public struct AxEngineHTTPError: Error, CustomStringConvertible, Sendable {
    public let statusCode: Int
    public let message: String
    public let payload: Data?

    public var description: String { "ax-engine: HTTP \(statusCode): \(message)" }
}
