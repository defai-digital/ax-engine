import Foundation
import AxEngine

private final class RejectRedirects: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

public struct RunningModel: Decodable, Identifiable, Sendable {
    public let id: String
    public let ownedBy: String?
    public let contextLength: Int?
    public let capabilities: Capabilities?
    public struct Capabilities: Decodable, Sendable {
        public let temperature: Bool?
        public let reasoning: Bool?
        public let attachment: Bool?
        public let toolcall: Bool?
    }
}

public struct EngineConnection: Sendable, Identifiable, CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
    public let id = UUID()
    public let baseURL: URL
    private let apiKey: String
    private static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.urlCache = nil
        config.httpCookieStorage = nil
        config.urlCredentialStorage = nil
        config.httpShouldSetCookies = false
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config, delegate: RejectRedirects(), delegateQueue: nil)
    }()
    public init(port: Int, apiKey: String = "") {
        baseURL = ServerURLs.baseURL(bindHost: "127.0.0.1", port: port)
        self.apiKey = apiKey
    }
    public var apiURL: URL { baseURL.appendingPathComponent("v1") }
    public var hasAPIKey: Bool { !apiKey.isEmpty }
    public var description: String { "EngineConnection(\(baseURL.absoluteString), apiKey: [redacted])" }
    public var debugDescription: String { description }
    public var customMirror: Mirror { Mirror(self, children: ["baseURL": baseURL, "hasAPIKey": hasAPIKey]) }
    @MainActor @discardableResult
    public func copyAPIKey(to clipboard: SensitiveClipboard) -> Bool {
        clipboard.copy(apiKey)
    }
    public var sdk: AxEngineClient {
        AxEngineClient(baseURL: baseURL, session: Self.session, headers: apiKey.isEmpty ? [:] : ["Authorization": "Bearer \(apiKey)"])
    }
    public func data(_ path: String, timeout: TimeInterval = 3) async throws -> Data {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.timeoutInterval = timeout
        if !apiKey.isEmpty { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        let (bytes, response) = try await Self.session.bytes(for: request)
        defer { bytes.task.cancel() }
        try Self.validate(response)
        let limit = 4 * 1024 * 1024
        guard response.expectedContentLength <= Int64(limit) else {
            throw AppFailure.message("Service response exceeds the display limit.")
        }
        var data = Data()
        for try await byte in bytes {
            guard data.count < limit else {
                throw AppFailure.message("Service response exceeds the display limit.")
            }
            data.append(byte)
        }
        return data
    }
    private static func validate(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { throw AppFailure.message("The service did not return HTTP.") }
        guard (200..<300).contains(http.statusCode) else {
            throw AppFailure.message(http.statusCode == 401 || http.statusCode == 403
                ? "The service rejected the API key. It has not been stopped."
                : "The service returned HTTP \(http.statusCode).")
        }
    }
    public func models() async throws -> [RunningModel] {
        struct Response: Decodable { let object: String; let data: [RunningModel] }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Response.self, from: await data("v1/models"))
        guard value.object == "list", !value.data.isEmpty,
              value.data.allSatisfy({ $0.ownedBy == "ax-engine" }) else {
            throw AppFailure.message("The endpoint did not identify an AX Engine model. Check the port and runtime version.")
        }
        return value.data
    }
    public func cancel(requestID: Int) async throws {
        guard requestID >= 0 else { return }
        var request = URLRequest(url: baseURL.appendingPathComponent("v1/requests/\(requestID)/cancel"))
        request.httpMethod = "POST"
        request.timeoutInterval = 3
        if !apiKey.isEmpty { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        let (_, response) = try await Self.session.data(for: request)
        try Self.validate(response)
    }
    public func metrics() async throws -> [String: Double] {
        Self.parseMetrics(String(decoding: try await data("metrics"), as: UTF8.self))
    }
    public static func parseMetrics(_ text: String) -> [String: Double] {
        var result: [String: Double] = [:]
        for line in text.split(separator: "\n") where !line.hasPrefix("#") {
            let fields = line.split(whereSeparator: \.isWhitespace)
            guard fields.count >= 2, !fields[0].contains("{"), let value = Double(fields[1]), value.isFinite else { continue }
            result[String(fields[0])] = value
        }
        return result
    }
    public static func requestID(_ id: String) -> Int? {
        guard id.hasPrefix("chatcmpl-"), let value = Int(id.dropFirst(9)), value >= 0 else { return nil }
        return value
    }
}
