import Foundation

public enum AppFailure: LocalizedError, Equatable {
    case message(String)
    public var errorDescription: String? { if case .message(let text) = self { text } else { nil } }
}

public struct AppSettings: Codable, Equatable, Sendable {
    public var executable = ""
    public var modelsDirectory = AppPaths.support.appendingPathComponent("Models").path
    public var port = 31418
    public var maxTokens = 1024
    public var temperature = 0.7
    public init() {}

    public func validate() throws {
        guard (1024...65535).contains(port), (1...32768).contains(maxTokens),
              temperature.isFinite, (0...2).contains(temperature),
              modelsDirectory.hasPrefix("/"), !modelsDirectory.contains("\0") else {
            throw AppFailure.message("Use an absolute model folder, port 1024–65535, 1–32768 output tokens and temperature 0–2.")
        }
        if !executable.isEmpty && (!executable.hasPrefix("/") || !FileManager.default.isExecutableFile(atPath: executable)) {
            throw AppFailure.message("Select an executable AX Engine runtime using its absolute path.")
        }
    }

    public static func load(from url: URL = AppPaths.settings) throws -> Self {
        guard FileManager.default.fileExists(atPath: url.path) else { return Self() }
        let value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        try value.validate()
        return value
    }

    public func save(to url: URL = AppPaths.settings) throws {
        try validate()
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(self).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    public func runtime(bundle: Bundle = .main) throws -> URL {
        let candidates = [executable, bundle.resourceURL?.appendingPathComponent("runtime/ax-engine").path ?? "",
                          "/opt/homebrew/bin/ax-engine", "/usr/local/bin/ax-engine"]
        guard let path = candidates.first(where: { !$0.isEmpty && FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw AppFailure.message("AX Engine runtime not found. Install AX Engine, or select its executable in Settings.")
        }
        return URL(fileURLWithPath: path)
    }
}

public enum AppPaths {
    public static let support = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/AX Engine", isDirectory: true)
    public static let settings = support.appendingPathComponent("settings.json")
    public static let conversation = support.appendingPathComponent("conversation.json")
}
