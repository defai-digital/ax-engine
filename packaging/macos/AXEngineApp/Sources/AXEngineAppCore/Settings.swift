import Foundation

public enum AppFailure: LocalizedError, Equatable {
    case message(String)
    public var errorDescription: String? { if case .message(let text) = self { text } else { nil } }
}

public struct AppSettings: Codable, Equatable, Sendable {
    public var executable = ""
    public var modelsDirectory = AppPaths.support.appendingPathComponent("Models").path
    public var port = 31418
    public var maxTokens = 2048
    public var temperature = 0.7
    public var generationProfile: GenerationProfile = .chatbot
    public var useRuntimeSampling = true
    public init() {}

    private enum CodingKeys: String, CodingKey {
        case executable, modelsDirectory, port, maxTokens, temperature, generationProfile, useRuntimeSampling
    }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        executable = try values.decode(String.self, forKey: .executable)
        modelsDirectory = try values.decode(String.self, forKey: .modelsDirectory)
        port = try values.decode(Int.self, forKey: .port)
        maxTokens = try values.decode(Int.self, forKey: .maxTokens)
        temperature = try values.decode(Double.self, forKey: .temperature)
        // Existing installs keep their exact behavior until a profile is selected.
        generationProfile = try values.decodeIfPresent(GenerationProfile.self, forKey: .generationProfile) ?? .custom
        useRuntimeSampling = try values.decodeIfPresent(Bool.self, forKey: .useRuntimeSampling) ?? false
    }

    public func hasRuntimeChanges(comparedTo saved: Self) -> Bool {
        executable != saved.executable || modelsDirectory != saved.modelsDirectory || port != saved.port
    }

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
        try RuntimeLocator.find(explicit: executable,
            bundled: bundle.bundleURL.appendingPathComponent("Contents/Helpers/AXEngineRuntime.bundle/Contents/MacOS/ax-engine").path,
            development: bundle.object(forInfoDictionaryKey: "AXEngineDevelopmentRuntime") as? String)
    }
}

public enum RuntimeLocator {
    public static func find(explicit: String = "", bundled: String? = nil, development: String? = nil,
                            environment: [String: String] = ProcessInfo.processInfo.environment,
                            home: URL = FileManager.default.homeDirectoryForCurrentUser,
                            systemDirectories: [String] = ["/opt/homebrew/bin", "/usr/local/bin"]) throws -> URL {
        let fm = FileManager.default
        func executable(_ path: String) -> Bool {
            var directory: ObjCBool = false
            return path.hasPrefix("/") && fm.fileExists(atPath: path, isDirectory: &directory)
                && !directory.boolValue && fm.isExecutableFile(atPath: path)
        }
        let selected = explicit.isEmpty ? (environment["AX_ENGINE_BIN"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines) : explicit
        if !selected.isEmpty {
            guard executable(selected) else { throw AppFailure.message("The selected runtime is unavailable. Check Settings or AX_ENGINE_BIN.") }
            return URL(fileURLWithPath: selected)
        }
        // A distribution uses its qualified runtime unless explicitly overridden.
        var candidates = [bundled].compactMap { $0 }
        candidates += (environment["PATH"] ?? "").split(separator: ":").filter { $0.hasPrefix("/") }.map { String($0) + "/ax-engine" }
        candidates += systemDirectories.map { $0 + "/ax-engine" }
        if let virtualEnv = environment["VIRTUAL_ENV"], virtualEnv.hasPrefix("/") {
            candidates.append(virtualEnv + "/bin/ax-engine")
        }
        candidates.append(home.appendingPathComponent(".local/bin/ax-engine").path)
        candidates.append(home.appendingPathComponent(".local/ax-engine/ax-engine").path)
        // AX Code Desktop delegates this discovery to its runtime. Use the same
        // managed install marker and complete payload boundary; do not scan or
        // adopt unfinished version directories or overwrite another app's install.
        let stateRoot = environment["XDG_STATE_HOME"].map { URL(fileURLWithPath: $0) } ?? home.appendingPathComponent(".local/state")
        let cacheRoot = environment["XDG_CACHE_HOME"].map { URL(fileURLWithPath: $0) } ?? home.appendingPathComponent(".cache")
        let marker = stateRoot.appendingPathComponent("ax-code/ax-engine/install.json")
        struct ManagedInstall: Decodable { let path: String; let version: String }
        if let data = try? Data(contentsOf: marker), data.count < 65_536,
           let install = try? JSONDecoder().decode(ManagedInstall.self, from: data),
           install.version.range(of: #"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$"#, options: .regularExpression) != nil {
            let directory = cacheRoot.appendingPathComponent("ax-code/ax-engine/bin").appendingPathComponent(install.version)
            let expected = directory.appendingPathComponent("ax-engine").standardizedFileURL
            if URL(fileURLWithPath: install.path).standardizedFileURL == expected,
               ["ax-engine", "ax-engine-server", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"].allSatisfy({
                   var isDirectory: ObjCBool = false
                   return fm.fileExists(atPath: directory.appendingPathComponent($0).path, isDirectory: &isDirectory) && !isDirectory.boolValue
               }) {
                candidates.append(expected.path)
            }
        }
        candidates += [development].compactMap { $0 }
        let pythonRoot = home.appendingPathComponent("Library/Python")
        let pythonVersions = ((try? fm.contentsOfDirectory(at: pythonRoot, includingPropertiesForKeys: nil)) ?? [])
            .sorted { $0.lastPathComponent.compare($1.lastPathComponent, options: .numeric) == .orderedDescending }
        candidates += pythonVersions.map { $0.appendingPathComponent("bin/ax-engine").path }
        guard let path = candidates.first(where: executable) else {
            throw AppFailure.message("Could not locate AX Engine automatically. Select your installed executable in Settings.")
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
