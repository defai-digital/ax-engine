import Foundation
import Combine

public struct CatalogModel: Decodable, Identifiable, Sendable, Equatable {
    public let alias: String
    public let repoId: String
    public let revision: String?
    public let mtpIncluded: Bool?
    public var id: String { repoId }
    public var target: String { repoId + (revision.map { "@" + $0 } ?? "") }
    public var folderName: String { repoId.replacingOccurrences(of: "/", with: "--") + (revision.map { "--" + $0 } ?? "") }
}

public struct LocalModel: Identifiable, Sendable, Equatable {
    public let path: String
    public let ready: Bool
    public let managed: Bool
    public var id: String { path }
    public var name: String {
        let url = URL(fileURLWithPath: path)
        if url.deletingLastPathComponent().lastPathComponent == "snapshots" {
            return url.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent
                .replacingOccurrences(of: "models--", with: "").replacingOccurrences(of: "--", with: "/")
                + " · " + String(url.lastPathComponent.prefix(8))
        }
        return url.lastPathComponent
    }
}

public struct DownloadItem: Identifiable, Sendable {
    public let id = UUID()
    public let model: CatalogModel
    public let destination: String
    public var state = "Queued"
    public var fraction: Double?
    public var detail = "Waiting"
}

public struct DownloadProgress: Equatable, Sendable {
    public var fraction: Double?
    public var detail: String?
    public var terminalReady = false
    public var destination: String?
    public static func parse(_ output: String) -> Self {
        var result = Self()
        for line in output.split(separator: "\n") {
            guard let object = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else { continue }
            if object["event"] as? String == "progress" {
                if let done = object["done"] as? Double, let total = object["total"] as? Double,
                   total > 0, done.isFinite, total.isFinite {
                    result.fraction = min(1, max(0, done / total))
                }
                result.detail = object["file"] as? String
            }
            if object["schema_version"] as? String == "ax.download_model.v1" {
                result.terminalReady = object["status"] as? String == "ready"
                result.destination = object["dest"] as? String
                if let errors = object["errors"] as? [String], !errors.isEmpty { result.detail = errors.joined(separator: "\n") }
            }
        }
        return result
    }
}

@MainActor
public final class ModelStore: ObservableObject {
    @Published public private(set) var catalog: [CatalogModel] = []
    @Published public private(set) var library: [LocalModel] = []
    @Published public private(set) var downloads: [DownloadItem] = []
    @Published public private(set) var error: String?
    @Published public private(set) var refreshing = false
    @Published public private(set) var downloading = false
    @Published public private(set) var removing = false
    public let downloadJob = ProcessJob()
    private let catalogJob = ProcessJob()
    private let removalJob = ProcessJob()
    private var queueTask: Task<Void, Never>?
    private var imported: [String] = []
    private let importsURL: URL
    public init(importsURL: URL = AppPaths.support.appendingPathComponent("imported-models.json")) {
        self.importsURL = importsURL
        if let data = try? Data(contentsOf: importsURL), let paths = try? JSONDecoder().decode([String].self, from: data) { imported = paths }
    }

    public func refresh(settings: AppSettings) async {
        guard !refreshing else { return }
        refreshing = true; error = nil; catalog = []
        defer { refreshing = false }
        do {
            let result = try await catalogJob.run(executable: settings.runtime(),
                arguments: ["download", "--list", "--json"], timeout: 30)
            guard result.status == 0, !result.truncated else { throw AppFailure.message("Cannot read runtime catalog: " + String(result.stderr.suffix(2048))) }
            struct Catalog: Decodable { let schemaVersion: String; let targets: [CatalogModel] }
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let value = try decoder.decode(Catalog.self, from: Data(result.stdout.utf8))
            guard value.schemaVersion == "ax.download_options.v1" else { throw AppFailure.message("Unsupported runtime catalog version.") }
            catalog = value.targets
        } catch { self.error = error.localizedDescription }
        await scan(settings: settings)
    }

    public func scan(settings: AppSettings) async {
        let root = settings.modelsDirectory, extra = imported
        library = await Task.detached(priority: .utility) { Self.scanLibrary(root: root, imported: extra) }.value
    }

    public nonisolated static func scanLibrary(root: String, imported: [String], environment: [String: String] = ProcessInfo.processInfo.environment) -> [LocalModel] {
        let fm = FileManager.default
        let rootURL = URL(fileURLWithPath: root)
        var paths = (try? fm.contentsOfDirectory(at: rootURL, includingPropertiesForKeys: nil)) ?? []
        paths += imported.map { URL(fileURLWithPath: $0) }
        let cache: URL
        if let path = environment["HF_HUB_CACHE"] { cache = URL(fileURLWithPath: path) }
        else if let path = environment["HF_HOME"] { cache = URL(fileURLWithPath: path).appendingPathComponent("hub") }
        else if let path = environment["XDG_CACHE_HOME"] { cache = URL(fileURLWithPath: path).appendingPathComponent("huggingface/hub") }
        else { cache = fm.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub") }
        for repo in (try? fm.contentsOfDirectory(at: cache, includingPropertiesForKeys: nil)) ?? [] where repo.lastPathComponent.hasPrefix("models--") {
            paths += (try? fm.contentsOfDirectory(at: repo.appendingPathComponent("snapshots"), includingPropertiesForKeys: nil)) ?? []
        }
        var seen = Set<String>()
        return paths.compactMap { path in
            let canonical = path.resolvingSymlinksInPath().standardizedFileURL.path
            guard seen.insert(canonical).inserted, fm.fileExists(atPath: path.appendingPathComponent("config.json").path) else { return nil }
            return LocalModel(path: canonical,
                ready: fm.fileExists(atPath: path.appendingPathComponent("model-manifest.json").path),
                managed: canRemove(path: canonical, root: root))
        }.sorted { $0.path < $1.path }
    }

    public nonisolated static func canRemove(path: String, root: String) -> Bool {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let parent = URL(fileURLWithPath: root).resolvingSymlinksInPath().standardizedFileURL
        return url == url.resolvingSymlinksInPath() && url.deletingLastPathComponent() == parent
            && url.path != parent.path && FileManager.default.fileExists(atPath: url.appendingPathComponent("model-manifest.json").path)
    }

    public func importFolder(_ url: URL, settings: AppSettings) async {
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) else { error = "Choose a model directory containing config.json."; return }
        let path = url.resolvingSymlinksInPath().path
        if !imported.contains(path) { imported.append(path) }
        do {
            try FileManager.default.createDirectory(at: importsURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(imported).write(to: importsURL, options: .atomic)
            await scan(settings: settings)
        } catch { self.error = error.localizedDescription }
    }

    public func remove(_ model: LocalModel, settings: AppSettings) async {
        guard !downloading, !removing, Self.canRemove(path: model.path, root: settings.modelsDirectory) else {
            error = "Only an idle, app-managed model folder can be removed here."; return
        }
        removing = true
        defer { removing = false }
        do {
            let result = try await removalJob.run(executable: settings.runtime(),
                arguments: ["models", "rm", model.path, "--yes", "--json"], timeout: 60)
            guard result.status == 0 else { throw AppFailure.message(String(result.stderr.suffix(2048))) }
            await scan(settings: settings)
        } catch { self.error = error.localizedDescription }
    }

    public func enqueue(_ model: CatalogModel, settings: AppSettings) {
        guard !removing else { error = "Wait for model removal to finish."; return }
        guard !downloads.contains(where: { $0.model.target == model.target && ["Queued", "Downloading"].contains($0.state) }) else { return }
        guard model.repoId.split(separator: "/").count == 2,
              model.repoId.range(of: #"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"#, options: .regularExpression) != nil,
              model.revision?.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil || model.revision == nil else {
            error = "The runtime supplied an invalid model identifier."; return
        }
        let destination = URL(fileURLWithPath: settings.modelsDirectory).appendingPathComponent(model.folderName).path
        downloads.append(.init(model: model, destination: destination))
        if queueTask == nil {
            downloading = true
            queueTask = Task { await runQueue(settings: settings) }
        }
    }

    public func retry(_ id: UUID, settings: AppSettings) {
        guard let item = downloads.first(where: { $0.id == id }), ["Failed", "Cancelled"].contains(item.state) else { return }
        downloads.removeAll { $0.id == id }; enqueue(item.model, settings: settings)
    }
    public func cancel(_ id: UUID) {
        guard let index = downloads.firstIndex(where: { $0.id == id }) else { return }
        if downloads[index].state == "Downloading" { downloadJob.cancel() }
        else if downloads[index].state == "Queued" { downloads[index].state = "Cancelled" }
    }
    public func shutdown() async {
        for index in downloads.indices where downloads[index].state == "Queued" { downloads[index].state = "Cancelled" }
        catalogJob.cancel(); removalJob.cancel(); downloadJob.cancel()
        await queueTask?.value
        while catalogJob.running || removalJob.running {
            await Task.detached { try? await Task.sleep(for: .milliseconds(50)) }.value
        }
    }
    private func runQueue(settings: AppSettings) async {
        defer { queueTask = nil; downloading = false }
        while let index = downloads.firstIndex(where: { $0.state == "Queued" }) {
            let id = downloads[index].id, item = downloads[index]
            downloads[index].state = "Downloading"
            do {
                try FileManager.default.createDirectory(atPath: settings.modelsDirectory, withIntermediateDirectories: true)
                let result = try await downloadJob.run(executable: settings.runtime(),
                    arguments: ["download", item.model.target, "--dest", item.destination, "--progress-json"]) { [weak self] value in
                        guard let self, let i = self.downloads.firstIndex(where: { $0.id == id }) else { return }
                        let progress = DownloadProgress.parse(String(value.stdout.suffix(65_536)))
                        self.downloads[i].fraction = progress.fraction
                        if let detail = progress.detail { self.downloads[i].detail = detail }
                    }
                let progress = DownloadProgress.parse(result.stdout)
                guard result.status == 0, progress.terminalReady, let destination = progress.destination,
                      URL(fileURLWithPath: destination).resolvingSymlinksInPath() == URL(fileURLWithPath: item.destination).resolvingSymlinksInPath(),
                      FileManager.default.fileExists(atPath: URL(fileURLWithPath: destination).appendingPathComponent("model-manifest.json").path),
                      FileManager.default.fileExists(atPath: URL(fileURLWithPath: destination).appendingPathComponent("config.json").path) else {
                    throw AppFailure.message(progress.detail ?? String(result.stderr.suffix(2048)))
                }
                if let i = downloads.firstIndex(where: { $0.id == id }) { downloads[i].state = "Ready"; downloads[i].fraction = 1 }
                await scan(settings: settings)
            } catch {
                if let i = downloads.firstIndex(where: { $0.id == id }) {
                    downloads[i].state = error is CancellationError ? "Cancelled" : "Failed"
                    downloads[i].detail = error is CancellationError ? "Download cancelled. Retry resumes through AX Engine." : error.localizedDescription
                }
            }
        }
    }
}
