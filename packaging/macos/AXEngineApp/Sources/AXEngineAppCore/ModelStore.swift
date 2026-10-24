import Foundation
import Combine

public struct CatalogModel: Decodable, Identifiable, Sendable, Equatable {
    public let alias: String
    public let repoId: String
    public let revision: String?
    public let mtpIncluded: Bool?
    public var id: String { alias + "|" + target }
    public var target: String { repoId + (revision.map { "@" + $0 } ?? "") }
    public var folderName: String { repoId.replacingOccurrences(of: "/", with: "--") + (revision.map { "--" + $0 } ?? "") }
}

public struct LocalModel: Codable, Identifiable, Sendable, Equatable {
    public let path: String
    public let ready: Bool
    public let managed: Bool
    public var source = "Local folder"
    public var displayName: String { name.components(separatedBy: " · ").first?.components(separatedBy: "/").last ?? name }
    public var id: String { path }
    public func isCatalogAssociated(_ catalog: [CatalogModel]) -> Bool {
        let url = URL(fileURLWithPath: path)
        if url.deletingLastPathComponent().lastPathComponent == "snapshots" {
            let repository = url.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent
            return catalog.contains { repository == "models--" + $0.repoId.replacingOccurrences(of: "/", with: "--") }
        }
        return managed && catalog.contains { url.lastPathComponent == $0.folderName }
    }
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
    @Published public private(set) var scanning = false
    @Published public private(set) var libraryIssue: String?
    @Published public private(set) var checkedPaths = Set<String>()
    @Published public private(set) var lastScanAt: Date?
    @Published public private(set) var discoveryAttempted = false
    private var scanGeneration = 0
    private var activeScan: ProcessJob?
    private let scannerURL: URL
    private let scanTimeout: TimeInterval
    private let scanEnvironment: [String: String]
    @Published public private(set) var downloading = false
    @Published public private(set) var removing = false
    public let downloadJob = ProcessJob()
    private let catalogJob = ProcessJob()
    private let removalJob = ProcessJob()
    private var queueTask: Task<Void, Never>?
    private var imported: [String] = []
    private let importsURL: URL
    public init(importsURL: URL = AppPaths.support.appendingPathComponent("imported-models.json"),
                scannerURL: URL? = nil, scanTimeout: TimeInterval = 20,
                scanEnvironment: [String: String] = [:]) {
        let sibling = Bundle.main.executableURL!.deletingLastPathComponent().appendingPathComponent("AXEngineModelScanner")
        self.scannerURL = scannerURL ?? (FileManager.default.isExecutableFile(atPath: sibling.path) ? sibling
            : Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("AXEngineModelScanner"))
        self.scanTimeout = scanTimeout
        self.scanEnvironment = scanEnvironment
        self.importsURL = importsURL
        if let data = try? Data(contentsOf: importsURL), let paths = try? JSONDecoder().decode([String].self, from: data) { imported = paths }
    }

    public func refresh(settings: AppSettings) async {
        guard !refreshing else { return }
        refreshing = true; error = nil
        async let libraryScan: Void = scan(settings: settings)
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
        refreshing = false
        await libraryScan
    }

    public func scan(settings: AppSettings) async {
        scanGeneration += 1
        let generation = scanGeneration
        let previous = activeScan
        previous?.cancel()
        scanning = true; libraryIssue = nil; checkedPaths = []
        // A superseding scan waits for the previous owned worker to be reaped;
        // repeated navigation cannot accumulate blocked filesystem workers.
        while previous?.running == true { await Task.detached { try? await Task.sleep(for: .milliseconds(50)) }.value }
        guard generation == scanGeneration else { return }
        let job = ProcessJob(); activeScan = job
        var parsedLines = 0
        var found: [String: LocalModel] = [:]
        var completed = false
        var invalid = false
        var warnings: [String] = []
        defer {
            if generation == scanGeneration {
                scanning = false; discoveryAttempted = true; activeScan = nil
            }
        }
        do {
            guard FileManager.default.isExecutableFile(atPath: scannerURL.path) else {
                libraryIssue = "The library scanner is unavailable. Rebuild or reinstall the app, then retry."
                return
            }
            let request = LibraryScanRequest(root: settings.modelsDirectory, imported: imported)
            let encoded = String(decoding: try JSONEncoder().encode(request), as: UTF8.self)
            let result = try await job.run(executable: scannerURL, arguments: [encoded],
                environment: scanEnvironment, timeout: scanTimeout) { [weak self] output in
                guard let self, generation == self.scanGeneration, !output.truncated else { return }
                // Ignore a partial trailing line. ProcessJob batches UI updates.
                let lines = output.stdout.split(separator: "\n", omittingEmptySubsequences: false).dropLast()
                for line in lines.dropFirst(parsedLines) {
                    guard let event = try? JSONDecoder().decode(LibraryScanEvent.self, from: Data(line.utf8)) else { invalid = true; continue }
                    if let item = event.model { found[item.path] = item; self.checkedPaths.insert(item.path) }
                    if let warning = event.warning, warnings.count < 8 { warnings.append(warning) }
                    if event.complete { completed = true }
                }
                parsedLines = lines.count
                if !found.isEmpty {
                    var rows = Dictionary(self.library.map { ($0.path, $0) }, uniquingKeysWith: { _, new in new })
                    rows.merge(found) { _, new in new }
                    self.library = rows.values.sorted { $0.path < $1.path }
                }
                if !warnings.isEmpty { self.libraryIssue = warnings.joined(separator: "\n") }
            }
            guard generation == scanGeneration else { return }
            guard result.status == 0, !result.truncated, completed, !invalid else {
                throw AppFailure.message("The library scan did not finish. Retry to check remaining folders.")
            }
            if warnings.isEmpty {
                library = found.values.sorted { $0.path < $1.path }; lastScanAt = Date()
            }
        } catch {
            guard generation == scanGeneration else { return }
            libraryIssue = error is CancellationError
                ? "Scan stopped. Results already found are kept; some folders have not been checked."
                : "Scan incomplete. A folder may be slow or unavailable. Results already found are kept. Retry to check remaining folders."
        }
    }

    public func cancelScan() {
        guard scanning else { return }
        scanGeneration += 1
        activeScan?.cancel()
        scanning = false; discoveryAttempted = true
        libraryIssue = "Scan stopped. Results already found are kept; some folders have not been checked."
    }

    public nonisolated static func scanLibrary(root: String, imported: [String], environment: [String: String] = ProcessInfo.processInfo.environment,
                                               emit: ((LocalModel) -> Void)? = nil,
                                               warning: ((String) -> Void)? = nil) -> [LocalModel] {
        let fm = FileManager.default
        var found: [LocalModel] = [], seen = Set<String>()
        func entries(_ url: URL, optional: Bool = false) -> [URL] {
            do { return try fm.contentsOfDirectory(at: url, includingPropertiesForKeys: nil).sorted { $0.path < $1.path } }
            catch {
                let failure = error as NSError
                if !(optional && failure.domain == NSCocoaErrorDomain && failure.code == NSFileReadNoSuchFileError) {
                    warning?("Could not read " + url.lastPathComponent + ". Check its location or storage connection.")
                }
                return []
            }
        }
        func add(_ path: URL, source: String, ready: Bool? = nil) {
            let canonical = path.resolvingSymlinksInPath().standardizedFileURL.path
            guard !seen.contains(canonical), fm.fileExists(atPath: path.appendingPathComponent("config.json").path)
                || fm.fileExists(atPath: path.appendingPathComponent("model-manifest.json").path) else { return }
            seen.insert(canonical)
            let item = LocalModel(path: canonical, ready: ready ?? modelReady(path),
                                  managed: canRemove(path: canonical, root: root), source: source)
            found.append(item); emit?(item)
        }
        func linked(_ url: URL) -> Bool { (try? fm.destinationOfSymbolicLink(atPath: url.path)) != nil || url.path.hasPrefix("/Volumes/") }
        var deferred: [() -> Void] = []
        func visit(_ url: URL, action: @escaping () -> Void) {
            if linked(url) { deferred.append(action) } else { action() }
        }
        let rootURL = URL(fileURLWithPath: root)
        visit(rootURL) { for path in entries(rootURL, optional: !root.hasPrefix("/Volumes/")) { add(path, source: "App folder") } }
        if let configured = environment["AX_ENGINE_MODELS_DIR"], configured != root {
            let url = URL(fileURLWithPath: configured)
            visit(url) { for path in entries(url) { add(path, source: "Configured folder") } }
        }
        for path in imported.map({ URL(fileURLWithPath: $0) }) {
            visit(path) {
                guard fm.fileExists(atPath: path.path) else { warning?("An imported folder is unavailable. Reconnect its storage or import it again."); return }
                add(path, source: "Imported")
            }
        }
        let cache: URL
        if let path = environment["HF_HUB_CACHE"] { cache = URL(fileURLWithPath: path) }
        else if let path = environment["HF_HOME"] { cache = URL(fileURLWithPath: path).appendingPathComponent("hub") }
        else if let path = environment["XDG_CACHE_HOME"] { cache = URL(fileURLWithPath: path).appendingPathComponent("huggingface/hub") }
        else { cache = fm.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub") }
        func repository(_ repo: URL) {
            let snapshots = entries(repo.appendingPathComponent("snapshots"), optional: true)
                .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            let ranked = snapshots.map { snapshot in
                (url: snapshot, ready: modelReady(snapshot),
                 modified: (try? snapshot.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
            }.sorted { left, right in
                if left.ready != right.ready { return !left.ready && right.ready }
                return left.modified == right.modified ? left.url.path < right.url.path : left.modified < right.modified
            }
            if let selected = ranked.last { add(selected.url, source: "Hugging Face cache", ready: selected.ready) }
        }
        visit(cache) {
            for repo in entries(cache, optional: !cache.path.hasPrefix("/Volumes/")) where repo.lastPathComponent.hasPrefix("models--") {
                visit(repo) { repository(repo) }
            }
        }
        // Deferred roots may enqueue linked repositories. Drain in order.
        var index = 0
        while index < deferred.count { deferred[index](); index += 1 }
        return found.sorted { $0.path < $1.path }
    }

    // Same readiness boundary as the TUI catalog: metadata plus complete
    // safetensor files. The runtime still performs format/admission validation.
    public nonisolated static func modelReady(_ path: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: path.appendingPathComponent("config.json").path)
            || fm.fileExists(atPath: path.appendingPathComponent("model-manifest.json").path) else { return false }
        let index = path.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: index.path) {
            guard let size = try? index.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 16 * 1024 * 1024,
                  let data = try? Data(contentsOf: index),
                  let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let map = payload["weight_map"] as? [String: String], !map.isEmpty else { return false }
            return Set(map.values).allSatisfy { shard in
                guard !shard.isEmpty, !shard.hasPrefix("/"),
                      shard.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !["", ".", ".."].contains(String($0)) }) else { return false }
                var directory: ObjCBool = false
                return fm.fileExists(atPath: path.appendingPathComponent(shard).path, isDirectory: &directory) && !directory.boolValue
            }
        }
        return ((try? fm.contentsOfDirectory(at: path, includingPropertiesForKeys: [.isRegularFileKey])) ?? [])
            .contains {
                var directory: ObjCBool = false
                return $0.pathExtension.lowercased() == "safetensors" && fm.fileExists(atPath: $0.path, isDirectory: &directory) && !directory.boolValue
            }
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
        cancelScan(); catalogJob.cancel(); removalJob.cancel(); downloadJob.cancel()
        await queueTask?.value
        while catalogJob.running || removalJob.running || activeScan?.running == true {
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
                      Self.modelReady(URL(fileURLWithPath: destination)) else {
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
