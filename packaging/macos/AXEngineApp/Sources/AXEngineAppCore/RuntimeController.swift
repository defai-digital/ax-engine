import Foundation
import Combine

// MTPLX's supervisor/client separation is retained, with AX Engine contracts.
// Ownership is a live Process handle; no PID-file adoption or port-based kills.
@MainActor
public final class RuntimeController: ObservableObject {
    @Published public private(set) var state = "Stopped"
    @Published public private(set) var models: [RunningModel] = []
    @Published public private(set) var metrics: [String: Double] = [:]
    @Published public private(set) var error: String?
    @Published public private(set) var connection: EngineConnection?
    @Published public private(set) var owned = false
    @Published public private(set) var busy = false
    @Published public private(set) var logs = ""
    @Published public private(set) var startedAt: Date?
    @Published public private(set) var lastSuccessfulCheck: Date?
    @Published public private(set) var selectedModel = ""
    @Published public private(set) var history = MetricHistory()
    private var mtpWindow = MTPCounterWindow()
    public let job = ProcessJob(terminationGrace: 8)
    private let logStore = BoundedLogStore(capacity: 250)
    private var generation = UUID()
    private var launch: Task<Void, Never>?
    private var polling: Task<Void, Never>?
    public var ready: Bool { connection != nil && !models.isEmpty && ["Running", "Attached"].contains(state) }
    private let portAvailable: @Sendable (Int) -> Bool
    public init(portAvailable: @escaping @Sendable (Int) -> Bool = PortProbe.isAvailable) {
        self.portAvailable = portAvailable
    }

    /// The last meaningful runtime diagnostic line, for an exit message.
    public nonisolated static func lastDiagnostic(_ stderr: String) -> String? {
        let lines = stderr.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }
        guard let line = lines.last(where: { !$0.isEmpty }) else { return nil }
        return line.count > 400 ? String(line.prefix(400)) + "…" : line
    }

    public func start(model: String, settings: AppSettings) {
        guard !busy, !owned, connection == nil else { return }
        do {
            try settings.validate()
            let executable = try settings.runtime()
            guard !model.isEmpty, !model.hasPrefix("-"), !model.contains("\0") else {
                throw AppFailure.message("Select an installed model or enter a model alias.")
            }
            // A second listener would fail to bind only after model load, and a
            // different service there could answer readiness probes instead.
            guard portAvailable(settings.port) else {
                throw AppFailure.message("Port \(settings.port) is already in use, possibly by another AX Engine service. Connect to it below, stop that service, or choose another port in Settings.")
            }
            let identity = UUID(); generation = identity
            history = MetricHistory(); mtpWindow.reset()
            let key = UUID().uuidString + UUID().uuidString
            let candidate = EngineConnection(port: settings.port, apiKey: key)
            busy = true; owned = true; state = "Starting"; error = nil; logs = ""
            selectedModel = model; startedAt = Date(); lastSuccessfulCheck = nil
            launch = Task { [self] in
                do {
                    let result = try await job.run(executable: executable,
                        arguments: ["serve", model, "--offline", "--host", "127.0.0.1", "--port", String(settings.port)],
                        environment: ["AX_ENGINE_API_KEY": key]) { [weak self] value in
                            self?.logs = String(value.stderr.suffix(32_768)).replacingOccurrences(of: key, with: "[redacted]")
                        }
                    guard generation == identity else { return }
                    let detail = Self.lastDiagnostic(result.stderr.replacingOccurrences(of: key, with: "[redacted]"))
                    error = "Runtime exited (\(result.status))" + (detail.map { ": " + $0 } ?? ". Check the runtime output.")
                } catch is CancellationError {
                    guard generation == identity else { return }
                } catch {
                    guard generation == identity else { return }
                    self.error = error.localizedDescription.replacingOccurrences(of: key, with: "[redacted]")
                }
                guard generation == identity else { return }
                polling?.cancel(); connection = nil; models = []; metrics = [:]
                history.append([:]); mtpWindow.reset()
                busy = false; owned = false; state = error == nil ? "Stopped" : "Failed"
            }
            polling = Task {
                let deadline = Date().addingTimeInterval(600)
                while !Task.isCancelled && generation == identity {
                    do {
                        let discovered = try await candidate.models()
                        guard !Task.isCancelled, generation == identity, job.running else { return }
                        models = discovered; connection = candidate; state = "Running"; busy = false; lastSuccessfulCheck = Date()
                        await monitor(candidate, identity: identity)
                        return
                    } catch {
                        guard !Task.isCancelled, generation == identity else { return }
                        if Date() > deadline {
                            self.error = "Runtime readiness timed out. Stop the runtime or inspect its output."
                            state = "Not ready"; busy = false
                            return
                        }
                    }
                    try? await Task.sleep(for: .seconds(1))
                }
            }
        } catch { self.error = error.localizedDescription; state = "Failed" }
    }

    public func attach(port: Int, apiKey: String) async {
        guard !busy, !owned, connection == nil else { return }
        guard (1024...65535).contains(port) else { error = "Choose a port between 1024 and 65535."; return }
        let identity = UUID(); generation = identity
        history = MetricHistory(); mtpWindow.reset()
        busy = true; state = "Connecting"; error = nil
        startedAt = Date(); lastSuccessfulCheck = nil; selectedModel = ""
        let candidate = EngineConnection(port: port, apiKey: apiKey)
        do {
            let discovered = try await candidate.models()
            guard generation == identity else { return }
            connection = candidate; models = discovered; state = "Attached"; busy = false; lastSuccessfulCheck = Date()
            polling = Task { await monitor(candidate, identity: identity) }
        } catch {
            guard generation == identity else { return }
            self.error = error.localizedDescription; state = "Failed"; busy = false
        }
    }

    private func monitor(_ client: EngineConnection, identity: UUID) async {
        while !Task.isCancelled && generation == identity {
            do {
                let latest = try await client.metrics()
                let discovered = try await client.models()
                guard !Task.isCancelled, generation == identity else { return }
                if models.map(\.id).sorted() != discovered.map(\.id).sorted() {
                    history = MetricHistory(); mtpWindow.reset()
                }
                let checked = Date()
                var plotted = latest
                plotted["mtpAcceptedPercent"] = mtpWindow.observe(latest, at: checked)
                history.append(plotted, at: checked)
                metrics = latest; models = discovered; error = nil; state = owned ? "Running" : "Attached"; lastSuccessfulCheck = checked
            } catch {
                guard !Task.isCancelled, generation == identity else { return }
                // A busy, unauthorized or changed-schema service is not dead.
                // Never auto-kill a serving model on a probe failure.
                metrics = [:]; history.append([:]); mtpWindow.reset()
                state = "Connection degraded"; self.error = error.localizedDescription
            }
            try? await Task.sleep(for: .seconds(2))
        }
    }

    public func stop() async {
        guard owned else { detach(); return }
        generation = UUID(); polling?.cancel(); polling = nil
        state = "Stopping"; busy = true
        history.append([:]); mtpWindow.reset()
        launch?.cancel(); job.cancel()
        await launch?.value
        await logStore.append(logs, stream: .stderr)
        connection = nil; models = []; metrics = [:]; owned = false; busy = false; state = "Stopped"
        error = nil; lastSuccessfulCheck = nil; startedAt = nil
    }

    /// Stop the app-owned server and start it again, optionally on another
    /// model. Settings such as the port are re-read for the new launch.
    public func restart(model: String? = nil, settings: AppSettings) async {
        guard owned, state != "Stopping" else { return }
        let target = model ?? selectedModel
        await stop()
        start(model: target, settings: settings)
    }

    public func detach() {
        guard !owned else { return }
        generation = UUID(); polling?.cancel(); polling = nil
        history.append([:]); mtpWindow.reset()
        connection = nil; models = []; metrics = [:]; busy = false; state = "Stopped"; error = nil; lastSuccessfulCheck = nil; startedAt = nil
    }
}
