import Foundation
import Combine
import AxEngine

/// One side of an A/B speed comparison (`.internal/ui/07-playground-compare.svg`).
/// Speed only: the same prompt and settings run on A, then on B.
public struct CompareSide: Equatable, Sendable {
    public enum Phase: Equatable, Sendable {
        case idle, waiting, loading, running, done, stopped
        case failed(String)
    }
    public var path = ""
    public var title = ""
    public var phase = Phase.idle
    public var text = ""
    public var firstTextSeconds: Double?
    public var totalSeconds: Double?
    public var performance: ChatPerformance?
    /// Accepted / drafted MTP tokens across the run, from the server's
    /// counters. Includes any other traffic the server handled meanwhile.
    public var draftAcceptedPercent: Double?
    /// Seconds since Send and the cumulative streamed characters at that time.
    public var samples: [Sample] = []
    public struct Sample: Equatable, Sendable {
        public let seconds: Double
        public let characters: Int
        public init(seconds: Double, characters: Int) { self.seconds = seconds; self.characters = characters }
    }
    public init(path: String = "", title: String = "") { self.path = path; self.title = title }

    public var done: Bool { phase == .done }

    /// Output tokens over time. The stream carries text, not token ids, so
    /// characters are scaled to the runtime's final decode token count; the
    /// curve starts flat until the first text.
    public var tokenCurve: [(seconds: Double, tokens: Double)] {
        guard let last = samples.last, last.characters > 0,
              let decoded = performance?.decodeTokens, decoded > 0 else { return [] }
        let scale = Double(decoded) / Double(last.characters)
        var points: [(Double, Double)] = [(0, 0)]
        if let first = samples.first { points.append((first.seconds, 0)) }
        points += samples.map { ($0.seconds, Double($0.characters) * scale) }
        if let total = totalSeconds, total > last.seconds { points.append((total, Double(last.characters) * scale)) }
        return points
    }
}

/// The verdict rows, declared only after both runs finish.
public struct CompareSummary: Equatable, Sendable {
    public enum Winner: Equatable, Sendable { case a, b, tie }
    /// Which side finished first and by how many seconds.
    public let finished: (winner: Winner, seconds: Double)?
    /// A's decode rate over B's.
    public let decodeRatio: Double?
    /// B's first-text time minus A's, in milliseconds (positive: A sooner).
    public let firstTextDeltaMs: Double?
    /// A's draft acceptance minus B's, in percentage points.
    public let draftDeltaPoints: Double?

    public init?(a: CompareSide, b: CompareSide) {
        guard a.done, b.done else { return nil }
        if let ta = a.totalSeconds, let tb = b.totalSeconds {
            let delta = tb - ta
            finished = (abs(delta) < 0.05 ? .tie : delta > 0 ? .a : .b, abs(delta))
        } else { finished = nil }
        if let da = a.performance?.decodeTokensPerSecond, let db = b.performance?.decodeTokensPerSecond, da > 0, db > 0 {
            decodeRatio = da / db
        } else { decodeRatio = nil }
        if let fa = a.firstTextSeconds, let fb = b.firstTextSeconds { firstTextDeltaMs = (fb - fa) * 1000 } else { firstTextDeltaMs = nil }
        if let pa = a.draftAcceptedPercent, let pb = b.draftAcceptedPercent { draftDeltaPoints = pa - pb } else { draftDeltaPoints = nil }
    }

    public static func == (lhs: CompareSummary, rhs: CompareSummary) -> Bool {
        lhs.finished?.winner == rhs.finished?.winner && lhs.finished?.seconds == rhs.finished?.seconds
            && lhs.decodeRatio == rhs.decodeRatio && lhs.firstTextDeltaMs == rhs.firstTextDeltaMs
            && lhs.draftDeltaPoints == rhs.draftDeltaPoints
    }
}

/// Holds both sides and streams one run. The app sequences the runs and the
/// server restarts between them; this store never touches the server state.
@MainActor
public final class CompareStore: ObservableObject {
    @Published public var prompt = ""
    @Published public var a = CompareSide()
    @Published public var b = CompareSide()
    /// Restart the server on A after B when A was serving at the start.
    @Published public var returnToA = true
    @Published public private(set) var running = false
    @Published public var status: String?
    /// A lasting message after the run, such as a failed restore.
    @Published public var notice: String?
    private var task: Task<Void, Never>?

    public init() {}

    public var summary: CompareSummary? { CompareSummary(a: a, b: b) }

    /// Runs `body` as the compare task; Cancel stops it and its stream.
    public func begin(_ body: @escaping @MainActor () async -> Void) {
        guard !running else { return }
        running = true
        task = Task {
            await body()
            running = false; task = nil
        }
    }
    public func cancel() { task?.cancel() }

    public func resetSides(aPath: String, aTitle: String, bPath: String, bTitle: String) {
        a = CompareSide(path: aPath, title: aTitle); a.phase = .waiting
        b = CompareSide(path: bPath, title: bTitle); b.phase = .waiting
        status = nil; notice = nil
    }

    public func update(_ isA: Bool, _ change: (inout CompareSide) -> Void) {
        if isA { change(&a) } else { change(&b) }
    }

    /// Streams the prompt once on the given connection and records timing.
    public func stream(isA: Bool, prompt: String, system systemPrompt: String, model: String, connection: EngineConnection,
                       settings: AppSettings, supportsTemperature: Bool) async {
        update(isA) { $0.phase = .running; $0.text = ""; $0.samples = []; $0.firstTextSeconds = nil
            $0.totalSeconds = nil; $0.performance = nil; $0.draftAcceptedPercent = nil }
        let before = try? await connection.metrics()
        var messages: [OpenAiChatMessage] = []
        if !systemPrompt.isEmpty { messages.append(.init(role: "system", content: systemPrompt)) }
        messages.append(.init(role: "user", content: prompt))
        var request = settings.chatRequest(model: model, messages: messages, supportsTemperature: supportsTemperature)
        request.streamOptions = .init(includeUsage: true)
        let started = Date()
        var text = "", samples: [CompareSide.Sample] = [], requestID: Int?
        var lastPublish = Date.distantPast
        do {
            for try await chunk in connection.sdk.streamChatCompletion(request) {
                try Task.checkCancellation()
                if requestID == nil { requestID = EngineConnection.requestID(chunk.id) }
                if let measurement = chunk.usage?.axEngine, let performance = ChatPerformance(measurement) {
                    update(isA) { $0.performance = performance }
                }
                for choice in chunk.choices {
                    if let delta = choice.delta.content, !delta.isEmpty {
                        let now = Date().timeIntervalSince(started)
                        if samples.isEmpty { update(isA) { $0.firstTextSeconds = now } }
                        guard text.utf8.count + delta.utf8.count <= 512 * 1024 else {
                            throw AppFailure.message("The reply exceeded 512 KiB; compare stopped.")
                        }
                        text += delta
                        samples.append(.init(seconds: now, characters: (samples.last?.characters ?? 0) + delta.count))
                    }
                    if choice.finishReason == "error" { throw AppFailure.message("The runtime reported a generation error.") }
                    if choice.finishReason == "tool_calls" { throw AppFailure.message("The model requested tools; compare does not run tools.") }
                }
                // Publish at most 20 times a second.
                if Date().timeIntervalSince(lastPublish) > 0.05 {
                    lastPublish = Date()
                    let snapshot = text, points = samples
                    update(isA) { $0.text = snapshot; $0.samples = points }
                }
            }
            try Task.checkCancellation()
            let total = Date().timeIntervalSince(started)
            let after = try? await connection.metrics()
            let draft = Self.draftAcceptance(before: before, after: after)
            update(isA) { $0.text = text; $0.samples = samples; $0.totalSeconds = total; $0.draftAcceptedPercent = draft; $0.phase = .done }
        } catch {
            // This task may already be cancelled, which would abort the
            // cancel request itself; send it from a fresh task.
            if let requestID { Task.detached { try? await connection.cancel(requestID: requestID) } }
            let cancelled = Task.isCancelled || error is CancellationError
            update(isA) { $0.text = text; $0.samples = samples
                $0.phase = cancelled ? .stopped : .failed(error.localizedDescription) }
        }
    }

    /// Accepted / drafted MTP tokens between two counter snapshots, or nil
    /// when the counters are missing, reset or did not move.
    public nonisolated static func draftAcceptance(before: [String: Double]?, after: [String: Double]?) -> Double? {
        guard let d0 = before?["ax_engine_mtp_draft_tokens_total"], let a0 = before?["ax_engine_mtp_accepted_tokens_total"],
              let d1 = after?["ax_engine_mtp_draft_tokens_total"], let a1 = after?["ax_engine_mtp_accepted_tokens_total"],
              d1 > d0, a1 >= a0, a1 - a0 <= d1 - d0 else { return nil }
        return 100 * (a1 - a0) / (d1 - d0)
    }
}
