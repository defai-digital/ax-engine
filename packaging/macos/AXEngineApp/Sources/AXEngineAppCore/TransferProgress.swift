import Foundation

/// Download phases reported by `ax-engine download --progress-json`. Older
/// runtimes omit the field; the GUI then infers a phase from the ratio.
public enum DownloadPhase: String, Sendable, Equatable {
    case inventory, waiting, transfer, validate, ready

    public var title: String {
        switch self {
        case .inventory: "Checking model files"
        case .waiting: "Waiting for the download service"
        case .transfer: "Downloading"
        case .validate: "Verifying model files"
        case .ready: "Ready"
        }
    }

    /// The legacy ratio contract: transfers span 0–90 %, validation 95 %.
    public static func inferred(fromFraction fraction: Double?) -> DownloadPhase? {
        guard let fraction else { return nil }
        if fraction >= 1 { return .ready }
        if fraction > 0.9 { return .validate }
        return .transfer
    }
}

/// Smoothed throughput over a short window, with a stall indication. Pure
/// value type so the estimate is testable without a running download.
public struct TransferRateEstimator: Sendable, Equatable {
    public static let window: TimeInterval = 12
    public static let stallAfter: TimeInterval = 30
    private var samples: [(time: Date, bytes: Int64)] = []
    private var lastAdvance: Date?
    public init() {}

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.samples.map(\.bytes) == rhs.samples.map(\.bytes) && lhs.samples.map(\.time) == rhs.samples.map(\.time)
    }

    /// Record the bytes transferred so far. Returns bytes per second when at
    /// least one second of history exists, otherwise nil.
    @discardableResult
    public mutating func observe(bytes: Int64, at time: Date = Date()) -> Double? {
        if let last = samples.last {
            // A resumed or restarted transfer may report fewer bytes; start over.
            if bytes < last.bytes || time < last.time { samples.removeAll(); lastAdvance = time }
            else if bytes > last.bytes { lastAdvance = time }
        } else {
            lastAdvance = time
        }
        samples.append((time, bytes))
        let cutoff = time.addingTimeInterval(-Self.window)
        while samples.count > 2, samples[1].time <= cutoff { samples.removeFirst() }
        return rate
    }

    /// Bytes per second across the retained window, or nil when unknown.
    public var rate: Double? {
        guard let first = samples.first, let last = samples.last else { return nil }
        let seconds = last.time.timeIntervalSince(first.time)
        guard seconds >= 1 else { return nil }
        return Double(last.bytes - first.bytes) / seconds
    }

    /// Seconds since bytes last advanced, when that exceeds the stall threshold.
    public func stalledFor(at time: Date = Date()) -> TimeInterval? {
        guard let lastAdvance else { return nil }
        let idle = time.timeIntervalSince(lastAdvance)
        return idle >= Self.stallAfter ? idle : nil
    }

    public static func remainingSeconds(remainingBytes: Int64, rate: Double?) -> TimeInterval? {
        guard remainingBytes >= 0, let rate, rate >= 1024 else { return nil }
        return Double(remainingBytes) / rate
    }
}

public enum TransferFormat {
    public static func bytes(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: max(0, value), countStyle: .file)
    }

    public static func rate(_ bytesPerSecond: Double?) -> String? {
        guard let bytesPerSecond, bytesPerSecond.isFinite, bytesPerSecond >= 1024 else { return nil }
        return bytes(Int64(bytesPerSecond)) + "/s"
    }

    /// "about 3 min left" style text; nil when no estimate is reliable.
    public static func remaining(_ seconds: TimeInterval?) -> String? {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return nil }
        if seconds < 10 { return "a few seconds left" }
        if seconds < 90 { return "about \(Int(seconds.rounded())) s left" }
        let minutes = Int((seconds / 60).rounded())
        if minutes < 90 { return "about \(minutes) min left" }
        let hours = Double(minutes) / 60
        return String(format: "about %.1f h left", hours)
    }

    public static func duration(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds.rounded()))
        let hours = total / 3600, minutes = (total % 3600) / 60, rest = total % 60
        if hours > 0 { return "\(hours) h \(minutes) min" }
        if minutes > 0 { return "\(minutes) min \(rest) s" }
        return "\(rest) s"
    }

    /// One line such as "4.2 GB of 20.4 GB · 85 MB/s · about 3 min left".
    /// `estimated` marks sizes derived from the catalog instead of the transfer.
    public static func summary(done: Int64?, total: Int64?, rate: Double?, remaining: TimeInterval?,
                               estimated: Bool = false) -> String? {
        var parts: [String] = []
        if let done, let total, total > 0 {
            parts.append((estimated ? "about " : "") + bytes(min(done, total)) + " of " + bytes(total))
        } else if let total, total > 0 {
            parts.append((estimated ? "about " : "") + bytes(total))
        }
        if let rate = Self.rate(rate) { parts.append(rate) }
        if let remaining = Self.remaining(remaining) { parts.append(remaining) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}
