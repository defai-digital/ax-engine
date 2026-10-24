import Foundation

public struct MetricSample: Identifiable, Sendable {
    public let id: Int
    public let time: Date
    public let values: [String: Double]
}

public struct MetricPlotPoint: Identifiable, Sendable {
    public let id: Int
    public let time: Date
    public let value: Double
    public let segment: Int
}

public enum ActivityHistoryRange: Double, CaseIterable, Sendable {
    case minute = 60, fiveMinutes = 300, fifteenMinutes = 900
    case thirtyMinutes = 1800, hour = 3600, fourHours = 14400
    public var label: String {
        switch self {
        case .minute: "1 min"
        case .fiveMinutes: "5 min"
        case .fifteenMinutes: "15 min"
        case .thirtyMinutes: "30 min"
        case .hour: "1 hour"
        case .fourHours: "4 hours"
        }
    }
    public var spokenLabel: String {
        switch self {
        case .minute: "1 minute"
        case .fiveMinutes: "5 minutes"
        case .fifteenMinutes: "15 minutes"
        case .thirtyMinutes: "30 minutes"
        case .hour: "1 hour"
        case .fourHours: "4 hours"
        }
    }
}

public struct MetricHistory: Sendable {
    public private(set) var samples: [MetricSample] = []
    private var sequence = 0
    public let capacity: Int
    public let duration: TimeInterval
    public init(capacity: Int = 180, duration: TimeInterval = 300) {
        self.capacity = max(1, capacity); self.duration = max(1, duration)
    }
    // Four hours at the five-second host cadence, with headroom for gap markers.
    public static func hostActivity() -> Self { Self(capacity: 6000, duration: ActivityHistoryRange.fourHours.rawValue) }
    public mutating func append(_ values: [String: Double], at time: Date = Date()) {
        if let last = samples.last, time <= last.time { samples.removeAll() }
        sequence += 1
        samples.append(MetricSample(id: sequence, time: time, values: values.filter { $0.value.isFinite && $0.value >= 0 }))
        samples.removeAll { time.timeIntervalSince($0.time) > duration }
        if samples.count > capacity { samples.removeFirst(samples.count - capacity) }
    }
    public func latest(_ key: String, now: Date = Date(), maxAge: TimeInterval = 12) -> Double? {
        guard let sample = samples.last, now >= sample.time, now.timeIntervalSince(sample.time) <= maxAge else { return nil }
        return sample.values[key]
    }
    /// Inspect an actual sample; do not interpolate across missing data or pauses.
    public func sample(_ key: String, near time: Date, now: Date = Date(), tolerance: TimeInterval = 6) -> MetricSample? {
        guard time <= now, now.timeIntervalSince(time) <= duration,
              let nearest = samples.filter({ $0.time <= now && now.timeIntervalSince($0.time) <= duration })
                .min(by: { abs($0.time.timeIntervalSince(time)) < abs($1.time.timeIntervalSince(time)) }),
              abs(nearest.time.timeIntervalSince(time)) <= tolerance,
              nearest.values[key] != nil else { return nil }
        return nearest
    }
    /// Bound rendering work using real first/min/max/last samples in time buckets.
    /// Keep original segment identities so reduction cannot bridge missing data.
    public func plotPoints(_ key: String, window: TimeInterval, now: Date = Date(), maxPoints: Int = 600) -> [MetricPlotPoint] {
        let start = now.addingTimeInterval(-max(1, window))
        let visible = points(key, now: now).filter { $0.time >= start }
        let budget = max(4, maxPoints)
        guard visible.count > budget else { return visible }
        let buckets = max(1, budget / 4)
        let bucketWidth = max(1, window) / Double(buckets)
        let groups = Dictionary(grouping: visible) {
            min(buckets - 1, max(0, Int($0.time.timeIntervalSince(start) / bucketWidth)))
        }
        return groups.keys.sorted().flatMap { bucket -> [MetricPlotPoint] in
            let values = groups[bucket]!
            let candidates = [values.first!, values.min(by: { $0.value < $1.value })!,
                              values.max(by: { $0.value < $1.value })!, values.last!]
            var seen = Set<Int>()
            return candidates.filter { seen.insert($0.id).inserted }.sorted { $0.time < $1.time }
        }
    }
    public func points(_ key: String, now: Date = Date(), maxGap: TimeInterval = 12) -> [MetricPlotPoint] {
        var segment = 0, points: [MetricPlotPoint] = []
        var previous: Date?
        for sample in samples where sample.time <= now && now.timeIntervalSince(sample.time) <= duration {
            guard let value = sample.values[key] else { segment += 1; previous = nil; continue }
            if let previous, sample.time.timeIntervalSince(previous) > maxGap { segment += 1 }
            points.append(MetricPlotPoint(id: sample.id, time: sample.time, value: value, segment: segment))
            previous = sample.time
        }
        return points
    }
}

public struct MTPCounterWindow: Sendable {
    private var previous: (time: Date, draft: Double, accepted: Double)?
    public init() {}
    public mutating func reset() { previous = nil }
    public mutating func observe(_ metrics: [String: Double], at time: Date = Date()) -> Double? {
        guard let draft = metrics["ax_engine_mtp_draft_tokens_total"],
              let accepted = metrics["ax_engine_mtp_accepted_tokens_total"],
              draft.isFinite, accepted.isFinite, draft >= 0, accepted >= 0, accepted <= draft else {
            reset(); return nil
        }
        defer { previous = (time, draft, accepted) }
        guard let previous, time > previous.time, time.timeIntervalSince(previous.time) <= 12,
              draft >= previous.draft, accepted >= previous.accepted else { return nil }
        let drafts = draft - previous.draft, accepts = accepted - previous.accepted
        guard drafts > 0, accepts <= drafts else { return nil }
        return 100 * accepts / drafts
    }
}
