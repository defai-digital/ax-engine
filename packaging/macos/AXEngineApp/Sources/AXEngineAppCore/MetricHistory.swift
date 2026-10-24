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

public struct MetricHistory: Sendable {
    public private(set) var samples: [MetricSample] = []
    private var sequence = 0
    public let capacity: Int
    public let duration: TimeInterval
    public init(capacity: Int = 180, duration: TimeInterval = 300) {
        self.capacity = max(1, capacity); self.duration = max(1, duration)
    }
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
