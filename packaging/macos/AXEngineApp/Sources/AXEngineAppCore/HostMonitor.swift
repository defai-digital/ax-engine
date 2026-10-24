import Foundation
import Combine
import Dispatch
import CoreFoundation

public struct HostProbe: Sendable {
    public let kind: String
    public let executable: URL
    public let arguments: [String]
    public init(kind: String, executable: URL, arguments: [String]) {
        self.kind = kind; self.executable = executable; self.arguments = arguments
    }
    public static let defaults: [HostProbe] = [
        .init(kind: "cpu", executable: URL(fileURLWithPath: "/bin/ps"), arguments: ["-A", "-o", "%cpu="]),
        .init(kind: "memory", executable: URL(fileURLWithPath: "/usr/bin/vm_stat"), arguments: []),
        .init(kind: "gpu", executable: URL(fileURLWithPath: "/usr/sbin/ioreg"), arguments: ["-a", "-r", "-d", "1", "-c", "IOAccelerator"]),
    ]
}

public enum HostMetricParser {
    public static func cpu(_ text: String, cores: Int) -> Double? {
        let fields = text.split(whereSeparator: \.isWhitespace)
        let values = fields.compactMap { Double($0) }
        guard cores > 0, !values.isEmpty, fields.count == values.count,
              values.allSatisfy({ $0.isFinite && $0 >= 0 }) else { return nil }
        let sum = values.reduce(0, +)
        guard sum.isFinite else { return nil }
        return min(100, sum / Double(cores))
    }
    public static func memory(_ text: String, total: Double) -> Double? {
        guard total > 0, total.isFinite,
              let pageRange = text.range(of: #"page size of ([0-9]+) bytes"#, options: .regularExpression),
              let page = Double(text[pageRange].split(separator: " ")[3]), page > 0 else { return nil }
        var values: [String: Double] = [:]
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: ":", maxSplits: 1)
            if parts.count == 2 {
                values[String(parts[0])] = Double(parts[1].trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: ".")))
            }
        }
        let keys = ["Pages active", "Pages wired down", "Pages occupied by compressor"]
        let counts = keys.compactMap { values[$0] }
        guard counts.count == keys.count, counts.allSatisfy({ $0.isFinite && $0 >= 0 }) else { return nil }
        let bytes = counts.reduce(0, +) * page
        guard bytes.isFinite, bytes <= total else { return nil }
        return 100 * bytes / total
    }
    public static func gpu(_ text: String) -> Double? {
        guard let value = try? PropertyListSerialization.propertyList(from: Data(text.utf8), format: nil) else { return nil }
        func samples(_ object: Any) -> [Double] {
            if let dictionary = object as? [String: Any] {
                var result: [Double] = []
                if let stats = dictionary["PerformanceStatistics"] as? [String: Any],
                   let number = stats["Device Utilization %"] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() {
                    let percent = number.doubleValue
                    if percent.isFinite && (0...100).contains(percent) { result.append(percent) }
                }
                for nested in dictionary.values { result += samples(nested) }
                return result
            }
            if let array = object as? [Any] { return array.flatMap(samples) }
            return []
        }
        return samples(value).max()
    }
}

@MainActor
public final class HostMonitor: ObservableObject {
    @Published public private(set) var history = MetricHistory()
    @Published public private(set) var active = false
    @Published public private(set) var collecting = false
    @Published public private(set) var pressure: String?
    @Published public private(set) var pressureAt: Date?
    private var task: Task<Void, Never>?
    private var jobs: [ProcessJob] = []
    private var generation = 0
    private var pressureSource: (any DispatchSourceMemoryPressure)?
    private let probes: [HostProbe]
    private let interval: Duration
    private let timeout: TimeInterval
    private let cores: Int
    private let memory: Double
    public init(probes: [HostProbe] = HostProbe.defaults, interval: Duration = .seconds(5),
                timeout: TimeInterval = 2, cores: Int = ProcessInfo.processInfo.processorCount,
                memory: Double = Double(ProcessInfo.processInfo.physicalMemory)) {
        self.probes = probes; self.interval = interval; self.timeout = timeout
        self.cores = cores; self.memory = memory
    }
    public static func shouldSample(enabled: Bool, foreground: Bool, visible: Bool) -> Bool {
        enabled && foreground && visible
    }
    public func setActive(_ value: Bool) {
        guard value != active else { return }
        active = value; generation += 1
        let identity = generation, previous = task
        previous?.cancel(); jobs.forEach { $0.cancel() }
        pressureSource?.cancel(); pressureSource = nil
        pressure = nil; pressureAt = nil
        if !value { collecting = false; history.append([:]); return }
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: .main)
        source.setEventHandler { [weak self] in
            Task { @MainActor in
                guard let self, self.active, self.generation == identity, let event = self.pressureSource?.data else { return }
                self.pressure = event.contains(.critical) ? "Critical" : event.contains(.warning) ? "Warning" : event.contains(.normal) ? "Normal" : nil
                self.pressureAt = Date()
            }
        }
        pressureSource = source; source.resume()
        task = Task { [weak self] in
            await previous?.value
            guard let self else { return }
            while self.active && self.generation == identity && !Task.isCancelled {
                self.collecting = true
                var values: [String: Double] = [:]
                // Absolute, fixed commands; never shell text or privilege escalation.
                for probe in self.probes {
                    guard self.generation == identity, !Task.isCancelled else { break }
                    let job = ProcessJob(); self.jobs = [job]
                    if let result = try? await job.run(executable: probe.executable, arguments: probe.arguments, timeout: self.timeout),
                       result.status == 0, !result.truncated {
                        switch probe.kind {
                        case "cpu": values["cpu"] = HostMetricParser.cpu(result.stdout, cores: self.cores)
                        case "memory": values["memory"] = HostMetricParser.memory(result.stdout, total: self.memory)
                        case "gpu": values["gpu"] = HostMetricParser.gpu(result.stdout)
                        default: break
                        }
                    }
                    self.jobs = []
                }
                guard self.generation == identity, !Task.isCancelled else { return }
                self.history.append(values); self.collecting = false
                do { try await Task.sleep(for: self.interval) } catch { return }
            }
        }
    }
    public func shutdown() async {
        setActive(false)
        await task?.value
        task = nil; jobs = []
    }
}
