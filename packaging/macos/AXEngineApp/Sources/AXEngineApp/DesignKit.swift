import SwiftUI
import Charts
import AXEngineAppCore

// Components from `.internal/ui/01-design-system.svg`. Pages compose these
// instead of styling controls inline, so every board reads as one system.

/// Uppercase section label ("OUTPUT SPEED · LIVE", "PINNED").
struct SectionLabel: View {
    let text: String
    var tint: Color = Brand.text3
    init(_ text: String, tint: Color = Brand.text3) { self.text = text; self.tint = tint }
    var body: some View {
        Text(text.uppercased()).font(Brand.Font.section).tracking(0.6).foregroundStyle(tint)
            .accessibilityAddTraits(.isHeader)
    }
}

/// Neutral spec chip ("AXQ MXFP4", "32k context").
struct Tag: View {
    let text: String
    var tint: Color? = nil
    init(_ text: String, tint: Color? = nil) { self.text = text; self.tint = tint }
    var body: some View {
        Text(text).font(.system(size: 11, weight: .medium)).lineLimit(1)
            .foregroundStyle(tint ?? Brand.text2)
            .padding(.horizontal, 8).padding(.vertical, 3.5)
            .background((tint ?? Brand.control).opacity(tint == nil ? 1 : 0.16), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .fixedSize()
    }
}

/// Status pill: a dot or symbol plus a word, tinted by meaning.
struct StatusPill: View {
    let text: String
    var tint: Color
    var systemImage: String? = nil
    var body: some View {
        HStack(spacing: 5) {
            if let systemImage { Image(systemName: systemImage).font(.system(size: 9, weight: .bold)) }
            else { Circle().fill(tint).frame(width: 6, height: 6) }
            Text(text).font(.system(size: 11.5, weight: .semibold)).lineLimit(1)
        }
        .fixedSize()
        .foregroundStyle(tint)
        .padding(.horizontal, 9).padding(.vertical, 4)
        .background(tint.opacity(0.15), in: Capsule())
        .accessibilityElement(children: .combine)
    }
}

/// Small labelled reading on an inset tile ("TTFT 312 ms").
struct StatTile: View {
    let title: String
    let value: String?
    var unit: String? = nil
    var tint: Color = Brand.text1
    var help: String? = nil
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.system(size: 11)).foregroundStyle(Brand.text3)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value ?? "—").font(.system(size: 14, weight: .semibold)).monospacedDigit()
                    .foregroundStyle(value == nil ? Brand.text3 : tint)
                if let unit, value != nil { Text(unit).font(.system(size: 11.5)).foregroundStyle(Brand.text2) }
            }
            .lineLimit(1).minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10).padding(.vertical, 9)
        .background(Brand.inset, in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
        .help(help ?? title)
        .accessibilityElement(children: .combine)
    }
}

/// Larger reading without a tile, for hero rows ("Prefill 1,840 tok/s").
struct StatReading: View {
    let title: String
    let value: String?
    var unit: String? = nil
    var tint: Color = Brand.text1
    var help: String? = nil
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.system(size: 11.5)).foregroundStyle(Brand.text3)
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value ?? "—").font(Brand.Font.stat).foregroundStyle(value == nil ? Brand.text3 : tint)
                if let unit, value != nil { Text(unit).font(.system(size: 15, weight: .medium)).foregroundStyle(Brand.text2) }
            }
        }
        .help(help ?? title)
        .accessibilityElement(children: .combine)
    }
}

/// Monospaced value on an inset field with a copy button.
struct CodeField: View {
    let text: String
    var placeholder = "—"
    @State private var copied = false
    var body: some View {
        HStack(spacing: 8) {
            Text(text.isEmpty ? placeholder : text).font(Brand.Font.code).lineLimit(1).truncationMode(.middle)
                .foregroundStyle(text.isEmpty ? Brand.text3 : Brand.text1).textSelection(.enabled)
            Spacer(minLength: 4)
            Button {
                copyText(text); copied = true
                Task { try? await Task.sleep(for: .seconds(2)); copied = false }
            } label: {
                Image(systemName: copied ? "checkmark" : "doc.on.doc").font(.system(size: 11))
                    .frame(width: 24, height: 22)
                    .background(Brand.control, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            }
            .buttonStyle(.plain).foregroundStyle(Brand.text2).disabled(text.isEmpty)
            .help(copied ? "Copied" : "Copy").accessibilityLabel(copied ? "Copied" : "Copy " + text)
        }
        .padding(.leading, 11).padding(.trailing, 5).padding(.vertical, 5)
        .background(Brand.inset, in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous).strokeBorder(Brand.hairlineColor))
    }
}

/// Filled accent, filled serve, neutral control and danger buttons.
struct AXButtonStyle: ButtonStyle {
    enum Kind { case primary, serve, secondary, danger, plain }
    var kind: Kind = .secondary
    var compact = false
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: compact ? 12 : 13, weight: .semibold))
            .labelStyle(.titleAndIcon)
            .lineLimit(1)
            .fixedSize()
            .foregroundStyle(foreground)
            .padding(.horizontal, compact ? 10 : 14).padding(.vertical, compact ? 5 : 7.5)
            .background(background.opacity(configuration.isPressed ? 0.75 : 1),
                        in: RoundedRectangle(cornerRadius: compact ? 7 : 8, style: .continuous))
            .opacity(enabled ? 1 : 0.45)
            .contentShape(Rectangle())
    }
    private var foreground: Color {
        switch kind {
        case .primary, .serve: .white
        case .secondary: Brand.text1
        case .danger: Brand.danger
        case .plain: Brand.accent
        }
    }
    private var background: Color {
        switch kind {
        case .primary: Brand.accent
        case .serve: Brand.serve
        case .secondary: Brand.control
        case .danger: Brand.danger.opacity(0.14)
        case .plain: .clear
        }
    }
}

extension ButtonStyle where Self == AXButtonStyle {
    static var axPrimary: AXButtonStyle { AXButtonStyle(kind: .primary) }
    static var axServe: AXButtonStyle { AXButtonStyle(kind: .serve) }
    static var axSecondary: AXButtonStyle { AXButtonStyle(kind: .secondary) }
    static var axDanger: AXButtonStyle { AXButtonStyle(kind: .danger) }
    static var axLink: AXButtonStyle { AXButtonStyle(kind: .plain, compact: true) }
    static func ax(_ kind: AXButtonStyle.Kind, compact: Bool = true) -> AXButtonStyle { AXButtonStyle(kind: kind, compact: compact) }
}

/// Segmented control drawn like the board: control track, raised selection.
struct AXSegmented<Value: Hashable>: View {
    let options: [(value: Value, title: String)]
    @Binding var selection: Value
    var compact = false
    var body: some View {
        HStack(spacing: 2) {
            ForEach(options.indices, id: \.self) { index in
                let option = options[index]
                let selected = option.value == selection
                Button { selection = option.value } label: {
                    Text(option.title).font(.system(size: compact ? 11.5 : 12.5, weight: selected ? .semibold : .medium))
                        .foregroundStyle(selected ? Brand.text1 : Brand.text2)
                        .padding(.horizontal, compact ? 10 : 14).padding(.vertical, compact ? 4 : 5.5)
                        .frame(minWidth: compact ? 0 : 56)
                        .background(selected ? Brand.inset : .clear, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .padding(2.5)
        .background(Brand.control, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

/// Stacked, labelled bar ("Memory bar · stacked, labelled").
struct StackedBar: View {
    struct Segment: Identifiable { let id: String; let value: Double; let color: Color }
    let segments: [Segment]
    let total: Double
    var height: CGFloat = 10
    var body: some View {
        GeometryReader { geometry in
            HStack(spacing: 2) {
                ForEach(segments) { segment in
                    RoundedRectangle(cornerRadius: 2, style: .continuous).fill(segment.color)
                        .frame(width: max(segment.value > 0 ? 3 : 0, geometry.size.width * CGFloat(max(0, segment.value) / max(total, 1))))
                }
                Spacer(minLength: 0)
            }
        }
        .frame(height: height)
        .background(Brand.control, in: RoundedRectangle(cornerRadius: height / 2, style: .continuous))
        .clipShape(RoundedRectangle(cornerRadius: height / 2, style: .continuous))
        .accessibilityHidden(true)
    }
}

/// Line-and-area sparkline over a metric history, with gaps where samples are missing.
struct Sparkline: View {
    let points: [MetricPlotPoint]
    let start: Date
    let end: Date
    var tint: Color = Brand.accent
    var body: some View {
        Chart {
            ForEach(points) { point in
                AreaMark(x: .value("Time", point.time), y: .value("Value", point.value),
                         series: .value("Segment", point.segment))
                    .foregroundStyle(LinearGradient(colors: [tint.opacity(0.32), tint.opacity(0.02)], startPoint: .top, endPoint: .bottom))
                LineMark(x: .value("Time", point.time), y: .value("Value", point.value),
                         series: .value("Segment", point.segment))
                    .foregroundStyle(tint).lineStyle(StrokeStyle(lineWidth: 1.8, lineCap: .round, lineJoin: .round))
            }
        }
        .chartXScale(domain: start...end)
        .chartXAxis(.hidden).chartYAxis(.hidden)
        .chartPlotStyle { $0.drawingGroup(opaque: false, colorMode: .linear) }
        .overlay {
            if points.isEmpty {
                Rectangle().fill(Brand.hairlineColor).frame(height: 1).frame(maxHeight: .infinity, alignment: .bottom)
            }
        }
        .accessibilityHidden(true)
    }
}

/// Letter avatar used in model tables ("Q", "G").
struct ModelAvatar: View {
    let name: String
    var body: some View {
        Text(String(name.trimmingCharacters(in: .whitespaces).prefix(1)).uppercased())
            .font(.system(size: 12, weight: .bold)).foregroundStyle(Brand.accent)
            .frame(width: 28, height: 28)
            .background(Brand.control, in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            .accessibilityHidden(true)
    }
}

/// Page scaffold: canvas background and the board's 24 pt gutters.
struct PageScroll<Content: View>: View {
    var maxWidth: CGFloat = 1320
    @ViewBuilder let content: () -> Content
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) { content() }
                .padding(.horizontal, 24).padding(.top, 8).padding(.bottom, 24)
                .frame(maxWidth: maxWidth, alignment: .leading)
                .frame(maxWidth: .infinity)
        }
        .background(Brand.canvas)
    }
}

enum Format {
    static func bytes(_ value: Double?) -> String? {
        guard let value, value.isFinite, value >= 0 else { return nil }
        return ByteCountFormatter.string(fromByteCount: Int64(min(value, Double(Int64.max))), countStyle: .memory)
    }
    static func count(_ value: Double?) -> String? {
        value.map { $0.formatted(.number.precision(.fractionLength(0))) }
    }
    static func rate(_ value: Double?) -> String? {
        guard let value, value.isFinite, value >= 0 else { return nil }
        return value >= 100 ? value.formatted(.number.precision(.fractionLength(0))) : String(format: "%.1f", value)
    }
    static func uptime(since start: Date?, now: Date = Date()) -> String? {
        guard let start else { return nil }
        let seconds = max(0, Int(now.timeIntervalSince(start)))
        let hours = seconds / 3600, minutes = (seconds % 3600) / 60
        return hours > 0 ? "\(hours)h \(minutes)m" : minutes > 0 ? "\(minutes)m" : "\(seconds)s"
    }
}

/// This Mac, as the sidebar header and System facts row describe it.
enum HostInfo {
    static let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
    static let chip: String = {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 1 else { return "Mac" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 else { return "Mac" }
        return String(decoding: bytes.prefix { $0 != 0 }.map(UInt8.init(bitPattern:)), as: UTF8.self)
    }()
    static let memory = ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory)
    static let macOS: String = {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return "\(version.majorVersion).\(version.minorVersion)" + (version.patchVersion > 0 ? ".\(version.patchVersion)" : "")
    }()
    /// "8.1 · M5 Max · 128 GB"
    static var summary: String {
        [version, chip.replacingOccurrences(of: "Apple ", with: ""), memory].compactMap { $0 }.joined(separator: " · ")
    }
}

/// Left-aligned rows that wrap to the proposed width (tag and pill rows).
struct WrapLayout: Layout {
    var spacing: CGFloat = 6
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(width: proposal.width ?? .infinity, subviews: subviews)
        return CGSize(width: rows.map(\.width).max() ?? 0, height: rows.last.map { $0.y + $0.height } ?? 0)
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for row in arrange(width: bounds.width, subviews: subviews) {
            var x = bounds.minX
            for index in row.items {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(at: CGPoint(x: x, y: bounds.minY + row.y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
        }
    }
    private struct Row { var items: [Int] = []; var width: CGFloat = 0; var height: CGFloat = 0; var y: CGFloat = 0 }
    private func arrange(width: CGFloat, subviews: Subviews) -> [Row] {
        var rows: [Row] = [], row = Row()
        for (index, subview) in subviews.enumerated() {
            let size = subview.sizeThatFits(.unspecified)
            if !row.items.isEmpty && row.width + spacing + size.width > width {
                rows.append(row); row = Row(y: row.y + row.height + spacing)
            }
            row.width += (row.items.isEmpty ? 0 : spacing) + size.width
            row.height = max(row.height, size.height)
            row.items.append(index)
        }
        if !row.items.isEmpty { rows.append(row) }
        return rows
    }
}
