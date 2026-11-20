import SwiftUI
import AXEngineAppCore

/// ⌘K: jump to a page, serve a downloaded model, or run a server action.
/// Every entry calls the same ApplicationModel action as its button, with
/// the same gates and confirmations.
struct CommandPalette: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var highlighted: String?
    /// Bumped by arrow keys only; hover must not scroll the row under the pointer.
    @State private var keyboardMoves = 0
    @FocusState private var focused: Bool

    private var entries: [PaletteEntry] {
        var list = Page.allCases.map { PaletteEntry(id: "page:" + $0.rawValue, title: "Go to " + $0.rawValue, section: "Navigate") }
        list.append(PaletteEntry(id: "settings", title: "Open Settings", section: "Navigate", keywords: ["preferences", "port"]))
        if runtime.owned {
            if model.canRestartServer { list.append(.init(id: "restart", title: "Restart Server", section: "Server")) }
            list.append(.init(id: "stop", title: "Stop Server", section: "Server", keywords: ["quit", "end"]))
        } else if runtime.connection != nil {
            list.append(.init(id: "stop", title: "Disconnect", section: "Server"))
        } else if model.canStartServer {
            list.append(.init(id: "start", title: "Start Server", section: "Server", keywords: ["serve", "run"]))
        }
        if let url = runtime.connection?.apiURL.absoluteString {
            list.append(.init(id: "copy-url", title: "Copy API URL", section: "Server", keywords: [url, "endpoint"]))
        }
        if !model.attachedToService && !model.connecting {
            for item in model.servableModels where !(runtime.owned && runtime.selectedModel == item.path) {
                list.append(.init(id: "serve:" + item.path, title: "Serve " + item.displayName, section: "Models",
                                  keywords: [item.repositoryName, item.name]))
            }
        }
        list.append(.init(id: "catalog", title: "Download a Model…", section: "Models", keywords: ["catalog", "hugging face"]))
        return list
    }

    var body: some View {
        let visible = PaletteEntry.filter(entries, query: query)
        let current = highlighted ?? visible.first?.id
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass").font(.system(size: 15)).foregroundStyle(Brand.text3)
                TextField("Search pages, models and actions", text: $query)
                    .textFieldStyle(.plain).font(.system(size: 16, weight: .medium)).focused($focused)
                    .onSubmit { if let id = current { run(id) } }
                    .onKeyPress(.downArrow) { move(1, in: visible); return .handled }
                    .onKeyPress(.upArrow) { move(-1, in: visible); return .handled }
                Text("esc").font(.system(size: 11)).foregroundStyle(Brand.text3)
            }
            .padding(.horizontal, 18).padding(.vertical, 15)
            Divider().overlay(Brand.hairlineColor)
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(sections(visible), id: \.self) { section in
                            SectionLabel(section).padding(.horizontal, 12).padding(.top, 10).padding(.bottom, 4)
                            ForEach(visible.filter { $0.section == section }) { entry in
                                row(entry, highlighted: current == entry.id).id(entry.id)
                            }
                        }
                        if visible.isEmpty {
                            Text("No matches").foregroundStyle(Brand.text3).padding(20)
                        }
                    }
                    .padding(10)
                }
                .onChange(of: keyboardMoves) { _, _ in if let id = highlighted { proxy.scrollTo(id) } }
            }
            Divider().overlay(Brand.hairlineColor)
            Text("↑↓ to move · ↩ to run · ⌘K anywhere").font(.system(size: 11)).foregroundStyle(Brand.text3)
                .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 18).padding(.vertical, 9)
        }
        .frame(width: 600, height: 440)
        .background(Brand.panel)
        .onAppear { focused = true }
        .onChange(of: query) { _, _ in highlighted = nil }
    }

    private func row(_ entry: PaletteEntry, highlighted: Bool) -> some View {
        Button { run(entry.id) } label: {
            HStack(spacing: 12) {
                Image(systemName: icon(entry.id)).font(.system(size: 12, weight: .medium)).frame(width: 18)
                    .foregroundStyle(highlighted ? .white : Brand.text2)
                Text(entry.title).font(.system(size: 13, weight: .medium)).lineLimit(1)
                    .foregroundStyle(highlighted ? .white : Brand.text1)
                Spacer()
                if let hint = hint(entry.id) {
                    Text(hint).font(.system(size: 11.5)).monospacedDigit().foregroundStyle(highlighted ? .white.opacity(0.85) : Brand.text3)
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(highlighted ? Brand.accent : .clear, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { if $0 { self.highlighted = entry.id } }
    }

    private func icon(_ id: String) -> String {
        if id.hasPrefix("page:"), let page = Page(rawValue: String(id.dropFirst(5))) { return page.icon }
        if id.hasPrefix("serve:") { return "play.fill" }
        switch id {
        case "settings": return "gearshape"
        case "restart": return "arrow.clockwise"
        case "start": return "play.fill"
        case "stop": return "stop.fill"
        case "copy-url": return "doc.on.doc"
        case "catalog": return "arrow.down.circle"
        default: return "circle"
        }
    }

    private func hint(_ id: String) -> String? {
        if id.hasPrefix("page:"), let page = Page(rawValue: String(id.dropFirst(5))), let index = Page.allCases.firstIndex(of: page) {
            return "⌘\(index + 1)"
        }
        if id.hasPrefix("serve:") {
            return model.servableModels.first { "serve:" + $0.path == id }?.sizeBytes.map(TransferFormat.bytes)
        }
        switch id {
        case "settings": return "⌘,"
        case "restart", "start": return "⇧⌘R"
        case "stop": return "⇧⌘."
        default: return nil
        }
    }

    private func sections(_ visible: [PaletteEntry]) -> [String] {
        var seen = Set<String>()
        return visible.map(\.section).filter { seen.insert($0).inserted }
    }

    private func move(_ step: Int, in visible: [PaletteEntry]) {
        guard !visible.isEmpty else { return }
        let current = visible.firstIndex { $0.id == (highlighted ?? visible.first?.id) } ?? 0
        highlighted = visible[max(0, min(visible.count - 1, current + step))].id
        keyboardMoves += 1
    }

    private func run(_ id: String) {
        dismiss()
        if id.hasPrefix("page:"), let page = Page(rawValue: String(id.dropFirst(5))) { model.page = page; return }
        if id.hasPrefix("serve:") { model.serve(String(id.dropFirst(6))); return }
        switch id {
        case "settings": model.openSettings()
        case "restart", "start": model.requestStartOrRestart()
        case "stop": model.requestStopServer()
        case "copy-url": if let url = runtime.connection?.apiURL.absoluteString { copyText(url) }
        case "catalog": model.catalogSearch = ""; model.modelsSection = "catalog"; model.page = .models
        default: break
        }
    }
}
