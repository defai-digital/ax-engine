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
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search pages, models and actions", text: $query)
                    .textFieldStyle(.plain).font(.title3).focused($focused)
                    .onSubmit { if let id = highlighted ?? visible.first?.id { run(id) } }
                    .onKeyPress(.downArrow) { move(1, in: visible); return .handled }
                    .onKeyPress(.upArrow) { move(-1, in: visible); return .handled }
                Text("esc").font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            .padding(14)
            Divider()
            ScrollViewReader { proxy in
                List {
                    ForEach(sections(visible), id: \.self) { section in
                        Section(section) {
                            ForEach(visible.filter { $0.section == section }) { entry in
                                Button { run(entry.id) } label: {
                                    Text(entry.title).frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
                                }
                                .buttonStyle(.plain)
                                .padding(.vertical, 3).padding(.horizontal, 6)
                                .background((highlighted ?? visible.first?.id) == entry.id ? Brand.accent.opacity(0.16) : .clear,
                                            in: RoundedRectangle(cornerRadius: 6))
                                .id(entry.id)
                            }
                        }
                    }
                    if visible.isEmpty { Text("No matches").foregroundStyle(.secondary) }
                }
                .listStyle(.plain)
                .onChange(of: highlighted) { _, id in if let id { proxy.scrollTo(id) } }
            }
        }
        .frame(width: 520, height: 400)
        .onAppear { focused = true }
        .onChange(of: query) { _, _ in highlighted = nil }
    }

    private func sections(_ visible: [PaletteEntry]) -> [String] {
        var seen = Set<String>()
        return visible.map(\.section).filter { seen.insert($0).inserted }
    }

    private func move(_ step: Int, in visible: [PaletteEntry]) {
        guard !visible.isEmpty else { return }
        let current = visible.firstIndex { $0.id == (highlighted ?? visible.first?.id) } ?? 0
        highlighted = visible[max(0, min(visible.count - 1, current + step))].id
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
