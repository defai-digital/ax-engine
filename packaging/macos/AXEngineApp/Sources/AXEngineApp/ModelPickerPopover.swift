import SwiftUI
import AXEngineAppCore

/// Searchable serve-model picker with Favorites and Recent sections and a
/// star per row. Arrow keys move the highlight, Return chooses it.
struct ModelPickerPopover: View {
    let models: [LocalModel]
    let selectedPath: String
    let favorites: [String]
    let recent: [String]
    let scanning: Bool
    let choose: (String) -> Void
    let toggleFavorite: (String) -> Void
    let browseCatalog: () -> Void
    @State private var query = ""
    @State private var highlighted: String?
    @State private var collapsed: Set<String> = []
    @FocusState private var searchFocused: Bool

    private var sections: ModelPickerSections {
        ModelPicker.sections(models: models, favorites: favorites, recent: recent, query: query)
    }
    /// Rows in display order, skipping collapsed sections.
    private var navigable: [LocalModel] {
        let grouped = sections
        return [("favorites", grouped.favorites), ("recent", grouped.recent), ("all", grouped.others)]
            .filter { !collapsed.contains($0.0) }.flatMap(\.1)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: browseCatalog) {
                Label("Download more models", systemImage: "plus").frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain).foregroundStyle(Brand.accentChrome).font(.callout.weight(.medium))
            .padding(.horizontal, 14).padding(.vertical, 10)
            Divider()
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search models", text: $query)
                    .textFieldStyle(.plain).focused($searchFocused)
                    .onSubmit(chooseHighlighted)
                    .onKeyPress(.downArrow) { move(1); return .handled }
                    .onKeyPress(.upArrow) { move(-1); return .handled }
                    .accessibilityLabel("Search downloaded models")
            }
            .padding(.horizontal, 14).padding(.vertical, 9)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        let grouped = sections
                        if grouped.isEmpty {
                            Text(emptyText).font(.callout).foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .center).padding(.vertical, 24)
                        }
                        section("favorites", title: "Favorites", symbol: "star.fill", rows: grouped.favorites)
                        section("recent", title: "Recent", symbol: "clock", rows: grouped.recent)
                        section("all", title: grouped.favorites.isEmpty && grouped.recent.isEmpty ? "Downloaded" : "Other downloaded",
                                symbol: "square.stack.3d.up", rows: grouped.others)
                    }.padding(6)
                }
                .onChange(of: highlighted) { _, path in
                    if let path { withAnimation(.easeOut(duration: 0.1)) { proxy.scrollTo(path) } }
                }
            }
            .frame(maxHeight: 380)
            Divider()
            Text("↑↓ navigate · Return select · ★ favorite")
                .font(.caption).foregroundStyle(.secondary)
                .padding(.horizontal, 14).padding(.vertical, 7)
        }
        .frame(width: 420)
        .onAppear {
            searchFocused = true
            highlighted = models.contains { $0.path == selectedPath } ? selectedPath : navigable.first?.path
        }
        .onChange(of: query) { _, _ in highlighted = navigable.first?.path }
    }

    private var emptyText: String {
        if !query.isEmpty { return "No downloaded model matches “\(query)”." }
        return scanning ? "Checking your library…" : "No downloaded models yet."
    }

    @ViewBuilder
    private func section(_ key: String, title: String, symbol: String, rows: [LocalModel]) -> some View {
        if !rows.isEmpty {
            let isCollapsed = collapsed.contains(key)
            Button {
                if isCollapsed { collapsed.remove(key) } else { collapsed.insert(key) }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: symbol).foregroundStyle(key == "favorites" ? Brand.accentChrome : .secondary)
                    Text(title.uppercased()).font(.caption.weight(.semibold)).tracking(0.6).foregroundStyle(.secondary)
                    Spacer()
                    Image(systemName: isCollapsed ? "chevron.right" : "chevron.down").font(.caption).foregroundStyle(.secondary)
                }
                .padding(.horizontal, 8).padding(.top, 8).padding(.bottom, 4).contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(title + (isCollapsed ? ", collapsed" : ", expanded"))
            if !isCollapsed {
                ForEach(rows) { row($0) }
            }
        }
    }

    private func row(_ item: LocalModel) -> some View {
        let selected = item.path == selectedPath
        let favorite = favorites.contains(item.path)
        return HStack(spacing: 8) {
            Button { choose(item.path) } label: {
                HStack(spacing: 8) {
                    Text(item.displayName).font(.callout.weight(selected ? .semibold : .regular))
                        .lineLimit(1).truncationMode(.middle)
                    if let size = item.sizeBytes {
                        Text(TransferFormat.bytes(size)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 4)
                    if selected { Image(systemName: "checkmark").foregroundStyle(Brand.accentChrome) }
                }.contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(item.path)
            Button { toggleFavorite(item.path) } label: {
                Image(systemName: favorite ? "star.fill" : "star")
                    .foregroundStyle(favorite ? Brand.accentChrome : .secondary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(favorite ? "Remove \(item.displayName) from favorites" : "Add \(item.displayName) to favorites")
            .help(favorite ? "Remove from favorites" : "Add to favorites")
        }
        .padding(.horizontal, 8).padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: 7).fill(
            highlighted == item.path ? Brand.accentChrome.opacity(0.14) : Color.clear))
        .onHover { if $0 { highlighted = item.path } }
        .id(item.path)
    }

    private func move(_ step: Int) {
        let rows = navigable.map(\.path)
        guard !rows.isEmpty else { return }
        let index = highlighted.flatMap { rows.firstIndex(of: $0) } ?? (step > 0 ? -1 : rows.count)
        highlighted = rows[min(max(index + step, 0), rows.count - 1)]
    }

    private func chooseHighlighted() {
        if let path = highlighted, navigable.contains(where: { $0.path == path }) { choose(path) }
        else if let first = navigable.first { choose(first.path) }
    }
}
