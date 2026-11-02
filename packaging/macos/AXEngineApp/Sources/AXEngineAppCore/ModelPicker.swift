import Foundation

/// Grouping for the serve-model picker: favorites first, then recently
/// served models, then everything else. A model appears in one section only.
public struct ModelPickerSections: Equatable, Sendable {
    public var favorites: [LocalModel] = []
    public var recent: [LocalModel] = []
    public var others: [LocalModel] = []
    public var all: [LocalModel] { favorites + recent + others }
    public var isEmpty: Bool { favorites.isEmpty && recent.isEmpty && others.isEmpty }
}

/// Favorites and recents are keyed by model path. Paths that are no longer
/// selectable (removed or incomplete) are kept but not shown, so a model that
/// comes back keeps its star.
public enum ModelPicker {
    public static let recentLimit = 5

    public static func sections(models: [LocalModel], favorites: [String], recent: [String], query: String = "") -> ModelPickerSections {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let visible = needle.isEmpty ? models : models.filter {
            $0.displayName.localizedCaseInsensitiveContains(needle) || $0.name.localizedCaseInsensitiveContains(needle)
        }
        let favoriteSet = Set(favorites)
        let byPath = Dictionary(visible.map { ($0.path, $0) }, uniquingKeysWith: { first, _ in first })
        var result = ModelPickerSections()
        result.favorites = visible.filter { favoriteSet.contains($0.path) }
            .sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }
        result.recent = recent.filter { !favoriteSet.contains($0) }.compactMap { byPath[$0] }
        let shown = Set(result.favorites.map(\.path) + result.recent.map(\.path))
        result.others = visible.filter { !shown.contains($0.path) }
        return result
    }

    /// Moves `path` to the front of the recent list, without duplicates.
    public static func recording(_ path: String, in recent: [String]) -> [String] {
        guard !path.isEmpty else { return recent }
        return Array(([path] + recent.filter { $0 != path }).prefix(recentLimit))
    }

    public static func toggling(_ path: String, in favorites: [String]) -> [String] {
        favorites.contains(path) ? favorites.filter { $0 != path } : favorites + [path]
    }

    /// The model to preselect when nothing valid is chosen: the most recently
    /// served one, then a favorite, then the first in the library.
    public static func preferred(models: [LocalModel], favorites: [String], recent: [String]) -> LocalModel? {
        let recentFirst = recent.lazy.compactMap { path in models.first { $0.path == path } }.first
        return recentFirst ?? sections(models: models, favorites: favorites, recent: []).favorites.first ?? models.first
    }
}
