import Testing
@testable import AXEngineAppCore

private func local(_ name: String) -> LocalModel { LocalModel(path: "/models/" + name, ready: true, managed: true) }

@Test func pickerGroupsFavoritesRecentAndOthersWithoutDuplicates() {
    let models = ["Zeta", "alpha", "Beta", "Gamma"].map(local)
    let grouped = ModelPicker.sections(models: models, favorites: ["/models/Zeta", "/models/alpha", "/models/gone"],
                                       recent: ["/models/Gamma", "/models/Zeta", "/models/gone"])
    // Favorites sort by name; a favorite never repeats under Recent; missing paths are hidden.
    #expect(grouped.favorites.map(\.displayName) == ["alpha", "Zeta"])
    #expect(grouped.recent.map(\.displayName) == ["Gamma"])
    #expect(grouped.others.map(\.displayName) == ["Beta"])
    #expect(grouped.all.count == models.count)
}

@Test func pickerSearchFiltersEverySection() {
    let models = ["Qwen3.8-27B", "Qwen3.8-Flash-Next", "Hy-MT2"].map(local)
    let grouped = ModelPicker.sections(models: models, favorites: ["/models/Hy-MT2"], recent: ["/models/Qwen3.8-27B"], query: "  flash ")
    #expect(grouped.favorites.isEmpty && grouped.recent.isEmpty)
    #expect(grouped.others.map(\.displayName) == ["Qwen3.8-Flash-Next"])
    #expect(ModelPicker.sections(models: models, favorites: [], recent: [], query: "nothing").isEmpty)
}

@Test func recentListMovesToFrontAndIsBounded() {
    var recent: [String] = []
    for name in ["a", "b", "c", "d", "e", "f"] { recent = ModelPicker.recording(name, in: recent) }
    #expect(recent == ["f", "e", "d", "c", "b"])
    #expect(ModelPicker.recording("c", in: recent) == ["c", "f", "e", "d", "b"])
    #expect(ModelPicker.recording("", in: recent) == recent)
    #expect(ModelPicker.toggling("x", in: ["y"]) == ["y", "x"])
    #expect(ModelPicker.toggling("y", in: ["y", "x"]) == ["x"])
}

@Test func preselectionPrefersRecentThenFavoriteThenFirst() {
    let models = ["a", "b", "c"].map(local)
    #expect(ModelPicker.preferred(models: models, favorites: ["/models/c"], recent: ["/models/gone", "/models/b"])?.displayName == "b")
    #expect(ModelPicker.preferred(models: models, favorites: ["/models/c"], recent: [])?.displayName == "c")
    #expect(ModelPicker.preferred(models: models, favorites: [], recent: [])?.displayName == "a")
    #expect(ModelPicker.preferred(models: [], favorites: [], recent: []) == nil)
}
