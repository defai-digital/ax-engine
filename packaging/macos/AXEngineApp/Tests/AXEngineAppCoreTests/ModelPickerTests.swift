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
    #expect(grouped.others.map(\.displayName) == ["Qwen3.8 Flash Next"])
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

@Test func labelsDropSharedWordsAndKeepVariantsAsTags() {
    let managed = LocalModel(path: "/m/AutomatosX--AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP--5ab39b24bfd7f65203be9b7823b1840486f58b6d",
                             ready: true, managed: true)
    #expect(managed.repositoryName == "AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP")
    #expect(managed.label == ModelLabel(title: "Tiel Coder 35B A3B", tags: ["MXFP4", "MTP"]))
    #expect(managed.displayName == "Tiel Coder 35B A3B · MXFP4 · MTP")
    let cached = LocalModel(path: "/hub/models--AutomatosX--AX-Clef-27B-MLX-AXQ-6bit/snapshots/0123456789abcdef", ready: true, managed: false)
    #expect(cached.label == ModelLabel(title: "Clef 27B", tags: ["6-bit"]))
    // Variants of one model differ only by tags.
    let mxfp8 = LocalModel(path: "/hub/models--AutomatosX--AX-Clef-27B-MLX-AXQ-MXFP8/snapshots/0123456789abcdef", ready: true, managed: false)
    #expect(mxfp8.label.title == cached.label.title && mxfp8.displayName != cached.displayName)
    #expect(ModelLabel(repositoryName: "AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-6bit-MTP")
            == ModelLabel(title: "Cyber Tiel Coder 35B A3B", tags: ["6-bit", "MTP"]))
    // Plain local folders keep their name; a tag-only name is not emptied.
    #expect(LocalModel(path: "/models/my-model", ready: true, managed: false).displayName == "my model")
    #expect(ModelLabel(repositoryName: "MLX-MTP").title == "MLX-MTP")
}
