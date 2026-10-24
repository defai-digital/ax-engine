import Foundation

public struct LibraryScanRequest: Codable, Sendable {
    public let root: String
    public let imported: [String]
    public init(root: String, imported: [String]) { self.root = root; self.imported = imported }
}

public struct LibraryScanEvent: Codable, Sendable {
    public let model: LocalModel?
    public let warning: String?
    public let complete: Bool
    public init(model: LocalModel? = nil, warning: String? = nil, complete: Bool = false) {
        self.model = model; self.warning = warning; self.complete = complete
    }
}
