import AppKit
import Foundation

/// User-initiated secret copying. Markers are hints, not a clipboard access boundary.
@MainActor
public final class SensitiveClipboard {
    private let pasteboard: NSPasteboard
    private let lifetime: Duration
    private var ownership: Int?
    private var expiry: Task<Void, Never>?

    public init(pasteboard: NSPasteboard = .general, lifetime: Duration = .seconds(60)) {
        self.pasteboard = pasteboard
        self.lifetime = lifetime
    }

    @discardableResult
    public func copy(_ value: String) -> Bool {
        guard !value.isEmpty, value.utf8.count <= 16_384,
              !value.unicodeScalars.contains(where: { $0.value < 32 || $0.value == 127 }) else { return false }
        let item = NSPasteboardItem()
        guard item.setString(value, forType: .string),
              item.setData(Data(), forType: .init("org.nspasteboard.ConcealedType")),
              item.setData(Data(), forType: .init("org.nspasteboard.TransientType")) else { return false }
        expiry?.cancel()
        let expected = pasteboard.prepareForNewContents(with: .currentHostOnly)
        ownership = expected
        guard pasteboard.writeObjects([item]), pasteboard.changeCount == expected else {
            clearIfOwned()
            return false
        }
        let delay = lifetime
        // Keep the cleanup alive even if the owning view is torn down first.
        expiry = Task { [self] in
            do { try await Task.sleep(for: delay) } catch { return }
            guard !Task.isCancelled else { return }
            clearIfOwned()
        }
        return true
    }

    /// Never clear another owner's newer clipboard item.
    public func clearIfOwned() {
        expiry?.cancel()
        expiry = nil
        if let ownership, pasteboard.changeCount == ownership {
            pasteboard.clearContents()
        }
        ownership = nil
    }
}
