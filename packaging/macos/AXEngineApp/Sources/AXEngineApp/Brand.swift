import SwiftUI
import AppKit
import AXEngineAppCore

/// Design tokens from the 2026-11 redesign. Indigo is the action and
/// selection accent; green only ever means "this app's server is serving";
/// blue marks speculative decoding and attached external services.
enum Brand {
    static let accent = dynamic(light: 0x5B57F0, dark: 0x7B78FF)
    static let serve = dynamic(light: 0x0E9F6E, dark: 0x34D399)
    static let boost = dynamic(light: 0x0077B6, dark: 0x5CC8FF)
    static let warn = dynamic(light: 0xB76E00, dark: 0xF5B544)
    static let danger = dynamic(light: 0xD93838, dark: 0xFF6B6B)
    static let typeBody = Color.primary
    static let cardSurface = Color(nsColor: .controlBackgroundColor)
    static let separator = Color(nsColor: .separatorColor)
    static let hairline: CGFloat = 0.5
    enum Radii { static let l: CGFloat = 14 }
    enum Elevation {
        struct Shadow { let color: Color; let radius: CGFloat; let x: CGFloat; let y: CGFloat }
        static let low = Shadow(color: .black.opacity(0.035), radius: 5, x: 0, y: 2)
    }
    static var logo: NSImage? {
        Bundle.main.url(forResource: "AppIcon", withExtension: "png").flatMap { NSImage(contentsOf: $0) }
    }

    static func tint(_ kind: ServerStatusKind) -> Color {
        switch kind {
        case .serving: serve
        case .external: boost
        case .transition: .secondary
        case .degraded: warn
        case .failed: danger
        case .stopped: .secondary
        }
    }

    private static func dynamic(light: UInt32, dark: UInt32) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let hex = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
                           blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
        })
    }
}
