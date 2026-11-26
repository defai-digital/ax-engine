import SwiftUI
import AppKit
import AXEngineAppCore

/// Design tokens from the 2026-11 redesign (`.internal/ui/01-design-system.svg`).
/// Indigo is the action and selection accent; green only ever means "this
/// app's server is serving"; blue marks speculative decoding and attached
/// external services. Surfaces step canvas → panel → inset → control.
enum Brand {
    static let accent = dynamic(light: 0x5B57F0, dark: 0x7B78FF)
    static let serve = dynamic(light: 0x0E9F6E, dark: 0x34D399)
    static let boost = dynamic(light: 0x0077B6, dark: 0x5CC8FF)
    static let warn = dynamic(light: 0xB76E00, dark: 0xF5B544)
    static let danger = dynamic(light: 0xD93838, dark: 0xFF6B6B)

    static let canvas = dynamic(light: 0xF5F6F8, dark: 0x0F1115)
    static let sidebar = dynamic(light: 0xECEEF2, dark: 0x13151A)
    static let panel = dynamic(light: 0xFFFFFF, dark: 0x171A20)
    static let inset = dynamic(light: 0xF4F5F8, dark: 0x1E222A)
    static let control = dynamic(light: 0xE9EBF0, dark: 0x262B35)
    static let hairlineColor = dynamic(light: 0xE1E4EA, dark: 0x2A2F3A)
    static let text1 = dynamic(light: 0x0E1116, dark: 0xF2F4F8)
    static let text2 = dynamic(light: 0x566070, dark: 0xA7AEBB)
    static let text3 = dynamic(light: 0x646D7C, dark: 0x8A93A3)

    /// Kept for call sites that predate the surface tokens.
    static let typeBody = text1
    static let cardSurface = panel
    static let separator = hairlineColor
    static let hairline: CGFloat = 1
    enum Radii {
        static let l: CGFloat = 14
        static let m: CGFloat = 10
        static let s: CGFloat = 7
    }
    enum Elevation {
        struct Shadow { let color: Color; let radius: CGFloat; let x: CGFloat; let y: CGFloat }
        static let low = Shadow(color: .black.opacity(0.06), radius: 10, x: 0, y: 4)
    }
    /// Type scale from the design system board.
    enum Font {
        static let hero = SwiftUI.Font.system(size: 56, weight: .bold).monospacedDigit()
        static let display = SwiftUI.Font.system(size: 32, weight: .bold)
        static let pageTitle = SwiftUI.Font.system(size: 20, weight: .bold)
        static let cardTitle = SwiftUI.Font.system(size: 14, weight: .semibold)
        static let body = SwiftUI.Font.system(size: 13)
        static let caption = SwiftUI.Font.system(size: 11.5)
        static let section = SwiftUI.Font.system(size: 10.5, weight: .semibold)
        static let code = SwiftUI.Font.system(size: 12.5, design: .monospaced)
        static let stat = SwiftUI.Font.system(size: 19, weight: .semibold).monospacedDigit()
    }
    @MainActor static let logo: NSImage? =
        Bundle.main.url(forResource: "AppIcon", withExtension: "png").flatMap { NSImage(contentsOf: $0) }

    static func tint(_ kind: ServerStatusKind) -> Color {
        switch kind {
        case .serving: serve
        case .external: boost
        case .transition: accent
        case .degraded: warn
        case .failed: danger
        case .stopped: text3
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
