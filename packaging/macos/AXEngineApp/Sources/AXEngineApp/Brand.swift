import SwiftUI
import AppKit

enum Brand {
    static let accentChrome = Color(red: 0.08, green: 0.59, blue: 0.40)
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
}
