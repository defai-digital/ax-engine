import SwiftUI

/// A capsule toggle shared by the Playground profile and serving mode rows.
struct SelectionChip: View {
    let title: String
    let selected: Bool
    let accessibilityLabel: String
    let help: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                if selected { Image(systemName: "checkmark.circle.fill") }
                Text(title)
            }
            .font(.callout.weight(selected ? .semibold : .regular))
            .padding(.horizontal, 12).padding(.vertical, 6)
            .foregroundStyle(selected ? Brand.accentChrome : Color.primary)
            .background(selected ? Brand.accentChrome.opacity(0.14) : Color.secondary.opacity(0.08), in: Capsule())
            .overlay(Capsule().strokeBorder(selected ? Brand.accentChrome.opacity(0.5) : Color.clear))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityAddTraits(selected ? [.isSelected] : [])
        .help(help)
    }
}
