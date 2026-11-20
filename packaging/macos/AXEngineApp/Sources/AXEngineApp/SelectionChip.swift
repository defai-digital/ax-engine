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
            Text(title)
                .font(.system(size: 12.5, weight: selected ? .semibold : .medium))
                .padding(.horizontal, 13).padding(.vertical, 5.5)
                .foregroundStyle(selected ? Brand.text1 : Brand.text2)
                .background(selected ? Brand.inset : Brand.control, in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(selected ? Brand.accent.opacity(0.55) : .clear))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityAddTraits(selected ? [.isSelected] : [])
        .help(help)
    }
}
