// Derived from MTPLX, Copyright 2026 Youssof Altoukhi (Apache-2.0).
// Source: apps/MTPLXApp/Sources/MTPLXAppHost/Views/Common/Card.swift
// Revision: 9882703f3105363ddc37eca9f97aa09a1d387112
// Modified for AX Engine: native presentation, bounded log messages and endpoint policy.
// See Resources/MTPLX-LICENSE.txt and Resources/MTPLX-NOTICE.txt.
import SwiftUI

// MARK: - Card
//
// Standard card surface. Brand.cardSurface fill, hairline border,
// `Brand.Elevation.low` drop shadow. Used as the unit of layout in
// every dashboard tab. `trailing` is intentionally type-erased to
// AnyView so the call site can supply any view in the secondary slot
// without forcing the Card itself to be generic over two View types
// (the alternative — `Card<Content, Trailing>` — complicated 30+ call
// sites for marginal compile-time benefit).

struct Card<Content: View>: View {
    let title: String?
    let subtitle: String?
    let trailing: AnyView?
    let content: Content
    let padding: CGFloat

    init(
        _ title: String? = nil,
        subtitle: String? = nil,
        padding: CGFloat = 16,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.trailing = nil
        self.padding = padding
        self.content = content()
    }

    init<Trailing: View>(
        _ title: String,
        subtitle: String? = nil,
        padding: CGFloat = 16,
        @ViewBuilder trailing: () -> Trailing,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.trailing = AnyView(trailing())
        self.padding = padding
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if title != nil || subtitle != nil {
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 2) {
                        if let title {
                            Text(title)
                                .font(.headline)
                                .foregroundStyle(.primary)
                        }
                        if let subtitle {
                            Text(subtitle)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer(minLength: 8)
                    if let trailing {
                        trailing
                    }
                }
            }
            content
        }
        .padding(padding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: Brand.Radii.l, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay {
                    RoundedRectangle(cornerRadius: Brand.Radii.l, style: .continuous)
                        .strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                }
                .shadow(
                    color: Brand.Elevation.low.color,
                    radius: Brand.Elevation.low.radius,
                    x: Brand.Elevation.low.x,
                    y: Brand.Elevation.low.y
                )
        }
    }
}
