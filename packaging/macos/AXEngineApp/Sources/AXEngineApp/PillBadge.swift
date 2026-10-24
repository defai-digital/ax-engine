// Derived from MTPLX, Copyright 2026 Youssof Altoukhi (Apache-2.0).
// Source: apps/MTPLXApp/Sources/MTPLXAppHost/Views/Common/PillBadge.swift
// Revision: 9882703f3105363ddc37eca9f97aa09a1d387112
// Modified for AX Engine: native presentation, bounded log messages and endpoint policy.
// See Resources/MTPLX-LICENSE.txt and Resources/MTPLX-NOTICE.txt.
import SwiftUI

// MARK: - PillBadge
//
// Soft pill used for state / profile / connection chips in the
// toolbar and bottom bars. `emphasized` swaps the foreground to the
// tint color so the badge reads as a status callout instead of a
// neutral chip.

struct PillBadge: View {
    let text: String
    var systemImage: String? = nil
    var tint: Color = .secondary
    var emphasized: Bool = false

    var body: some View {
        HStack(spacing: 4) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.caption2)
            }
            Text(text)
                .font(.caption.weight(.medium))
        }
        .foregroundStyle(emphasized ? tint : .primary)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background {
            Capsule(style: .continuous)
                .fill(tint.opacity(emphasized ? 0.18 : 0.10))
                .overlay {
                    Capsule(style: .continuous)
                        .strokeBorder(tint.opacity(0.25), lineWidth: Brand.hairline)
                }
        }
    }
}
