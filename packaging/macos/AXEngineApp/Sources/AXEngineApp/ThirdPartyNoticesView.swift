import SwiftUI
import AppKit

struct ThirdPartyNoticesView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 14) {
                if let logo = Brand.logo {
                    Image(nsImage: logo).resizable().frame(width: 56, height: 56)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("AX Engine").font(.title2.bold())
                    Text("Third-Party Notices").foregroundStyle(.secondary)
                }
            }
            Text("AX Engine uses selected open-source components. These acknowledgements identify their authors and licenses.")
            Divider()
            Text("Native UI components").font(.headline)
            Text("Selected interface and support components are adapted from MTPLX by Youssof Altoukhi, Copyright 2026, under Apache-2.0. AX Engine provides the inference runtime.")
            Text("Powered by MTPLX")
            Link("MTPLX source repository", destination: URL(string: "https://github.com/youssofal/mtplx")!)
            HStack {
                Button("Read License") { openResource("MTPLX-LICENSE") }
                Button("Read Original Notice") { openResource("MTPLX-NOTICE") }
            }
        }
        .textSelection(.enabled)
        .padding(28)
        .frame(width: 560)
    }

    private func openResource(_ name: String) {
        if let url = Bundle.main.url(forResource: name, withExtension: "txt") {
            NSWorkspace.shared.open(url)
        }
    }
}
