import Foundation

/// Splits prose into ATX headings and text runs; inline syntax stays inline.
public enum MarkdownBlocks: Equatable, Sendable {
    case heading(Int, String)
    case text(String)
    public static func split(_ prose: String) -> [MarkdownBlocks] {
        var blocks: [MarkdownBlocks] = [], run: [Substring] = []
        func flush() {
            let text = run.joined(separator: "\n").trimmingCharacters(in: .newlines)
            if !text.isEmpty { blocks.append(.text(text)) }
            run = []
        }
        for line in prose.split(separator: "\n", omittingEmptySubsequences: false) {
            let hashes = line.prefix { $0 == "#" }.count
            if (1...6).contains(hashes), line.dropFirst(hashes).first == " " {
                flush()
                blocks.append(.heading(hashes, String(line.dropFirst(hashes + 1)).trimmingCharacters(in: .whitespaces)))
            } else {
                run.append(line)
            }
        }
        flush()
        return blocks
    }
}

