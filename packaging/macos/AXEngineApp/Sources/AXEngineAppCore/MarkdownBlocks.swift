import Foundation

/// Splits a reply into ATX headings, fenced code and runs of inline Markdown.
public enum MarkdownBlocks: Equatable, Sendable {
    case heading(Int, String)
    case text(String)
    case code(language: String?, body: String)

    public static func split(_ prose: String) -> [MarkdownBlocks] {
        var blocks: [MarkdownBlocks] = [], run: [Substring] = []
        var fence: (marker: Character, width: Int, indent: Int, language: String?)?
        func flushText() {
            let text = run.joined(separator: "\n").trimmingCharacters(in: .newlines)
            if !text.isEmpty { blocks.append(.text(text)) }
            run.removeAll()
        }
        let lines = prose.split(omittingEmptySubsequences: false) { $0 == "\n" || $0 == "\r\n" || $0 == "\r" }
        for line in lines {
            let indent = line.prefix { $0 == " " }.count
            let trimmed = line.dropFirst(indent)
            if let open = fence {
                let width = trimmed.prefix { $0 == open.marker }.count
                if indent <= 3, width >= open.width, trimmed.dropFirst(width).allSatisfy({ $0 == " " || $0 == "\t" }) {
                    let body = run.isEmpty ? "" : run.joined(separator: "\n") + "\n"
                    blocks.append(.code(language: open.language, body: body))
                    run.removeAll()
                    fence = nil
                } else {
                    run.append(line.dropFirst(min(open.indent, indent)))
                }
                continue
            }
            if indent <= 3, let marker = trimmed.first, marker == "`" || marker == "~" {
                let width = trimmed.prefix { $0 == marker }.count
                let info = trimmed.dropFirst(width)
                if width >= 3, marker != "`" || !info.contains("`") {
                    flushText()
                    let language = info.split(whereSeparator: \.isWhitespace).first.map(String.init)
                    fence = (marker, width, indent, language)
                    continue
                }
            }
            let hashes = line.prefix { $0 == "#" }.count
            if (1...6).contains(hashes), line.dropFirst(hashes).first == " " {
                flushText()
                blocks.append(.heading(hashes, String(line.dropFirst(hashes + 1)).trimmingCharacters(in: .whitespaces)))
            } else {
                run.append(line)
            }
        }
        if let fence {
            blocks.append(.code(language: fence.language, body: run.joined(separator: "\n")))
        } else {
            flushText()
        }
        return blocks
    }
}
