import Foundation

/// View-only filtering of the bounded runtime output: newest lines, an
/// optional text match and an optional warnings-and-errors view. The
/// underlying output and Copy are unaffected.
public enum LogFilter {
    public static let problemMarkers = ["error", "warn", "panic", "fatal", "failed", "traceback", "exception"]

    public static func isProblem(_ line: Substring) -> Bool {
        let lower = line.lowercased()
        return problemMarkers.contains { lower.contains($0) }
    }

    /// Returns the matching lines (at most `lastLines`, newest kept) and how
    /// many lines matched before that limit.
    public static func apply(_ text: String, query: String = "", problemsOnly: Bool = false,
                             lastLines: Int? = nil) -> (text: String, matched: Int, total: Int) {
        let lines = text.split(whereSeparator: \.isNewline)
        let needle = query.trimmingCharacters(in: .whitespaces)
        let matched = lines.filter { line in
            (needle.isEmpty || line.localizedCaseInsensitiveContains(needle)) && (!problemsOnly || isProblem(line))
        }
        let shown = lastLines.map { matched.suffix(max(0, $0)) } ?? matched[...]
        return (shown.joined(separator: "\n"), matched.count, lines.count)
    }
}
