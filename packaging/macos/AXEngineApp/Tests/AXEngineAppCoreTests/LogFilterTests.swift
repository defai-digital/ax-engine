import Testing
@testable import AXEngineAppCore

@Test func logFilterMatchesTextProblemsAndKeepsNewestLines() {
    let log = "loading weights\nWARN slow disk\nready on 127.0.0.1\nerror: bad request\n\nPanicked at main\n"
    #expect(LogFilter.apply(log).total == 5)
    let problems = LogFilter.apply(log, problemsOnly: true)
    #expect(problems.text == "WARN slow disk\nerror: bad request\nPanicked at main" && problems.matched == 3)
    #expect(LogFilter.apply(log, query: "READY").text == "ready on 127.0.0.1")
    let newest = LogFilter.apply(log, problemsOnly: true, lastLines: 2)
    #expect(newest.text == "error: bad request\nPanicked at main" && newest.matched == 3)
    #expect(LogFilter.apply(log, query: "disk", problemsOnly: true).text == "WARN slow disk")
    #expect(LogFilter.apply(log, query: "absent").text.isEmpty)
}
