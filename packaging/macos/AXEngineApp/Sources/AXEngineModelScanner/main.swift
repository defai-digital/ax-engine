import Foundation
import AXEngineAppCore

// Separate from the UI process so slow/unavailable filesystem mounts can be
// cancelled with a bounded deadline. This helper only reads file metadata.
let arguments = CommandLine.arguments
if arguments.count == 3, arguments[1] == "--volume", arguments[2].utf8.count <= 256 * 1024 {
    do {
        let value = try StorageSnapshot.read(path: arguments[2])
        FileHandle.standardOutput.write(try JSONEncoder().encode(value))
        exit(0)
    } catch { exit(1) }
}
guard arguments.count == 2, arguments[1].utf8.count <= 256 * 1024 else { exit(2) }
let request = try JSONDecoder().decode(LibraryScanRequest.self, from: Data(arguments[1].utf8))
guard request.root.hasPrefix("/"), request.imported.allSatisfy({ $0.hasPrefix("/") }) else { exit(2) }
func emit(_ event: LibraryScanEvent) {
    if var data = try? JSONEncoder().encode(event) {
        data.append(10)
        FileHandle.standardOutput.write(data)
    }
}
_ = ModelStore.scanLibrary(root: request.root, imported: request.imported,
                          emit: { emit(LibraryScanEvent(model: $0)) },
                          warning: { emit(LibraryScanEvent(warning: $0)) })
emit(LibraryScanEvent(complete: true))
