import AppKit

/// Shares one graceful shutdown between user quits and updater restart requests.
@MainActor
final class AppQuitCoordinator {
    private var stopping = false
    private var confirming = false

    func request(needsConfirmation: Bool, confirm: () -> Bool,
                 shutdown: @escaping @MainActor () async -> Void,
                 reply: @escaping @MainActor () -> Void) -> NSApplication.TerminateReply {
        guard !confirming else { return .terminateCancel }
        guard !stopping else { return .terminateLater }
        if needsConfirmation {
            confirming = true
            let approved = confirm()
            confirming = false
            if !approved { return .terminateCancel }
        }
        stopping = true
        Task {
            await shutdown()
            reply()
        }
        return .terminateLater
    }
}
