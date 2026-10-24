import Foundation

public struct DashboardStatus: Equatable, Sendable {
    public enum Action: Sendable { case models, catalog, server, chat }
    public let title: String
    public let detail: String
    public let actionTitle: String
    public let action: Action

    public static func resolve(state: String, ready: Bool, owned: Bool, hasConnection: Bool,
                               hasFiles: Bool, discovering: Bool, discoveryAttempted: Bool,
                               discoveryIssue: Bool, hasSelection: Bool = false) -> Self {
        switch state {
        case "Starting", "Connecting":
            return Self(title: state == "Starting" ? "Starting your model" : "Checking the connection",
                        detail: "Waiting for the runtime to report a model. You can inspect progress in Server.",
                        actionTitle: "View startup", action: .server)
        case "Stopping":
            return Self(title: "Stopping the local server", detail: "Waiting for app-owned work to finish.", actionTitle: "View server", action: .server)
        case "Failed", "Not ready", "Connection degraded":
            return Self(title: state == "Connection degraded" ? "Connection could not be verified" : "The runtime needs attention",
                        detail: owned ? "Inspect the app-owned process and its output in Server." : "Check the connection in Server. External services are never stopped by this app.",
                        actionTitle: "Review server", action: .server)
        default: break
        }
        if ready {
            return Self(title: "Your model is ready", detail: owned ? "Running in this app. Open Chat or use the local API." : "Connected to an external service. Disconnecting leaves that service running.",
                        actionTitle: "Open Chat", action: .chat)
        }
        if hasConnection || owned {
            return Self(title: "No model is ready yet", detail: "The runtime has not reported a usable model. Check Server for details.", actionTitle: "Review server", action: .server)
        }
        if hasFiles && hasSelection {
            return Self(title: "Review your selected model", detail: "Open Server to review the model and explicitly start it. Compatibility is checked at startup.",
                        actionTitle: "Review startup", action: .server)
        }
        if hasFiles {
            return Self(title: "Choose a downloaded model", detail: "Weight files were found. AX Engine checks compatibility when you start a model.", actionTitle: "Choose model", action: .models)
        }
        if discovering || !discoveryAttempted {
            return Self(title: "Finding your local models", detail: "Folders appear as they are checked. You can browse the library while discovery continues.", actionTitle: "View library", action: .models)
        }
        if discoveryIssue {
            return Self(title: "Some folders could not be checked", detail: "Your library may be incomplete. Review its status before downloading another copy.", actionTitle: "Review library", action: .models)
        }
        return Self(title: "Start with a model", detail: "Browse the runtime catalog or import an existing folder. Downloads and startup happen only when you choose them.", actionTitle: "Browse Catalog", action: .catalog)
    }
}
