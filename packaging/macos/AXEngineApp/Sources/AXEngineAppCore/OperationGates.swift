import Foundation

/// Why a library row cannot be removed right now. Pure so the GUI shows the
/// same sentence inline, in tooltips and in tests.
public enum RemovalGate {
    public struct State: Sendable, Equatable {
        public var removable: Bool
        public var source: String
        public var downloading = false
        public var removing = false
        public var runtimeBusy = false
        public var servedByThisApp = false
        public var attachedToExternalService = false
        public init(removable: Bool, source: String, downloading: Bool = false, removing: Bool = false,
                    runtimeBusy: Bool = false, servedByThisApp: Bool = false, attachedToExternalService: Bool = false) {
            self.removable = removable; self.source = source; self.downloading = downloading; self.removing = removing
            self.runtimeBusy = runtimeBusy; self.servedByThisApp = servedByThisApp
            self.attachedToExternalService = attachedToExternalService
        }
    }

    /// Nil when removal can proceed. Permanent reasons come before transient ones.
    public static func reason(_ state: State) -> String? {
        if !state.removable { return permanentReason(source: state.source) }
        if state.servedByThisApp { return "Stop the server to remove the model it is serving." }
        if state.downloading { return "Wait for downloads to finish before removing models." }
        if state.removing { return "Another model operation is in progress." }
        if state.runtimeBusy { return "Wait for the server to finish starting or stopping." }
        if state.attachedToExternalService { return "Disconnect from the attached service first; it may be serving this model." }
        return nil
    }

    /// True when the reason will not clear by itself (the row is not app-removable).
    public static func isPermanent(_ state: State) -> Bool { !state.removable }

    static func permanentReason(source: String) -> String {
        switch source {
        case "Imported": "Imported folders are never deleted by AX Engine. Remove the folder yourself if you no longer need it."
        case "Configured folder": "This folder comes from AX_ENGINE_MODELS_DIR and is managed outside the app."
        case "Hugging Face cache": "Only AutomatosX repositories in the Hugging Face cache can be removed here."
        default: "This folder was not downloaded by AX Engine, so the app does not delete it."
        }
    }
}

/// Why Start Server is unavailable, or nil when it can run.
public enum ServerGate {
    public struct State: Sendable, Equatable {
        public var selectedPath: String
        public var selectable: Bool
        public var libraryEmpty: Bool
        public var scanning = false
        public var runtimeBusy = false
        public var removing = false
        public var attached = false
        public init(selectedPath: String, selectable: Bool, libraryEmpty: Bool, scanning: Bool = false,
                    runtimeBusy: Bool = false, removing: Bool = false, attached: Bool = false) {
            self.selectedPath = selectedPath; self.selectable = selectable; self.libraryEmpty = libraryEmpty
            self.scanning = scanning; self.runtimeBusy = runtimeBusy; self.removing = removing; self.attached = attached
        }
    }

    public static func startBlockReason(_ state: State) -> String? {
        if state.attached { return "Disconnect from the attached service to start your own server." }
        if state.runtimeBusy { return "The server is still starting or stopping." }
        if state.removing { return "Wait for the model operation to finish." }
        if state.selectedPath.isEmpty {
            return state.libraryEmpty
                ? "No installed model yet. Download one from Models › Catalog first."
                : "Choose an installed model to start."
        }
        if !state.selectable {
            return state.scanning
                ? "The library scan has not checked this model yet."
                : "The selected model is incomplete or no longer in the library. Choose another model."
        }
        return nil
    }

    /// Restart button title for an app-owned server, naming what the restart
    /// changes so a plain restart and a configuration switch read differently.
    public static func restartTitle(modelChanged: Bool, modeChanged: Bool, capChanged: Bool = false, mode: ServingMode) -> String {
        switch (modelChanged, modeChanged) {
        case (true, true): "Restart with Selected Model in \(mode.title) Mode"
        case (true, false): capChanged ? "Restart with Selected Model and New Request Cap" : "Restart with Selected Model"
        case (false, true): "Restart in \(mode.title) Mode"
        case (false, false): capChanged ? "Restart with New Request Cap" : "Restart Server"
        }
    }

    /// Explains a saved change that the running server does not use yet.
    public static func pendingRestartNotice(modelChanged: Bool, modeChanged: Bool, running: ServingMode, selected: ServingMode,
                                            runningCap: Int? = nil, selectedCap: Int? = nil) -> String? {
        let capChanged = runningCap != nil && selectedCap != nil && runningCap != selectedCap
        let capNote = capChanged ? " The request cap changes from \(runningCap ?? 0) to \(selectedCap ?? 0) at the same time." : ""
        switch (modelChanged, modeChanged) {
        case (true, true):
            return "The selected model and serving mode differ from the running server. Restart to switch; it keeps serving in \(running.title) mode until then." + capNote
        case (true, false):
            return "The selected model differs from the running one. Restart to switch; the current server keeps serving until then." + capNote
        case (false, true):
            return "\(selected.title) mode is saved for the next launch. The running server keeps \(running.title) mode until you restart." + capNote
        case (false, false):
            return capChanged
                ? "Up to \(selectedCap ?? 0) concurrent requests is saved for the next launch. The running server still admits \(runningCap ?? 0) until you restart."
                : nil
        }
    }

    /// Human text for the server state pill, with elapsed time for transitions.
    public static func stateDetail(state: String, startedAt: Date?, now: Date = Date()) -> String? {
        guard let startedAt else { return nil }
        let elapsed = TransferFormat.duration(now.timeIntervalSince(startedAt))
        switch state {
        case "Starting": return "Loading the model · \(elapsed)"
        case "Connecting": return "Connecting · \(elapsed)"
        case "Running", "Attached": return "Up for \(elapsed)"
        case "Connection degraded": return "Last healthy response is stale · up for \(elapsed)"
        default: return nil
        }
    }
}
