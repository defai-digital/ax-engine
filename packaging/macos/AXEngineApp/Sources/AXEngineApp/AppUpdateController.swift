import Combine
import Foundation
import Sparkle
import SwiftUI

@MainActor
protocol AppUpdateDriver: AnyObject {
    var canCheck: AnyPublisher<Bool, Never> { get }
    var automaticChecks: AnyPublisher<Bool, Never> { get }
    func start() throws
    func check()
    func setAutomaticChecks(_ enabled: Bool)
}

@MainActor
private final class SparkleUpdateDriver: AppUpdateDriver {
    private let controller = SPUStandardUpdaterController(
        startingUpdater: false, updaterDelegate: nil, userDriverDelegate: nil)
    var canCheck: AnyPublisher<Bool, Never> {
        controller.updater.publisher(for: \.canCheckForUpdates).eraseToAnyPublisher()
    }
    var automaticChecks: AnyPublisher<Bool, Never> {
        controller.updater.publisher(for: \.automaticallyChecksForUpdates).eraseToAnyPublisher()
    }
    func start() throws { try controller.updater.start() }
    func check() { controller.checkForUpdates(nil) }
    func setAutomaticChecks(_ enabled: Bool) {
        controller.updater.automaticallyChecksForUpdates = enabled
    }
}

@MainActor
final class AppUpdateController: ObservableObject {
    @Published private(set) var canCheck = false
    @Published private(set) var automaticChecks = false
    @Published private(set) var unavailableReason: String?
    private var driver: (any AppUpdateDriver)?
    private var subscriptions = Set<AnyCancellable>()
    private var started = false
    private let makeDriver: () -> any AppUpdateDriver
    private let info: [String: Any]

    init(info: [String: Any] = Bundle.main.infoDictionary ?? [:],
         makeDriver: @escaping () -> any AppUpdateDriver = { SparkleUpdateDriver() }) {
        self.info = info
        self.makeDriver = makeDriver
        unavailableReason = Self.configurationError(info)
    }

    static func configurationError(_ info: [String: Any]) -> String? {
        guard info["AXEngineDistributionBuild"] as? Bool == true else {
            return "Updates are available in installed release builds. Development builds do not update themselves."
        }
        guard info["SUFeedURL"] as? String == "https://download.automatosx.com/ax-engine-desktop/appcast.xml",
              let key = info["SUPublicEDKey"] as? String, Data(base64Encoded: key)?.count == 32,
              info["SUAllowsAutomaticUpdates"] as? Bool == false,
              info["SUEnableSystemProfiling"] as? Bool == false else {
            return "Update configuration is invalid. Install a signed AX Engine release to restore updates."
        }
        return nil
    }

    func start() {
        guard !started, unavailableReason == nil else { return }
        started = true
        let candidate = makeDriver()
        do {
            try candidate.start()
            driver = candidate
            candidate.canCheck.sink { [weak self] in self?.canCheck = $0 }.store(in: &subscriptions)
            candidate.automaticChecks.sink { [weak self] in self?.automaticChecks = $0 }.store(in: &subscriptions)
        } catch {
            unavailableReason = "Updates could not start: \(error.localizedDescription)"
        }
    }

    func check() {
        guard canCheck, unavailableReason == nil else { return }
        driver?.check()
    }

    func retryStart() {
        guard driver == nil, Self.configurationError(info) == nil else { return }
        started = false
        unavailableReason = nil
        start()
    }

    var canRetryStart: Bool { unavailableReason != nil && Self.configurationError(info) == nil }

    func setAutomaticChecks(_ enabled: Bool) { driver?.setAutomaticChecks(enabled) }
}

struct CheckForAppUpdates: View {
    @ObservedObject var updates: AppUpdateController
    var body: some View {
        Button("Check for Updates…", action: updates.check).disabled(!updates.canCheck)
    }
}

struct AppUpdatesSettings: View {
    @ObservedObject var updates: AppUpdateController
    var body: some View {
        Section("App updates") {
            LabeledContent("Installed version", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Development")
            if let reason = updates.unavailableReason {
                Text(reason).foregroundStyle(.secondary)
                if updates.canRetryStart { Button("Retry", action: updates.retryStart) }
            } else {
                Toggle("Automatically check for updates", isOn: Binding(
                    get: { updates.automaticChecks }, set: { enabled in updates.setAutomaticChecks(enabled) }))
                CheckForAppUpdates(updates: updates)
                Text("AX Engine checks its release service for updates. You choose when to download and restart to install. Models, settings and conversations stay on this Mac.")
                    .foregroundStyle(.secondary)
                Text("Restarting stops app-owned inference and model operations. You can cancel the restart and finish your work first.")
                    .foregroundStyle(.secondary)
            }
        }
    }
}
