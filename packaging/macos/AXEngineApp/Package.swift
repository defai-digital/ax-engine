// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AXEngineApp",
    platforms: [.macOS("26.0")],
    products: [.executable(name: "AXEngineApp", targets: ["AXEngineApp"])],
    dependencies: [.package(path: "../../../sdk/swift")],
    targets: [
        .target(name: "AXEngineAppCore", dependencies: [.product(name: "AxEngine", package: "swift")]),
        .executableTarget(name: "AXEngineApp", dependencies: ["AXEngineAppCore"]),
        .testTarget(name: "AXEngineAppCoreTests", dependencies: ["AXEngineAppCore"], resources: [.copy("Fixtures")]),
    ]
)
