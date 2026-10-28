// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AXEngineApp",
    platforms: [.macOS("26.0")],
    products: [.executable(name: "AXEngineApp", targets: ["AXEngineApp"]),
               .executable(name: "AXEngineModelScanner", targets: ["AXEngineModelScanner"])],
    dependencies: [.package(path: "../../../sdk/swift"),
                   .package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.10.1")],
    targets: [
        .target(name: "AXEngineAppCore", dependencies: [.product(name: "AxEngine", package: "swift")]),
        .executableTarget(name: "AXEngineApp", dependencies: ["AXEngineAppCore", .product(name: "Sparkle", package: "Sparkle")],
                          linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"])]),
        .executableTarget(name: "AXEngineModelScanner", dependencies: ["AXEngineAppCore"]),
        .testTarget(name: "AXEngineAppTests", dependencies: ["AXEngineApp"]),
        .testTarget(name: "AXEngineAppCoreTests", dependencies: ["AXEngineAppCore"], resources: [.copy("Fixtures")]),
    ]
)
