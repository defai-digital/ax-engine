// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ax-engine-swift-examples",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../../sdk/swift"),
    ],
    targets: [
        .executableTarget(
            name: "Chat",
            dependencies: [.product(name: "AxEngine", package: "ax-engine-swift")],
            path: "Sources/Chat"
        ),
        .executableTarget(
            name: "Stream",
            dependencies: [.product(name: "AxEngine", package: "ax-engine-swift")],
            path: "Sources/Stream"
        ),
    ]
)
