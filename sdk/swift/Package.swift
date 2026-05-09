// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ax-engine-swift",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "AxEngine", targets: ["AxEngine"]),
    ],
    targets: [
        .target(
            name: "AxEngine",
            path: "Sources/AxEngine"
        ),
        .testTarget(
            name: "AxEngineTests",
            dependencies: ["AxEngine"],
            path: "Tests/AxEngineTests"
        ),
    ]
)
