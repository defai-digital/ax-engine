// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "mlx-swift-bench",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../../.internal/reference/mlx-swift-lm"),
    ],
    targets: [
        .executableTarget(
            name: "mlx-swift-bench",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "BenchmarkHelpers", package: "mlx-swift-lm"),
            ],
            path: "Sources"
        ),
        .executableTarget(
            name: "mlx-swift-embed-bench",
            dependencies: [
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "BenchmarkHelpers", package: "mlx-swift-lm"),
            ],
            path: "EmbedSources"
        ),
    ]
)
