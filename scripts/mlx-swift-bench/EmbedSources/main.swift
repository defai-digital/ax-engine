// mlx-swift-embed-bench: Embedding throughput benchmark for mlx-swift-lm MLXEmbedders.
//
// Reads pre-tokenized sentences from a JSON file, runs the embedding model,
// and outputs per-trial timing statistics to stdout as JSON.
//
// Input JSON (--token-ids-path):
//   [{"label": "sentence text", "token_ids": [1, 2, ..., eos_id]}, ...]
//
// Output JSON (stdout):
//   {"trials": [{"ms_per_sentence": 12.3, "tokens_per_sec": 842.5, "peak_memory_gb": 1.23}]}
import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import BenchmarkHelpers

// MARK: - Args

struct BenchArgs: @unchecked Sendable {
    var modelDir: URL
    var tokenIdsPath: URL
    var trials: Int = 5
    var delay: Double = 2.0
}

func parseArgs() -> BenchArgs {
    var raw = Array(CommandLine.arguments.dropFirst())
    var a = BenchArgs(
        modelDir: URL(fileURLWithPath: "."),
        tokenIdsPath: URL(fileURLWithPath: ".")
    )
    var modelSet = false
    var tokenIdsSet = false

    while !raw.isEmpty {
        let flag = raw.removeFirst()
        switch flag {
        case "--model":
            a.modelDir = URL(fileURLWithPath: raw.removeFirst())
            modelSet = true
        case "--token-ids-path":
            a.tokenIdsPath = URL(fileURLWithPath: raw.removeFirst())
            tokenIdsSet = true
        case "--trials":
            a.trials = Int(raw.removeFirst())!
        case "--delay":
            a.delay = Double(raw.removeFirst())!
        default:
            if !raw.isEmpty && !raw[0].hasPrefix("--") {
                raw.removeFirst()
            }
        }
    }

    if !modelSet || !tokenIdsSet {
        fputs("error: --model and --token-ids-path are required\n", stderr)
        exit(1)
    }
    return a
}

// MARK: - Input

struct SentenceTokenIds: Decodable {
    let label: String
    let token_ids: [Int]
}

func loadSentenceTokenIds(from url: URL) throws -> [SentenceTokenIds] {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode([SentenceTokenIds].self, from: data)
}

// MARK: - Output

struct TrialResult: Codable, Sendable {
    let ms_per_sentence: Double
    let tokens_per_sec: Double
    let peak_memory_gb: Double
}

struct BenchOutput: Codable {
    let trials: [TrialResult]
}

// MARK: - Benchmark

func runOneTrial(
    container: EmbedderModelContainer,
    sentences: [SentenceTokenIds],
    trialIndex: Int
) async throws -> TrialResult {
    let totalTokens = sentences.reduce(0) { $0 + $1.token_ids.count }

    return try await container.perform { (context: EmbedderModelContext) throws -> TrialResult in
        let start = CFAbsoluteTimeGetCurrent()

        for sentence in sentences {
            let ids = sentence.token_ids.map { Int32($0) }
            let tokenArray = MLXArray(ids).expandedDimensions(axis: 0)  // [1, seq]
            let output = context.model(
                tokenArray,
                positionIds: nil,
                tokenTypeIds: nil,
                attentionMask: nil
            )
            let embedding = context.pooling(output, normalize: true)
            embedding.eval()
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let elapsedMs = elapsed * 1000.0
        let msPerSentence = elapsedMs / Double(sentences.count)
        let tokensPerSec = elapsed > 0 ? Double(totalTokens) / elapsed : 0
        let peakGb = Double(Memory.activeMemory) / 1_073_741_824.0

        fputs(
            "    trial \(trialIndex): \(String(format: "%.1f", msPerSentence))ms/sentence "
                + "\(String(format: "%.1f", tokensPerSec)) tok/s "
                + "\(String(format: "%.2f", peakGb))GB\n",
            stderr
        )

        return TrialResult(
            ms_per_sentence: msPerSentence,
            tokens_per_sec: tokensPerSec,
            peak_memory_gb: peakGb
        )
    }
}

func runBenchmark(args: BenchArgs) async throws {
    fputs("  [mlx-swift-embed-bench] loading model from \(args.modelDir.path)\n", stderr)
    let container = try await EmbedderModelFactory.shared.loadContainer(
        from: args.modelDir,
        using: NoOpTokenizerLoader()
    )

    let sentences = try loadSentenceTokenIds(from: args.tokenIdsPath)
    let totalTokens = sentences.reduce(0) { $0 + $1.token_ids.count }
    fputs(
        "  [mlx-swift-embed-bench] \(sentences.count) sentences, \(totalTokens) total tokens, "
            + "\(args.trials) trials\n",
        stderr
    )

    fputs("  [mlx-swift-embed-bench] warmup\n", stderr)
    _ = try await runOneTrial(container: container, sentences: sentences, trialIndex: 0)
    Memory.clearCache()

    var results: [TrialResult] = []
    for i in 1 ... args.trials {
        if args.delay > 0 {
            try await Task.sleep(nanoseconds: UInt64(args.delay * 1_000_000_000))
        }
        let result = try await runOneTrial(container: container, sentences: sentences, trialIndex: i)
        results.append(result)
        Memory.clearCache()
    }

    let output = BenchOutput(trials: results)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let json = try encoder.encode(output)
    print(String(data: json, encoding: .utf8)!)
}

// MARK: - Entry

let benchArgs = parseArgs()
Task {
    do {
        try await runBenchmark(args: benchArgs)
        exit(0)
    } catch {
        fputs("error: \(error)\n", stderr)
        exit(1)
    }
}
RunLoop.main.run()
