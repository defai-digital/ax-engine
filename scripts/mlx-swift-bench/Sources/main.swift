// mlx-swift-bench: Throughput benchmark adapter for mlx-swift-lm.
//
// Outputs JSON to stdout:
//   {"trials": [{"prefill_tok_s": ..., "decode_tok_s": ..., "peak_memory_gb": ...}, ...]}
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import BenchmarkHelpers

// MARK: - Args

struct BenchArgs: @unchecked Sendable {
    var modelDir: URL
    var promptTokenIdsPath: URL
    var generationTokens: Int = 128
    var trials: Int = 5
    var delay: Double = 5.0
    var prefillStepSize: Int = 512
}

func parseArgs() -> BenchArgs {
    var raw = Array(CommandLine.arguments.dropFirst())
    var a = BenchArgs(
        modelDir: URL(fileURLWithPath: "."),
        promptTokenIdsPath: URL(fileURLWithPath: ".")
    )
    var modelSet = false
    var tokPathSet = false

    while !raw.isEmpty {
        let flag = raw.removeFirst()
        switch flag {
        case "--model":
            a.modelDir = URL(fileURLWithPath: raw.removeFirst())
            modelSet = true
        case "--prompt-token-ids", "--prompt-tokens-path", "--prompt_token_ids_path":
            a.promptTokenIdsPath = URL(fileURLWithPath: raw.removeFirst())
            tokPathSet = true
        case "--generation-tokens":
            a.generationTokens = Int(raw.removeFirst())!
        case "--trials":
            a.trials = Int(raw.removeFirst())!
        case "--delay":
            a.delay = Double(raw.removeFirst())!
        case "--prefill-step-size":
            a.prefillStepSize = Int(raw.removeFirst())!
        default:
            if !raw.isEmpty && !raw[0].hasPrefix("--") {
                raw.removeFirst()
            }
        }
    }

    if !modelSet || !tokPathSet {
        fputs("error: --model and --prompt-token-ids are required\n", stderr)
        exit(1)
    }
    return a
}

// MARK: - Token loading

func loadTokenIds(from url: URL) throws -> [Int32] {
    let data = try Data(contentsOf: url)
    let obj = try JSONSerialization.jsonObject(with: data)
    if let dict = obj as? [String: Any] {
        if let ids = dict["token_ids"] as? [Int] { return ids.map { Int32($0) } }
        if let ids = dict["prompt_token_ids"] as? [Int] { return ids.map { Int32($0) } }
    }
    if let bare = obj as? [Int] { return bare.map { Int32($0) } }
    fputs("error: cannot find token_ids array in \(url.path)\n", stderr)
    exit(1)
}

func shouldUseVLMFactory(modelDir: URL) -> Bool {
    let configURL = modelDir.appendingPathComponent("config.json")
    guard
        let data = try? Data(contentsOf: configURL),
        let obj = try? JSONSerialization.jsonObject(with: data),
        let dict = obj as? [String: Any],
        dict["model_type"] as? String == "gemma4",
        let textConfig = dict["text_config"] as? [String: Any],
        textConfig["enable_moe_block"] as? Bool == true
    else {
        return false
    }

    return true
}

func loadModelContainer(modelDir: URL, useVLMFactory: Bool) async throws -> ModelContainer {
    if useVLMFactory {
        fputs("  [mlx-swift-bench] using MLXVLM factory for Gemma4 MoE\n", stderr)
        return try await VLMModelFactory.shared.loadContainer(
            from: modelDir,
            using: NoOpTokenizerLoader()
        )
    }

    fputs("  [mlx-swift-bench] using MLXLLM factory\n", stderr)
    return try await LLMModelFactory.shared.loadContainer(
        from: modelDir,
        using: NoOpTokenizerLoader()
    )
}

// MARK: - Output types

struct TrialResult: Codable, Sendable {
    let prefill_tok_s: Double
    let decode_tok_s: Double
    let peak_memory_gb: Double
}

struct BenchOutput: Codable {
    let trials: [TrialResult]
}

// MARK: - Benchmark

func runOneTrial(
    container: ModelContainer,
    tokenIds: [Int32],
    generationTokens: Int,
    prefillStepSize: Int,
    trialIndex: Int,
    useBatchInput: Bool
) async throws -> TrialResult {
    let params = GenerateParameters(
        maxTokens: generationTokens,
        temperature: 0,
        prefillStepSize: prefillStepSize
    )
    let promptTokenCount = tokenIds.count

    return try await container.perform { (context: ModelContext) throws -> TrialResult in
        let flatPromptArray = MLXArray(tokenIds)
        let promptArray = useBatchInput ? flatPromptArray.expandedDimensions(axis: 0) : flatPromptArray
        let input = LMInput(tokens: promptArray)

        // Time prefill by measuring the init (prefill happens inside init).
        let prefillStart = Date.timeIntervalSinceReferenceDate
        let iter = try TokenIterator(input: input, model: context.model, parameters: params)
        let prefillTime = Date.timeIntervalSinceReferenceDate - prefillStart

        var genCount = 0
        let genStart = Date.timeIntervalSinceReferenceDate
        for _ in iter {
            genCount += 1
            if genCount >= generationTokens { break }
        }
        let genTime = Date.timeIntervalSinceReferenceDate - genStart

        let prefill_tps = prefillTime > 0 ? Double(promptTokenCount) / prefillTime : 0
        let decode_tps = genCount > 0 && genTime > 0 ? Double(genCount) / genTime : 0
        let peak_gb = Double(Memory.activeMemory) / 1_073_741_824.0

        fputs(
            "    trial \(trialIndex): prefill=\(String(format: "%.1f", prefill_tps)) tok/s "
                + "decode=\(String(format: "%.1f", decode_tps)) tok/s "
                + "peak_mem=\(String(format: "%.2f", peak_gb)) GB\n",
            stderr
        )
        return TrialResult(
            prefill_tok_s: prefill_tps,
            decode_tok_s: decode_tps,
            peak_memory_gb: peak_gb
        )
    }
}

func runBenchmark(args: BenchArgs) async throws {
    fputs("  [mlx-swift-bench] loading model from \(args.modelDir.path)\n", stderr)
    let useVLMFactory = shouldUseVLMFactory(modelDir: args.modelDir)
    let container = try await loadModelContainer(
        modelDir: args.modelDir,
        useVLMFactory: useVLMFactory
    )

    let tokenIds = try loadTokenIds(from: args.promptTokenIdsPath)
    fputs(
        "  [mlx-swift-bench] prompt_tokens=\(tokenIds.count) "
            + "generation_tokens=\(args.generationTokens) trials=\(args.trials)\n",
        stderr
    )

    // Warmup
    fputs("  [mlx-swift-bench] warmup\n", stderr)
    _ = try await runOneTrial(
        container: container,
        tokenIds: tokenIds,
        generationTokens: args.generationTokens,
        prefillStepSize: args.prefillStepSize,
        trialIndex: 0,
        useBatchInput: useVLMFactory
    )
    Memory.clearCache()

    // Timed trials
    var results: [TrialResult] = []
    for i in 1...args.trials {
        if args.delay > 0 {
            try await Task.sleep(nanoseconds: UInt64(args.delay * 1_000_000_000))
        }
        let result = try await runOneTrial(
            container: container,
            tokenIds: tokenIds,
            generationTokens: args.generationTokens,
            prefillStepSize: args.prefillStepSize,
            trialIndex: i,
            useBatchInput: useVLMFactory
        )
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
