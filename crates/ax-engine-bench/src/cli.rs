pub(crate) fn usage() -> String {
    let text = r#"AX Engine v6 benchmark CLI

Usage:
  ax-engine-bench generate [--model-id <id>] (--prompt <text> | --tokens <ids>) [--multimodal-inputs-json <json> | --multimodal-inputs-file <path>] [--max-output-tokens <n>] [--mlx] [--support-tier <tier>] [--llama-cli-path <path>] [--llama-model-path <path>] [--llama-server-url <url>] [--mlx-lm-server-url <url>] [--mlx-model-artifacts-dir <path>] [--json]
  ax-engine-bench stream [--model-id <id>] (--prompt <text> | --tokens <ids>) [--multimodal-inputs-json <json> | --multimodal-inputs-file <path>] [--max-output-tokens <n>] [--mlx] [--support-tier <tier>] [--llama-cli-path <path>] [--llama-model-path <path>] [--llama-server-url <url>] [--mlx-lm-server-url <url>] [--mlx-model-artifacts-dir <path>] [--json]
  ax-engine-bench scenario --manifest <path> --output-root <path> [--json] [--no-trace]
  ax-engine-bench replay --manifest <path> --output-root <path> [--json] [--no-trace]
  ax-engine-bench compare --baseline <path> --candidate <path> --output-root <path> [--json]
  ax-engine-bench matrix-compare --baseline <path> --candidate <path> --output-root <path> [--json]
  ax-engine-bench baseline --source <path> --name <name> --output-root <path> [--json]
  ax-engine-bench matrix --manifest <path> --output-root <path> [--json] [--no-trace]
  ax-engine-bench doctor [--json] [--mlx-model-artifacts-dir <path>]
  ax-engine-bench generate-manifest <model-dir> [--json] [--validate]
  ax-engine-bench metal-build [--manifest <path>] [--output-dir <path>]
  ax-engine-bench serving-stress --workload <name> [--mlx-model-artifacts-dir <path>] [--model-id <id>] [--prefill-tokens <n>] [--decode-tokens <n>] [--concurrent-short-requests <n>] [--short-prefix-tokens <n>] [--seed <n>] [--output-path <path>] [--json]
"#;

    text.to_string()
}

pub(crate) fn generate_manifest_usage() -> String {
    "Usage: ax-engine-bench generate-manifest <model-dir> [--json] [--validate]\n\n\
     Generates model-manifest.json for an MLX model snapshot. Required before \
     ax-engine can load the model. With --validate, reads the generated \
     model-manifest.json back through the native model artifact validator."
        .to_string()
}
