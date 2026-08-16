pub(crate) fn usage() -> String {
    let text = format!(
        r#"AX Engine v{} benchmark CLI

Usage:
  ax-engine-bench generate [--model-id <id>] (--prompt <text> | --tokens <ids>) [--multimodal-inputs-json <json> | --multimodal-inputs-file <path>] [--max-output-tokens <n>] [--ignore-eos] [--mlx] [--support-tier <tier>] [--llama-cli-path <path>] [--llama-model-path <path>] [--llama-server-url <url>] [--mlx-lm-server-url <url>] [--mlx-model-artifacts-dir <path>] [--json]
  ax-engine-bench stream [--model-id <id>] (--prompt <text> | --tokens <ids>) [--multimodal-inputs-json <json> | --multimodal-inputs-file <path>] [--max-output-tokens <n>] [--ignore-eos] [--mlx] [--support-tier <tier>] [--llama-cli-path <path>] [--llama-model-path <path>] [--llama-server-url <url>] [--mlx-lm-server-url <url>] [--mlx-model-artifacts-dir <path>] [--json]
  ax-engine-bench scenario --manifest <path> --output-root <path> [--json] [--no-trace]
  ax-engine-bench replay --manifest <path> --output-root <path> [--json] [--no-trace]
  ax-engine-bench compare --baseline <path> --candidate <path> --output-root <path> [--json]
  ax-engine-bench matrix-compare --baseline <path> --candidate <path> --output-root <path> [--json]
  ax-engine-bench baseline --source <path> --name <name> --output-root <path> [--json]
  ax-engine-bench matrix --manifest <path> --output-root <path> [--json] [--no-trace]
  ax-engine-bench doctor [--json] [--mlx-model-artifacts-dir <path>]
  ax-engine-bench generate-manifest <model-dir> [--force] [--json] [--validate]
  ax-engine-bench generate-manifest [--force] [--json] [--validate] -- <model-dir>
  ax-engine-bench metal-build [--manifest <path>] [--output-dir <path>]
  ax-engine-bench serving-stress --workload <name> [--mlx-model-artifacts-dir <path>] [--model-id <id>] [--prefill-tokens <n>] [--decode-tokens <n>] [--concurrent-short-requests <n>] [--short-prefix-tokens <n>] [--seed <n>] [--output-path <path>] [--json]
"#,
        env!("CARGO_PKG_VERSION")
    );

    text
}

pub(crate) fn generate_manifest_usage() -> String {
    "Usage: ax-engine-bench generate-manifest <model-dir> [--force] [--json] [--validate]\n\n\
     For a model directory beginning with '-', put -- before <model-dir>.\n\n\
     Generates model-manifest.json for an MLX model snapshot. Required before \
     ax-engine can load the model. With --force, replaces an existing manifest. \
     With --validate, reads the generated model-manifest.json back through the \
     native model artifact validator."
        .to_string()
}
