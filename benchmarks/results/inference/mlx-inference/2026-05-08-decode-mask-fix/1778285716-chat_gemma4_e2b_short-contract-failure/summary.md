# Benchmark Contract Failure

- run_id: `1778285716-chat_gemma4_e2b_short-contract-failure`
- command: `ax-engine-bench scenario`
- manifest: `benchmarks/manifests/scenario/chat_gemma4_e2b_short.json`
- status: `contract_failure`
- code: `contract_validation_failed`
- selected_backend: `mlx`
- support_tier: `mlx_preview`
- resolution_policy: `mlx_only`
- backend_adapter: `"none"`
- scenario: `chat`

Failure reason:

failed to create benchmark SDK session from manifest runtime: MLX mode requires validated Metal runtime artifacts; deterministic fallback is internal-only and must be explicitly enabled

Recommended action:

Review the manifest/runtime contract and rerun with a supported workload shape or runtime configuration.
