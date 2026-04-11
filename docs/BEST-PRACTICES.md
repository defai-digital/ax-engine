# Use Cases & Best Practices

## When to Use AX Engine

Use AX Engine when:

- You need **maximum decode performance** on Apple Silicon M3+ for supported
  model families such as Qwen 3.5, Qwen 3 Coder, and Gemma 4.
- Building **on-device, privacy-first** applications (chatbots, agents, local RAG, coding assistants).
- You want **predictable low-latency** inference with fused Metal kernels and UMA-aware memory paths.
- Running **single-node or embedded production workloads** via the Rust SDK or Python bindings.
- Doing **benchmarking or research** comparing native Apple Silicon performance vs portable engines.

**Do not use** AX Engine if:
- You need broad model support beyond the current 4 families.
- Targeting non-Apple Silicon platforms.
- Using older M1/M2 Macs (officially unsupported).
- You need multi-node routing, orchestration, or governed deployment controls; use AX Serving for that layer.

See [ax-engine-vs-llama-cpp.md](ax-engine-vs-llama-cpp.md) for detailed comparison.

## Product Boundary

Treat AX Engine as the runtime layer:

- AX Engine owns inference, SDKs, bindings, CLI, and performance tooling.
- AX Serving owns production serving, routing, orchestration, and fleet controls.
- A lightweight AX Engine HTTP surface may exist for local or single-node use,
  but it should stay thin and should not replace AX Serving.

## Best Practices

### Model & Quantization
- Preferred: **Q4_K_M** (best speed/quality trade-off)
- Alternative: Q5_K_M for higher quality, Q8_0 for maximum accuracy
- Avoid: Q4_0, Q5_0, non-K quants

### Performance Tuning
- Use `AX_METAL_F16_KV_CACHE=auto` (default)
- For long context (>4k): consider `AX_HYBRID_DECODE=cpu` if memory pressure appears
- Enable `--verbose` during development to see prefill/decode plans

### CLI Usage
- Always use `--chat` with instruct-tuned models
- Set `--ctx-size` explicitly based on your workload (default is model-specific)
- Use `--deterministic` in benchmarks
- Prefer `--interactive` for testing

### Application Integration
- **Rust**: Use `ax-engine-sdk` crate for clean high-level API
- **Python**: Use `ax-engine-py` bindings
- **JavaScript/HTTP**: Use `packages/ax-engine-js` only against AX-compatible
  endpoints; treat first-party HTTP serving as a planned single-node surface
- Avoid shelling out to the CLI in production services
- Reuse `Model` and `Session` objects when possible

### Serving & Deployment
- Keep AX Engine embedded for local, edge, and single-node deployments
- Keep orchestration, auth, policy, and fleet concerns in AX Serving
- Do not let AX Engine accumulate cluster-control responsibilities

### Memory & Thermal
- Monitor peak RSS with `--verbose`
- Keep context size reasonable (4k–8k is sweet spot for most models)
- Be aware of thermal throttling on sustained high-throughput workloads
- Use `AX_METAL_RESIDENCY=1` on macOS 15+ for better weight residency

### Development Workflow
1. Start with `cargo check --workspace`
2. Test changes with `cargo test -p ax-engine-core`
3. Validate performance with `./benchmarks/run_apple_to_apple.py --ax-only`
4. Run full validation before PR: `cargo test --workspace && cargo clippy --workspace --tests -- -D warnings`

For environment variables see [ENV_VARS.md](ENV_VARS.md).
For benchmarking methodology see [BENCHMARKING.md](../BENCHMARKING.md).
