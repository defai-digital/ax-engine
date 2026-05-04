# AX Engine v4

AX Engine v4 is a Mac-first LLM inference core focused on Apple M4-or-newer Silicon.

Status:

- core engine contracts (request lifecycle, scheduler, KV cache, runner
  integration) are implemented
- benchmark tooling with scenario, replay, compare, matrix, direct inference,
  and MLX inference-stack comparison commands is operational
- preview SDK, local HTTP server, Python bindings, and a JavaScript preview
  client are available
- repo-owned MLX inference is the AX-owned runtime path
- non-MLX local inference routes through the delegated `llama.cpp` contract

Repository layout:

- `crates/` — implementation workspace (5 crates)
- `docs/` — public-facing documentation
- `benchmarks/` — canonical benchmark manifests and benchmark assets
- `javascript/` — repo-local JavaScript preview client package
- `python/` — Python package wrapper, type stubs, tests, examples
- `scripts/` — E2E smoke check scripts

Public docs:

- `docs/ARCHITECTURE.md`
- `docs/README.md`
- `docs/GETTING-STARTED.md`
- `docs/JAVASCRIPT.md`
- `docs/CLI.md`
- `docs/PYTHON.md`
- `docs/SERVER.md`
- `docs/SUPPORTED-MODELS.md`
- `docs/BENCHMARKS.md`

Benchmarking now has a strict split: use `scripts/bench_mlx_inference_stack.py`
for AX Engine MLX mode versus `mlx_lm.benchmark` and optional `mlx-swift-lm`
JSON-adapter comparisons; use `ax-bench` for checked-in scenario, replay,
matrix, baseline, and delegated-route contract artifacts. llama.cpp manifests
validate non-MLX delegation behavior and must not be used as AX-owned MLX
throughput baselines.

For a repo-owned Python packaging smoke check that bootstraps a temporary
virtual environment, installs `maturin`, builds the preview package, runs the
checked-in Python examples, and then runs both the installed-package preview
tests and the Python wrapper tests, use
`bash scripts/check-python-preview.sh`.

For a repo-owned server smoke check that starts the preview binary and exercises
its health, runtime, one-shot generate, cancel, and SSE streaming paths over
real HTTP, use `bash scripts/check-server-preview.sh`.

## Contributing

AX Engine welcomes public contributions. See [CONTRIBUTING.md](CONTRIBUTING.md)
for guidelines.

## Community

- Website: [automatosx.com](https://automatosx.com)
- Discord: [Join us](https://discord.com/invite/cTavsMgu)
- Email: enquiry@defai.digital

## License

MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 [DEFAI Private Limited](https://defai.digital)

---

Migration note:

Compared with the earlier AX Engine repository, v4 currently contains the new
engine-core bring-up path, benchmark manifests, a manifest-driven benchmark
CLI, a preview `ax-engine-sdk` crate for backend-resolution and session
contract bring-up, a preview `ax-engine-server` local HTTP adapter built on
that SDK contract, a preview `ax-engine-py` / `python/ax_engine` binding
surface for token-based generation, request lifecycle control, and in-process
streaming evaluation flows, plus a thin repo-local JavaScript preview client
over the checked-in HTTP surface.
A thin user-facing inference CLI is now present through `ax-bench generate` and
`ax-bench stream`, while broader migrated binding surfaces are still expected to
arrive as thin layers above the core rather than as the design center of v4.
