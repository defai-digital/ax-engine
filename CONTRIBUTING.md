# Contributing to AX Engine

AX Engine welcomes community input primarily through reports, requests, and
evidence. We are not looking for unsolicited coding contributions at this stage.

The most helpful contributions are:

- Issue tickets for bugs, regressions, confusing behavior, install problems, or
  compatibility gaps
- Wishlist requests for models, routes, SDK behavior, server behavior, tools, or
  documentation
- Reproducible performance benchmark results, especially with raw artifacts
- Benchmark methodology feedback
- Documentation corrections or clarification requests

We generally do not accept unsolicited code pull requests. This is especially
important for runtime, scheduler, model graph, MLX kernel, KV/cache, n-gram,
quantization, serving, and performance-tuning changes. A change that improves
one benchmark row can easily regress correctness, TTFT, memory pressure,
direct-vs-n-gram behavior, long-context scaling, serving stability, or another
model family.

If you believe a code change is necessary, open an issue first. Maintainers may
invite a scoped pull request after the expected behavior, evidence, and
validation gate are agreed.

## What To Send

### Issue Tickets

For bugs or regressions, include:

- The AX Engine version, commit, or release build
- Host details: macOS version, chip, memory size, and whether the machine was
  plugged in or thermally constrained
- The exact command, request payload, model ID, model artifact path, and backend
  route used
- Expected behavior and actual behavior
- Logs, JSON responses, screenshots, or terminal output when available
- Whether the issue reproduces with direct AX mode, AX n-gram mode, delegated
  `mlx_lm`, or `llama.cpp`

### Wishlist Requests

For product or model requests, include:

- The user workflow you want to support
- The model family, quantization, context length, and serving mode you care about
- Why the request matters compared with the currently documented routes
- Any reference behavior from `mlx_lm`, `llama.cpp`, a paper, or another runtime

### Benchmark Results

Benchmark reports are especially useful when they include:

- The raw artifact directory or attached JSON/log files
- The exact command used to run the benchmark
- AX Engine version or commit
- Host details: macOS version, chip, memory, power state, and thermal notes
- Model ID, local artifact path, quantization, and tokenizer source
- Prompt tokens, generated tokens, repetitions, warmup/cooldown, and seed
- Rows for direct AX mode and AX n-gram mode when both are relevant
- Matching `mlx_lm.benchmark` baseline results when comparing repo-owned MLX
  performance
- Any `mlx_swift_lm` or `llama.cpp` comparison rows you want maintainers to
  consider

For the current benchmark contract and result interpretation, start with
`docs/BENCHMARKS.md`. Community benchmark artifacts can follow the layout
described in `benchmarks/community-results/README.md`.

## Performance Change Policy

Performance work is evidence-first. Do not send isolated speed patches without a
shared issue and validation plan. Runtime changes must preserve correctness,
route identity, direct AX baseline behavior, default n-gram behavior,
long-context behavior, server/SDK contracts, memory behavior, and public
benchmark reproducibility.

Maintainers may still use community reports to create focused internal patches or
invite a narrow PR, but the default public contribution path is to submit the
issue, wishlist, benchmark artifact, or reproduction evidence first.

## Local Reproduction

These commands are useful for maintainers and for invited changes:

- Rust stable toolchain with Rust 1.85 or newer
- Python 3.10+ (for Python bindings, optional)
- macOS 14 (Sonoma) or later with Apple M2 Max-or-newer Silicon and 32 GB RAM minimum (for full runtime testing)

```bash
cargo test --quiet
cargo clippy --all-targets --all-features -- -D warnings
bash scripts/check-bench-preview.sh
bash scripts/check-bench-doctor.sh
bash scripts/check-bench-mlx.sh
bash scripts/check-bench-replay.sh
bash scripts/check-metal-kernel-contract.sh
bash scripts/check-server-preview.sh
bash scripts/check-python-preview.sh
```

## License

By submitting issues, benchmark artifacts, reproduction details, documentation
suggestions, or invited changes, you agree that maintainers may use that material
in AX Engine documentation, tests, benchmarks, and project work under the MIT
License unless you clearly state otherwise.

## Contact

- Website: https://automatosx.com
- Discord: https://discord.com/invite/cTavsMgu
- Email: enquiry@defai.digital
- Company: DEFAI Private Limited (https://defai.digital)
