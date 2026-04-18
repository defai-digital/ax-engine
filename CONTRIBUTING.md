# Contributing to AX Engine

AX Engine welcomes contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a branch for your change
4. Make your changes
5. Run the tests: `cargo test --workspace`
6. Run clippy: `cargo clippy --all-targets -- -D warnings`
7. Submit a pull request

## Build Requirements

- Rust stable toolchain
- Python 3.10+ (for Python bindings, optional)
- macOS with Apple M4-or-newer Silicon (for full runtime testing)

## Running Tests

```bash
# All workspace tests
cargo test --workspace

# Individual crates
cargo test -p ax-engine-core
cargo test -p ax-bench
cargo test -p ax-engine-sdk
cargo test -p ax-engine-server
cargo test -p ax-engine-py

# E2E smoke checks
bash scripts/check-bench-preview.sh
bash scripts/check-bench-doctor.sh
bash scripts/check-metal-kernel-contract.sh
bash scripts/check-server-preview.sh
bash scripts/check-python-preview.sh
```

## Code Style

- Follow existing patterns in the codebase
- `unsafe_code = "forbid"` is enforced workspace-wide
- All clippy warnings must be resolved before merging
- Add tests for new functionality

## Areas for Contribution

- Metal kernel implementation and optimization
- Model family support (Qwen, Gemma, and beyond)
- Benchmark workload coverage
- Python and server ergonomics
- Documentation improvements

## Reporting Issues

Please open an issue on the repository with:

- A clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.

## Contact

- Website: https://automatosx.com
- Discord: https://discord.com/invite/cTavsMgu
- Email: enquiry@defai.digital
- Company: DEFAI Private Limited (https://defai.digital)
