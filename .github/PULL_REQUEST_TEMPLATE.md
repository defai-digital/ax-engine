## Related issue

AX Engine generally does not accept unsolicited code PRs (see
[CONTRIBUTING.md](../CONTRIBUTING.md)) — this is especially true for runtime,
scheduler, model graph, MLX kernel, KV/cache, n-gram, quantization, and
serving changes. If this PR wasn't invited by a maintainer after an issue
discussion, please open an issue first instead.

Closes #

## What changed and why

## Local reproduction

- [ ] `cargo fmt --check`
- [ ] `cargo test --quiet --no-fail-fast`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] Relevant `scripts/check-*.sh` gates for the area touched (see
      CONTRIBUTING.md's "Local Reproduction" section)

## Performance-sensitive change?

If this touches runtime, scheduler, model graphs, MLX kernels, KV/cache,
n-gram, or quantization: attach raw benchmark artifacts (not just summary
numbers) per CONTRIBUTING.md's "Performance Change Policy".
