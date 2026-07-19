# Release Workflow

AX Engine builds and validates public runtime artifacts on macOS 26 Apple
Silicon. Linux jobs are limited to artifact transfer, PyPI upload, Homebrew
formula metadata, supply-chain checks, and CI aggregation; they do not build or
execute AX Engine.

## Publish a release

Commit and push the synchronized version change, then wait for the exact commit
to pass the `CI` workflow. Run the canonical publisher from a clean checkout:

```bash
scripts/publish-github-release.sh v6.9.0
```

The publisher performs this sequence:

1. Verify the tag/version contract, clean worktree, and successful `CI` run for
   the exact 40-character source commit.
2. Reuse an unexpired release candidate for that commit or dispatch
   `.github/workflows/release-candidate.yml` and wait for it.
3. Verify the candidate manifest and SHA-256 digest of every standalone binary.
4. Developer ID sign and notarize the binaries locally (required for real
   publishes; dry-run may skip notarization), then package and minisign the
   GitHub release assets.
5. Push the tag, create the GitHub release as a **draft**, upload assets, then
   independently re-download and verify checksum / minisign / codesign /
   notarization on the uploaded bytes before flipping draft → published.
6. Dispatch `brew-release.yml` only after the release is published and verified.
   Homebrew refuses draft tags.
7. Let the tag-triggered PyPI workflow promote the exact-SHA candidate wheel.
   If no candidate exists, that workflow fails over to the original macOS wheel
   build and smoke-test path.

This promotes the verified candidate without recompiling it; Apple signing and
notarization are required for real publishes (fail-closed). The
`ax-engine-server` binary is built with `--profile release-server`
(`panic = "unwind"`) so generation-worker panic containment works; bench/CLI
keep plain `--release`. Re-uploading to an existing draft requires
`--clobber-assets`.

## Operator options

- `--dry-run` runs local gates, build, packaging, and minisign without changing
  GitHub or submitting notarization.
- `--full-local-checks` repeats all local gates in addition to requiring green
  GitHub CI for a real publish.
- `--local-build` bypasses candidate reuse and builds the GitHub binaries
  locally. PyPI may need its macOS fallback build after the tag is pushed.
- `--skip-build` accepts existing `target/release` binaries. Use only when their
  provenance has been verified separately.
- `--skip-brew-dispatch` publishes the GitHub assets without updating the tap.

`--skip-checks` skips local gates only. It never bypasses the exact-SHA GitHub
CI requirement for a real publish.

## Build-cache policy

The Rust toolchain is pinned in `rust-toolchain.toml`, and release-candidate and
PyPI fallback builds share the `release-macos-arm64` Rust cache key. Candidate
builds dispatched from `main` populate the default-branch cache scope so later
tag workflows can restore it.

Release optimization remains unchanged: public binaries retain fat LTO and one
codegen unit. Any future move to thin LTO must pass the benchmark convention of
two warmups plus five measured runs with the median reported.
