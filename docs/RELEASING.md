# Release Workflow

AX Engine builds and validates public runtime artifacts on macOS 26 Apple
Silicon. Linux jobs are limited to artifact transfer, PyPI upload, Homebrew
formula metadata, supply-chain checks, and CI aggregation; they do not build or
execute AX Engine.

## Publish a release

Commit and push the synchronized version change, then wait for the exact commit
to pass the `CI` workflow. Run the canonical publisher from a clean checkout:

```bash
# Draft notes first (required for real publishes — not empty gh --generate-notes)
cp docs/releases/TEMPLATE.md /tmp/ax-engine-notes-v6.9.0.md
# edit /tmp/ax-engine-notes-v6.9.0.md — see docs/releases/README.md

scripts/publish-github-release.sh v6.9.0 \
  --notes-file /tmp/ax-engine-notes-v6.9.0.md
```

**Release notes policy:** GitHub Releases are the sole public changelog. Do not
reintroduce a root `CHANGELOG.md` that duplicates release bodies. Style guide
and template: [docs/releases/](releases/README.md).

The publisher performs this sequence:

1. Verify the tag/version contract, clean worktree, and successful `CI` run for
   the exact 40-character source commit.
2. Reuse an unexpired release candidate for that commit or dispatch
   `.github/workflows/release-candidate.yml` and wait for it.
3. Verify the candidate manifest and SHA-256 digest of every standalone binary,
   the pinned `libmlx.dylib` / `libjaccl.dylib`, `mlx.metallib`, and the MLX
   license. Runtime acquisition also requires matching pinned `mlx` and
   `mlx-metal` arm64 wheel tags and verifies each dylib's `minos` against that
   wheel platform tag; this is an artifact-integrity check, not a NAX feature
   proxy.
4. Assemble a relocatable payload, rewriting only the AX binaries to remove
   builder-host rpaths and install `@loader_path` plus
   `@loader_path/../libexec`. The pinned-wheel MLX dylibs and metallib must
   remain byte-for-byte identical through this relocation step. The publisher
   then Developer ID re-signs the private dylib copies before the binaries so
   hardened-runtime library validation retains the same-Team-ID boundary. It
   verifies that signing preserved both dylibs' install names, dependencies,
   and rpaths, and that `mlx.metallib` remains byte-identical. A
   clean-environment `ax-engine doctor` must report ready on the release host.
   Real publishes notarize the complete payload; dry-runs use ad-hoc signatures
   and skip notarization.
5. Push the tag, create the GitHub release as a **draft**, upload assets, then
   independently re-download and verify checksum / minisign / runtime digests /
   rpaths / clean-environment startup / codesign / notarization on the uploaded
   bytes before flipping draft → published.
6. Dispatch `brew-release.yml` only after the release is published and verified.
   Homebrew refuses draft tags. That workflow updates formula metadata only and
   installs the same signed runtime to the formula's private `libexec/`
   directory. It must not rewrite or ad-hoc re-sign the binaries. See
   `packaging/homebrew/README.md`.
7. Let the tag-triggered PyPI workflow promote the exact-SHA candidate wheel.
   If no candidate exists, that workflow fails over to the original macOS wheel
   build and smoke-test path.

This promotes the verified candidate without recompiling it; Apple signing and
notarization are required for real publishes (fail-closed). The
`ax-engine-server` binary is built with `--profile release-server`
(`panic = "unwind"`) so generation-worker panic containment works; bench/CLI
keep plain `--release`. Re-uploading to an existing draft requires
`--clobber-assets`.

The archive keeps binaries, `libmlx.dylib`, `libjaccl.dylib`, and
`mlx.metallib` colocated so direct extraction remains backward-compatible.
`mlx.metallib` and both dylibs must come from the exact MLX version in
`mlx.version`; do not substitute Homebrew MLX during publication. The pin
accepts two forms: a wheel semver (`0.32.1`, installed via pip) or an admitted
source build (`git:<sha>@<version>`, whose build recipe and qmm-admission
evidence live in `docs/performance/mlx-main-admission-2026-07-28.md`); either
way the shipped dylibs must come from exactly that runtime. For the current
wheel pin, the release manifest records both the upstream pinned-wheel digests
and the final packaged digests. The dylibs are therefore private signed
derivations of the upstream files, not byte-for-byte claims after signing.

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

## Tag vs release parity

Git tags and GitHub Releases are separate objects. The publisher **pushes the
tag first**, creates a **draft** release, uploads assets, verifies the uploaded
bytes, then flips draft → published. If that sequence stops mid-flight you can
end up with:

- an **orphan tag** (visible on `/tags`, missing from `/releases`), or
- a **stuck draft** (assets uploaded but never published; drafts are hidden
  from most visitors on the public Releases page).

After every publish — and whenever the tags page and Releases page disagree —
run:

```bash
python3 scripts/check_github_release_parity.py
python3 scripts/check_github_release_parity.py --strict   # also fail on drafts
```

### Recover orphan tags

**Notes-only backfill** (when re-building notarized macOS assets is not
worthwhile; state that clearly in the body):

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file path/to/notes.md --verify-tag
```

Write operator-facing notes (required for real publishes). Use
[docs/releases/TEMPLATE.md](releases/TEMPLATE.md) and the style guide in
[docs/releases/README.md](releases/README.md). Pass them with
`--notes-file /path/to/notes.md`. Always include a compare link:
`https://github.com/defai-digital/ax-engine/compare/vPREV...vX.Y.Z`.

Do **not** maintain a parallel root `CHANGELOG.md` — GitHub Releases are the
only public changelog.

**Full asset publish** for a tag that already exists at the intended commit:

```bash
scripts/publish-github-release.sh vX.Y.Z --skip-tag-push
```

That path still requires a green exact-SHA `CI` run, signed/notarized payload
verification, and will refuse to replace an already-published release.

### Recover stuck drafts

Finish after verification, or delete and re-run the publisher with
`--clobber-assets` if the draft assets are wrong:

```bash
gh release edit vX.Y.Z --draft=false          # only after assets verify
# or
gh release delete vX.Y.Z                      # then re-publish
```

Publishing an **older** draft (or creating a backfilled release) can steal
GitHub's **Latest** badge away from the current public release. After any
historical publish, re-pin Latest explicitly:

```bash
gh release edit v6.11.1 --latest   # replace with the real current release tag
gh api repos/defai-digital/ax-engine/releases/latest --jq .tag_name
```

Do not push version tags outside `scripts/publish-github-release.sh` unless you
immediately create the matching GitHub release.

### Notarization evidence and Gatekeeper propagation

The publisher captures the JSON response from `notarytool submit --wait`, then
downloads the corresponding notarization log. Publication fails unless the log
reports `Accepted`, has no issues, and its `ticketContents` contains the exact
arm64 CDHash for every signed release Mach-O. The same check is repeated against
the independently downloaded GitHub archive, and the submission ID is recorded
in the signed release manifest.

This is intentionally independent of Gatekeeper's ticket CDN. Standalone
command-line binaries cannot carry a stapled ticket, and a newly accepted ticket
can take time to become visible through `codesign -R=notarized` or `spctl`.
Treating that propagation delay as a notarization failure can strand an
otherwise valid draft. The Apple notarization log is the fail-closed publication
evidence; a clean-machine download and online Gatekeeper check remain required
post-publication deployment tests.

For a draft created by an older publisher, fetch the official log and compare
its ticket CDHashes with the uploaded payload before publishing:

```bash
xcrun notarytool log <submission-id> --keychain-profile ax-notary
codesign -dv --verbose=4 payload/ax-engine
codesign --verify --strict payload/ax-engine
```

Do not publish when the log is missing an image, reports any issue, or contains
a different CDHash. After publication, verify the downloaded artifact on a
separate Mac with:

```bash
codesign --verify --strict --check-notarization \
  -R=notarized /path/to/ax-engine
```

## Build-cache policy

The Rust toolchain is pinned in `rust-toolchain.toml`, and release-candidate and
PyPI fallback builds share the `release-macos-arm64` Rust cache key. Candidate
builds dispatched from `main` populate the default-branch cache scope so later
tag workflows can restore it.

Release optimization remains unchanged: public binaries retain fat LTO and one
codegen unit. Any future move to thin LTO must pass the benchmark convention of
two warmups plus five measured runs with the median reported.
