# Homebrew packaging notes

**Homebrew is the primary end-user deploy path** for the `ax-engine` CLI,
server, and bench tools on macOS Apple Silicon. The Python wheel remains the
SDK / library channel (`import ax_engine`).

The live formula lives in
[`defai-digital/homebrew-tap`](https://github.com/defai-digital/homebrew-tap).
`Formula/ax-engine.rb` here is the canonical copy used for review and as a
reference when updating the tap.

## Linkage model

| Consumer | `libmlx` source | How it resolves |
|----------|-----------------|-----------------|
| Source / pip wheel builds | pip / venv MLX | `mlx-sys` embeds absolute LC_RPATH to that dylib (NAX-correct) |
| GitHub release tarball | pinned pip MLX from the release candidate | preserves the upstream dylib load commands, Developer ID re-signs private copies, and bundles them with the byte-identical `mlx.metallib`; binaries use `@loader_path` |
| Homebrew install | same private signed runtime from the release tarball | installs binaries to `bin/` and the private runtime to `libexec/`; binaries also carry `@loader_path/../libexec` |

Do **not** bake build-host, Python, or `/opt/homebrew` paths into a release.
Release builds intentionally track the pinned pip MLX runtime for performance
parity with `mlx-lm`. Standalone preparation carries those exact candidate
bytes into staging without changing either dylib's load commands. Production
signing then replaces the dylibs' upstream ad-hoc signatures with the AX
Developer ID signature required by hardened-runtime library validation. The
release manifest records the before-signing upstream digests and final packaged
digests separately. Homebrew must not rewrite the signed Mach-O files. The
formula's `preserve_rpath` directive prevents Homebrew's formula installer from
changing the `@rpath` dylib IDs to Cellar paths and replacing the Developer ID
signatures with ad-hoc signatures. Installing under `libexec/` alone does not
prevent that rewrite. The formula places the bundled runtime in its private
`libexec/` directory so a separately installed `mlx` formula cannot collide
with AX Engine's pinned dylibs.

The standalone archive keeps binaries and runtime files colocated for
backward-compatible direct extraction. Its two relative rpaths support both
layouts:

- `@loader_path` — direct archive extraction.
- `@loader_path/../libexec` — Homebrew `bin/` plus private `libexec/`.

`mlx.metallib` must remain next to the loaded `libmlx.dylib` in both layouts.
The release also ships `MLX-LICENSE.txt`. Homebrew users consume the
precompiled runtime and therefore do not need Xcode or the Metal Toolchain.

## Required install markers

The legacy `scripts/brew-release.sh` preview delegates to the canonical
publisher. `.github/workflows/brew-release.yml` fails if the tap formula is
missing:

- `preserve_rpath`
- `libexec.install "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"`
- `doc.install "MLX-LICENSE.txt"`

The workflow also fails if `preserve_rpath` is disabled, or if obsolete
install-time relinking or tap-local MLX dependencies return.

When changing install logic, update both this mirror and the tap formula.
