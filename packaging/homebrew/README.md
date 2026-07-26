# Homebrew packaging notes

**Homebrew is the primary end-user deploy path** for the `ax-engine` CLI,
server, and bench tools on macOS Apple Silicon. The Python wheel remains the
SDK / library channel (`import ax_engine`).

The live formula lives in
[`defai-digital/homebrew-ax-engine`](https://github.com/defai-digital/homebrew-ax-engine).
`Formula/ax-engine.rb` here is the canonical copy used for review and as a
reference when updating the tap.

## Linkage model

| Consumer | `libmlx` source | How it resolves |
|----------|-----------------|-----------------|
| Source / pip wheel builds | pip / venv MLX | `mlx-sys` embeds absolute LC_RPATH to that dylib (NAX-correct) |
| GitHub release tarball | pinned pip MLX from the release candidate | bundles `libmlx.dylib`, `libjaccl.dylib`, and colocated `mlx.metallib`; binaries use `@loader_path` |
| Homebrew install | same signed runtime from the release tarball | installs binaries to `bin/` and the private runtime to `libexec/`; binaries also carry `@loader_path/../libexec` |

Do **not** bake build-host, Python, or `/opt/homebrew` paths into a release.
Release builds intentionally track the pinned pip MLX bytes for performance
parity with `mlx-lm`; the immutable candidate carries those exact runtime
files into the final archive. Homebrew must not rewrite the signed Mach-O
files. It places the bundled runtime in the formula-private `libexec/`
directory so a separately installed `mlx` formula cannot collide with AX
Engine's pinned dylibs.

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

- `libexec.install "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"`
- `doc.install "MLX-LICENSE.txt"`

The workflow also fails if obsolete install-time relinking or tap-local MLX
dependencies return.

When changing install logic, update both this mirror and the tap formula.
