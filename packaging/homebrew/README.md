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
| GitHub release tarball | same as release builder | ships `@rpath/libmlx.dylib` + builder-host rpath |
| Homebrew install | tap `defai-digital/ax-engine/mlx` | formula rewrites load commands at install time |

Do **not** bake `/opt/homebrew/opt/mlx` into the notarized release archive.
Release builds intentionally track pip MLX for performance parity with
`mlx-lm`. Homebrew is the only place that should re-point prebuilt binaries
at formula-owned libraries (`MachO::Tools.change_install_name` + ad-hoc
`codesign`).

## Metal Toolchain preflight

The tap's `Formula/mlx.rb` (no mirror here) declares a custom
`MetalToolchainRequirement` that runs `xcrun metal --version` before the
source build and aborts with `xcodebuild -downloadComponent metalToolchain`
instructions when it fails. Since Xcode 26 the Metal Toolchain is a separate
download; without the preflight the mlx kernel build dies mid-compile with an
opaque error (issue #68). Keep the requirement when touching the tap formula.

## Required install markers

`scripts/brew-release.sh` and `.github/workflows/brew-release.yml` fail if the
tap formula is missing:

- `relink_release_binaries_to_tap_mlx!`
- `change_install_name`
- `libmlx.dylib`
- `defai-digital/ax-engine/mlx`
- `codesign`

When changing install logic, update both this mirror and the tap formula.
