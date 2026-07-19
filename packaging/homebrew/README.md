# Homebrew packaging notes

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

## Required install markers

`scripts/brew-release.sh` and `.github/workflows/brew-release.yml` fail if the
tap formula is missing:

- `relink_release_binaries_to_tap_mlx!`
- `change_install_name`
- `libmlx.dylib`
- `defai-digital/ax-engine/mlx`
- `codesign`

When changing install logic, update both this mirror and the tap formula.
