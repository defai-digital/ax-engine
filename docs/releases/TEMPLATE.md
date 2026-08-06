## What's New in vX.Y.Z

One short paragraph: the user-facing theme of this release (speed, downloads,
serving, packaging, …). Mention hardware/OS only if the release changes
requirements.

### Section title (operator benefit)

- Bullet that states what changed and why it matters
- Prefer verbs: download, serve, decode, fail closed, opt-in
- Name flags/env vars when operators must act: `` `AX_…` ``, `` `--flag` ``

### Fixes

- High-impact fixes first (data loss, wrong models, panics, regressions)
- Skip pure refactors unless they change behavior

### Install

Signed macOS arm64 standalone archive is attached when this is a full asset
release. Prefer Homebrew or the documented install path so dylibs, metallib,
and notarization stay intact.

```bash
# example after download
shasum -a 256 -c ax-engine-vX.Y.Z-macos-arm64.tar.gz.sha256
```

See also: `docs/GETTING-STARTED.md`, `docs/RELEASING.md`, and the
[releases page](https://github.com/defai-digital/ax-engine/releases).

---

**Full Changelog**: https://github.com/defai-digital/ax-engine/compare/vPREV...vX.Y.Z
