# Release notes

**GitHub Releases are the only public changelog** for AX Engine:

[https://github.com/defai-digital/ax-engine/releases](https://github.com/defai-digital/ax-engine/releases)

Do **not** keep a parallel `CHANGELOG.md` (or per-version note copies that
restate published releases). That double source drifts: recent tags shipped
with empty auto-generated bodies while the file claimed full Keep-a-Changelog
sections.

## What lives where

| Surface | Role |
| --- | --- |
| [GitHub Releases](https://github.com/defai-digital/ax-engine/releases) | **Source of truth** for shipped versions: narrative, install pointers, compare links |
| This directory | **Authoring aid only**: style guide + template for the next release |
| `docs/RELEASING.md` | Publish procedure (sign, notarize, parity); requires human notes |
| Commit history / PR titles | Raw detail for maintainers; not the user-facing changelog |

## Style (match strong historical notes)

Good examples on GitHub: **v6.12.0**, **v6.11.1**. Weak anti-pattern: a body
that is only `**Full Changelog**: compare/…` (v6.13.0 / v6.13.1 before
backfill).

Write for an operator who just downloaded the binary or `brew upgrade`d:

1. **Lead with one paragraph** — what this release is for (user benefit), not
   every internal ticket.
2. **Theme sections with clear headings** — downloads, performance, serving,
   packaging, fixes. Use short bullets; expand only when it changes operator
   behavior.
3. **Call out breaking or opt-in flags** explicitly (env vars, defaults, remove
   flags).
4. **Link docs and artifacts**, not internal design-doc walls.
5. **End with** `Full Changelog` compare URL (and install note if assets ship).

Prefer plain language over implementation monologues. Keep engine-internal
details (file paths, counter names) only when they help debug or configure.

## Draft the next release

```bash
# 1) Copy the template and fill it while cutting the release
cp docs/releases/TEMPLATE.md /tmp/ax-engine-notes-vX.Y.Z.md
# 2) Optionally seed with the commit range
git log --oneline vPREV..HEAD
# 3) Publish (required for real publishes)
scripts/publish-github-release.sh vX.Y.Z --notes-file /tmp/ax-engine-notes-vX.Y.Z.md
```

Backfill or edit an existing release body:

```bash
gh release edit vX.Y.Z --notes-file /tmp/ax-engine-notes-vX.Y.Z.md
```

## Template

See [TEMPLATE.md](TEMPLATE.md).
