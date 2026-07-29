#!/usr/bin/env python3
"""Report version tags that lack a published GitHub release (and stuck drafts).

Tags are not releases. ``scripts/publish-github-release.sh`` pushes the tag
before creating a draft and only flips draft → published after independent
asset verification. A failed or abandoned publish therefore leaves an orphan
tag visible on /tags with no entry on /releases.

Usage (operator / post-publish):

    python3 scripts/check_github_release_parity.py
    python3 scripts/check_github_release_parity.py --repo defai-digital/ax-engine
    python3 scripts/check_github_release_parity.py --strict   # also fail on drafts

Exit codes:
  0  every matching tag has a non-draft release (or only allowed exceptions)
  1  parity problems reported
  2  usage / environment error (missing gh, network, etc.)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

DEFAULT_REPO = "defai-digital/ax-engine"
# Annotated / lightweight version tags only (matches publisher contract).
TAG_PATTERN = re.compile(r"^v\d+\.\d+(\.\d+)?([.-].+)?$")


@dataclass(frozen=True)
class ReleaseInfo:
    tag_name: str
    is_draft: bool
    is_prerelease: bool


@dataclass(frozen=True)
class ParityReport:
    tags: tuple[str, ...]
    releases: tuple[ReleaseInfo, ...]
    orphan_tags: tuple[str, ...]
    draft_tags: tuple[str, ...]
    published_without_tag: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.orphan_tags and not self.draft_tags


def is_version_tag(name: str) -> bool:
    return bool(TAG_PATTERN.match(name))


def compute_parity(
    tags: Iterable[str],
    releases: Iterable[ReleaseInfo],
) -> ParityReport:
    version_tags = tuple(sorted({t for t in tags if is_version_tag(t)}, key=_version_key))
    release_list = tuple(releases)
    by_tag = {r.tag_name: r for r in release_list}

    orphan = tuple(t for t in version_tags if t not in by_tag)
    drafts = tuple(
        sorted(
            (r.tag_name for r in release_list if r.is_draft and is_version_tag(r.tag_name)),
            key=_version_key,
        )
    )
    published_without_tag = tuple(
        sorted(
            (
                r.tag_name
                for r in release_list
                if not r.is_draft
                and is_version_tag(r.tag_name)
                and r.tag_name not in set(version_tags)
            ),
            key=_version_key,
        )
    )
    return ParityReport(
        tags=version_tags,
        releases=release_list,
        orphan_tags=orphan,
        draft_tags=drafts,
        published_without_tag=published_without_tag,
    )


def _version_key(tag: str) -> tuple:
    """Sort key for vX.Y.Z(-suffix) tags; non-semver falls back to string."""
    body = tag[1:] if tag.startswith("v") else tag
    main, _, suffix = body.partition("-")
    parts = main.split(".")
    nums: list[object] = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(p)
    return (tuple(nums), suffix)


def parse_ls_remote_tags(stdout: str) -> list[str]:
    tags: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # <sha>\trefs/tags/<name>  (skip peeled ^{} lines)
        try:
            _sha, ref = line.split(None, 1)
        except ValueError:
            continue
        if ref.endswith("^{}"):
            continue
        prefix = "refs/tags/"
        if not ref.startswith(prefix):
            continue
        tags.append(ref[len(prefix) :])
    return tags


def parse_gh_release_json(payload: str | Sequence[dict]) -> list[ReleaseInfo]:
    data = json.loads(payload) if isinstance(payload, str) else list(payload)
    out: list[ReleaseInfo] = []
    for item in data:
        out.append(
            ReleaseInfo(
                tag_name=str(item["tagName"]),
                is_draft=bool(item.get("isDraft", False)),
                is_prerelease=bool(item.get("isPrerelease", False)),
            )
        )
    return out


def _run(cmd: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(cmd),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"required command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"command failed ({exc.returncode}): {' '.join(cmd)}"
            + (f"\n{detail}" if detail else "")
        ) from exc
    return completed.stdout


def fetch_remote_tags(repo: str, remote: str | None = None) -> list[str]:
    if remote:
        stdout = _run(["git", "ls-remote", "--tags", remote])
    else:
        # Prefer origin; fall back to the GitHub HTTPS URL for the repo.
        try:
            stdout = _run(["git", "ls-remote", "--tags", "origin"])
        except RuntimeError:
            stdout = _run(
                ["git", "ls-remote", "--tags", f"https://github.com/{repo}.git"]
            )
    return parse_ls_remote_tags(stdout)


def fetch_releases(repo: str, limit: int = 500) -> list[ReleaseInfo]:
    stdout = _run(
        [
            "gh",
            "release",
            "list",
            "--repo",
            repo,
            "--limit",
            str(limit),
            "--json",
            "tagName,isDraft,isPrerelease",
        ]
    )
    return parse_gh_release_json(stdout)


def format_report(report: ParityReport, *, strict_drafts: bool) -> str:
    lines: list[str] = [
        f"version tags: {len(report.tags)}",
        f"releases:     {len(report.releases)} "
        f"({sum(1 for r in report.releases if not r.is_draft)} published, "
        f"{sum(1 for r in report.releases if r.is_draft)} draft)",
    ]
    if report.orphan_tags:
        lines.append("orphan tags (tag exists, no GitHub release):")
        for tag in report.orphan_tags:
            lines.append(f"  - {tag}")
    else:
        lines.append("orphan tags: none")

    if report.draft_tags:
        label = "stuck drafts" if strict_drafts else "draft releases (warn)"
        lines.append(f"{label}:")
        for tag in report.draft_tags:
            lines.append(f"  - {tag}")
    else:
        lines.append("draft releases: none")

    if report.published_without_tag:
        lines.append("published releases without a matching remote tag:")
        for tag in report.published_without_tag:
            lines.append(f"  - {tag}")

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that every version tag has a GitHub Release "
            "(detect orphan tags and stuck drafts)."
        )
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"GitHub owner/name (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--remote",
        default=None,
        help="git remote name or URL for ls-remote (default: origin)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="max releases to fetch from GitHub (default: 500)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when any draft release exists (default: warn only)",
    )
    parser.add_argument(
        "--tags-file",
        default=None,
        help="offline: path to git ls-remote --tags style output",
    )
    parser.add_argument(
        "--releases-file",
        default=None,
        help="offline: path to gh release list --json tagName,isDraft,isPrerelease",
    )
    args = parser.parse_args(argv)

    try:
        if args.tags_file:
            tags = parse_ls_remote_tags(
                pathlib.Path(args.tags_file).read_text(encoding="utf-8")
            )
        else:
            tags = fetch_remote_tags(args.repo, remote=args.remote)

        if args.releases_file:
            releases = parse_gh_release_json(
                pathlib.Path(args.releases_file).read_text(encoding="utf-8")
            )
        else:
            releases = fetch_releases(args.repo, limit=args.limit)
    except (OSError, RuntimeError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = compute_parity(tags, releases)
    sys.stdout.write(format_report(report, strict_drafts=args.strict))

    if report.orphan_tags:
        return 1
    if args.strict and report.draft_tags:
        return 1
    if report.draft_tags:
        print(
            "note: draft releases present; re-run with --strict to fail, "
            "or finish/delete them after publish verification.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
