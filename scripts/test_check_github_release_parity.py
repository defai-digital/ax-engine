#!/usr/bin/env python3
"""Unit tests for scripts/check_github_release_parity.py."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import check_github_release_parity as parity  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_parse_ls_remote_skips_peeled(self) -> None:
        raw = """\
abc123\trefs/tags/v6.8.2
def456\trefs/tags/v6.8.2^{}
aaa111\trefs/tags/v6.8.1
not-a-tag\trefs/heads/main
"""
        self.assertEqual(
            parity.parse_ls_remote_tags(raw),
            ["v6.8.2", "v6.8.1"],
        )

    def test_parse_releases_json(self) -> None:
        payload = json.dumps(
            [
                {"tagName": "v6.8.2", "isDraft": False, "isPrerelease": False},
                {"tagName": "v6.4.5", "isDraft": True, "isPrerelease": False},
            ]
        )
        releases = parity.parse_gh_release_json(payload)
        self.assertEqual(len(releases), 2)
        self.assertFalse(releases[0].is_draft)
        self.assertTrue(releases[1].is_draft)


class ParityTests(unittest.TestCase):
    def test_orphan_and_draft_detection(self) -> None:
        tags = ["v6.8.1", "v6.8.2", "v5.1.7", "not-a-version"]
        releases = [
            parity.ReleaseInfo("v6.8.1", is_draft=False, is_prerelease=False),
            parity.ReleaseInfo("v6.4.5", is_draft=True, is_prerelease=False),
        ]
        report = parity.compute_parity(tags, releases)
        self.assertEqual(report.orphan_tags, ("v5.1.7", "v6.8.2"))
        self.assertEqual(report.draft_tags, ("v6.4.5",))
        self.assertFalse(report.ok)

    def test_clean_parity(self) -> None:
        tags = ["v6.8.1", "v6.8.2"]
        releases = [
            parity.ReleaseInfo("v6.8.1", is_draft=False, is_prerelease=False),
            parity.ReleaseInfo("v6.8.2", is_draft=False, is_prerelease=False),
        ]
        report = parity.compute_parity(tags, releases)
        self.assertEqual(report.orphan_tags, ())
        self.assertEqual(report.draft_tags, ())
        self.assertTrue(report.ok)

    def test_published_without_tag(self) -> None:
        report = parity.compute_parity(
            ["v1.0"],
            [parity.ReleaseInfo("v1.1", is_draft=False, is_prerelease=False)],
        )
        self.assertEqual(report.published_without_tag, ("v1.1",))

    def test_exact_allowed_orphan_does_not_hide_other_orphans(self) -> None:
        report = parity.compute_parity(
            ["v1.0", "v1.1", "v1.2"],
            [parity.ReleaseInfo("v1.0", is_draft=False, is_prerelease=False)],
            allowed_orphans=["v1.1"],
        )
        self.assertEqual(report.allowed_orphan_tags, ("v1.1",))
        self.assertEqual(report.orphan_tags, ("v1.2",))
        self.assertFalse(report.ok)


class CliTests(unittest.TestCase):
    def test_offline_cli_fails_on_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            tags = root / "tags.txt"
            tags.write_text(
                "abc\trefs/tags/v1.0\n"
                "def\trefs/tags/v1.1\n"
                "ghi\trefs/tags/v1.1^{}\n",
                encoding="utf-8",
            )
            releases = root / "releases.json"
            releases.write_text(
                json.dumps(
                    [
                        {
                            "tagName": "v1.0",
                            "isDraft": False,
                            "isPrerelease": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            code = parity.main(
                [
                    "--tags-file",
                    str(tags),
                    "--releases-file",
                    str(releases),
                ]
            )
            self.assertEqual(code, 1)

    def test_offline_cli_accepts_only_the_named_historical_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            tags = root / "tags.txt"
            tags.write_text(
                "abc\trefs/tags/v1.0\n" "def\trefs/tags/v1.1\n",
                encoding="utf-8",
            )
            releases = root / "releases.json"
            releases.write_text(
                json.dumps(
                    [
                        {
                            "tagName": "v1.0",
                            "isDraft": False,
                            "isPrerelease": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            code = parity.main(
                [
                    "--tags-file",
                    str(tags),
                    "--releases-file",
                    str(releases),
                    "--strict",
                    "--allow-orphan",
                    "v1.1",
                ]
            )
            self.assertEqual(code, 0)

    def test_offline_cli_strict_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            tags = root / "tags.txt"
            tags.write_text("abc\trefs/tags/v1.0\n", encoding="utf-8")
            releases = root / "releases.json"
            releases.write_text(
                json.dumps(
                    [
                        {
                            "tagName": "v1.0",
                            "isDraft": True,
                            "isPrerelease": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            warn_code = parity.main(
                ["--tags-file", str(tags), "--releases-file", str(releases)]
            )
            self.assertEqual(warn_code, 0)
            strict_code = parity.main(
                [
                    "--tags-file",
                    str(tags),
                    "--releases-file",
                    str(releases),
                    "--strict",
                ]
            )
            self.assertEqual(strict_code, 1)


if __name__ == "__main__":
    unittest.main()
